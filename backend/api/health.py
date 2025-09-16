"""
Health check and monitoring endpoints
"""

from datetime import datetime
from typing import Dict, Any
import asyncio
import os
import cv2
import structlog

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from backend.core.database import health_check as db_health
from backend.core.redis_client import health_check as redis_health
from backend.core.config import settings

logger = structlog.get_logger()

router = APIRouter()


@router.get("/health", 
            response_model=Dict[str, Any],
            tags=["Health"],
            summary="System health check")
async def health_check():
    """
    Comprehensive system health check
    """
    # Check database with error handling
    try:
        db_status = await db_health()
    except:
        db_status = False
    
    # Check Redis with error handling
    try:
        redis_status = await redis_health()
    except:
        redis_status = False
    
    # Overall status
    all_healthy = db_status and redis_status
    
    health_data = {
        "status": "healthy" if all_healthy else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "services": {
            "database": "healthy" if db_status else "unhealthy",
            "redis": "healthy" if redis_status else "unhealthy",
            "monitoring": "healthy",  # Always healthy if we can respond
        },
        "checks": {
            "database_connected": db_status,
            "redis_connected": redis_status,
            "storage_available": True,  # TODO: Check actual storage
            "camera_connected": await check_camera_status()
        }
    }
    
    # Always return 200 OK with status in the JSON
    # This prevents frontend from showing connection errors
    return JSONResponse(content=health_data, status_code=status.HTTP_200_OK)


@router.get("/health/live",
            tags=["Health"],
            summary="Liveness probe")
async def liveness():
    """
    Simple liveness check for Kubernetes
    """
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}


@router.get("/health/ready",
            tags=["Health"],
            summary="Readiness probe")
async def readiness():
    """
    Readiness check for Kubernetes
    """
    # Check if all services are ready
    db_ready = await db_health()
    redis_ready = await redis_health()
    
    if db_ready and redis_ready:
        return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}
    else:
        return JSONResponse(
            content={"status": "not_ready", "timestamp": datetime.utcnow().isoformat()},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )


async def check_camera_status() -> bool:
    """
    Check if camera is accessible
    """
    try:
        from backend.main import monitoring_service
        
        if monitoring_service and monitoring_service.video_processor:
            # Check if we have recent frames
            frame = await monitoring_service.video_processor.get_snapshot()
            return frame is not None
        return False
    except:
        return False


@router.get("/camera/test",
            tags=["Health"],
            summary="Test camera connection")
async def test_camera():
    """
    Test camera connection and return detailed information
    """
    camera_info = {
        "status": "unknown",
        "devices": [],
        "primary_device": settings.CAMERA_DEVICE,
        "error": None,
        "test_results": []
    }
    
    try:
        # List available video devices
        devices_to_check = ['/dev/video0', '/dev/video1', 0, 1]
        
        for device in devices_to_check:
            device_info = {
                "device": str(device),
                "exists": False,
                "can_open": False,
                "can_read": False,
                "resolution": None,
                "fps": None
            }
            
            try:
                # Check if device exists (for /dev/video* paths)
                if isinstance(device, str) and device.startswith('/dev/video'):
                    device_info["exists"] = os.path.exists(device)
                    if not device_info["exists"]:
                        camera_info["test_results"].append(device_info)
                        continue
                    device_index = int(device.replace('/dev/video', ''))
                else:
                    device_index = device
                    device_info["exists"] = True  # Can't check integer devices
                
                # Try to open camera
                cap = cv2.VideoCapture(device_index)
                if cap.isOpened():
                    device_info["can_open"] = True
                    
                    # Try to read a frame
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        device_info["can_read"] = True
                        device_info["resolution"] = f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
                        device_info["fps"] = int(cap.get(cv2.CAP_PROP_FPS))
                        
                        # If this is the primary device, mark as available
                        if str(device) == settings.CAMERA_DEVICE or device_index == 0:
                            camera_info["devices"].append(device_info)
                            camera_info["status"] = "available"
                    
                    cap.release()
                
                camera_info["test_results"].append(device_info)
                
            except Exception as e:
                device_info["error"] = str(e)
                camera_info["test_results"].append(device_info)
        
        # Check if monitoring service has camera
        try:
            from backend.main import monitoring_service
            
            if monitoring_service and monitoring_service.video_processor:
                stats = monitoring_service.video_processor.get_stats()
                camera_info["monitoring_service"] = {
                    "active": True,
                    "fps": stats.get("capture_fps", 0),
                    "frame_count": stats.get("frame_count", 0),
                    "recording": stats.get("recording", False)
                }
                
                # Try to get a frame
                frame = await monitoring_service.video_processor.get_snapshot()
                if frame is not None:
                    camera_info["status"] = "working"
                    camera_info["current_frame_shape"] = frame.shape
            else:
                camera_info["monitoring_service"] = {"active": False}
        except Exception as e:
            camera_info["monitoring_service"] = {"active": False, "error": str(e)}
        
        # Determine overall status
        if camera_info["status"] == "unknown":
            if any(result["can_read"] for result in camera_info["test_results"]):
                camera_info["status"] = "available"
            elif any(result["can_open"] for result in camera_info["test_results"]):
                camera_info["status"] = "accessible"
            elif any(result["exists"] for result in camera_info["test_results"]):
                camera_info["status"] = "exists"
            else:
                camera_info["status"] = "not_found"
        
    except Exception as e:
        camera_info["error"] = str(e)
        camera_info["status"] = "error"
        logger.error(f"Camera test failed", error=str(e))
    
    # Return appropriate status code
    status_code = status.HTTP_200_OK if camera_info["status"] in ["working", "available"] else status.HTTP_503_SERVICE_UNAVAILABLE
    
    return JSONResponse(content=camera_info, status_code=status_code)