"""
Health check and monitoring endpoints
"""

from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from backend.core.database import health_check as db_health
from backend.core.redis_client import health_check as redis_health
from backend.core.config import settings

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
            "camera_connected": True,   # TODO: Check camera status
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