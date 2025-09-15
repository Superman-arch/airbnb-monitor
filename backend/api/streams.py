"""
Video streaming API endpoints
"""

import asyncio
import base64
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse, Response
from starlette.websockets import WebSocketState

from backend.core.config import settings
from backend.core.redis_client import Cache, CacheKeys

router = APIRouter()


@router.websocket("/ws/video")
async def video_stream_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video streaming
    """
    from backend.main import monitoring_service, ws_manager
    
    if not monitoring_service:
        await websocket.close(code=1003, reason="Service unavailable")
        return
    
    client_id = None
    
    try:
        # Accept connection
        client_id = await ws_manager.connect(websocket, channel="video")
        
        # Send initial message
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "message": "Video stream connected",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Stream frames
        while True:
            if websocket.client_state != WebSocketState.CONNECTED:
                break
            
            # Get current frame with overlays
            frame = await monitoring_service.get_current_frame()
            
            if frame is not None:
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Send frame
                await websocket.send_json({
                    "type": "frame",
                    "data": frame_base64,
                    "timestamp": datetime.utcnow().isoformat(),
                    "frame_number": monitoring_service.frame_count
                })
            
            # Control frame rate
            await asyncio.sleep(1.0 / 30)  # 30 FPS max
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error", error=str(e))
    finally:
        if client_id and ws_manager:
            await ws_manager.disconnect(client_id, "video")


@router.get("/video/live")
async def video_stream_mjpeg():
    """
    MJPEG video stream endpoint
    """
    from backend.main import monitoring_service
    
    if not monitoring_service:
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    async def generate():
        """Generate MJPEG frames"""
        while True:
            frame = await monitoring_service.get_current_frame()
            
            if frame is not None:
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()
                
                # Yield MJPEG frame
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                )
            
            await asyncio.sleep(1.0 / 30)  # 30 FPS max
    
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/video/snapshot")
async def get_snapshot(
    quality: int = Query(85, ge=50, le=100),
    width: Optional[int] = Query(None, ge=320, le=3840),
    height: Optional[int] = Query(None, ge=240, le=2160)
):
    """
    Get current frame snapshot
    """
    from backend.main import monitoring_service
    
    if not monitoring_service:
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    # Get current frame
    frame = await monitoring_service.get_current_frame()
    
    if frame is None:
        raise HTTPException(status_code=503, detail="No frame available")
    
    # Resize if requested
    if width or height:
        current_height, current_width = frame.shape[:2]
        
        if width and not height:
            # Calculate height maintaining aspect ratio
            height = int(current_height * (width / current_width))
        elif height and not width:
            # Calculate width maintaining aspect ratio
            width = int(current_width * (height / current_height))
        
        if width and height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    
    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    
    return Response(
        content=buffer.tobytes(),
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }
    )


@router.get("/video/hls/playlist.m3u8")
async def get_hls_playlist():
    """
    HLS playlist for video streaming
    """
    # Generate HLS playlist
    playlist = """#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:1
#EXT-X-MEDIA-SEQUENCE:0
#EXT-X-PLAYLIST-TYPE:EVENT

#EXTINF:1.0,
/api/streams/video/hls/segment0.ts
#EXTINF:1.0,
/api/streams/video/hls/segment1.ts
#EXTINF:1.0,
/api/streams/video/hls/segment2.ts
"""
    
    return Response(
        content=playlist,
        media_type="application/vnd.apple.mpegurl",
        headers={
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*"
        }
    )


@router.get("/video/hls/segment{segment_id}.ts")
async def get_hls_segment(segment_id: int):
    """
    HLS video segment
    """
    # This would normally return actual video segments
    # For now, return a placeholder
    return Response(
        content=b"",
        media_type="video/mp2t",
        headers={
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*"
        }
    )


@router.post("/video/record/start")
async def start_recording():
    """
    Start video recording
    """
    from backend.main import monitoring_service
    
    if not monitoring_service:
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    if monitoring_service.video_processor:
        monitoring_service.video_processor.start_recording()
        return {"status": "recording_started", "timestamp": datetime.utcnow().isoformat()}
    
    raise HTTPException(status_code=503, detail="Video processor not available")


@router.post("/video/record/stop")
async def stop_recording():
    """
    Stop video recording
    """
    from backend.main import monitoring_service
    
    if not monitoring_service:
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    if monitoring_service.video_processor:
        monitoring_service.video_processor.stop_recording()
        return {"status": "recording_stopped", "timestamp": datetime.utcnow().isoformat()}
    
    raise HTTPException(status_code=503, detail="Video processor not available")


@router.get("/video/status")
async def get_video_status():
    """
    Get video stream status
    """
    from backend.main import monitoring_service
    
    if not monitoring_service or not monitoring_service.video_processor:
        return {
            "status": "offline",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    stats = monitoring_service.video_processor.get_stats()
    
    return {
        "status": "online",
        "capture_fps": stats.get("capture_fps", 0),
        "recording": stats.get("recording", False),
        "queue_size": stats.get("queue_size", 0),
        "frame_count": stats.get("frame_count", 0),
        "timestamp": datetime.utcnow().isoformat()
    }