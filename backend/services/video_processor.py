"""
Video processing service with hardware acceleration
"""

import asyncio
import time
from typing import Optional, Tuple, Any
import cv2
import numpy as np
from threading import Thread, Lock
from queue import Queue, Empty
import structlog

from backend.core.config import Settings

logger = structlog.get_logger()


class VideoProcessor:
    """
    Hardware-accelerated video processing for Jetson
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.camera = None
        self.capture_thread = None
        self.frame_queue = Queue(maxsize=settings.MAX_FRAME_QUEUE_SIZE)
        self.running = False
        self.frame_count = 0
        self.last_frame = None
        self.frame_lock = Lock()
        
        # Performance tracking
        self.capture_fps = 0
        self.last_fps_time = time.time()
        self.fps_frame_count = 0
        
        # Video writer for recording
        self.video_writer = None
        self.recording = False
        
    async def start(self):
        """
        Start video capture
        """
        try:
            # Initialize camera
            if not self._init_camera():
                raise RuntimeError("Failed to initialize camera")
            
            # Start capture thread
            self.running = True
            self.capture_thread = Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            # Start recording if enabled
            if self.settings.ENVIRONMENT == "production":
                self._start_recording()
            
            logger.info("Video processor started", 
                       resolution=self.settings.VIDEO_RESOLUTION,
                       fps=self.settings.VIDEO_FPS)
            
        except Exception as e:
            logger.error(f"Failed to start video processor", error=str(e))
            raise
    
    async def stop(self):
        """
        Stop video capture
        """
        self.running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        
        if self.camera:
            self.camera.release()
        
        if self.video_writer:
            self.video_writer.release()
        
        logger.info("Video processor stopped")
    
    def _init_camera(self) -> bool:
        """
        Initialize camera with optimal settings for Jetson
        """
        try:
            # Determine camera source from settings
            camera_source = getattr(self.settings, 'CAMERA_DEVICE', '/dev/video0')
            
            # Convert device path to index if needed
            if isinstance(camera_source, str) and camera_source.startswith('/dev/video'):
                camera_index = int(camera_source.replace('/dev/video', ''))
                camera_source = camera_index
            
            # Check for CSI camera on Jetson
            if self.settings.JETSON_MODE and camera_source == "csi://0":
                # GStreamer pipeline for CSI camera with hardware acceleration
                pipeline = self._get_csi_pipeline()
                self.camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            else:
                # USB camera
                logger.info(f"Initializing USB camera at device {camera_source}")
                self.camera = cv2.VideoCapture(camera_source)
                
                if not self.camera.isOpened():
                    raise RuntimeError(f"Failed to open camera at {camera_source}")
                
                # Set camera properties
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.VIDEO_RESOLUTION[0])
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.VIDEO_RESOLUTION[1])
                self.camera.set(cv2.CAP_PROP_FPS, self.settings.VIDEO_FPS)
                
                # Set buffer size to reduce latency
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Use MJPEG for better performance
                self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            
            # Test camera
            ret, frame = self.camera.read()
            if not ret or frame is None:
                raise RuntimeError(f"Cannot read from camera at {camera_source}")
            
            logger.info("Camera initialized successfully",
                       camera_source=camera_source,
                       actual_width=frame.shape[1],
                       actual_height=frame.shape[0])
            
            return True
            
        except Exception as e:
            logger.error(f"Camera initialization failed", error=str(e))
            return False
    
    def _get_csi_pipeline(self) -> str:
        """
        Get GStreamer pipeline for CSI camera on Jetson
        """
        width, height = self.settings.VIDEO_RESOLUTION
        fps = self.settings.VIDEO_FPS
        
        if self.settings.USE_NVMM_BUFFERS:
            # Hardware accelerated pipeline with NVMM buffers
            return (
                f"nvarguscamerasrc ! "
                f"video/x-raw(memory:NVMM), "
                f"width={width}, height={height}, "
                f"format=NV12, framerate={fps}/1 ! "
                f"nvvidconv flip-method=0 ! "
                f"video/x-raw, width={width}, height={height}, format=BGRx ! "
                f"videoconvert ! "
                f"video/x-raw, format=BGR ! appsink"
            )
        else:
            # Standard pipeline
            return (
                f"v4l2src device=/dev/video0 ! "
                f"video/x-raw, width={width}, height={height}, framerate={fps}/1 ! "
                f"videoconvert ! appsink"
            )
    
    def _capture_loop(self):
        """
        Main capture loop running in separate thread
        """
        consecutive_failures = 0
        
        while self.running:
            try:
                ret, frame = self.camera.read()
                
                if ret and frame is not None:
                    # Reset failure counter
                    consecutive_failures = 0
                    
                    # Update last frame
                    with self.frame_lock:
                        self.last_frame = frame.copy()
                    
                    # Add to queue if not full
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame)
                    else:
                        # Skip frame if queue is full
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put(frame)
                        except Empty:
                            pass
                    
                    # Record if enabled
                    if self.recording and self.video_writer:
                        self.video_writer.write(frame)
                    
                    # Update FPS
                    self._update_fps()
                    
                else:
                    consecutive_failures += 1
                    if consecutive_failures > 30:  # 1 second at 30fps
                        logger.error("Camera read failed repeatedly, attempting reconnect")
                        self._reconnect_camera()
                        consecutive_failures = 0
                    
            except Exception as e:
                logger.error(f"Error in capture loop", error=str(e))
                time.sleep(0.1)
    
    def _reconnect_camera(self):
        """
        Attempt to reconnect to camera
        """
        try:
            if self.camera:
                self.camera.release()
            
            time.sleep(1)
            self._init_camera()
            
        except Exception as e:
            logger.error(f"Camera reconnection failed", error=str(e))
    
    def _update_fps(self):
        """
        Update FPS calculation
        """
        self.fps_frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.capture_fps = self.fps_frame_count / (current_time - self.last_fps_time)
            self.fps_frame_count = 0
            self.last_fps_time = current_time
    
    def _start_recording(self):
        """
        Start video recording
        """
        try:
            width, height = self.settings.VIDEO_RESOLUTION
            
            # Video codec - use hardware acceleration if available
            if self.settings.JETSON_MODE:
                # Use hardware encoder on Jetson
                fourcc = cv2.VideoWriter_fourcc(*'H264')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            # Create output filename with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.settings.STORAGE_PATH / "videos" / f"recording_{timestamp}.mp4"
            
            self.video_writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                self.settings.VIDEO_FPS,
                (width, height)
            )
            
            self.recording = True
            logger.info("Recording started", output=str(output_path))
            
        except Exception as e:
            logger.error(f"Failed to start recording", error=str(e))
    
    async def get_frame(self) -> Optional[np.ndarray]:
        """
        Get next frame from queue
        """
        try:
            if not self.frame_queue.empty():
                return self.frame_queue.get_nowait()
            
            # Return last frame if queue is empty
            with self.frame_lock:
                if self.last_frame is not None:
                    return self.last_frame.copy()
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting frame", error=str(e))
            return None
    
    async def get_snapshot(self) -> Optional[np.ndarray]:
        """
        Get current frame snapshot
        """
        with self.frame_lock:
            if self.last_frame is not None:
                return self.last_frame.copy()
        return None
    
    def get_stats(self) -> dict:
        """
        Get video processor statistics
        """
        return {
            "capture_fps": round(self.capture_fps, 1),
            "queue_size": self.frame_queue.qsize(),
            "max_queue_size": self.frame_queue.maxsize,
            "recording": self.recording,
            "frame_count": self.frame_count
        }
    
    def start_recording(self):
        """
        Start recording video
        """
        if not self.recording:
            self._start_recording()
    
    def stop_recording(self):
        """
        Stop recording video
        """
        if self.recording and self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
            logger.info("Recording stopped")