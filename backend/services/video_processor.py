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

import os
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
        
        # Camera initialization settings
        self.camera_init_timeout = int(os.environ.get('CAMERA_INIT_TIMEOUT', '30'))
        self.camera_retry_count = int(os.environ.get('CAMERA_RETRY_COUNT', '5'))
        self.camera_device_fallbacks = ['/dev/video0', '/dev/video1', 0, 1]  # Try multiple devices
        
    async def start(self):
        """
        Start video capture with retry logic
        """
        last_error = None
        
        for attempt in range(self.camera_retry_count):
            try:
                logger.info(f"Attempting to start video processor (attempt {attempt + 1}/{self.camera_retry_count})")
                
                # Initialize camera with timeout
                if await self._init_camera_with_timeout():
                    # Start capture thread
                    self.running = True
                    self.capture_thread = Thread(target=self._capture_loop, daemon=True)
                    self.capture_thread.start()
                    
                    # Verify camera is working
                    await asyncio.sleep(1)  # Give camera time to start
                    test_frame = await self.get_snapshot()
                    if test_frame is None:
                        raise RuntimeError("Camera started but no frames available")
                    
                    # Start recording if enabled
                    if self.settings.ENVIRONMENT == "production":
                        self._start_recording()
                    
                    logger.info("Video processor started successfully", 
                               resolution=self.settings.VIDEO_RESOLUTION,
                               fps=self.settings.VIDEO_FPS,
                               device=self.camera_source_used)
                    return
                
            except Exception as e:
                last_error = e
                logger.warning(f"Camera initialization attempt {attempt + 1} failed", error=str(e))
                
                # Clean up failed attempt
                if self.camera:
                    self.camera.release()
                    self.camera = None
                
                if attempt < self.camera_retry_count - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        # All attempts failed
        error_msg = f"Failed to start video processor after {self.camera_retry_count} attempts"
        logger.error(error_msg, last_error=str(last_error))
        raise RuntimeError(f"{error_msg}: {last_error}")
    
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
    
    async def _init_camera_with_timeout(self) -> bool:
        """
        Initialize camera with timeout
        """
        try:
            # Run camera initialization with timeout
            return await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, self._init_camera),
                timeout=self.camera_init_timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Camera initialization timed out after {self.camera_init_timeout} seconds")
            return False
        except Exception as e:
            logger.error(f"Camera initialization failed", error=str(e))
            return False
    
    def _init_camera(self) -> bool:
        """
        Initialize camera with optimal settings for Jetson
        """
        # Try primary device from settings first
        primary_device = getattr(self.settings, 'CAMERA_DEVICE', '/dev/video0')
        devices_to_try = [primary_device] + [d for d in self.camera_device_fallbacks if d != primary_device]
        
        for camera_source in devices_to_try:
            try:
                logger.info(f"Trying camera device: {camera_source}")
                
                # Check if device exists (for /dev/video* paths)
                if isinstance(camera_source, str) and camera_source.startswith('/dev/video'):
                    import os
                    if not os.path.exists(camera_source):
                        logger.debug(f"Device {camera_source} does not exist, skipping")
                        continue
                    
                    # Convert device path to index
                    camera_index = int(camera_source.replace('/dev/video', ''))
                    camera_source_to_use = camera_index
                else:
                    camera_source_to_use = camera_source
                
                # Check for CSI camera on Jetson
                if self.settings.JETSON_MODE and camera_source == "csi://0":
                    # GStreamer pipeline for CSI camera with hardware acceleration
                    pipeline = self._get_csi_pipeline()
                    self.camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                else:
                    # USB camera
                    logger.info(f"Opening camera at index/device {camera_source_to_use}")
                    self.camera = cv2.VideoCapture(camera_source_to_use)
                    
                    if not self.camera.isOpened():
                        logger.debug(f"Failed to open camera at {camera_source_to_use}")
                        self.camera = None
                        continue
                    
                    # Set camera properties for USB camera
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.VIDEO_RESOLUTION[0])
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.VIDEO_RESOLUTION[1])
                    self.camera.set(cv2.CAP_PROP_FPS, self.settings.VIDEO_FPS)
                    
                    # Set buffer size to reduce latency
                    self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # Use MJPEG for better performance
                    self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                
                # Test camera with multiple attempts
                for test_attempt in range(3):
                    ret, frame = self.camera.read()
                    if ret and frame is not None:
                        # Get actual camera properties
                        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        actual_fps = int(self.camera.get(cv2.CAP_PROP_FPS))
                        
                        logger.info("Camera initialized successfully",
                                   camera_source=camera_source,
                                   actual_width=actual_width,
                                   actual_height=actual_height,
                                   actual_fps=actual_fps,
                                   frame_shape=frame.shape)
                        
                        self.camera_source_used = camera_source
                        return True
                    
                    time.sleep(0.5)  # Wait before retry
                
                # Camera opened but couldn't read frames
                logger.warning(f"Camera at {camera_source} opened but cannot read frames")
                self.camera.release()
                self.camera = None
                
            except Exception as e:
                logger.debug(f"Failed to initialize camera {camera_source}", error=str(e))
                if self.camera:
                    self.camera.release()
                    self.camera = None
                continue
        
        logger.error("Failed to initialize any camera device")
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
        reconnect_attempts = 0
        max_reconnect_attempts = 3
        
        while self.running:
            try:
                if self.camera is None:
                    logger.warning("Camera is None in capture loop, attempting to reinitialize")
                    if reconnect_attempts < max_reconnect_attempts:
                        time.sleep(2 ** reconnect_attempts)  # Exponential backoff
                        if self._init_camera():
                            reconnect_attempts = 0
                            logger.info("Camera reinitialized successfully in capture loop")
                        else:
                            reconnect_attempts += 1
                            continue
                    else:
                        logger.error("Max reconnection attempts reached, stopping capture loop")
                        break
                
                ret, frame = self.camera.read()
                
                if ret and frame is not None:
                    # Reset failure counter
                    consecutive_failures = 0
                    reconnect_attempts = 0
                    
                    # Update last frame
                    with self.frame_lock:
                        self.last_frame = frame.copy()
                        self.frame_count += 1
                    
                    # Add to queue if not full
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame)
                    else:
                        # Skip oldest frame if queue is full
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
                        logger.error(f"Camera read failed {consecutive_failures} times, attempting reconnect")
                        self._reconnect_camera()
                        consecutive_failures = 0
                    else:
                        time.sleep(0.033)  # ~30fps timing
                    
            except Exception as e:
                logger.error(f"Error in capture loop", error=str(e), exc_info=True)
                consecutive_failures += 1
                time.sleep(0.1)
    
    def _reconnect_camera(self):
        """
        Attempt to reconnect to camera
        """
        try:
            logger.info("Attempting camera reconnection")
            
            if self.camera:
                self.camera.release()
                self.camera = None
            
            time.sleep(2)  # Give device time to reset
            
            if self._init_camera():
                logger.info("Camera reconnected successfully")
                # Clear any old frames from queue
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except Empty:
                        break
            else:
                logger.error("Camera reconnection failed")
            
        except Exception as e:
            logger.error(f"Error during camera reconnection", error=str(e))
    
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