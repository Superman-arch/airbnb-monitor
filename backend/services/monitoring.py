"""
Core monitoring service that orchestrates all detection and tracking
"""

import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
import cv2
import numpy as np

import structlog
from prometheus_client import Counter, Gauge, Histogram

# Import existing detection modules from parent directory
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from detection.door_inference_detector import DoorInferenceDetector
from detection.door_detector import DoorDetector
from tracking.person_tracker import PersonTracker
from tracking.journey_manager import JourneyManager
from detection.motion_detector import MotionZoneDetector
from notifications.webhook_handler import WebhookHandler

from backend.core.config import Settings
from backend.websocket.manager import WebSocketManager
from backend.services.video_processor import VideoProcessor
from backend.services.overlay_renderer import OverlayRenderer

logger = structlog.get_logger()

# Prometheus metrics
frames_processed = Counter('frames_processed_total', 'Total frames processed')
detection_duration = Histogram('detection_duration_seconds', 'Detection processing time')
active_persons = Gauge('active_persons', 'Number of people currently tracked')
active_doors = Gauge('active_doors', 'Number of doors being monitored')
fps_gauge = Gauge('processing_fps', 'Current processing FPS')


class MonitoringService:
    """
    Main monitoring service orchestrating all components
    """
    
    def __init__(self, settings: Settings, ws_manager: WebSocketManager):
        self.settings = settings
        self.ws_manager = ws_manager
        
        # Initialize components
        self.door_detector = None
        self.person_tracker = None
        self.journey_manager = None
        self.zone_detector = None
        self.webhook_handler = None
        self.video_processor = None
        self.overlay_renderer = None
        
        # State management
        self.running = False
        self.paused = False
        self.recording = True
        
        # Performance tracking
        self.frame_count = 0
        self.fps_history = []
        self.last_fps_time = time.time()
        self.last_stats_broadcast = time.time()
        
        # Frame management
        self.current_frame = None
        self.display_frame = None
        self.frame_lock = asyncio.Lock()
        
        # Processing intervals
        self.person_detect_interval = settings.PERSON_DETECT_INTERVAL
        self.door_detect_interval = settings.DOOR_DETECT_INTERVAL
        self.zone_detect_interval = settings.ZONE_DETECT_INTERVAL
        
        logger.info("Monitoring service initialized")
    
    async def start(self):
        """
        Start the monitoring service
        """
        try:
            logger.info("Starting monitoring service...")
            
            # Initialize detection components
            await self._initialize_components()
            
            # Start processing
            self.running = True
            
            # Start background tasks
            asyncio.create_task(self._process_frames())
            asyncio.create_task(self._broadcast_stats())
            asyncio.create_task(self._cleanup_stale_tracks())
            
            # Broadcast startup event
            await self.ws_manager.broadcast_event("system_startup", {
                "message": "Monitoring system started",
                "version": self.settings.VERSION,
                "environment": self.settings.ENVIRONMENT
            })
            
            logger.info("Monitoring service started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start monitoring service", error=str(e))
            raise
    
    async def stop(self):
        """
        Stop the monitoring service
        """
        logger.info("Stopping monitoring service...")
        self.running = False
        
        # Stop video processor
        if self.video_processor:
            await self.video_processor.stop()
        
        # Save states
        if self.door_detector and hasattr(self.door_detector, 'save_door_states'):
            self.door_detector.save_door_states()
        
        # Broadcast shutdown event
        await self.ws_manager.broadcast_event("system_shutdown", {
            "message": "Monitoring system shutting down"
        })
        
        logger.info("Monitoring service stopped")
    
    async def _initialize_components(self):
        """
        Initialize all detection and tracking components
        """
        # Convert settings to dict for compatibility with existing modules
        config = {
            'camera': {
                'resolution': list(self.settings.VIDEO_RESOLUTION),
                'fps': self.settings.VIDEO_FPS
            },
            'door_detection': {
                'enabled': self.settings.DOOR_DETECTION_ENABLED,
                'use_inference': True,
                'inference_url': self.settings.DOOR_INFERENCE_URL,
                'model_id': self.settings.DOOR_MODEL_ID,
                'confidence_threshold': self.settings.DOOR_CONFIDENCE_THRESHOLD,
                'state_confirmation_frames': self.settings.DOOR_STATE_CONFIRMATION_FRAMES,
                'min_seconds_between_changes': self.settings.DOOR_MIN_SECONDS_BETWEEN_CHANGES
            },
            'detection': {
                'person_model': self.settings.YOLO_MODEL,
                'confidence_threshold': self.settings.DETECTION_CONFIDENCE,
                'nms_threshold': self.settings.NMS_THRESHOLD
            },
            'tracking': {
                'tracker': 'bytetrack'
            },
            'zones': {
                'auto_detect': True
            },
            'notifications': {
                'webhook_url': self.settings.DEFAULT_WEBHOOK_URL,
                'person_webhook_url': self.settings.PERSON_WEBHOOK_URL,
                'door_webhook_url': self.settings.DOOR_WEBHOOK_URL,
                'enabled': self.settings.WEBHOOK_ENABLED
            }
        }
        
        # Initialize door detector
        try:
            from detection.door_inference_detector import DoorInferenceDetector
            self.door_detector = DoorInferenceDetector(config)
            logger.info("Using inference-based door detection")
        except ImportError:
            from detection.door_detector import DoorDetector
            self.door_detector = DoorDetector(config)
            logger.info("Using edge-based door detection")
        
        # Initialize person tracker
        self.person_tracker = PersonTracker(config)
        
        # Initialize journey manager
        self.journey_manager = JourneyManager(config)
        
        # Initialize zone detector
        self.zone_detector = MotionZoneDetector(config)
        
        # Initialize webhook handler
        self.webhook_handler = WebhookHandler(config)
        
        # Initialize video processor with error handling and validation
        try:
            logger.info("Initializing video processor...")
            self.video_processor = VideoProcessor(self.settings)
            
            # Start video processor with validation
            await self.video_processor.start()
            
            # Validate camera is working
            logger.info("Validating camera functionality...")
            test_frame = await self.video_processor.get_snapshot()
            
            if test_frame is not None:
                logger.info("Video processor started and validated successfully",
                           frame_shape=test_frame.shape,
                           frame_size=f"{test_frame.shape[1]}x{test_frame.shape[0]}")
                
                # Send startup notification
                await self.ws_manager.broadcast_event("camera_initialized", {
                    "status": "connected",
                    "resolution": f"{test_frame.shape[1]}x{test_frame.shape[0]}",
                    "message": "Camera initialized successfully"
                })
            else:
                logger.warning("Video processor started but no frames available")
                await self.ws_manager.broadcast_event("camera_warning", {
                    "status": "degraded",
                    "message": "Camera started but frames not available"
                })
                
        except Exception as e:
            logger.error(f"Failed to initialize video processor: {str(e)}")
            logger.warning("System will continue without video processing")
            self.video_processor = None
            
            # Send error notification
            await self.ws_manager.broadcast_event("camera_error", {
                "status": "failed",
                "error": str(e),
                "message": "Failed to initialize camera"
            })
        
        # Initialize overlay renderer
        self.overlay_renderer = OverlayRenderer(self.settings)
        
        logger.info("All components initialized")
    
    async def _process_frames(self):
        """
        Main frame processing loop
        """
        while self.running:
            try:
                if self.paused:
                    await asyncio.sleep(0.1)
                    continue
                
                # Get frame from video processor
                if not self.video_processor:
                    await asyncio.sleep(0.1)
                    continue
                    
                frame = await self.video_processor.get_frame()
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue
                
                # Start timing
                start_time = time.time()
                
                # Update current frame
                async with self.frame_lock:
                    self.current_frame = frame.copy()
                
                # Process detections based on intervals
                detections = {
                    'doors': [],
                    'persons': [],
                    'zones': []
                }
                
                # Door detection
                if self.frame_count % self.door_detect_interval == 0:
                    detections['doors'] = await self._detect_doors(frame)
                
                # Person detection
                if self.frame_count % self.person_detect_interval == 0:
                    detections['persons'] = await self._detect_persons(frame)
                
                # Zone detection
                if self.frame_count % self.zone_detect_interval == 0:
                    detections['zones'] = await self._detect_zones(frame)
                
                # Render overlays
                display_frame = self.overlay_renderer.render(frame, detections)
                
                # Update display frame
                async with self.frame_lock:
                    self.display_frame = display_frame
                
                # Broadcast frame to WebSocket clients (video stream)
                await self._broadcast_frame(display_frame)
                
                # Update metrics
                self.frame_count += 1
                frames_processed.inc()
                processing_time = time.time() - start_time
                detection_duration.observe(processing_time)
                
                # Calculate FPS
                self._update_fps()
                
            except Exception as e:
                logger.error(f"Error processing frame", error=str(e))
                await asyncio.sleep(0.01)
    
    async def _detect_doors(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect doors in frame
        """
        try:
            doors = self.door_detector.detect(frame)
            
            # Process door events
            for door in doors:
                if door.get('state_changed'):
                    # Create event
                    event = {
                        'event_type': 'door',
                        'door_id': door['id'],
                        'action': door['current_state'],
                        'confidence': door.get('confidence', 1.0),
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    
                    # Broadcast door update
                    await self.ws_manager.broadcast_door_update(door)
                    
                    # Send webhook
                    if self.webhook_handler:
                        await self.webhook_handler.send_door_event(event)
                    
                    # Log event
                    await self.ws_manager.broadcast_log(
                        'info',
                        f"Door {door['id']} {door['current_state']}",
                        {'door_id': door['id'], 'state': door['current_state']}
                    )
            
            # Update metric
            active_doors.set(len(doors))
            
            return doors
            
        except Exception as e:
            logger.error(f"Error detecting doors", error=str(e))
            return []
    
    async def _detect_persons(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect and track persons in frame
        """
        try:
            # Detect persons
            persons = self.person_tracker.update(frame)
            
            # Update journey tracking
            for person in persons:
                # Check zone
                zone = self._get_person_zone(person)
                
                # Update journey
                self.journey_manager.update_person(
                    person['track_id'],
                    'camera_1',
                    zone,
                    datetime.utcnow()
                )
                
                # Check for events
                events = self.journey_manager.get_person_events(person['track_id'])
                for event in events:
                    # Broadcast person event
                    await self.ws_manager.broadcast_person_update(event)
                    
                    # Send webhook
                    if self.webhook_handler:
                        await self.webhook_handler.send_person_event(event)
            
            # Update metric
            active_persons.set(len(persons))
            
            return persons
            
        except Exception as e:
            logger.error(f"Error detecting persons", error=str(e))
            return []
    
    async def _detect_zones(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect motion zones
        """
        try:
            zones = self.zone_detector.detect_motion_zones(frame)
            
            # Broadcast zone updates
            for zone in zones:
                if zone.get('state_changed'):
                    await self.ws_manager.broadcast_zone_update(zone)
            
            return zones
            
        except Exception as e:
            logger.error(f"Error detecting zones", error=str(e))
            return []
    
    def _get_person_zone(self, person: Dict[str, Any]) -> Optional[str]:
        """
        Determine which zone a person is in
        """
        if not self.zone_detector:
            return None
        
        # Get person center point
        bbox = person.get('bbox', [])
        if len(bbox) == 4:
            cx = bbox[0] + bbox[2] / 2
            cy = bbox[1] + bbox[3] / 2
            
            # Check zones
            zones = self.zone_detector.get_zones()
            for zone in zones:
                if self._point_in_polygon(cx, cy, zone.coordinates):
                    return zone.id
        
        return None
    
    def _point_in_polygon(self, x: float, y: float, polygon: List[List[float]]) -> bool:
        """
        Check if point is inside polygon
        """
        n = len(polygon)
        inside = False
        j = n - 1
        
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            
            j = i
        
        return inside
    
    def _update_fps(self):
        """
        Update FPS calculation
        """
        current_time = time.time()
        time_diff = current_time - self.last_fps_time
        
        if time_diff >= 1.0:  # Update every second
            fps = self.frame_count / time_diff
            self.fps_history.append(fps)
            
            # Keep only last 10 values
            if len(self.fps_history) > 10:
                self.fps_history.pop(0)
            
            # Update metric
            fps_gauge.set(fps)
            
            # Reset counters
            self.frame_count = 0
            self.last_fps_time = current_time
    
    async def _broadcast_frame(self, frame: np.ndarray):
        """
        Broadcast processed frame to WebSocket clients
        """
        try:
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_data = buffer.tobytes()
            
            # Create frame message
            message = {
                'type': 'frame',
                'data': frame_data.hex(),  # Convert to hex string
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Broadcast to video channel
            await self.ws_manager.broadcast_to_channel('video', message)
            
        except Exception as e:
            logger.error(f"Error broadcasting frame", error=str(e))
    
    async def _broadcast_stats(self):
        """
        Periodically broadcast system statistics
        """
        while self.running:
            try:
                await asyncio.sleep(1)  # Broadcast every second
                
                # Calculate average FPS
                avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
                
                # Gather stats
                stats = {
                    'fps': round(avg_fps, 1),
                    'frame': self.frame_count,
                    'doors': active_doors._value.get(),
                    'persons': active_persons._value.get(),
                    'zones': len(self.zone_detector.get_zones()) if self.zone_detector else 0,
                    'memory': self._get_memory_usage(),
                    'gpu': self._get_gpu_usage(),
                    'latency': int(1000 / max(avg_fps, 1)),  # ms
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                # Broadcast stats
                await self.ws_manager.broadcast_stats(stats)
                
            except Exception as e:
                logger.error(f"Error broadcasting stats", error=str(e))
    
    async def _cleanup_stale_tracks(self):
        """
        Periodically clean up stale person tracks
        """
        while self.running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                if self.journey_manager:
                    # Clean up inactive persons
                    removed = self.journey_manager.cleanup_inactive_persons(
                        inactive_threshold=timedelta(minutes=5)
                    )
                    
                    if removed:
                        logger.info(f"Cleaned up {len(removed)} inactive person tracks")
                
            except Exception as e:
                logger.error(f"Error cleaning up tracks", error=str(e))
    
    def _get_memory_usage(self) -> float:
        """
        Get current memory usage percentage
        """
        try:
            import psutil
            return psutil.virtual_memory().percent
        except:
            return 0.0
    
    def _get_gpu_usage(self) -> float:
        """
        Get GPU usage percentage (Jetson specific)
        """
        try:
            if self.settings.JETSON_MODE:
                # Use jetson-stats if available
                from jtop import jtop
                with jtop() as jetson:
                    if jetson.ok():
                        return jetson.stats['GPU']
            return 0.0
        except:
            return 0.0
    
    async def get_current_frame(self) -> Optional[np.ndarray]:
        """
        Get current processed frame
        """
        async with self.frame_lock:
            if self.display_frame is not None:
                return self.display_frame.copy()
            elif self.current_frame is not None:
                return self.current_frame.copy()
            else:
                return None
    
    async def calibrate_doors(self, frame: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calibrate door zones
        """
        if frame is None:
            frame = await self.get_current_frame()
        
        if frame is None:
            return {'success': False, 'error': 'No frame available'}
        
        if hasattr(self.door_detector, 'calibrate_zones'):
            result = self.door_detector.calibrate_zones(frame)
            
            # Broadcast calibration result
            await self.ws_manager.broadcast_event('door_calibration', result)
            
            return result
        
        return {'success': False, 'error': 'Calibration not supported'}
    
    def pause(self):
        """Pause monitoring"""
        self.paused = True
        logger.info("Monitoring paused")
    
    def resume(self):
        """Resume monitoring"""
        self.paused = False
        logger.info("Monitoring resumed")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current monitoring status
        """
        return {
            'running': self.running,
            'paused': self.paused,
            'recording': self.recording,
            'frame_count': self.frame_count,
            'fps': sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0,
            'doors_detected': active_doors._value.get(),
            'persons_tracked': active_persons._value.get()
        }