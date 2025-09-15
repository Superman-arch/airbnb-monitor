#!/usr/bin/env python3
"""
Optimized Airbnb Monitor for Jetson Nano
Achieves 10-15 FPS by using frame skipping and optimized processing
"""

import cv2
import yaml
import numpy as np
import signal
import sys
import time
from datetime import datetime
from threading import Thread, Lock
from typing import Dict, Any, List, Optional, Tuple
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Optimize for Jetson
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Import our modules
from detection.motion_detector import MotionZoneDetector
from utils.logger import logger
from storage.door_persistence import DoorPersistence
from core.state_manager import state_manager

# Try to import inference detector first, fall back to edge detection
try:
    from detection.door_inference_detector import DoorInferenceDetector
    INFERENCE_AVAILABLE = True
    logger.log('INFO', "Using Roboflow Inference for door detection", 'MAIN')
except ImportError:
    from detection.door_detector import DoorDetector
    INFERENCE_AVAILABLE = False
    logger.log('WARNING', "Using edge detection for doors (inference not available)", 'MAIN')

from tracking.person_tracker import PersonTracker
from tracking.journey_manager import JourneyManager
from notifications.webhook_handler import WebhookHandler
from storage.video_manager import CircularVideoBuffer

# Check if web interface is available but don't import yet to avoid circular imports
WEB_AVAILABLE = False
SOCKETIO_AVAILABLE = False

try:
    import flask
    import flask_cors
    WEB_AVAILABLE = True
    print("[INIT] Flask and Flask-CORS available")
    
    # Check SocketIO separately as it might be broken
    try:
        import flask_socketio
        SOCKETIO_AVAILABLE = True
        print("[INIT] Flask-SocketIO available - WebSocket support enabled")
    except (ImportError, AttributeError) as e:
        SOCKETIO_AVAILABLE = False
        print(f"[INIT] Flask-SocketIO not available ({e}) - using regular Flask")
        
except ImportError as e:
    print(f"[INIT] Web interface not available: {e}")
    WEB_AVAILABLE = False

# Create stub functions that will be replaced when web is initialized
def broadcast_log(msg, level='info'):
    print(f"[{level.upper()}] {msg}")

def broadcast_stats(stats):
    pass

def broadcast_event(event):
    if isinstance(event, dict):
        print(f"Event: {event.get('event', 'unknown')}")
    else:
        print(f"Event: {event}")

# These will be set when web interface is initialized
app = None
socketio = None

# Color definitions for drawing (BGR format for OpenCV)
COLORS = {
    'zone': (255, 255, 0),      # Yellow for zones
    'door_open': (0, 0, 255),   # Red for open doors
    'door_closed': (0, 255, 0),  # Green for closed doors
    'door_unknown': (128, 128, 128),  # Gray for unknown state
    'person': (255, 0, 255),     # Magenta for persons
    'motion': (0, 255, 255),     # Cyan for motion
    'text': (255, 255, 255),     # White for text
    'background': (0, 0, 0)      # Black for background
}


class OptimizedAirbnbMonitor:
    """Optimized version for better FPS on Jetson Nano."""
    
    def __init__(self, config_path: str = "config/settings_optimized.yaml"):
        """Initialize optimized monitor."""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        logger.log('INFO', "Initializing Optimized Airbnb Monitor...", 'MAIN')
        logger.log('INFO', "Target FPS: 10-15 with door detection", 'MAIN')
        
        # Initialize components
        self.zone_detector = MotionZoneDetector(self.config)
        
        # Use inference detector if available
        if INFERENCE_AVAILABLE:
            self.door_detector = DoorInferenceDetector(self.config)
        else:
            self.door_detector = DoorDetector(self.config)
            
        self.person_tracker = PersonTracker(self.config)
        self.journey_manager = JourneyManager(self.config)
        self.webhook_handler = WebhookHandler(self.config)
        self.video_buffer = CircularVideoBuffer(self.config)
        
        # Initialize unified state manager
        self.state_manager = state_manager
        self.state_manager.reset()  # Start fresh
        
        # Camera settings
        self.camera = None
        self.camera_id = "camera_1"
        self.resolution = tuple(self.config['camera']['resolution'])
        self.fps = self.config['camera']['fps']
        
        # Optimization settings
        self.person_detect_interval = 3  # Detect persons every N frames
        self.zone_detect_interval = 30   # Update zones every N frames
        self.frame_counter = 0
        
        # State
        self.running = False
        self.recording_enabled = True
        
        # Performance monitoring
        self.fps_history = []
        self.last_fps_time = time.time()
        self.last_fps_frame = 0
        
        # Thread safety
        self.frame_lock = Lock()
        self.current_frame = None
        self.display_frame = None  # Frame with detection overlays
        
        # Web server
        self.web_thread = None
        self.web_app = None
        self.socketio = None
        
    def initialize_camera(self):
        """Initialize camera with optimized settings."""
        camera_source = self.config['camera']['source']
        
        if isinstance(camera_source, int):
            # USB camera
            print(f"Initializing USB camera at index {camera_source}")
            self.camera = cv2.VideoCapture(camera_source)
            
            # Set buffer size to reduce latency
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
        elif camera_source.startswith('csi'):
            # CSI camera for Jetson Nano
            print("Initializing CSI camera")
            gst_pipeline = self._get_gstreamer_pipeline()
            self.camera = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        else:
            self.camera = cv2.VideoCapture(camera_source)
        
        # Set camera properties for speed
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.camera.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Verify camera
        ret, frame = self.camera.read()
        if not ret:
            raise RuntimeError("Failed to read from camera")
        
        print(f"Camera initialized: {frame.shape}")
        return True
    
    def _get_gstreamer_pipeline(self):
        """Optimized GStreamer pipeline for Jetson Nano CSI camera."""
        return (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM), "
            f"width={self.resolution[0]}, height={self.resolution[1]}, "
            f"format=NV12, framerate={self.fps}/1 ! "
            f"nvvidconv flip-method=0 ! "
            f"video/x-raw, format=BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=BGR ! "
            f"appsink max-buffers=1 drop=true"
        )
    
    def _auto_calibrate_doors(self):
        """Auto-calibrate door zones on startup."""
        try:
            # Check if we have saved door configurations first
            if hasattr(self.door_detector, 'doors') and self.door_detector.doors:
                logger.log('INFO', f"Using {len(self.door_detector.doors)} saved door configurations", 'MAIN')
                return True
            
            # Capture a few frames to get a stable image
            logger.log('INFO', "Capturing frames for door calibration...", 'MAIN')
            frames = []
            for i in range(5):
                ret, frame = self.camera.read()
                if ret:
                    frames.append(frame)
                    time.sleep(0.2)
            
            if frames:
                # Use the middle frame for calibration
                calibration_frame = frames[len(frames)//2]
                
                # Store frame for web access
                with self.frame_lock:
                    self.current_frame = calibration_frame.copy()
                
                # Attempt auto-calibration
                if hasattr(self.door_detector, 'auto_calibrate'):
                    success = self.door_detector.auto_calibrate(calibration_frame)
                    if success:
                        logger.log('INFO', f"✓ Auto-calibration successful! {len(self.door_detector.doors)} door(s) configured", 'MAIN')
                        return True
                
                # Fallback to regular calibration
                if hasattr(self.door_detector, 'calibrate_zones'):
                    result = self.door_detector.calibrate_zones(calibration_frame)
                    
                    if result.get('success'):
                        zones_found = result.get('zones_found', 0)
                        logger.log('INFO', f"✓ Calibration successful! Found {zones_found} door(s)", 'MAIN')
                        
                        # Log calibration event
                        logger.log_calibration(result)
                        
                        # Broadcast to web if available
                        if WEB_AVAILABLE:
                            try:
                                broadcast_log(f"Auto-calibration complete: {zones_found} doors detected", 'info')
                            except:
                                pass
                        return True
                    else:
                        logger.log('WARNING', f"Calibration failed: {result.get('error', 'Unknown error')}", 'MAIN')
                else:
                    logger.log('WARNING', "Door detector does not support zone calibration", 'MAIN')
            else:
                logger.log('ERROR', "Could not capture frames for calibration", 'MAIN')
                
        except Exception as e:
            logger.log('ERROR', f"Error during auto-calibration: {e}", 'MAIN')
        
        return False
    
    def start(self):
        """Start the optimized monitoring system."""
        print("Starting Optimized Airbnb Monitor...")
        
        # Initialize components
        if not self.zone_detector.initialize():
            print("Failed to initialize zone detector")
            return False
        
        if not self.person_tracker.initialize():
            print("Failed to initialize person tracker")
            return False
        
        # Initialize door detector (inference needs initialization)
        if hasattr(self.door_detector, 'initialize'):
            if not self.door_detector.initialize():
                logger.log('WARNING', "Door detector initialization failed", 'MAIN')
                logger.log('WARNING', "Continuing without door detection...", 'MAIN')
        
        # Initialize camera
        if not self.initialize_camera():
            print("Failed to initialize camera")
            return False
        
        # Auto-calibrate doors on startup if configured
        if self.config.get('door_detection', {}).get('auto_calibrate_on_startup', True):
            logger.log('INFO', "Auto-calibrating door zones on startup...", 'MAIN')
            self._auto_calibrate_doors()
        
        # Start door learning phase (only for edge detection without zones)
        if hasattr(self.door_detector, 'start_learning'):
            # Only start learning if no zones are configured
            if not self.door_detector.get_zones():
                self.door_detector.start_learning()
            else:
                logger.log('INFO', f"Skipping learning phase - {len(self.door_detector.get_zones())} zones already configured", 'MAIN')
        
        # Start services
        self.webhook_handler.start()
        self.video_buffer.start(self.resolution, self.fps)
        
        # Start web server
        web_started = self.start_web_server()
        if not web_started:
            print("\n⚠️  WARNING: Web interface failed to start!")
            print("   You can still use the system but won't have web dashboard access.")
            print("   To fix: pip3 install flask flask-cors flask-socketio\n")
        else:
            print("\n✅ Web interface is running!")
            print("   Open your browser and go to http://192.168.86.246:5000\n")
        
        self.running = True
        self.state_manager.set_system_ready(True)
        
        # Start processing
        self.process_loop_optimized()
        
        return True
    
    def start_web_server(self):
        """Start the Flask web server in a separate thread."""
        if not WEB_AVAILABLE:
            print("[WEB] Flask not installed - web interface disabled")
            print("[WEB] To enable web interface, install dependencies:")
            print("[WEB]   sudo pip3 install flask flask-cors flask-socketio")
            return False
        
        # Check if port is available
        from utils.network_utils import is_port_available, get_local_ip
        web_config = self.config.get('web', {})
        port = web_config.get('port', 5000)
        
        if not is_port_available('0.0.0.0', port):
            print(f"[WEB ERROR] Port {port} is already in use!")
            print(f"[WEB ERROR] Another process might be using this port.")
            print(f"[WEB ERROR] Try: sudo lsof -i :{port}  (to see what's using it)")
            print(f"[WEB ERROR] Or change the port in config/settings_optimized.yaml")
            return False
        
        def run_web_server():
            print("[WEB] Initializing web server...")
            try:
                # Import web components here to avoid circular imports
                print("[WEB] Importing Flask components...")
                from web.app import app as web_app
                from web.app import init_app
                
                # Try to import socketio if available
                web_socketio = None
                if SOCKETIO_AVAILABLE:
                    try:
                        from web.app import socketio as web_socketio
                        print("[WEB] SocketIO imported successfully")
                    except (ImportError, AttributeError) as e:
                        print(f"[WEB] SocketIO import failed: {e}")
                        web_socketio = None
                
                print("[WEB] Flask components imported successfully")
                
                # Import broadcast functions and update globals
                global app, socketio, broadcast_event, broadcast_log, broadcast_stats
                try:
                    from web.app import broadcast_event as web_broadcast_event
                    from web.app import broadcast_log as web_broadcast_log
                    from web.app import broadcast_stats as web_broadcast_stats
                    
                    # Update global functions
                    broadcast_event = web_broadcast_event
                    broadcast_log = web_broadcast_log
                    broadcast_stats = web_broadcast_stats
                    print("[WEB] Broadcast functions loaded")
                except ImportError as e:
                    print(f"[WEB] Warning: Could not import broadcast functions: {e}")
                    # Keep using stub functions
                    pass
                
                # Set global references
                app = web_app
                socketio = web_socketio
                
                # Initialize the web app with monitor instance
                print("[WEB] Initializing app with monitor instance...")
                init_app(self.config, monitor=self)
                
                # Pass reference to this monitor for frame access
                app.config['monitor'] = self
                app.config['zone_detector'] = self.zone_detector
                app.config['journey_manager'] = self.journey_manager
                
                # Get host and port from config
                web_config = self.config.get('web', {})
                host = web_config.get('host', '0.0.0.0')
                port = web_config.get('port', 5000)
                
                # Get actual IP address for display
                import socket
                hostname = socket.gethostname()
                try:
                    local_ip = socket.gethostbyname(hostname)
                except:
                    local_ip = '127.0.0.1'
                
                print(f"[WEB] Starting Flask server...")
                print(f"[WEB] Host: {host}, Port: {port}")
                print(f"[WEB] Access the dashboard at:")
                print(f"[WEB]   - http://localhost:{port} (from this machine)")
                print(f"[WEB]   - http://{local_ip}:{port} (from network)")
                print(f"[WEB]   - http://192.168.86.246:{port} (your Jetson IP)")
                
                # Set Flask to not show warnings
                import logging
                log = logging.getLogger('werkzeug')
                log.setLevel(logging.ERROR)
                
                # Try to run with SocketIO if available, otherwise use regular Flask
                if SOCKETIO_AVAILABLE and socketio is not None:
                    print("[WEB] Starting Flask with SocketIO (WebSocket support)...")
                    try:
                        print(f"[WEB] Calling socketio.run() on host={host}, port={port}...")
                        socketio.run(app, host=host, port=port, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
                        print("[WEB] Flask server with SocketIO stopped normally")
                    except Exception as e:
                        print(f"[WEB ERROR] SocketIO failed: {e}")
                        print("[WEB] Falling back to regular Flask...")
                        app.run(host=host, port=port, debug=False, use_reloader=False)
                else:
                    print("[WEB] Starting Flask without SocketIO (no WebSocket support)...")
                    print(f"[WEB] Running app.run() on host={host}, port={port}...")
                    app.run(host=host, port=port, debug=False, use_reloader=False)
                    print("[WEB] Flask server stopped normally")
                
            except ImportError as e:
                print(f"[WEB ERROR] Missing dependencies: {e}")
                print("[WEB ERROR] Install with: pip3 install flask flask-cors flask-socketio")
                return
            except Exception as e:
                print(f"[WEB ERROR] Failed to start web server: {e}")
                import traceback
                print("[WEB ERROR] Full traceback:")
                traceback.print_exc()
                return
        
        # Start web server in separate thread (NOT daemon so it keeps running)
        print("[WEB] Creating web server thread...")
        self.web_thread = Thread(target=run_web_server, daemon=False)  # Changed to non-daemon
        self.web_thread.start()
        
        # Give the server more time to start (Flask can be slow)
        import time
        print("[WEB] Waiting for server to initialize (this may take up to 10 seconds)...")
        
        # First just check if thread is alive
        for i in range(5):
            time.sleep(1)
            if not self.web_thread.is_alive():
                print(f"[WEB] ✗ Web server thread died after {i+1} seconds")
                return False
            print(f"[WEB] Thread still alive after {i+1} seconds...")
        
        # Thread is alive, now test if we can actually connect
        from utils.network_utils import test_connection
        print("[WEB] Thread is running, testing connection to localhost...")
        
        # Try more times with longer timeout as Flask might be slow to start
        for attempt in range(5):
            if test_connection('localhost', port, timeout=3):
                print(f"[WEB] ✓ Successfully connected to web server on port {port}!")
                print(f"[WEB] Connection established after {attempt+1} attempts")
                return True
            print(f"[WEB] Connection attempt {attempt+1}/5 failed, retrying...")
            time.sleep(2)
        
        # If thread is alive but can't connect, something is wrong
        print("[WEB] ⚠️  Web server thread is running but not accepting connections")
        print("[WEB] This might be a Flask initialization issue")
        return False
    
    def process_loop_optimized(self):
        """Optimized processing loop with frame skipping."""
        print("Optimized processing started. Press 'q' to quit")
        logger.log('INFO', "Door learning phase: 30 seconds...", 'MAIN')
        
        # Tracking variables
        last_person_detection = 0
        last_zone_detection = 0
        tracked_persons = []  # Maintain state between detections
        active_persons = 0  # Track active person count
        zones = []
        door_learning_complete = False
        last_processing_time = 0  # Track processing latency
        
        while self.running:
            loop_start = time.time()
            
            # Read frame
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to read frame")
                continue
            
            self.frame_counter += 1
            timestamp = datetime.now()
            
            # Store current frame for web display
            with self.frame_lock:
                self.current_frame = frame.copy()
            
            # ALWAYS: Door detection
            if INFERENCE_AVAILABLE:
                # Inference detection (no learning phase)
                _, door_events = self.door_detector.process_frame(frame)
            else:
                # Edge detection with learning phase
                learning_complete, door_events = self.door_detector.process_frame(frame)
                
                if learning_complete and not door_learning_complete:
                    door_learning_complete = True
                    logger.log('INFO', f"Door learning complete! Found {len(self.door_detector.doors)} doors", 'MAIN')
                    
                    # Send door discovery events
                    for event in door_events:
                        self.webhook_handler.send_event(event, frame)
            
            # Handle door state change events
            for event in door_events:
                # Update unified state manager
                door_id = event.get('door_id', 'unknown')
                if 'open' in event.get('event', ''):
                    door_state = 'open'
                elif 'closed' in event.get('event', ''):
                    door_state = 'closed'
                else:
                    door_state = event.get('current_state', 'unknown')
                
                # Update state manager
                self.state_manager.update_door(
                    door_id=door_id,
                    state=door_state,
                    confidence=event.get('confidence', 0),
                    bbox=event.get('bbox'),
                    zone=event.get('zone_id')
                )
                
                if event['event'] in ['door_opened', 'door_closed', 'door_moving', 'door_left_open', 'door_discovered']:
                    # Send door event to specific webhook
                    self.send_door_event(event, frame)
                    door_name = event.get('door_name', event.get('door_id', 'unknown'))
                    log_msg = f"Door event: {door_name} - {event['event']}"
                    logger.log('INFO', log_msg, 'DOOR')
                    if WEB_AVAILABLE:
                        try:
                            broadcast_log(log_msg, 'warning' if 'open' in event['event'] else 'info')
                        except:
                            pass  # Ignore broadcast errors
                    
                    # Broadcast to web interface
                    if WEB_AVAILABLE:
                        try:
                            # Store door event in journey manager for persistence
                            self.journey_manager.store_door_event(event)
                            # Ensure event type is set
                            event['event_type'] = 'door'
                            # Broadcast to web clients
                            broadcast_event(event)
                            # Also send as log
                            door_name = event.get('door_name', event.get('door_id', 'unknown'))
                            broadcast_log(f"Door {event['event'].replace('door_', '')}: {door_name}", 'warning' if 'open' in event['event'] else 'info')
                        except Exception as e:
                            # Don't crash on broadcast errors
                            if self.frame_counter % 300 == 0:
                                logger.log('ERROR', f"Failed to broadcast door event: {e}", 'MAIN')
            
            # PERIODIC: Zone detection (every 30 frames)
            if self.frame_counter - last_zone_detection >= self.zone_detect_interval:
                motion_detections = self.zone_detector.detect(frame)
                zones = self.zone_detector.get_zones()
                last_zone_detection = self.frame_counter
            
            # PERIODIC: Person detection (every 3 frames)
            if self.frame_counter - last_person_detection >= self.person_detect_interval:
                detection_start = time.time()
                tracked_persons = self.person_tracker.detect(frame)
                last_person_detection = self.frame_counter
                active_persons = len(tracked_persons)  # Update active count
                last_processing_time = int((time.time() - detection_start) * 1000)  # ms
                
                # Process person events
                if tracked_persons and zones:
                    events = self.journey_manager.process_tracks(
                        tracked_persons, self.camera_id, zones, timestamp
                    )
                    
                    for event in events:
                        if event['action'] in ['entry', 'exit']:
                            # Send person event to specific webhook
                            self.send_person_event(event, frame)
                            print(f"Person event: {event['action']} - {event.get('person_id', 'unknown')}")
                            
                            # Broadcast to web interface
                            if WEB_AVAILABLE:
                                try:
                                    event['event_type'] = 'person'
                                    broadcast_event(event)
                                    broadcast_log(f"Person {event['action']}: {event.get('person_id', 'unknown')}", 'warning')
                                except Exception as e:
                                    if self.frame_counter % 300 == 0:
                                        print(f"Failed to broadcast person event: {e}")
            
            # ALWAYS: Record video (if enabled)
            if self.recording_enabled:
                self.video_buffer.add_frame(frame, timestamp)
            
            # Draw overlays
            display_frame = frame.copy()
            display_frame = self.draw_all_detections(display_frame, tracked_persons)
            
            # Store display frame for web
            with self.frame_lock:
                self.display_frame = display_frame.copy()
            
            # Calculate and display FPS
            self.update_fps()
            if self.frame_counter % 30 == 0:
                current_fps = self.get_current_fps()
                num_doors = len(self.door_detector.doors) if hasattr(self.door_detector, 'doors') else 0
                stats_msg = f"FPS: {current_fps:.1f} | Doors: {num_doors} | Persons: {active_persons} | Frame: {self.frame_counter}"
                print(stats_msg)
                
                # Always broadcast stats and log
                if WEB_AVAILABLE:
                    broadcast_log(stats_msg, 'info')
                
                # Broadcast stats to web interface (with error handling)
                if WEB_AVAILABLE:
                    try:
                        broadcast_stats({
                            'fps': current_fps,
                            'doors': len(self.door_detector.doors),
                            'persons': active_persons,  # Use maintained count
                            'zones': len(zones),
                            'frame': self.frame_counter,
                            'latency': last_processing_time,
                            'memory': self._get_memory_usage(),
                            'gpu': self._get_gpu_usage()
                        })
                        broadcast_log(stats_msg, 'info')
                    except Exception as e:
                        # Don't crash if web broadcast fails
                        if self.frame_counter % 300 == 0:  # Log every 10 seconds
                            print(f"Warning: Could not broadcast stats: {e}")
            
            # Show frame (optional - disable for better performance)
            if self.config.get('display', {}).get('enabled', False):
                self.draw_stats(display_frame)
                cv2.imshow('Optimized Airbnb Monitor', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord('r'):
                    logger.log('INFO', "Resetting door detection...", 'MAIN')
                    self.door_detector.reset()
            
            # Frame rate limiting (optional)
            elapsed = time.time() - loop_start
            target_time = 1.0 / self.fps
            if elapsed < target_time:
                time.sleep(target_time - elapsed)
    
    def send_door_event(self, event: Dict[str, Any], frame: np.ndarray):
        """Send door-specific event to webhook."""
        # Use door-specific webhook if configured
        webhook_url = self.config.get('notifications', {}).get('door_webhook_url')
        if webhook_url:
            # Temporarily override webhook URL
            original_url = self.webhook_handler.webhook_url
            self.webhook_handler.webhook_url = webhook_url
            self.webhook_handler.send_event(event, frame)
            self.webhook_handler.webhook_url = original_url
        else:
            # Use default webhook
            self.webhook_handler.send_event(event, frame)
    
    def send_person_event(self, event: Dict[str, Any], frame: np.ndarray):
        """Send person-specific event to webhook."""
        # Use person-specific webhook if configured
        webhook_url = self.config.get('notifications', {}).get('person_webhook_url')
        if webhook_url:
            # Temporarily override webhook URL
            original_url = self.webhook_handler.webhook_url
            self.webhook_handler.webhook_url = webhook_url
            
            # Get journey data
            journey = self.journey_manager.get_person_journey(event['person_id'])
            self.webhook_handler.send_event(event, frame, journey)
            
            self.webhook_handler.webhook_url = original_url
        else:
            # Use default webhook
            journey = self.journey_manager.get_person_journey(event['person_id'])
            self.webhook_handler.send_event(event, frame, journey)
    
    def update_fps(self):
        """Update FPS calculation."""
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            fps = (self.frame_counter - self.last_fps_frame) / (current_time - self.last_fps_time)
            self.fps_history.append(fps)
            if len(self.fps_history) > 10:
                self.fps_history.pop(0)
            self.last_fps_time = current_time
            self.last_fps_frame = self.frame_counter
    
    def get_current_fps(self) -> float:
        """Get current average FPS."""
        if self.fps_history:
            return sum(self.fps_history) / len(self.fps_history)
        return 0.0
    
    def _get_memory_usage(self) -> int:
        """Get memory usage percentage."""
        try:
            import psutil
            return int(psutil.virtual_memory().percent)
        except:
            # Fallback: try reading from /proc/meminfo
            try:
                with open('/proc/meminfo', 'r') as f:
                    lines = f.readlines()
                    total = int(lines[0].split()[1])
                    available = int(lines[2].split()[1])
                    used_percent = int(((total - available) / total) * 100)
                    return used_percent
            except:
                return 0
    
    def _get_gpu_usage(self) -> int:
        """Get GPU usage percentage (Jetson specific)."""
        try:
            # Try nvidia-smi first (for desktop GPUs)
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=0.5)
            if result.returncode == 0:
                return int(result.stdout.strip())
        except:
            pass
        
        try:
            # Try Jetson-specific tegrastats
            import subprocess
            result = subprocess.run(['cat', '/sys/devices/gpu.0/load'], 
                                  capture_output=True, text=True, timeout=0.1)
            if result.returncode == 0:
                # Value is in per-mille (0-1000), convert to percentage
                return int(int(result.stdout.strip()) / 10)
        except:
            pass
        
        return 0  # Return 0 if no GPU stats available
    
    def draw_all_detections(self, frame: np.ndarray, tracked_persons: List[Dict]) -> np.ndarray:
        """Draw comprehensive detection overlays warehouse-style."""
        overlay = frame.copy()
        height, width = frame.shape[:2]
        
        # Define colors for different object types
        COLORS = {
            'door_open': (0, 255, 0),      # Green
            'door_closed': (0, 0, 255),    # Red  
            'door_unknown': (0, 255, 255), # Yellow
            'person': (255, 165, 0),       # Orange
            'zone': (255, 0, 255),         # Magenta
        }
        
        # Draw door zones first (background layer)
        if hasattr(self.door_detector, 'zone_mapper') and self.door_detector.zone_mapper:
            for zone in self.door_detector.zone_mapper.zones:
                # Handle both dict and object zones
                if isinstance(zone, dict):
                    bbox = zone.get('bbox', (0, 0, 0, 0))
                else:
                    bbox = zone.bbox if hasattr(zone, 'bbox') else (0, 0, 0, 0)
                
                x, y, w, h = bbox
                
                # Validate coordinates
                if x < 0 or y < 0 or w <= 0 or h <= 0:
                    continue
                
                # Ensure within frame bounds
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                w = min(w, width - x)
                h = min(h, height - y)
                
                # Draw semi-transparent zone
                zone_overlay = overlay.copy()
                cv2.rectangle(zone_overlay, (x, y), (x + w, y + h), COLORS['zone'], 2)
                cv2.addWeighted(zone_overlay, 0.3, overlay, 0.7, 0, overlay)
        
        # Draw door detections with labels
        if hasattr(self.door_detector, 'doors'):
            doors_dict = self.door_detector.doors
            # Handle both dict and list formats
            if isinstance(doors_dict, dict):
                doors_list = doors_dict.values()
            else:
                doors_list = doors_dict
            
            for door in doors_list:
                # Extract door_id from door object
                door_id = getattr(door, 'id', 'unknown')
                x, y, w, h = door.bbox
                
                # Validate and adjust coordinates
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                w = min(w, width - x)
                h = min(h, height - y)
                
                if w <= 0 or h <= 0:
                    continue
                
                # Determine color based on state
                if door.current_state == 'open':
                    color = COLORS['door_open']
                    state_text = "OPEN"
                elif door.current_state == 'closed':
                    color = COLORS['door_closed']
                    state_text = "CLOSED"
                else:
                    color = COLORS['door_unknown']
                    state_text = "UNKNOWN"
                
                # Draw bounding box
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
                
                # Get door name
                door_name = getattr(door, 'zone_name', None) or door_id
                
                # Draw label background
                label = f"{door_name} | {state_text}"
                if hasattr(door, 'confidence') and door.confidence > 0:
                    label += f" {door.confidence:.0%}"
                
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                label_y = y - 10 if y > 30 else y + h + 20
                
                # Semi-transparent background for label
                label_bg = overlay.copy()
                cv2.rectangle(label_bg, 
                            (x, label_y - label_size[1] - 4),
                            (x + label_size[0] + 8, label_y + 4),
                            color, -1)
                cv2.addWeighted(label_bg, 0.6, overlay, 0.4, 0, overlay)
                
                # Draw text
                cv2.putText(overlay, label, (x + 4, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Draw ID badge
                id_text = door_id.replace('door_', 'D')
                cv2.rectangle(overlay, (x + w - 30, y + 2), (x + w - 2, y + 22), color, -1)
                cv2.putText(overlay, id_text, (x + w - 28, y + 17),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw person tracking with IDs
        for person in tracked_persons:
            try:
                person_id = person.get('id', 'unknown')
                track = person.get('track')
                
                if track:
                    # Get bounding box - try different methods
                    bbox = None
                    if hasattr(track, 'tlbr'):
                        bbox = track.tlbr()  # Call as method
                    elif hasattr(track, '_tlbr'):
                        bbox = track._tlbr  # Access internal attribute
                    elif hasattr(track, 'tlwh'):
                        # Convert tlwh to tlbr
                        tlwh = track.tlwh
                        bbox = [tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]]
                    
                    if bbox is not None:
                        x1, y1, x2, y2 = map(int, bbox)
                        
                        # Validate and adjust coordinates
                        x1 = max(0, min(x1, width - 1))
                        y1 = max(0, min(y1, height - 1))
                        x2 = max(0, min(x2, width - 1))
                        y2 = max(0, min(y2, height - 1))
                        
                        if x2 <= x1 or y2 <= y1:
                            continue
                            
                        color = COLORS['person']
                        
                        # Draw bounding box
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw tracking trail if available
                        if hasattr(track, 'centroid_history') and len(track.centroid_history) > 1:
                            points = np.array(track.centroid_history[-10:], np.int32)
                            cv2.polylines(overlay, [points], False, color, 1)
                        
                        # Draw label
                        label = f"Person #{person_id}"
                        confidence = person.get('confidence', 0)
                        if confidence > 0:
                            label += f" {confidence:.0%}"
                        
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        
                        # Semi-transparent background
                        label_bg = overlay.copy()
                        cv2.rectangle(label_bg,
                                    (x1, y1 - label_size[1] - 4),
                                    (x1 + label_size[0] + 8, y1 + 4),
                                    color, -1)
                        cv2.addWeighted(label_bg, 0.6, overlay, 0.4, 0, overlay)
                        
                        # Draw text
                        cv2.putText(overlay, label, (x1 + 4, y1 - 2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        
                        # Draw ID badge
                        cv2.rectangle(overlay, (x2 - 30, y1 + 2), (x2 - 2, y1 + 22), color, -1)
                        cv2.putText(overlay, f"P{person_id}", (x2 - 28, y1 + 17),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            except Exception as e:
                # Skip this person if there's an error
                print(f"Warning: Could not draw person {person.get('id', 'unknown')}: {e}")
                continue
        
        # Draw stats overlay
        self.draw_stats_overlay(overlay)
        
        return overlay
    
    def draw_stats_overlay(self, frame: np.ndarray):
        """Draw statistics overlay on frame."""
        height, width = frame.shape[:2]
        
        # Create semi-transparent background for stats
        stats_bg = frame.copy()
        cv2.rectangle(stats_bg, (10, 10), (250, 100), (0, 0, 0), -1)
        cv2.addWeighted(stats_bg, 0.5, frame, 0.5, 0, frame)
        
        # Draw stats text
        y_offset = 30
        stats = [
            f"FPS: {self.get_current_fps():.1f}",
            f"Doors: {len(self.door_detector.doors)}",
            f"Persons: {len(self.person_tracker.tracker.tracked_stracks) if hasattr(self.person_tracker.tracker, 'tracked_stracks') else 0}",
            f"Frame: {self.frame_counter}"
        ]
        
        for stat in stats:
            cv2.putText(frame, stat, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 20
    
    def draw_stats(self, frame: np.ndarray):
        """Draw performance statistics on frame."""
        fps = self.get_current_fps()
        
        stats = [
            f"FPS: {fps:.1f} (Target: 10-15)",
            f"Doors: {len(self.door_detector.doors)}",
            f"Persons: {len(self.person_tracker.tracker.tracked_stracks)}",
            f"Frame: {self.frame_counter}",
            f"Skip: 1/{self.person_detect_interval}"
        ]
        
        y = 30
        for stat in stats:
            color = (0, 255, 0) if fps >= 10 else (0, 165, 255) if fps >= 5 else (0, 0, 255)
            cv2.putText(frame, stat, (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y += 20
        
        # Draw door states
        door_states = self.door_detector.get_door_states()
        if door_states:
            y += 10
            cv2.putText(frame, "Doors:", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 20
            for door_id, state in door_states.items():
                color = (0, 255, 0) if state == "open" else (0, 0, 255)
                cv2.putText(frame, f"  {door_id}: {state}", (10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                y += 15
    
    def get_current_frame(self):
        """Get current frame for web display."""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None
    
    def stop(self):
        """Stop the monitoring system."""
        print("\nStopping Optimized Airbnb Monitor...")
        
        self.running = False
        
        # Save door configurations
        self.door_detector.save_doors()
        
        # Stop services
        self.webhook_handler.stop()
        self.video_buffer.stop()
        
        # Cleanup
        if self.camera:
            self.camera.release()
        
        self.zone_detector.cleanup()
        self.person_tracker.cleanup()
        
        cv2.destroyAllWindows()
        
        # Print statistics
        print(f"\nSession Statistics:")
        print(f"  Total frames: {self.frame_counter}")
        print(f"  Average FPS: {self.get_current_fps():.1f}")
        logger.log('INFO', f"  Doors detected: {len(self.door_detector.doors)}", 'STATS')
        
        stats = self.webhook_handler.get_statistics()
        print(f"  Notifications sent: {stats['notifications_sent']}")
        
        print("Shutdown complete")


def signal_handler(sig, frame):
    """Handle shutdown signal."""
    print("\nShutdown signal received")
    sys.exit(0)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimized Airbnb Monitor')
    parser.add_argument('--config', default='config/settings_optimized.yaml',
                       help='Path to configuration file')
    parser.add_argument('--display', action='store_true',
                       help='Show video display (reduces FPS)')
    
    args = parser.parse_args()
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Check for optimized config, use regular config if not exists
    if not os.path.exists(args.config):
        print(f"Optimized config not found at {args.config}")
        # Try regular config
        if os.path.exists('config/settings.yaml'):
            print("Using config/settings.yaml instead")
            args.config = 'config/settings.yaml'
        else:
            print("ERROR: No configuration file found!")
            print("Please create config/settings.yaml or config/settings_optimized.yaml")
            sys.exit(1)
    
    # Create monitor
    monitor = OptimizedAirbnbMonitor(args.config)
    
    # Enable display if requested
    if args.display:
        monitor.config['display'] = {'enabled': True}
    
    try:
        monitor.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()