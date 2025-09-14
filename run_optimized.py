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
from typing import Dict, Any
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Optimize for Jetson
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Import our modules
from detection.motion_detector import MotionZoneDetector
from detection.door_detector import DoorDetector
from tracking.person_tracker import PersonTracker
from tracking.journey_manager import JourneyManager
from notifications.webhook_handler import WebhookHandler
from storage.video_manager import CircularVideoBuffer

# Import web interface
try:
    from flask import Flask
    from flask_cors import CORS
    from flask_socketio import SocketIO
    from web.app import app, init_app, update_frame
    WEB_AVAILABLE = True
except ImportError:
    print("Warning: Flask not available, web interface disabled")
    WEB_AVAILABLE = False


class OptimizedAirbnbMonitor:
    """Optimized version for better FPS on Jetson Nano."""
    
    def __init__(self, config_path: str = "config/settings_optimized.yaml"):
        """Initialize optimized monitor."""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print("Initializing Optimized Airbnb Monitor...")
        print("Target FPS: 10-15 with door detection")
        
        # Initialize components
        self.zone_detector = MotionZoneDetector(self.config)
        self.door_detector = DoorDetector(self.config)
        self.person_tracker = PersonTracker(self.config)
        self.journey_manager = JourneyManager(self.config)
        self.webhook_handler = WebhookHandler(self.config)
        self.video_buffer = CircularVideoBuffer(self.config)
        
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
        
        # Initialize camera
        if not self.initialize_camera():
            print("Failed to initialize camera")
            return False
        
        # Start door learning phase
        self.door_detector.start_learning()
        
        # Start services
        self.webhook_handler.start()
        self.video_buffer.start(self.resolution, self.fps)
        
        # Start web server
        self.start_web_server()
        
        self.running = True
        
        # Start processing
        self.process_loop_optimized()
        
        return True
    
    def start_web_server(self):
        """Start the Flask web server in a separate thread."""
        if not WEB_AVAILABLE:
            print("Web interface not available (Flask not installed)")
            return
        
        def run_web_server():
            try:
                # Initialize the web app with our components
                init_app(self.config)
                
                # Pass reference to this monitor for frame access
                app.config['monitor'] = self
                
                # Get host and port from config
                web_config = self.config.get('web', {})
                host = web_config.get('host', '0.0.0.0')
                port = web_config.get('port', 5000)
                
                print(f"Starting web interface at http://{host}:{port}")
                
                # Run Flask app (blocking call)
                app.run(host=host, port=port, debug=False, use_reloader=False)
                
            except Exception as e:
                print(f"Web server error: {e}")
        
        # Start web server in separate thread
        self.web_thread = Thread(target=run_web_server, daemon=True)
        self.web_thread.start()
        print("Web server thread started")
    
    def process_loop_optimized(self):
        """Optimized processing loop with frame skipping."""
        print("Optimized processing started. Press 'q' to quit")
        print("Door learning phase: 30 seconds...")
        
        # Tracking variables
        last_person_detection = 0
        last_zone_detection = 0
        tracked_persons = []
        zones = []
        door_learning_complete = False
        
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
            
            # ALWAYS: Door detection (lightweight edge detection)
            learning_complete, door_events = self.door_detector.process_frame(frame)
            
            if learning_complete and not door_learning_complete:
                door_learning_complete = True
                print(f"Door learning complete! Found {len(self.door_detector.doors)} doors")
                
                # Send door discovery events
                for event in door_events:
                    self.webhook_handler.send_event(event, frame)
            
            # Handle door state change events
            for event in door_events:
                if event['event'] in ['door_opened', 'door_closed', 'door_moving', 'door_left_open', 'door_discovered']:
                    # Send door event to specific webhook
                    self.send_door_event(event, frame)
                    print(f"Door event: {event['event']} - {event.get('door_id', 'unknown')}")
            
            # PERIODIC: Zone detection (every 30 frames)
            if self.frame_counter - last_zone_detection >= self.zone_detect_interval:
                motion_detections = self.zone_detector.detect(frame)
                zones = self.zone_detector.get_zones()
                last_zone_detection = self.frame_counter
            
            # PERIODIC: Person detection (every 3 frames)
            if self.frame_counter - last_person_detection >= self.person_detect_interval:
                tracked_persons = self.person_tracker.detect(frame)
                last_person_detection = self.frame_counter
                
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
            
            # ALWAYS: Record video (if enabled)
            if self.recording_enabled:
                self.video_buffer.add_frame(frame, timestamp)
            
            # Draw overlays
            display_frame = frame.copy()
            display_frame = self.door_detector.draw_doors(display_frame)
            if tracked_persons:
                display_frame = self.person_tracker.draw_tracks(display_frame, tracked_persons)
            
            # Calculate and display FPS
            self.update_fps()
            if self.frame_counter % 30 == 0:
                current_fps = self.get_current_fps()
                print(f"FPS: {current_fps:.1f} | Doors: {len(self.door_detector.doors)} | "
                      f"Persons: {len(tracked_persons)} | Frame: {self.frame_counter}")
            
            # Show frame (optional - disable for better performance)
            if self.config.get('display', {}).get('enabled', False):
                self.draw_stats(display_frame)
                cv2.imshow('Optimized Airbnb Monitor', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord('r'):
                    print("Resetting door detection...")
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
        print(f"  Doors detected: {len(self.door_detector.doors)}")
        
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