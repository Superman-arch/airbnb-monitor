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

# Try to import inference detector first, fall back to edge detection
try:
    from detection.door_inference_detector import DoorInferenceDetector
    INFERENCE_AVAILABLE = True
    print("Using Roboflow Inference for door detection")
except ImportError:
    from detection.door_detector import DoorDetector
    INFERENCE_AVAILABLE = False
    print("Using edge detection for doors (inference not available)")

from tracking.person_tracker import PersonTracker
from tracking.journey_manager import JourneyManager
from notifications.webhook_handler import WebhookHandler
from storage.video_manager import CircularVideoBuffer

# Import web interface
try:
    from flask import Flask
    from flask_cors import CORS
    from web.app import app, broadcast_event, broadcast_log, broadcast_stats
    WEB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Web interface not available: {e}")
    WEB_AVAILABLE = False
    broadcast_log = lambda msg, level='info': None  # No-op if web not available
    broadcast_stats = lambda stats: None


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
        
        # Use inference detector if available
        if INFERENCE_AVAILABLE:
            self.door_detector = DoorInferenceDetector(self.config)
        else:
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
                print("Warning: Door detector initialization failed")
                print("Continuing without door detection...")
        
        # Initialize camera
        if not self.initialize_camera():
            print("Failed to initialize camera")
            return False
        
        # Start door learning phase (only for edge detection)
        if hasattr(self.door_detector, 'start_learning'):
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
                # Initialize the web app with monitor instance
                from web.app import init_app
                init_app(self.config, monitor=self)
                
                # Pass reference to this monitor for frame access
                app.config['monitor'] = self
                app.config['zone_detector'] = self.zone_detector
                app.config['journey_manager'] = self.journey_manager
                
                # Get host and port from config
                web_config = self.config.get('web', {})
                host = web_config.get('host', '0.0.0.0')
                port = web_config.get('port', 5000)
                
                print(f"Starting web interface at http://{host}:{port}")
                
                # Run Flask app (blocking call)
                app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)
                
            except Exception as e:
                print(f"Web server error: {e}")
                import traceback
                traceback.print_exc()
        
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
            
            # ALWAYS: Door detection
            if INFERENCE_AVAILABLE:
                # Inference detection (no learning phase)
                _, door_events = self.door_detector.process_frame(frame)
            else:
                # Edge detection with learning phase
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
                    door_name = event.get('door_name', event.get('door_id', 'unknown'))
                    log_msg = f"Door event: {door_name} - {event['event']}"
                    print(log_msg)
                    if WEB_AVAILABLE:
                        broadcast_log(log_msg, 'warning' if 'open' in event['event'] else 'info')
                    
                    # Broadcast to web interface
                    if WEB_AVAILABLE:
                        try:
                            # Store door event in journey manager for persistence
                            self.journey_manager.store_door_event(event)
                            # Broadcast to web clients
                            broadcast_event(event)
                        except Exception as e:
                            print(f"Failed to broadcast door event: {e}")
            
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
                            
                            # Broadcast to web interface
                            if WEB_AVAILABLE:
                                try:
                                    broadcast_event(event)
                                except Exception as e:
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
                stats_msg = f"FPS: {current_fps:.1f} | Doors: {len(self.door_detector.doors)} | Persons: {len(tracked_persons)} | Frame: {self.frame_counter}"
                print(stats_msg)
                
                # Broadcast stats to web interface
                if WEB_AVAILABLE:
                    broadcast_stats({
                        'fps': current_fps,
                        'doors': len(self.door_detector.doors),
                        'persons': len(tracked_persons),
                        'frame': self.frame_counter
                    })
                    broadcast_log(stats_msg, 'info')
            
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
                x, y, w, h = zone.bbox
                # Draw semi-transparent zone
                zone_overlay = overlay.copy()
                cv2.rectangle(zone_overlay, (x, y), (x + w, y + h), COLORS['zone'], 2)
                cv2.addWeighted(zone_overlay, 0.3, overlay, 0.7, 0, overlay)
        
        # Draw door detections with labels
        if hasattr(self.door_detector, 'doors'):
            for door_id, door in self.door_detector.doors.items():
                x, y, w, h = door.bbox
                
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