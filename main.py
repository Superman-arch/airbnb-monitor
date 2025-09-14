#!/usr/bin/env python3
"""
Airbnb Monitoring System - Main Application
For Jetson Nano with USB webcam or CSI camera
"""

import cv2
import yaml
import argparse
import signal
import sys
from datetime import datetime
from threading import Thread
import numpy as np

# Import our modules
from detection.motion_detector import MotionZoneDetector
from tracking.person_tracker import PersonTracker
from tracking.journey_manager import JourneyManager
from notifications.webhook_handler import WebhookHandler
from storage.video_manager import CircularVideoBuffer


class AirbnbMonitor:
    """Main application for Airbnb monitoring system."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize the monitoring system."""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print("Initializing Airbnb Monitoring System...")
        
        # Initialize components
        self.zone_detector = MotionZoneDetector(self.config)
        self.person_tracker = PersonTracker(self.config)
        self.journey_manager = JourneyManager(self.config)
        self.webhook_handler = WebhookHandler(self.config)
        self.video_buffer = CircularVideoBuffer(self.config)
        
        # Camera settings
        self.camera = None
        self.camera_id = "camera_1"
        self.resolution = tuple(self.config['camera']['resolution'])
        self.fps = self.config['camera']['fps']
        
        # State
        self.running = False
        self.display_enabled = True
        self.recording_enabled = True
        
        # Statistics
        self.frame_count = 0
        self.start_time = None
        
    def initialize_camera(self):
        """Initialize camera (USB or CSI)."""
        camera_source = self.config['camera']['source']
        
        if isinstance(camera_source, int):
            # USB camera
            print(f"Initializing USB camera at index {camera_source}")
            self.camera = cv2.VideoCapture(camera_source)
            
        elif camera_source.startswith('csi'):
            # CSI camera for Jetson Nano
            print("Initializing CSI camera")
            gst_pipeline = self._get_gstreamer_pipeline()
            self.camera = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            
        else:
            # RTSP or file
            print(f"Initializing camera from: {camera_source}")
            self.camera = cv2.VideoCapture(camera_source)
        
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.camera.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Verify camera is working
        ret, frame = self.camera.read()
        if not ret:
            raise RuntimeError("Failed to read from camera")
        
        print(f"Camera initialized: {frame.shape}")
        return True
    
    def _get_gstreamer_pipeline(self):
        """Get GStreamer pipeline for Jetson Nano CSI camera."""
        return (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM), "
            f"width={self.resolution[0]}, height={self.resolution[1]}, "
            f"format=NV12, framerate={self.fps}/1 ! "
            f"nvvidconv flip-method=0 ! "
            f"video/x-raw, format=BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=BGR ! appsink"
        )
    
    def start(self):
        """Start the monitoring system."""
        print("Starting Airbnb Monitoring System...")
        
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
        
        # Start services
        self.webhook_handler.start()
        self.video_buffer.start(self.resolution, self.fps)
        
        self.running = True
        self.start_time = datetime.now()
        
        # Start processing loop
        self.process_loop()
        
        return True
    
    def process_loop(self):
        """Main processing loop."""
        print("Processing started. Press 'q' to quit, 's' to save zones, 't' for test webhook")
        
        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to read frame")
                continue
            
            self.frame_count += 1
            timestamp = datetime.now()
            
            # Detect motion zones
            motion_detections = self.zone_detector.detect(frame)
            zones = self.zone_detector.get_zones()
            
            # Detect and track persons
            tracked_persons = self.person_tracker.detect(frame)
            
            # Process person journeys and generate events
            events = self.journey_manager.process_tracks(
                tracked_persons, self.camera_id, zones, timestamp
            )
            
            # Handle events (send notifications)
            for event in events:
                # Store event in database
                snapshot_path = None
                if event['action'] == 'entry':
                    # Save snapshot for entry events
                    snapshot_path = self.video_buffer.save_snapshot(frame, event['person_id'])
                
                self.journey_manager.store_event(event, snapshot_path)
                
                # Send webhook notification
                journey = self.journey_manager.get_person_journey(event['person_id'])
                self.webhook_handler.send_event(event, frame, journey)
                
                # Log event
                print(f"Event: Person {event['person_id']} {event['action']} "
                      f"zone {event.get('zone_name', 'unknown')} at {timestamp:%H:%M:%S}")
            
            # Add frame to video buffer
            if self.recording_enabled:
                self.video_buffer.add_frame(frame, timestamp)
            
            # Display frame with overlays
            if self.display_enabled:
                display_frame = self.draw_overlays(frame, zones, tracked_persons, motion_detections)
                cv2.imshow('Airbnb Monitor', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord('s'):
                    self.zone_detector.save_zones()
                    print("Zones saved")
                elif key == ord('t'):
                    self.webhook_handler.send_test_notification()
                elif key == ord('h'):
                    self.show_help()
                elif key == ord('m'):
                    # Show motion heatmap
                    heatmap = self.zone_detector.get_motion_heatmap()
                    if heatmap is not None:
                        cv2.imshow('Motion Heatmap', heatmap)
    
    def draw_overlays(self, frame, zones, tracked_persons, motion_detections):
        """Draw overlays on frame."""
        overlay = frame.copy()
        
        # Draw zones
        for zone in zones:
            # Draw zone polygon
            pts = zone.coordinates.astype(np.int32)
            cv2.polylines(overlay, [pts], True, (0, 255, 255), 2)
            
            # Draw zone label
            M = cv2.moments(pts)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                label = f"{zone.name} ({zone.type})"
                cv2.putText(overlay, label, (cx - 50, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw tracked persons
        overlay = self.person_tracker.draw_tracks(overlay, tracked_persons)
        
        # Draw motion detections (optional, can be noisy)
        if False:  # Set to True to see motion regions
            for detection in motion_detections:
                x1, y1, x2, y2 = detection['bbox']
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 1)
        
        # Draw statistics
        self.draw_stats(overlay)
        
        return overlay
    
    def draw_stats(self, frame):
        """Draw statistics on frame."""
        if self.start_time:
            elapsed = datetime.now() - self.start_time
            fps = self.frame_count / elapsed.total_seconds()
        else:
            fps = 0
        
        # Get current occupancy
        occupancy = {}
        for zone in self.zone_detector.get_zones():
            occupants = self.journey_manager.get_zone_occupancy(zone.id)
            if occupants:
                occupancy[zone.name] = len(occupants)
        
        # Draw stats
        y = 30
        stats = [
            f"FPS: {fps:.1f}",
            f"Zones: {len(self.zone_detector.get_zones())}",
            f"Persons: {len(self.person_tracker.tracker.tracked_stracks)}",
            f"Time: {datetime.now():%H:%M:%S}"
        ]
        
        for stat in stats:
            cv2.putText(frame, stat, (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 25
        
        # Draw occupancy
        if occupancy:
            y += 10
            cv2.putText(frame, "Occupancy:", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 25
            for zone_name, count in occupancy.items():
                cv2.putText(frame, f"  {zone_name}: {count}", (10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y += 20
    
    def show_help(self):
        """Show help information."""
        print("\n=== Airbnb Monitor Controls ===")
        print("q - Quit")
        print("s - Save zones")
        print("t - Send test webhook")
        print("h - Show this help")
        print("m - Show motion heatmap")
        print("===============================\n")
    
    def stop(self):
        """Stop the monitoring system."""
        print("\nStopping Airbnb Monitoring System...")
        
        self.running = False
        
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
        if self.start_time:
            elapsed = datetime.now() - self.start_time
            print(f"\nSession Statistics:")
            print(f"  Duration: {elapsed}")
            print(f"  Frames processed: {self.frame_count}")
            print(f"  Average FPS: {self.frame_count / elapsed.total_seconds():.1f}")
            
            stats = self.webhook_handler.get_statistics()
            print(f"  Notifications sent: {stats['notifications_sent']}")
            print(f"  Notifications failed: {stats['notifications_failed']}")
        
        print("Shutdown complete")


def signal_handler(sig, frame):
    """Handle shutdown signal."""
    print("\nShutdown signal received")
    sys.exit(0)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Airbnb Monitoring System')
    parser.add_argument('--config', default='config/settings.yaml',
                       help='Path to configuration file')
    parser.add_argument('--no-display', action='store_true',
                       help='Run without display (headless)')
    parser.add_argument('--no-recording', action='store_true',
                       help='Disable video recording')
    parser.add_argument('--test-webhook', action='store_true',
                       help='Send test webhook and exit')
    
    args = parser.parse_args()
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create monitor
    monitor = AirbnbMonitor(args.config)
    
    # Handle test webhook
    if args.test_webhook:
        monitor.webhook_handler.webhook_url = input("Enter webhook URL: ")
        if monitor.webhook_handler.send_test_notification():
            print("Test successful!")
        else:
            print("Test failed!")
        return
    
    # Set options
    monitor.display_enabled = not args.no_display
    monitor.recording_enabled = not args.no_recording
    
    try:
        # Start monitoring
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