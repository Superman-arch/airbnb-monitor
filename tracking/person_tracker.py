"""Person detection and tracking using YOLOv8 and ByteTrack."""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import os

from tracking.bytetrack import ByteTracker


class PersonTracker:
    """Person detection and tracking using YOLOv8."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize person tracker."""
        self.config = config
        
        # Model settings
        self.model_path = config.get('detection', {}).get('person_model', 'yolov8n.pt')
        self.confidence_threshold = config.get('detection', {}).get('confidence_threshold', 0.45)
        self.nms_threshold = config.get('detection', {}).get('nms_threshold', 0.45)
        
        # Tracking settings
        tracking_config = config.get('tracking', {})
        self.tracker = ByteTracker(
            track_thresh=tracking_config.get('track_thresh', 0.25),
            match_thresh=tracking_config.get('match_thresh', 0.8),
            track_buffer=tracking_config.get('track_buffer', 30),
            min_box_area=tracking_config.get('min_box_area', 100)
        )
        
        self.model = None
        self.person_class_id = 0  # COCO person class
        
    def initialize(self) -> bool:
        """Initialize the YOLOv8 model."""
        try:
            # Load YOLOv8 model
            self.model = YOLO(self.model_path)
            
            # Download model if not exists
            if not os.path.exists(self.model_path):
                print(f"Downloading YOLOv8 model: {self.model_path}")
                self.model = YOLO(self.model_path)
            
            self.is_initialized = True
            print(f"Person tracker initialized with model: {self.model_path}")
            return True
            
        except Exception as e:
            print(f"Failed to initialize person tracker: {e}")
            return False
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect and track persons in frame.
        
        Returns:
            List of tracked persons with format:
            [{
                'bbox': [x1, y1, x2, y2],
                'confidence': float,
                'class': 'person',
                'track_id': int,
                'center': (x, y)
            }]
        """
        if not self.is_initialized or self.model is None:
            return []
        
        # Run YOLOv8 detection
        results = self.model(frame, conf=self.confidence_threshold, iou=self.nms_threshold)
        
        # Extract person detections
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                # Check if detection is a person
                class_id = int(boxes.cls[i])
                if class_id == self.person_class_id:
                    # Get bbox in xyxy format
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    confidence = float(boxes.conf[i])
                    
                    # Add to detections for tracking
                    detections.append([x1, y1, x2, y2, confidence, class_id])
        
        # Convert to numpy array for ByteTrack
        if len(detections) > 0:
            detections = np.array(detections)
        else:
            detections = np.empty((0, 6))
        
        # Update tracker
        tracks = self.tracker.update(detections)
        
        # Format output
        tracked_persons = []
        for track in tracks:
            x1, y1, x2, y2 = track.tlbr
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            tracked_persons.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': track.score,
                'class': 'person',
                'track_id': track.track_id,
                'center': (center_x, center_y)
            })
        
        return tracked_persons
    
    def draw_tracks(self, frame: np.ndarray, tracks: List[Dict[str, Any]]) -> np.ndarray:
        """Draw tracking results on frame."""
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            track_id = track['track_id']
            confidence = track['confidence']
            
            # Draw bounding box
            color = self._get_color(track_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID and confidence
            label = f"ID:{track_id} ({confidence:.2f})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw label background
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 4), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw center point
            cx, cy = track['center']
            cv2.circle(frame, (cx, cy), 3, color, -1)
        
        return frame
    
    def _get_color(self, track_id: int) -> Tuple[int, int, int]:
        """Get consistent color for track ID."""
        # Generate color based on track ID
        np.random.seed(track_id)
        color = np.random.randint(0, 255, 3).tolist()
        return tuple(color)
    
    def get_track_by_id(self, track_id: int) -> Optional[Dict[str, Any]]:
        """Get track information by ID."""
        for track in self.tracker.tracked_stracks:
            if track.track_id == track_id:
                x1, y1, x2, y2 = track.tlbr
                return {
                    'track_id': track_id,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'frame_id': track.frame_id,
                    'start_frame': track.start_frame
                }
        return None
    
    def reset_tracker(self):
        """Reset the tracker."""
        tracking_config = self.config.get('tracking', {})
        self.tracker = ByteTracker(
            track_thresh=tracking_config.get('track_thresh', 0.25),
            match_thresh=tracking_config.get('match_thresh', 0.8),
            track_buffer=tracking_config.get('track_buffer', 30),
            min_box_area=tracking_config.get('min_box_area', 100)
        )
    
    def cleanup(self):
        """Cleanup resources."""
        self.model = None
        self.tracker = None