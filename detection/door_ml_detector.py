"""ML-based door detection using Roboflow/YOLOv8 model."""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import os
import json

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")

try:
    from roboflow import Roboflow
    ROBOFLOW_AVAILABLE = True
except ImportError:
    ROBOFLOW_AVAILABLE = False
    print("Warning: roboflow not installed. Install with: pip install roboflow")


class DoorML:
    """Represents a detected door using ML."""
    
    def __init__(self, door_id: str, bbox: Tuple[int, int, int, int]):
        """Initialize a door."""
        self.id = door_id
        self.bbox = bbox  # (x, y, width, height)
        self.current_state = "unknown"
        self.previous_state = "unknown"
        self.last_change = datetime.now()
        self.open_duration = timedelta()
        self.change_history = deque(maxlen=100)
        self.confidence = 0.0
        
        # Temporal filtering
        self.state_buffer = deque(maxlen=5)  # Fewer frames needed with ML
        self.pending_state = None
        self.pending_state_count = 0
        self.last_event_time = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'bbox': self.bbox,
            'state': self.current_state,
            'confidence': self.confidence,
            'last_change': self.last_change.isoformat(),
            'open_duration': str(self.open_duration)
        }


class DoorMLDetector:
    """ML-based door detection using YOLOv8/Roboflow model."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize ML door detector."""
        self.config = config
        door_config = config.get('door_detection', {})
        
        # Model settings
        self.model_path = door_config.get('model_path', 'models/door_model.pt')
        self.use_roboflow = door_config.get('use_roboflow', False)
        self.roboflow_api_key = door_config.get('roboflow_api_key', '')
        self.confidence_threshold = door_config.get('confidence_threshold', 0.6)
        
        # Temporal filtering (less aggressive with ML)
        self.state_confirmation_frames = door_config.get('state_confirmation_frames', 3)
        self.min_seconds_between_changes = door_config.get('min_seconds_between_changes', 1.0)
        
        # State
        self.model = None
        self.doors = {}  # track_id -> DoorML
        self.frame_count = 0
        self.door_counter = 0
        
        # Class mappings
        self.class_names = {
            'open_door': 'open',
            'open': 'open',
            'opened': 'open',
            'closed_door': 'closed',
            'closed': 'closed',
            'door': 'unknown'
        }
        
    def initialize(self):
        """Initialize the ML model."""
        if self.use_roboflow and ROBOFLOW_AVAILABLE and self.roboflow_api_key:
            try:
                # Initialize Roboflow
                rf = Roboflow(api_key=self.roboflow_api_key)
                project = rf.workspace().project("is-my-door-open")
                self.model = project.version(2).model
                print("Roboflow door detection model loaded")
                return True
            except Exception as e:
                print(f"Failed to load Roboflow model: {e}")
                
        # Fallback to local YOLOv8 model
        if YOLO_AVAILABLE:
            try:
                if os.path.exists(self.model_path):
                    self.model = YOLO(self.model_path)
                    print(f"Loaded local door model from {self.model_path}")
                else:
                    # Use default YOLOv8 and filter for doors
                    print("Door model not found, using default YOLOv8n")
                    self.model = YOLO('yolov8n.pt')
                return True
            except Exception as e:
                print(f"Failed to load YOLO model: {e}")
                
        print("WARNING: No ML model available, door detection disabled")
        return False
    
    def process_frame(self, frame: np.ndarray) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Process a frame for door detection using ML.
        
        Returns:
            (always False for compatibility, door_events)
        """
        if self.model is None:
            return False, []
            
        self.frame_count += 1
        events = []
        
        # Run inference
        detections = self._detect_doors(frame)
        
        # Process each detection
        current_door_ids = set()
        
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            
            # Map class to state
            state = self.class_names.get(class_name, 'unknown')
            
            # Generate door ID based on location (spatial tracking)
            door_id = self._get_door_id_for_location(bbox)
            current_door_ids.add(door_id)
            
            # Get or create door object
            if door_id not in self.doors:
                self.doors[door_id] = DoorML(door_id, bbox)
                # Send discovery event
                events.append({
                    'event': 'door_discovered',
                    'door_id': door_id,
                    'bbox': bbox,
                    'initial_state': state,
                    'timestamp': datetime.now(),
                    'confidence': confidence
                })
                print(f"Discovered {door_id} at {bbox}, state: {state}")
            
            door = self.doors[door_id]
            door.confidence = confidence
            
            # Update bbox (doors might shift slightly)
            door.bbox = bbox
            
            # Apply temporal filtering for state changes
            door.state_buffer.append(state)
            
            # Check if state is changing
            if state != door.current_state and state != 'unknown':
                if door.pending_state != state:
                    door.pending_state = state
                    door.pending_state_count = 1
                else:
                    door.pending_state_count += 1
                
                # Confirm state change
                time_since_last_change = (datetime.now() - door.last_change).total_seconds()
                time_since_last_event = (datetime.now() - door.last_event_time).total_seconds()
                
                if (door.pending_state_count >= self.state_confirmation_frames and
                    time_since_last_change >= self.min_seconds_between_changes and
                    time_since_last_event >= 0.5):
                    
                    # State change confirmed
                    door.previous_state = door.current_state
                    door.current_state = state
                    door.last_change = datetime.now()
                    door.last_event_time = datetime.now()
                    
                    # Create event
                    event = {
                        'event': f'door_{state}',
                        'door_id': door_id,
                        'timestamp': door.last_change,
                        'previous_state': door.previous_state,
                        'current_state': state,
                        'confidence': confidence,
                        'bbox': bbox
                    }
                    events.append(event)
                    
                    print(f"{door_id}: {door.previous_state} -> {state} (confidence: {confidence:.2f})")
                    
                    # Reset pending
                    door.pending_state = None
                    door.pending_state_count = 0
            
            # Track open duration
            if door.current_state == "open":
                door.open_duration = datetime.now() - door.last_change
                
                # Alert if door left open
                if door.open_duration.total_seconds() > 300:  # 5 minutes
                    if self.frame_count % 150 == 0:  # Every 5 seconds
                        events.append({
                            'event': 'door_left_open',
                            'door_id': door_id,
                            'duration_seconds': door.open_duration.total_seconds(),
                            'timestamp': datetime.now()
                        })
        
        # Remove doors that are no longer detected
        for door_id in list(self.doors.keys()):
            if door_id not in current_door_ids:
                # Door no longer visible, but keep tracking it
                pass
        
        return False, events
    
    def _detect_doors(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run ML inference to detect doors."""
        detections = []
        
        if self.use_roboflow and hasattr(self.model, 'predict'):
            # Roboflow inference
            try:
                result = self.model.predict(frame, confidence=self.confidence_threshold * 100).json()
                
                for pred in result.get('predictions', []):
                    x = pred['x'] - pred['width'] / 2
                    y = pred['y'] - pred['height'] / 2
                    
                    detections.append({
                        'bbox': (int(x), int(y), int(pred['width']), int(pred['height'])),
                        'confidence': pred['confidence'],
                        'class': pred['class']
                    })
            except Exception as e:
                print(f"Roboflow inference error: {e}")
                
        elif YOLO_AVAILABLE and self.model:
            # YOLOv8 inference
            try:
                results = self.model(frame, conf=self.confidence_threshold, verbose=False)
                
                for r in results:
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            
                            # Get class name
                            if hasattr(self.model, 'names'):
                                class_name = self.model.names.get(cls, 'door')
                            else:
                                class_name = 'door'
                            
                            # Filter for door-related classes
                            if 'door' in class_name.lower() or class_name in self.class_names:
                                detections.append({
                                    'bbox': (int(x1), int(y1), int(x2-x1), int(y2-y1)),
                                    'confidence': conf,
                                    'class': class_name
                                })
            except Exception as e:
                print(f"YOLO inference error: {e}")
        
        return detections
    
    def _get_door_id_for_location(self, bbox: Tuple[int, int, int, int]) -> str:
        """Get or create door ID based on spatial location."""
        x, y, w, h = bbox
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Check if this location matches an existing door
        for door_id, door in self.doors.items():
            dx, dy, dw, dh = door.bbox
            dcenter_x = dx + dw // 2
            dcenter_y = dy + dh // 2
            
            # If centers are within 50 pixels, consider it the same door
            if abs(center_x - dcenter_x) < 50 and abs(center_y - dcenter_y) < 50:
                return door_id
        
        # New door location
        self.door_counter += 1
        return f"door_{self.door_counter:03d}"
    
    def draw_doors(self, frame: np.ndarray) -> np.ndarray:
        """Draw door overlays on frame."""
        overlay = frame.copy()
        
        for door in self.doors.values():
            x, y, w, h = door.bbox
            
            # Choose color based on state
            if door.current_state == "open":
                color = (0, 255, 0)  # Green
                label_text = "OPEN"
            elif door.current_state == "closed":
                color = (0, 0, 255)  # Red
                label_text = "CLOSED"
            else:
                color = (0, 255, 255)  # Yellow
                label_text = "UNKNOWN"
            
            # Draw rectangle
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            
            # Draw label with confidence
            label = f"{door.id}: {label_text}"
            if door.confidence > 0:
                label += f" ({door.confidence:.0%})"
            
            if door.current_state == "open" and door.open_duration.total_seconds() > 0:
                label += f" ({int(door.open_duration.total_seconds())}s)"
            
            # Draw background for label
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(overlay, (x, y - label_size[1] - 4), 
                         (x + label_size[0], y), color, -1)
            
            # Draw text
            cv2.putText(overlay, label, (x, y - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return overlay
    
    def get_doors(self) -> List[DoorML]:
        """Get list of detected doors."""
        return list(self.doors.values())
    
    def get_door_states(self) -> Dict[str, str]:
        """Get current state of all doors."""
        return {door.id: door.current_state for door in self.doors.values()}
    
    def save_doors(self, filepath: str = "config/detected_doors_ml.json"):
        """Save detected doors to file."""
        door_data = [door.to_dict() for door in self.doors.values()]
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(door_data, f, indent=2, default=str)
    
    def reset(self):
        """Reset detector."""
        self.doors = {}
        self.door_counter = 0
        self.frame_count = 0
    
    # Compatibility methods with old DoorDetector
    def start_learning(self):
        """No learning phase needed with ML."""
        pass