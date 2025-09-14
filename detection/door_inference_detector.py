"""Door detection using Roboflow Inference Server (local, offline)."""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import json
import os

try:
    from inference_sdk import InferenceHTTPClient
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False
    print("Warning: inference-sdk not installed. Install with: pip install inference-sdk")


class DoorInference:
    """Represents a door detected by inference."""
    
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
        
        # Temporal filtering (minimal with ML)
        self.state_buffer = deque(maxlen=3)  # Only 3 frames needed
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


class DoorInferenceDetector:
    """Door detection using local Roboflow Inference Server."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize inference-based door detector."""
        self.config = config
        door_config = config.get('door_detection', {})
        
        # Inference settings
        self.inference_url = door_config.get('inference_url', 'http://localhost:9001')
        self.model_id = door_config.get('model_id', 'is-my-door-open/2')
        self.confidence_threshold = door_config.get('confidence_threshold', 0.6)
        
        # Temporal filtering (minimal)
        self.state_confirmation_frames = door_config.get('state_confirmation_frames', 2)
        self.min_seconds_between_changes = door_config.get('min_seconds_between_changes', 1.0)
        
        # State
        self.client = None
        self.doors = {}  # location_key -> DoorInference
        self.frame_count = 0
        self.door_counter = 0
        self.initialized = False
        
        # Performance optimization
        self.process_every_n_frames = door_config.get('process_every_n_frames', 2)
        
    def initialize(self):
        """Initialize the inference client."""
        if not INFERENCE_AVAILABLE:
            print("ERROR: inference-sdk not available")
            return False
            
        try:
            # Connect to local inference server
            self.client = InferenceHTTPClient(
                api_url=self.inference_url
            )
            
            # Test connection
            print(f"Connecting to inference server at {self.inference_url}...")
            
            # Try a simple health check
            import requests
            response = requests.get(f"{self.inference_url}/health", timeout=5)
            if response.status_code == 200:
                print("✓ Connected to Roboflow Inference Server")
                self.initialized = True
                return True
            else:
                print(f"✗ Server responded with status {response.status_code}")
                
        except Exception as e:
            print(f"✗ Failed to connect to inference server: {e}")
            print(f"Make sure the server is running at {self.inference_url}")
            print("Start it with: ./start_inference.sh")
            
        return False
    
    def process_frame(self, frame: np.ndarray) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Process frame using inference server.
        
        Returns:
            (always False for compatibility, door_events)
        """
        if not self.initialized or self.client is None:
            return False, []
        
        self.frame_count += 1
        
        # Skip frames for performance
        if self.frame_count % self.process_every_n_frames != 0:
            return False, []
        
        events = []
        
        try:
            # Run inference on the frame
            result = self.client.infer(
                frame,
                model_id=self.model_id
            )
            
            # Process predictions
            predictions = result.get('predictions', [])
            current_door_ids = set()
            
            for pred in predictions:
                # Extract detection data
                x = pred['x'] - pred['width'] / 2
                y = pred['y'] - pred['height'] / 2
                bbox = (int(x), int(y), int(pred['width']), int(pred['height']))
                
                class_name = pred['class']
                confidence = pred['confidence']
                
                # Skip low confidence detections
                if confidence < self.confidence_threshold:
                    continue
                
                # Map class to state
                if 'open' in class_name.lower():
                    state = 'open'
                elif 'closed' in class_name.lower():
                    state = 'closed'
                else:
                    state = 'unknown'
                
                # Get door ID based on location
                door_id = self._get_door_id_for_location(bbox)
                current_door_ids.add(door_id)
                
                # Get or create door
                if door_id not in self.doors:
                    self.doors[door_id] = DoorInference(door_id, bbox)
                    # Discovery event
                    events.append({
                        'event': 'door_discovered',
                        'door_id': door_id,
                        'bbox': bbox,
                        'initial_state': state,
                        'timestamp': datetime.now(),
                        'confidence': confidence,
                        'event_type': 'door'
                    })
                    print(f"Discovered {door_id}: {state} ({confidence:.0%})")
                
                door = self.doors[door_id]
                door.confidence = confidence
                door.bbox = bbox  # Update position
                
                # Check for state change
                door.state_buffer.append(state)
                
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
                        
                        # Confirmed state change
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
                            'bbox': bbox,
                            'event_type': 'door'
                        }
                        events.append(event)
                        
                        print(f"{door_id}: {door.previous_state} -> {state} ({confidence:.0%})")
                        
                        # Reset pending
                        door.pending_state = None
                        door.pending_state_count = 0
                
                # Track open duration
                if door.current_state == "open":
                    door.open_duration = datetime.now() - door.last_change
                    
                    # Alert if left open
                    if door.open_duration.total_seconds() > 300:  # 5 minutes
                        if self.frame_count % 150 == 0:
                            events.append({
                                'event': 'door_left_open',
                                'door_id': door_id,
                                'duration_seconds': door.open_duration.total_seconds(),
                                'timestamp': datetime.now(),
                                'event_type': 'door'
                            })
            
        except Exception as e:
            print(f"Inference error: {e}")
            # Server might be down, try to reconnect on next frame
            self.initialized = False
        
        return False, events
    
    def _get_door_id_for_location(self, bbox: Tuple[int, int, int, int]) -> str:
        """Get door ID based on spatial location."""
        x, y, w, h = bbox
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Check existing doors
        for door_id, door in self.doors.items():
            dx, dy, dw, dh = door.bbox
            dcenter_x = dx + dw // 2
            dcenter_y = dy + dh // 2
            
            # Same door if centers within 100 pixels
            if abs(center_x - dcenter_x) < 100 and abs(center_y - dcenter_y) < 100:
                return door_id
        
        # New door
        self.door_counter += 1
        return f"door_{self.door_counter:03d}"
    
    def draw_doors(self, frame: np.ndarray) -> np.ndarray:
        """Draw door overlays."""
        overlay = frame.copy()
        
        for door in self.doors.values():
            x, y, w, h = door.bbox
            
            # Color based on state
            if door.current_state == "open":
                color = (0, 255, 0)  # Green
                icon = "🔓"
            elif door.current_state == "closed":
                color = (0, 0, 255)  # Red  
                icon = "🔒"
            else:
                color = (0, 255, 255)  # Yellow
                icon = "🚪"
            
            # Draw bounding box
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            
            # Label
            label = f"{door.id}: {door.current_state.upper()}"
            if door.confidence > 0:
                label += f" ({door.confidence:.0%})"
            if door.current_state == "open" and door.open_duration.total_seconds() > 0:
                label += f" ({int(door.open_duration.total_seconds())}s)"
            
            # Label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(overlay, (x, y - label_size[1] - 4),
                         (x + label_size[0], y), color, -1)
            
            # Draw text
            cv2.putText(overlay, label, (x, y - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw confidence bar
            if door.confidence > 0:
                bar_width = int(w * door.confidence)
                cv2.rectangle(overlay, (x, y + h + 2),
                            (x + bar_width, y + h + 6), color, -1)
        
        # Show inference status
        status = "Inference: ✓ Connected" if self.initialized else "Inference: ✗ Disconnected"
        cv2.putText(overlay, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   (0, 255, 0) if self.initialized else (0, 0, 255), 2)
        
        return overlay
    
    def get_doors(self) -> List[DoorInference]:
        """Get list of doors."""
        return list(self.doors.values())
    
    def get_door_states(self) -> Dict[str, str]:
        """Get current door states."""
        return {door.id: door.current_state for door in self.doors.values()}
    
    def save_doors(self, filepath: str = "config/detected_doors_inference.json"):
        """Save doors to file."""
        door_data = [door.to_dict() for door in self.doors.values()]
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(door_data, f, indent=2, default=str)
    
    def reset(self):
        """Reset detector."""
        self.doors = {}
        self.door_counter = 0
        self.frame_count = 0
    
    # Compatibility with old DoorDetector
    def start_learning(self):
        """No learning needed - ML works immediately."""
        pass