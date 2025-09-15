"""Door detection using Roboflow Inference Server (local, offline)."""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from storage.door_persistence import DoorPersistence
from utils.logger import logger

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
        self.confidence_threshold = max(0.7, door_config.get('confidence_threshold', 0.7))  # Min 70% confidence
        
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
        
        # VLM Zone integration
        self.zone_mapper = None
        self._init_zone_mapper()
        
        # Persistence
        self.persistence = DoorPersistence()
        self._load_saved_doors()
        
        # Auto-calibration flag
        self.auto_calibrated = False
    
    def _init_zone_mapper(self):
        """Initialize the VLM zone mapper."""
        try:
            from .vlm_zone_mapper import VLMZoneMapper
            self.zone_mapper = VLMZoneMapper(self.config)
            if self.zone_mapper.zones:
                logger.log('INFO', f"Loaded {len(self.zone_mapper.zones)} predefined door zones", 'DOOR_DETECTOR')
        except Exception as e:
            logger.log('WARNING', f"Zone mapper not available: {e}", 'DOOR_DETECTOR')
            self.zone_mapper = None
    
    def _load_saved_doors(self):
        """Load previously saved door configurations."""
        saved_doors = self.persistence.get_all_doors()
        if saved_doors:
            for door_id, door_config in saved_doors.items():
                if door_config.get('active', True):
                    bbox = tuple(door_config['bbox'])
                    door = DoorInference(door_id, bbox)
                    door.confidence = door_config.get('confidence', 0.0)
                    door.zone_id = door_config.get('zone')
                    self.doors[door_id] = door
            logger.log('INFO', f"Loaded {len(self.doors)} saved door configurations", 'DOOR_DETECTOR')
    
    def calibrate_zones(self, frame: np.ndarray) -> Dict[str, Any]:
        """Calibrate door zones using multiple methods."""
        import time
        calibration_start = time.time()
        zones_found = 0
        zones = []
        
        # Try VLM first if available
        if self.zone_mapper:
            result = self.zone_mapper.calibrate(frame)
            if result.get('success'):
                return result
        
        # Fallback: Try inference detection
        if self.initialized and self.client:
            try:
                print("Attempting door detection with inference...")
                # Run inference on the frame
                result = self.client.infer(
                    frame,
                    model_id=self.model_id
                )
                
                predictions = result.get('predictions', [])
                # Sort by confidence and take only high-confidence detections
                high_conf_preds = [p for p in predictions if p['confidence'] >= 0.75]
                high_conf_preds.sort(key=lambda x: x['confidence'], reverse=True)
                
                for pred in high_conf_preds[:5]:  # Max 5 doors
                    # Extract and validate detection
                    center_x = pred['x']
                    center_y = pred['y']
                    width = pred['width']
                    height = pred['height']
                    
                    # Validate dimensions (doors should be taller than wide)
                    if height >= width * 1.5 and width >= 30 and height >= 60:
                        x = int(center_x - width / 2)
                        y = int(center_y - height / 2)
                        
                        # Generate consistent ID based on position
                        door_id = f"door_{int(center_x/100)}_{int(center_y/100)}"
                        
                        zone = {
                            'id': door_id,
                            'name': f"Door {zones_found + 1}",
                            'bbox': (x, y, int(width), int(height)),
                            'confidence': pred['confidence'],
                            'description': pred.get('class', 'door'),
                            'state': 'open' if 'open' in pred.get('class', '').lower() else 'closed'
                        }
                        zones.append(zone)
                        zones_found += 1
                        
                        # Save to persistence
                        self.persistence.add_door(door_id, zone['bbox'], zone['confidence'])
            except Exception as e:
                print(f"Inference detection failed: {e}")
        
        calibration_time = time.time() - calibration_start
        
        return {
            'success': zones_found > 0,
            'zones_found': zones_found,
            'zones': zones,
            'calibration_time': calibration_time,
            'method': 'inference' if zones_found > 0 else 'none',
            'error': 'No doors detected' if zones_found == 0 else None
        }
    
    def get_zones(self) -> List[Dict[str, Any]]:
        """Get all defined zones."""
        if self.zone_mapper:
            return self.zone_mapper.get_zones()
        return []
        
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
                # Extract detection data - inference returns center coordinates
                center_x = pred['x']
                center_y = pred['y']
                width = pred['width']
                height = pred['height']
                
                # Convert center coordinates to top-left corner
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                
                # Ensure coordinates are within frame bounds
                x = max(0, x)
                y = max(0, y)
                width = min(int(width), frame.shape[1] - x)
                height = min(int(height), frame.shape[0] - y)
                
                bbox = (x, y, width, height)
                
                class_name = pred['class']
                confidence = pred['confidence']
                
                # Skip low confidence detections (min 70%)
                if confidence < max(0.7, self.confidence_threshold):
                    continue
                
                # Validate door dimensions (doors should be taller than wide)
                if height < width * 1.2:  # Doors are typically at least 1.2x taller than wide
                    continue
                
                # Skip very small detections (likely false positives)
                if width < 20 or height < 40:
                    continue
                
                # Skip detections at edges (often partial/incorrect)
                edge_margin = 10
                if x < edge_margin or y < edge_margin or (x + width) > frame.shape[1] - edge_margin or (y + height) > frame.shape[0] - edge_margin:
                    continue
                
                # Map class to state with high confidence
                is_open = 'open' in class_name.lower()
                is_closed = 'closed' in class_name.lower()
                
                if is_open and confidence >= 0.7:
                    state = 'open'
                elif is_closed and confidence >= 0.7:
                    state = 'closed'
                else:
                    state = 'unknown'
                
                # Get door ID based on location and zone mapping
                door_id = self._get_door_id_for_location(bbox)
                current_door_ids.add(door_id)
                
                # Try to match with a zone
                zone = None
                zone_name = None
                if self.zone_mapper:
                    zone = self.zone_mapper.get_zone_for_detection(bbox)
                    if zone:
                        door_id = zone.id  # Use zone ID as door ID
                        zone_name = zone.name
                
                # Get or create door
                if door_id not in self.doors:
                    self.doors[door_id] = DoorInference(door_id, bbox)
                    # Store zone info
                    if zone:
                        self.doors[door_id].zone_id = zone.id
                        self.doors[door_id].zone_name = zone.name
                    
                    # Save new door to persistence
                    self.persistence.add_door(door_id, bbox, confidence, zone.id if zone else None)
                    
                    # Discovery event
                    events.append({
                        'event': 'door_discovered',
                        'door_id': door_id,
                        'door_name': zone_name or f"Door {door_id}",
                        'zone_id': zone.id if zone else None,
                        'bbox': bbox,
                        'initial_state': state,
                        'timestamp': datetime.now(),
                        'confidence': confidence,
                        'event_type': 'door'
                    })
                    logger.log('INFO', f"Discovered {zone_name or door_id}: {state} ({confidence:.0%})", 'DOOR_DETECTOR')
                
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
                        
                        # Update persistence
                        is_open = (state == 'open')
                        self.persistence.update_door_state(door_id, is_open, confidence)
                        
                        # Log door event
                        logger.log_door_event(door_id, state, confidence, getattr(door, 'zone_id', None))
                        
                        # Create event
                        event = {
                            'event': f'door_{state}',
                            'door_id': door_id,
                            'door_name': getattr(door, 'zone_name', None) or f"Door {door_id}",
                            'zone_id': getattr(door, 'zone_id', None),
                            'timestamp': door.last_change,
                            'previous_state': door.previous_state,
                            'current_state': state,
                            'confidence': confidence,
                            'bbox': bbox,
                            'event_type': 'door'
                        }
                        events.append(event)
                        
                        door_display = getattr(door, 'zone_name', None) or door_id
                        logger.log('INFO', f"{door_display}: {door.previous_state} -> {state} ({confidence:.0%})", 'DOOR_DETECTOR')
                        
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
            logger.log('ERROR', f"Inference error: {e}", 'DOOR_DETECTOR')
            # Server might be down, try to reconnect on next frame
            self.initialized = False
        
        return False, events
    
    def _get_door_id_for_location(self, bbox: Tuple[int, int, int, int]) -> str:
        """Get door ID based on spatial location - consistent across sessions."""
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
        
        # Generate consistent ID based on position
        # This ensures same door gets same ID across restarts
        grid_x = center_x // 100
        grid_y = center_y // 100
        return f"door_{grid_x}_{grid_y}"
    
    def draw_doors(self, frame: np.ndarray) -> np.ndarray:
        """Draw door overlays with zones."""
        overlay = frame.copy()
        
        # First draw zones if available
        if self.zone_mapper:
            # Prepare active doors info for zone drawing
            active_doors = {}
            for door_id, door in self.doors.items():
                active_doors[door_id] = {
                    'zone_id': getattr(door, 'zone_id', None),
                    'state': door.current_state
                }
            overlay = self.zone_mapper.draw_zones(overlay, active_doors)
        
        # Then draw current detections
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
            
            # Label with zone name if available
            door_name = getattr(door, 'zone_name', None) or door.id
            label = f"{door_name}: {door.current_state.upper()}"
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
    
    def get_doors_dict(self) -> Dict[str, Any]:
        """Get all detected doors with their current states as dictionary."""
        doors_dict = {}
        for door_id, door in self.doors.items():
            door_info = door.to_dict()
            
            # Get the latest state from persistence if available
            saved_state = self.persistence.door_states.get(door_id, {})
            if saved_state:
                # Use saved state if we haven't detected it recently
                if door.current_state == 'unknown' and 'state' in saved_state:
                    door_info['state'] = saved_state['state']
                    door_info['is_open'] = saved_state.get('is_open', False)
            
            # Ensure we have open/closed state not just configured
            if door_info['state'] not in ['open', 'closed']:
                door_info['state'] = 'unknown'
                door_info['is_open'] = False
            else:
                door_info['is_open'] = (door_info['state'] == 'open')
            
            doors_dict[door_id] = door_info
        
        return doors_dict
    
    def get_door_states(self) -> Dict[str, str]:
        """Get current door states."""
        return {door.id: door.current_state for door in self.doors.values()}
    
    def save_doors(self, filepath: str = "config/detected_doors_inference.json"):
        """Save doors to file."""
        door_data = [door.to_dict() for door in self.doors.values()]
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(door_data, f, indent=2, default=str)
    
    def save_door_states(self):
        """Save door states to file."""
        try:
            states = {}
            for door_id, door in self.doors.items():
                # Use spatial hash as key if available
                key = getattr(door, 'spatial_hash', door_id)
                states[key] = {
                    'id': door.id,
                    'bbox': door.bbox,
                    'zone_name': getattr(door, 'zone_name', None),
                    'zone_id': getattr(door, 'zone_id', None),
                    'metadata': getattr(door, 'metadata', {}),
                    'last_state': door.current_state,
                    'last_change': door.last_change.isoformat()
                }
            
            filepath = getattr(self, 'doors_file', 'config/door_inference_states.json')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(states, f, indent=2)
        except Exception as e:
            print(f"Failed to save door states: {e}")
    
    def load_door_states(self):
        """Load door states from file."""
        try:
            filepath = getattr(self, 'doors_file', 'config/door_inference_states.json')
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    states = json.load(f)
                
                for key, state in states.items():
                    door = DoorInference(state['id'], tuple(state['bbox']))
                    if hasattr(door, 'spatial_hash'):
                        door.spatial_hash = key
                    door.zone_name = state.get('zone_name')
                    door.zone_id = state.get('zone_id')
                    if hasattr(door, 'metadata'):
                        door.metadata = state.get('metadata', {})
                    door.current_state = state.get('last_state', 'unknown')
                    self.doors[key] = door
                
                self.door_counter = len(self.doors)
                print(f"Loaded {len(self.doors)} doors from saved state")
        except Exception as e:
            print(f"Failed to load door states: {e}")
    
    def reset(self):
        """Reset detector."""
        self.doors = {}
        self.door_counter = 0
        self.frame_count = 0
        filepath = getattr(self, 'doors_file', 'config/door_inference_states.json')
        if os.path.exists(filepath):
            os.remove(filepath)
    
    # Compatibility with old DoorDetector
    def start_learning(self):
        """No learning needed - ML works immediately."""
        pass