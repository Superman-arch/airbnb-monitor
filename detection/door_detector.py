"""Fallback door detection using edge detection and computer vision."""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import json
import os
import hashlib


class Door:
    """Represents a door detected through edge detection."""
    
    def __init__(self, door_id: str, bbox: Tuple[int, int, int, int], spatial_hash: str = None):
        """Initialize a door."""
        self.id = door_id
        self.spatial_hash = spatial_hash or self._generate_spatial_hash(bbox)
        self.bbox = bbox  # (x, y, width, height)
        self.current_state = "unknown"
        self.previous_state = "unknown"
        self.last_change = datetime.now()
        self.open_duration = timedelta()
        self.change_history = deque(maxlen=100)
        self.confidence = 0.0
        self.zone_name = None
        self.zone_id = None
        
        # Reference templates for state detection
        self.closed_template = None
        self.open_template = None
        self.edge_history = deque(maxlen=10)
        
        # Temporal filtering
        self.state_buffer = deque(maxlen=5)
        self.pending_state = None
        self.pending_state_count = 0
        self.last_event_time = datetime.now()
        
        # Persistence
        self.metadata = {
            'name': door_id,
            'type': 'unknown',
            'features': [],
            'created': datetime.now().isoformat()
        }
    
    def _generate_spatial_hash(self, bbox: Tuple[int, int, int, int]) -> str:
        """Generate unique hash based on spatial location."""
        x, y, w, h = bbox
        # Create hash from approximate location (grid-based)
        grid_x = x // 50
        grid_y = y // 50
        location_str = f"door_{grid_x}_{grid_y}"
        return hashlib.md5(location_str.encode()).hexdigest()[:8]
    
    def update_state(self, new_state: str, confidence: float = 0.0):
        """Update door state with temporal filtering."""
        self.state_buffer.append((new_state, confidence))
        
        # Check for consistent state
        if len(self.state_buffer) >= 3:
            recent_states = [s[0] for s in list(self.state_buffer)[-3:]]
            avg_confidence = sum(s[1] for s in list(self.state_buffer)[-3:]) / 3
            
            if all(s == recent_states[0] for s in recent_states):
                if recent_states[0] != self.current_state:
                    # State change confirmed
                    self.previous_state = self.current_state
                    self.current_state = recent_states[0]
                    self.confidence = avg_confidence
                    self.last_change = datetime.now()
                    
                    # Track open duration
                    if self.current_state == 'open':
                        self.open_duration = timedelta()
                    elif self.previous_state == 'open':
                        self.open_duration = datetime.now() - self.last_change
                    
                    # Add to history
                    self.change_history.append({
                        'timestamp': self.last_change.isoformat(),
                        'from_state': self.previous_state,
                        'to_state': self.current_state,
                        'confidence': self.confidence
                    })
                    
                    return True  # State changed
        return False  # No state change
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'door_id': self.id,
            'spatial_hash': self.spatial_hash,
            'bbox': self.bbox,
            'state': self.current_state,
            'confidence': self.confidence,
            'last_change': self.last_change.isoformat(),
            'open_duration': str(self.open_duration),
            'zone_name': self.zone_name,
            'zone_id': self.zone_id,
            'metadata': self.metadata
        }


class DoorDetector:
    """Edge-detection based door detector (fallback when inference unavailable)."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize door detector."""
        self.config = config
        door_config = config.get('door_detection', {})
        
        # Detection parameters
        self.learning_duration = door_config.get('learning_duration', 30)
        self.min_door_height = door_config.get('min_door_height', 120)
        self.min_door_width = door_config.get('min_door_width', 30)
        self.max_door_width = door_config.get('max_door_width', 150)
        self.edge_threshold1 = door_config.get('edge_threshold1', 50)
        self.edge_threshold2 = door_config.get('edge_threshold2', 150)
        self.change_threshold = door_config.get('change_threshold', 0.3)
        self.min_edge_density = door_config.get('min_edge_density', 0.1)
        self.closed_threshold = door_config.get('closed_threshold', 0.8)
        self.open_threshold = door_config.get('open_threshold', 0.2)
        self.noise_kernel_size = door_config.get('noise_kernel_size', 3)
        
        # State
        self.doors = {}  # door_id -> Door object
        self.learning = False
        self.learning_start = None
        self.frame_count = 0
        self.door_counter = 0
        self.initialized = False
        
        # Persistence
        self.doors_file = "config/door_states.json"
        self.load_door_states()
        
        # Zone integration
        self.zone_mapper = None
        self._init_zone_mapper()
    
    def _init_zone_mapper(self):
        """Initialize the VLM zone mapper if available."""
        try:
            from .vlm_zone_mapper import VLMZoneMapper
            self.zone_mapper = VLMZoneMapper(self.config)
            if self.zone_mapper.zones:
                print(f"Loaded {len(self.zone_mapper.zones)} predefined door zones")
                # Map zones to doors
                self._map_zones_to_doors()
        except Exception as e:
            print(f"Zone mapper not available: {e}")
            self.zone_mapper = None
    
    def _map_zones_to_doors(self):
        """Map VLM zones to detected doors."""
        if not self.zone_mapper:
            return
        
        for zone in self.zone_mapper.zones:
            # Find door that matches this zone's bbox
            zone_x, zone_y, zone_w, zone_h = zone.bbox
            
            for door_id, door in self.doors.items():
                door_x, door_y, door_w, door_h = door.bbox
                
                # Check if bboxes overlap significantly
                x_overlap = max(0, min(door_x + door_w, zone_x + zone_w) - max(door_x, zone_x))
                y_overlap = max(0, min(door_y + door_h, zone_y + zone_h) - max(door_y, zone_y))
                overlap_area = x_overlap * y_overlap
                door_area = door_w * door_h
                
                if overlap_area > 0.5 * door_area:
                    # This door matches this zone
                    door.zone_id = zone.id
                    door.zone_name = zone.name
                    door.metadata['name'] = zone.name
                    door.metadata['type'] = zone.description
                    print(f"Mapped door {door_id} to zone {zone.name}")
    
    def calibrate_zones(self, frame: np.ndarray) -> Dict[str, Any]:
        """Calibrate door zones using VLM if available."""
        if self.zone_mapper:
            result = self.zone_mapper.calibrate(frame)
            if result.get('success'):
                # Re-map zones to doors after calibration
                self._map_zones_to_doors()
            return result
        return {'success': False, 'error': 'Zone mapper not available'}
    
    def get_zones(self) -> List[Dict[str, Any]]:
        """Get all defined zones."""
        if self.zone_mapper:
            return self.zone_mapper.get_zones()
        return []
    
    def get_doors(self) -> List[Door]:
        """Get all detected doors."""
        return list(self.doors.values())
    
    def initialize(self) -> bool:
        """Initialize the door detector."""
        print("Initializing edge-detection based door detector...")
        self.initialized = True
        return True
    
    def start_learning(self):
        """Start learning phase to detect doors."""
        self.learning = True
        self.learning_start = datetime.now()
        print(f"Starting door learning phase for {self.learning_duration} seconds...")
    
    def detect_potential_doors(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect potential door regions using edge detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (self.noise_kernel_size, self.noise_kernel_size), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, self.edge_threshold1, self.edge_threshold2)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        potential_doors = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by door-like dimensions
            if (h >= self.min_door_height and 
                self.min_door_width <= w <= self.max_door_width and
                h > w * 1.5):  # Doors are typically taller than wide
                
                # Check edge density in region
                roi_edges = edges[y:y+h, x:x+w]
                edge_density = np.sum(roi_edges > 0) / (w * h)
                
                if edge_density >= self.min_edge_density:
                    potential_doors.append((x, y, w, h))
        
        return potential_doors
    
    def process_frame(self, frame: np.ndarray) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Process frame to detect doors and their states.
        
        Returns:
            (is_learning, door_events)
        """
        self.frame_count += 1
        events = []
        
        # Check if still in learning phase
        if self.learning:
            elapsed = (datetime.now() - self.learning_start).total_seconds()
            if elapsed < self.learning_duration:
                # Detect potential doors
                potential_doors = self.detect_potential_doors(frame)
                
                for bbox in potential_doors:
                    x, y, w, h = bbox
                    spatial_hash = self._generate_spatial_hash(bbox)
                    
                    # Check if this door already exists (by spatial hash)
                    existing_door = None
                    for door_id, door in self.doors.items():
                        if door.spatial_hash == spatial_hash:
                            existing_door = door
                            break
                    
                    if not existing_door:
                        # New door discovered
                        door_id = f"door_{self.door_counter}"
                        self.door_counter += 1
                        
                        door = Door(door_id, bbox, spatial_hash)
                        self.doors[spatial_hash] = door
                        
                        # Store initial template
                        door.closed_template = self._extract_door_features(frame, bbox)
                        
                        print(f"Discovered door: {door_id} at position {bbox}")
                        
                        events.append({
                            'event': 'door_discovered',
                            'door_id': door_id,
                            'bbox': bbox,
                            'timestamp': datetime.now().isoformat()
                        })
                
                # Show progress
                if self.frame_count % 30 == 0:
                    remaining = self.learning_duration - elapsed
                    print(f"Learning doors... {remaining:.1f}s remaining. Found {len(self.doors)} doors.")
                
                return True, events
            else:
                # Learning phase complete
                self.learning = False
                print(f"Learning complete. Detected {len(self.doors)} doors.")
                self.save_door_states()
                return False, events
        
        # Normal operation - detect door states
        for spatial_hash, door in self.doors.items():
            state, confidence = self._detect_door_state(frame, door)
            
            if door.update_state(state, confidence):
                # State changed
                event = {
                    'event': f'door_{door.current_state}',
                    'event_type': 'door',
                    'action': door.current_state,
                    'door_id': door.id,
                    'door_name': door.zone_name or door.id,
                    'confidence': confidence,
                    'timestamp': datetime.now().isoformat(),
                    'bbox': door.bbox
                }
                events.append(event)
                
                print(f"Door {door.id} is now {door.current_state} (confidence: {confidence:.2f})")
        
        # Save states periodically
        if self.frame_count % 300 == 0:  # Every ~10 seconds at 30fps
            self.save_door_states()
        
        return False, events
    
    def _generate_spatial_hash(self, bbox: Tuple[int, int, int, int]) -> str:
        """Generate unique hash based on spatial location."""
        x, y, w, h = bbox
        grid_x = x // 50
        grid_y = y // 50
        location_str = f"door_{grid_x}_{grid_y}"
        return hashlib.md5(location_str.encode()).hexdigest()[:8]
    
    def _extract_door_features(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract feature template from door region."""
        x, y, w, h = bbox
        roi = frame[y:y+h, x:x+w]
        
        # Convert to grayscale and extract edges
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.edge_threshold1, self.edge_threshold2)
        
        return edges
    
    def _detect_door_state(self, frame: np.ndarray, door: Door) -> Tuple[str, float]:
        """Detect if door is open or closed."""
        current_features = self._extract_door_features(frame, door.bbox)
        
        if door.closed_template is not None:
            # Compare with closed template
            similarity = self._calculate_similarity(current_features, door.closed_template)
            
            if similarity > self.closed_threshold:
                return 'closed', similarity
            elif similarity < self.open_threshold:
                return 'open', 1.0 - similarity
            else:
                return 'unknown', 0.5
        
        return 'unknown', 0.0
    
    def _calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate similarity between two feature templates."""
        if features1.shape != features2.shape:
            # Resize if shapes don't match
            features2 = cv2.resize(features2, (features1.shape[1], features1.shape[0]))
        
        # Calculate normalized correlation
        correlation = cv2.matchTemplate(features1.astype(np.float32), 
                                      features2.astype(np.float32), 
                                      cv2.TM_CCORR_NORMED)
        
        return float(np.max(correlation))
    
    def save_door_states(self):
        """Save door states to file."""
        try:
            states = {}
            for spatial_hash, door in self.doors.items():
                states[spatial_hash] = {
                    'id': door.id,
                    'bbox': door.bbox,
                    'zone_name': door.zone_name,
                    'zone_id': door.zone_id,
                    'metadata': door.metadata,
                    'last_state': door.current_state,
                    'last_change': door.last_change.isoformat()
                }
            
            os.makedirs(os.path.dirname(self.doors_file), exist_ok=True)
            with open(self.doors_file, 'w') as f:
                json.dump(states, f, indent=2)
        except Exception as e:
            print(f"Failed to save door states: {e}")
    
    def load_door_states(self):
        """Load door states from file."""
        try:
            if os.path.exists(self.doors_file):
                with open(self.doors_file, 'r') as f:
                    states = json.load(f)
                
                for spatial_hash, state in states.items():
                    door = Door(state['id'], tuple(state['bbox']), spatial_hash)
                    door.zone_name = state.get('zone_name')
                    door.zone_id = state.get('zone_id')
                    door.metadata = state.get('metadata', door.metadata)
                    door.current_state = state.get('last_state', 'unknown')
                    self.doors[spatial_hash] = door
                
                self.door_counter = len(self.doors)
                print(f"Loaded {len(self.doors)} doors from saved state")
        except Exception as e:
            print(f"Failed to load door states: {e}")
    
    def reset(self):
        """Reset door detection."""
        self.doors.clear()
        self.door_counter = 0
        self.learning = False
        self.learning_start = None
        self.frame_count = 0
        if os.path.exists(self.doors_file):
            os.remove(self.doors_file)
        print("Door detector reset")