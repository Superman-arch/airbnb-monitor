"""Automatic door detection and state monitoring using edge detection."""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import json
import os


class Door:
    """Represents a detected door."""
    
    def __init__(self, door_id: str, bbox: Tuple[int, int, int, int], 
                 baseline_edges: np.ndarray):
        """Initialize a door."""
        self.id = door_id
        self.bbox = bbox  # (x, y, width, height)
        self.baseline_edges = baseline_edges
        self.current_state = "unknown"
        self.previous_state = "unknown"
        self.last_change = datetime.now()
        self.open_duration = timedelta()
        self.change_history = deque(maxlen=100)
        self.confidence = 0.0
        
    def get_region(self, frame: np.ndarray) -> np.ndarray:
        """Extract door region from frame."""
        x, y, w, h = self.bbox
        return frame[y:y+h, x:x+w]
    
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


class DoorDetector:
    """Automatic door detection and state monitoring."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize door detector."""
        self.config = config
        door_config = config.get('door_detection', {})
        
        # Learning parameters
        self.learning_duration = door_config.get('learning_duration', 30)
        self.learning_fps = door_config.get('learning_fps', 10)
        self.min_door_height = door_config.get('min_door_height', 150)
        self.min_door_width = door_config.get('min_door_width', 40)
        self.max_door_width = door_config.get('max_door_width', 200)
        
        # Detection parameters
        self.edge_threshold1 = door_config.get('edge_threshold1', 50)
        self.edge_threshold2 = door_config.get('edge_threshold2', 150)
        self.change_threshold = door_config.get('change_threshold', 0.3)
        self.min_edge_density = door_config.get('min_edge_density', 0.1)
        
        # State
        self.learning_phase = True
        self.learning_frames = []
        self.learning_start_time = None
        self.doors = []
        self.frame_count = 0
        
        # Edge accumulator for learning
        self.edge_accumulator = None
        self.frame_height = None
        self.frame_width = None
        
    def start_learning(self):
        """Start the learning phase."""
        self.learning_phase = True
        self.learning_start_time = datetime.now()
        self.learning_frames = []
        self.edge_accumulator = None
        print(f"Starting door learning phase ({self.learning_duration} seconds)...")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Process a frame for door detection.
        
        Returns:
            (learning_complete, door_events)
        """
        if self.frame_height is None:
            self.frame_height, self.frame_width = frame.shape[:2]
            self.edge_accumulator = np.zeros((self.frame_height, self.frame_width), dtype=np.float32)
        
        if self.learning_phase:
            return self._learning_phase(frame)
        else:
            return self._detection_phase(frame)
    
    def _learning_phase(self, frame: np.ndarray) -> Tuple[bool, List[Dict[str, Any]]]:
        """Process frame during learning phase."""
        self.frame_count += 1
        
        # Calculate edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.edge_threshold1, self.edge_threshold2)
        
        # Accumulate edges
        self.edge_accumulator += edges.astype(np.float32) / 255.0
        
        # Store frame for baseline creation
        if self.frame_count % 3 == 0:  # Store every 3rd frame
            self.learning_frames.append(frame.copy())
        
        # Check if learning complete
        elapsed = (datetime.now() - self.learning_start_time).total_seconds()
        
        if elapsed >= self.learning_duration:
            # Learning complete, detect doors
            self._detect_doors()
            self.learning_phase = False
            print(f"Learning complete. Detected {len(self.doors)} doors.")
            
            # Return door discovery events
            events = []
            for door in self.doors:
                events.append({
                    'event': 'door_discovered',
                    'door_id': door.id,
                    'bbox': door.bbox,
                    'initial_state': door.current_state,
                    'timestamp': datetime.now()
                })
            
            return True, events
        
        # Still learning
        progress = int((elapsed / self.learning_duration) * 100)
        if self.frame_count % 30 == 0:
            print(f"Learning environment: {progress}% ({elapsed:.1f}/{self.learning_duration}s)")
        
        return False, []
    
    def _detect_doors(self):
        """Detect doors from accumulated edge data."""
        # Normalize edge accumulator
        if self.frame_count > 0:
            avg_edges = self.edge_accumulator / self.frame_count
        else:
            return
        
        # Threshold to get stable edges
        stable_edges = (avg_edges > 0.3).astype(np.uint8) * 255
        
        # Find vertical lines using Hough transform
        lines = cv2.HoughLinesP(stable_edges, 1, np.pi/180, 100, 
                                minLineLength=self.min_door_height,
                                maxLineGap=20)
        
        if lines is None:
            print("No vertical lines detected")
            return
        
        # Group lines into potential doors
        door_candidates = self._group_lines_into_doors(lines, stable_edges)
        
        # Create door objects with baselines
        for i, (x, y, w, h) in enumerate(door_candidates):
            # Get baseline from last learning frame
            if self.learning_frames:
                baseline_frame = self.learning_frames[-1]
                door_region = baseline_frame[y:y+h, x:x+w]
                
                # Calculate baseline edges
                gray_region = cv2.cvtColor(door_region, cv2.COLOR_BGR2GRAY)
                baseline_edges = cv2.Canny(gray_region, self.edge_threshold1, self.edge_threshold2)
                
                # Create door object
                door_id = f"door_{i+1:03d}"
                door = Door(door_id, (x, y, w, h), baseline_edges)
                
                # Determine initial state
                door.current_state = self._determine_door_state(baseline_edges)
                door.previous_state = door.current_state
                
                self.doors.append(door)
                print(f"Detected {door_id} at ({x},{y}) size {w}x{h}, state: {door.current_state}")
    
    def _group_lines_into_doors(self, lines: np.ndarray, 
                                edge_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Group detected lines into door rectangles."""
        door_candidates = []
        
        # Find vertical lines
        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Check if line is mostly vertical (within 20 degrees)
            if angle > 70 or angle < 20:
                vertical_lines.append((min(x1, x2), min(y1, y2), 
                                     max(x1, x2), max(y1, y2)))
        
        # Sort lines by x coordinate
        vertical_lines.sort(key=lambda l: l[0])
        
        # Group nearby vertical lines into doors
        i = 0
        while i < len(vertical_lines):
            x1, y1, x2, y2 = vertical_lines[i]
            
            # Look for parallel line nearby (potential door frame)
            door_width = 0
            door_x = x1
            door_y = min(y1, y2)
            door_height = abs(y2 - y1)
            
            # Search for matching vertical line
            for j in range(i + 1, len(vertical_lines)):
                x1_next, y1_next, x2_next, y2_next = vertical_lines[j]
                distance = abs(x1_next - x1)
                
                if self.min_door_width <= distance <= self.max_door_width:
                    # Found potential door frame
                    door_width = distance
                    door_y = min(door_y, y1_next, y2_next)
                    door_height = max(door_height, abs(y2_next - y1_next))
                    break
            
            if door_width > 0 and door_height >= self.min_door_height:
                # Valid door candidate
                # Check edge density in region
                door_region = edge_image[door_y:door_y+door_height, 
                                        door_x:door_x+door_width]
                edge_density = np.mean(door_region > 0)
                
                if edge_density >= self.min_edge_density:
                    door_candidates.append((door_x, door_y, door_width, door_height))
                    i = j + 1 if door_width > 0 else i + 1
                else:
                    i += 1
            else:
                i += 1
        
        return door_candidates
    
    def _detection_phase(self, frame: np.ndarray) -> Tuple[bool, List[Dict[str, Any]]]:
        """Process frame during detection phase."""
        events = []
        
        for door in self.doors:
            # Extract door region
            door_region = door.get_region(frame)
            
            # Calculate current edges
            gray_region = cv2.cvtColor(door_region, cv2.COLOR_BGR2GRAY)
            current_edges = cv2.Canny(gray_region, self.edge_threshold1, self.edge_threshold2)
            
            # Compare to baseline
            similarity = self._calculate_edge_similarity(door.baseline_edges, current_edges)
            door.confidence = similarity
            
            # Determine current state
            new_state = self._determine_state_from_similarity(similarity)
            
            # Check for state change
            if new_state != door.current_state:
                # State changed
                door.previous_state = door.current_state
                door.current_state = new_state
                door.last_change = datetime.now()
                
                # Record change
                door.change_history.append({
                    'timestamp': door.last_change,
                    'from_state': door.previous_state,
                    'to_state': new_state,
                    'confidence': similarity
                })
                
                # Create event
                event = {
                    'event': f'door_{new_state}',
                    'door_id': door.id,
                    'timestamp': door.last_change,
                    'previous_state': door.previous_state,
                    'current_state': new_state,
                    'confidence': similarity,
                    'bbox': door.bbox
                }
                events.append(event)
                
                print(f"{door.id}: {door.previous_state} -> {new_state} (confidence: {similarity:.2f})")
            
            # Track open duration
            if door.current_state == "open":
                door.open_duration = datetime.now() - door.last_change
                
                # Check for door left open alert
                if door.open_duration.total_seconds() > 300:  # 5 minutes
                    if self.frame_count % 150 == 0:  # Alert every 5 seconds
                        events.append({
                            'event': 'door_left_open',
                            'door_id': door.id,
                            'duration_seconds': door.open_duration.total_seconds(),
                            'timestamp': datetime.now()
                        })
        
        return False, events
    
    def _calculate_edge_similarity(self, baseline: np.ndarray, 
                                  current: np.ndarray) -> float:
        """Calculate similarity between edge images."""
        if baseline.shape != current.shape:
            current = cv2.resize(current, (baseline.shape[1], baseline.shape[0]))
        
        # Method 1: Direct pixel comparison
        intersection = cv2.bitwise_and(baseline, current)
        union = cv2.bitwise_or(baseline, current)
        
        intersection_pixels = np.sum(intersection > 0)
        union_pixels = np.sum(union > 0)
        
        if union_pixels == 0:
            return 0.0
        
        similarity = intersection_pixels / union_pixels
        return similarity
    
    def _determine_door_state(self, edges: np.ndarray) -> str:
        """Determine if door is open or closed based on edge pattern."""
        # Calculate edge density
        edge_density = np.mean(edges > 0)
        
        # High edge density suggests closed door (visible door edges)
        # Low edge density suggests open door (no door blocking view)
        if edge_density > 0.15:
            return "closed"
        else:
            return "open"
    
    def _determine_state_from_similarity(self, similarity: float) -> str:
        """Determine door state from similarity score."""
        # High similarity to baseline = same state as baseline
        # Low similarity = state changed
        
        if similarity > 0.7:
            return "closed"  # Similar to baseline (usually closed)
        elif similarity < 0.3:
            return "open"    # Very different from baseline
        else:
            return "moving"  # Transitioning
    
    def draw_doors(self, frame: np.ndarray) -> np.ndarray:
        """Draw door overlays on frame."""
        overlay = frame.copy()
        
        for door in self.doors:
            x, y, w, h = door.bbox
            
            # Choose color based on state
            if door.current_state == "open":
                color = (0, 255, 0)  # Green
            elif door.current_state == "closed":
                color = (0, 0, 255)  # Red
            else:
                color = (0, 255, 255)  # Yellow (moving/unknown)
            
            # Draw rectangle
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{door.id}: {door.current_state}"
            if door.current_state == "open" and door.open_duration.total_seconds() > 0:
                label += f" ({int(door.open_duration.total_seconds())}s)"
            
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(overlay, (x, y - label_size[1] - 4), 
                         (x + label_size[0], y), color, -1)
            cv2.putText(overlay, label, (x, y - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw confidence bar
            if door.confidence > 0:
                bar_width = int(w * door.confidence)
                cv2.rectangle(overlay, (x, y + h + 2), 
                            (x + bar_width, y + h + 6), color, -1)
        
        # Draw learning progress if in learning phase
        if self.learning_phase and self.learning_start_time:
            elapsed = (datetime.now() - self.learning_start_time).total_seconds()
            progress = min(100, int((elapsed / self.learning_duration) * 100))
            
            text = f"Learning doors: {progress}%"
            cv2.putText(overlay, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Progress bar
            bar_length = 200
            bar_filled = int(bar_length * progress / 100)
            cv2.rectangle(overlay, (10, 40), (10 + bar_length, 50), (100, 100, 100), -1)
            cv2.rectangle(overlay, (10, 40), (10 + bar_filled, 50), (0, 255, 255), -1)
        
        return overlay
    
    def get_doors(self) -> List[Door]:
        """Get list of detected doors."""
        return self.doors
    
    def get_door_states(self) -> Dict[str, str]:
        """Get current state of all doors."""
        return {door.id: door.current_state for door in self.doors}
    
    def save_doors(self, filepath: str = "config/detected_doors.json"):
        """Save detected doors to file."""
        door_data = [door.to_dict() for door in self.doors]
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(door_data, f, indent=2, default=str)
    
    def reset(self):
        """Reset detector for new learning phase."""
        self.doors = []
        self.learning_frames = []
        self.edge_accumulator = None
        self.frame_count = 0
        self.start_learning()