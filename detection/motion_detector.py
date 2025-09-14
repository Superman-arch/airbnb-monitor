"""Motion-based zone detection for doors."""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
import os

from core.base_detector import BaseDetector, Zone


class MotionZoneDetector(BaseDetector):
    """Detects high-traffic zones using motion accumulation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize motion detector."""
        super().__init__(config)
        self.motion_threshold = config.get('zones', {}).get('motion_threshold', 25)
        self.min_zone_area = config.get('zones', {}).get('min_zone_area', 5000)
        self.learning_period = config.get('zones', {}).get('learning_period', 86400)
        
        # Motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=16
        )
        
        # Motion accumulation for zone detection
        self.motion_accumulator = None
        self.frame_count = 0
        self.start_time = datetime.now()
        
        # Detected zones
        self.zones = []
        self.zones_file = "./config/detected_zones.json"
        
    def initialize(self) -> bool:
        """Initialize the motion detector."""
        try:
            # Load existing zones if available
            if os.path.exists(self.zones_file):
                self.load_zones()
            
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize motion detector: {e}")
            return False
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect motion and accumulate for zone detection.
        
        Returns both motion regions and detected zones.
        """
        if self.motion_accumulator is None:
            h, w = frame.shape[:2]
            self.motion_accumulator = np.zeros((h, w), dtype=np.float32)
        
        # Apply background subtraction
        fgmask = self.bg_subtractor.apply(frame)
        
        # Remove shadows
        fgmask[fgmask == 127] = 0
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        
        # Accumulate motion
        self.motion_accumulator += (fgmask > 0).astype(np.float32)
        self.frame_count += 1
        
        # Find contours of current motion
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                detections.append({
                    'bbox': [x, y, x + w, y + h],
                    'confidence': min(area / 5000, 1.0),  # Normalize confidence
                    'class': 'motion',
                    'area': area
                })
        
        # Auto-detect zones if in learning period
        if self.config.get('zones', {}).get('auto_detect', True):
            self._update_zones()
        
        return detections
    
    def _update_zones(self):
        """Update detected zones based on motion accumulation."""
        # Only update zones periodically
        if self.frame_count % 300 != 0:  # Every 10 seconds at 30fps
            return
        
        # Normalize accumulator
        if self.motion_accumulator.max() > 0:
            normalized = (self.motion_accumulator / self.motion_accumulator.max() * 255).astype(np.uint8)
        else:
            return
        
        # Apply threshold to get high-traffic areas
        _, thresh = cv2.threshold(normalized, 50, 255, cv2.THRESH_BINARY)
        
        # Find contours of high-traffic areas
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        new_zones = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > self.min_zone_area:
                # Simplify contour
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Create zone
                zone_id = f"auto_{i}_{datetime.now():%Y%m%d_%H%M%S}"
                coordinates = approx.reshape(-1, 2).tolist()
                
                # Check if zone already exists
                if not self._zone_exists(coordinates):
                    zone = Zone(zone_id, coordinates, zone_type="detected")
                    new_zones.append(zone)
        
        # Add new zones
        self.zones.extend(new_zones)
        
        # Save zones periodically
        if self.frame_count % 3000 == 0:  # Every 100 seconds
            self.save_zones()
    
    def _zone_exists(self, coordinates: List) -> bool:
        """Check if a similar zone already exists."""
        new_center = np.mean(coordinates, axis=0)
        
        for zone in self.zones:
            existing_center = np.mean(zone.coordinates, axis=0)
            distance = np.linalg.norm(new_center - existing_center)
            
            if distance < 100:  # Within 100 pixels
                return True
        
        return False
    
    def add_manual_zone(self, coordinates: List, name: str, zone_type: str) -> Zone:
        """Manually add a zone."""
        zone_id = f"manual_{len(self.zones)}_{datetime.now():%Y%m%d_%H%M%S}"
        zone = Zone(zone_id, coordinates, name, zone_type)
        self.zones.append(zone)
        self.save_zones()
        return zone
    
    def remove_zone(self, zone_id: str) -> bool:
        """Remove a zone by ID."""
        self.zones = [z for z in self.zones if z.id != zone_id]
        self.save_zones()
        return True
    
    def update_zone(self, zone_id: str, name: str = None, zone_type: str = None) -> bool:
        """Update zone properties."""
        for zone in self.zones:
            if zone.id == zone_id:
                if name:
                    zone.name = name
                if zone_type:
                    zone.type = zone_type
                self.save_zones()
                return True
        return False
    
    def get_zones(self) -> List[Zone]:
        """Get all detected zones."""
        return self.zones
    
    def get_zone_at_point(self, x: int, y: int) -> Optional[Zone]:
        """Get zone containing a point."""
        for zone in self.zones:
            if zone.contains_point(x, y):
                return zone
        return None
    
    def get_zones_for_bbox(self, bbox: List[int]) -> List[Zone]:
        """Get zones overlapping with a bounding box."""
        overlapping = []
        for zone in self.zones:
            if zone.contains_bbox(bbox):
                overlapping.append(zone)
        return overlapping
    
    def save_zones(self):
        """Save zones to file."""
        zones_data = [zone.to_dict() for zone in self.zones]
        os.makedirs(os.path.dirname(self.zones_file), exist_ok=True)
        with open(self.zones_file, 'w') as f:
            json.dump(zones_data, f, indent=2, default=str)
    
    def load_zones(self):
        """Load zones from file."""
        try:
            with open(self.zones_file, 'r') as f:
                zones_data = json.load(f)
            
            self.zones = []
            for zone_data in zones_data:
                zone = Zone(
                    zone_data['id'],
                    zone_data['coordinates'],
                    zone_data.get('name'),
                    zone_data.get('type', 'unknown')
                )
                zone.traffic_count = zone_data.get('traffic_count', 0)
                zone.unique_persons = set(zone_data.get('unique_persons', []))
                self.zones.append(zone)
        except Exception as e:
            print(f"Error loading zones: {e}")
            self.zones = []
    
    def get_motion_heatmap(self) -> np.ndarray:
        """Get motion heatmap for visualization."""
        if self.motion_accumulator is None:
            return None
        
        # Normalize and convert to color
        normalized = (self.motion_accumulator / self.motion_accumulator.max() * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        return heatmap
    
    def cleanup(self):
        """Cleanup resources."""
        self.save_zones()
        self.bg_subtractor = None
        self.motion_accumulator = None