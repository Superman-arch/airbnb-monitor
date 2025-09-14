"""Base detector class for modular detection system."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np


class BaseDetector(ABC):
    """Abstract base class for detection modules."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize detector with configuration."""
        self.config = config
        self.is_initialized = False
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the detector."""
        pass
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of detections with format:
            [{
                'bbox': [x1, y1, x2, y2],
                'confidence': float,
                'class': str,
                'id': optional int
            }]
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources."""
        pass


class Zone:
    """Represents a detection zone (door area)."""
    
    def __init__(self, zone_id: str, coordinates: List[Tuple[int, int]], 
                 name: str = None, zone_type: str = "unknown"):
        """
        Initialize a zone.
        
        Args:
            zone_id: Unique identifier for the zone
            coordinates: List of (x, y) points defining the zone polygon
            name: Human-readable name for the zone
            zone_type: Type of zone (main_entrance, room_door, common_area)
        """
        self.id = zone_id
        self.coordinates = np.array(coordinates)
        self.name = name or f"Zone_{zone_id}"
        self.type = zone_type
        self.traffic_count = 0
        self.unique_persons = set()
        self.last_activity = None
        
    def contains_point(self, x: int, y: int) -> bool:
        """Check if a point is inside the zone."""
        return cv2.pointPolygonTest(self.coordinates, (x, y), False) >= 0
    
    def contains_bbox(self, bbox: List[int], threshold: float = 0.3) -> bool:
        """
        Check if a bounding box overlaps with the zone.
        
        Args:
            bbox: [x1, y1, x2, y2]
            threshold: Minimum overlap ratio to consider inside
        """
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        
        # Check corners and center
        points = [
            (x1, y1), (x2, y1), (x1, y2), (x2, y2),
            ((x1 + x2) // 2, (y1 + y2) // 2)
        ]
        
        inside_count = sum(1 for p in points if self.contains_point(*p))
        return inside_count >= 2  # At least 2 points inside
    
    def update_statistics(self, person_id: str):
        """Update zone statistics with person activity."""
        self.traffic_count += 1
        self.unique_persons.add(person_id)
        self.last_activity = datetime.now()
    
    def classify_type(self) -> str:
        """Auto-classify zone type based on traffic patterns."""
        if self.traffic_count > 100:
            return "main_entrance"
        elif len(self.unique_persons) < 5:
            return "room_door"
        else:
            return "common_area"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert zone to dictionary for serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'coordinates': self.coordinates.tolist(),
            'traffic_count': self.traffic_count,
            'unique_persons': len(self.unique_persons),
            'last_activity': self.last_activity.isoformat() if self.last_activity else None
        }


import cv2
from datetime import datetime