"""VLM-based zone mapping for intelligent door identification."""

import cv2
import numpy as np
import json
import os
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import base64
from io import BytesIO
from PIL import Image

# Try to import VLM libraries
VLM_AVAILABLE = False
VLM_TYPE = None

try:
    # Try Jetson-optimized inference
    from jetson_inference import detectNet, poseNet
    from jetson_utils import cudaFromNumpy, cudaToNumpy
    VLM_AVAILABLE = True
    VLM_TYPE = "jetson"
    print("Using Jetson inference for VLM")
except ImportError:
    try:
        # Try llamafile/llava approach
        import requests
        if os.path.exists("/usr/local/bin/llamafile"):
            VLM_AVAILABLE = True
            VLM_TYPE = "llamafile"
            print("Using llamafile for VLM")
    except:
        pass

if not VLM_AVAILABLE:
    # Only show warning if VLM is explicitly enabled in config
    import os
    if os.environ.get('VLM_VERBOSE', '').lower() == 'true':
        print("Info: VLM backend not available. This is optional for zone calibration.")


@dataclass
class DoorZone:
    """Represents a door zone identified by VLM."""
    id: str
    name: str
    description: str
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    vlm_context: str
    image_region: Optional[str] = None  # Base64 encoded image of the door
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DoorZone':
        """Create from dictionary."""
        return cls(**data)


class VLMZoneMapper:
    """Maps door zones using Vision Language Models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize VLM zone mapper."""
        self.config = config
        self.zones: List[DoorZone] = []
        self.zones_file = "config/door_zones.json"
        self.vlm_endpoint = config.get('vlm', {}).get('endpoint', 'http://localhost:8080/completion')
        self.vlm_model = config.get('vlm', {}).get('model', 'phi-3.5-vision')
        
        # Load existing zones
        self.load_zones()
        
        # Initialize VLM based on available backend
        self._init_vlm()
    
    def _init_vlm(self):
        """Initialize the VLM backend."""
        if VLM_TYPE == "jetson":
            # Jetson-specific initialization
            self.detector = detectNet("ssd-mobilenet-v2", threshold=0.5)
        elif VLM_TYPE == "llamafile":
            # Check if llamafile server is running
            try:
                response = requests.get(self.vlm_endpoint.replace('/completion', '/health'))
                if response.status_code == 200:
                    print(f"VLM server ready at {self.vlm_endpoint}")
            except:
                print(f"VLM server not responding at {self.vlm_endpoint}")
    
    def analyze_scene(self, frame: np.ndarray) -> List[DoorZone]:
        """Analyze scene with VLM to identify door zones."""
        print("Analyzing scene with VLM to identify doors...")
        
        if not VLM_AVAILABLE:
            print("VLM not available, using fallback detection")
            return self._fallback_detection(frame)
        
        if VLM_TYPE == "llamafile":
            return self._analyze_with_llamafile(frame)
        elif VLM_TYPE == "jetson":
            return self._analyze_with_jetson(frame)
        else:
            return self._fallback_detection(frame)
    
    def _analyze_with_llamafile(self, frame: np.ndarray) -> List[DoorZone]:
        """Analyze using llamafile/Phi Vision."""
        # Convert frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Prepare VLM prompt
        prompt = """<image>
Analyze this image and identify all doors visible. For each door:
1. Describe its location in the image (left, right, center, top, bottom)
2. Describe what type of door it is (entrance, interior, bathroom, bedroom, etc.)
3. Describe any distinguishing features
4. Estimate the bounding box coordinates as percentages of image size

Respond in JSON format:
{
  "doors": [
    {
      "location": "left side",
      "type": "entrance door",
      "description": "wooden door with glass panel",
      "features": "has doorknob on right, appears to be main entrance",
      "bbox_percent": {"x": 10, "y": 20, "width": 15, "height": 60}
    }
  ]
}"""
        
        try:
            # Call VLM API
            response = requests.post(
                self.vlm_endpoint,
                json={
                    "prompt": prompt,
                    "image": img_base64,
                    "temperature": 0.1,
                    "max_tokens": 500
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return self._parse_vlm_response(result, frame.shape)
            else:
                print(f"VLM API error: {response.status_code}")
                return self._fallback_detection(frame)
                
        except Exception as e:
            print(f"VLM analysis failed: {e}")
            return self._fallback_detection(frame)
    
    def _analyze_with_jetson(self, frame: np.ndarray) -> List[DoorZone]:
        """Analyze using Jetson inference."""
        # Convert to CUDA
        cuda_img = cudaFromNumpy(frame)
        
        # Detect objects
        detections = self.detector.Detect(cuda_img)
        
        zones = []
        for idx, detection in enumerate(detections):
            # Check if detection might be a door (based on aspect ratio and size)
            width = detection.Right - detection.Left
            height = detection.Bottom - detection.Top
            aspect_ratio = height / width if width > 0 else 0
            
            # Doors typically have aspect ratio between 1.5 and 3.0
            if 1.5 <= aspect_ratio <= 3.0:
                zone = DoorZone(
                    id=f"door_{idx+1}",
                    name=f"Door {idx+1}",
                    description=f"Detected door-like object",
                    bbox=(int(detection.Left), int(detection.Top), int(width), int(height)),
                    confidence=detection.Confidence,
                    vlm_context="Detected by Jetson AI"
                )
                zones.append(zone)
        
        return zones
    
    def _parse_vlm_response(self, response: Dict, image_shape: Tuple) -> List[DoorZone]:
        """Parse VLM response and create door zones."""
        zones = []
        height, width = image_shape[:2]
        
        try:
            # Extract JSON from response
            if 'content' in response:
                content = response['content']
            else:
                content = response.get('response', '{}')
            
            # Parse JSON
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                doors = data.get('doors', [])
                
                for idx, door in enumerate(doors):
                    # Convert percentage bbox to pixels
                    bbox_pct = door.get('bbox_percent', {})
                    x = int(bbox_pct.get('x', 0) * width / 100)
                    y = int(bbox_pct.get('y', 0) * height / 100)
                    w = int(bbox_pct.get('width', 20) * width / 100)
                    h = int(bbox_pct.get('height', 40) * height / 100)
                    
                    # Create zone
                    zone = DoorZone(
                        id=f"door_{idx+1}",
                        name=self._generate_door_name(door),
                        description=door.get('description', 'Door'),
                        bbox=(x, y, w, h),
                        confidence=0.85,  # VLM confidence estimate
                        vlm_context=door.get('features', '')
                    )
                    zones.append(zone)
                    
        except Exception as e:
            print(f"Failed to parse VLM response: {e}")
        
        return zones
    
    def _generate_door_name(self, door_info: Dict) -> str:
        """Generate a meaningful name for the door."""
        door_type = door_info.get('type', '').lower()
        location = door_info.get('location', '').lower()
        
        # Prioritize type-based naming
        if 'entrance' in door_type or 'main' in door_type:
            return "Main Entrance"
        elif 'bathroom' in door_type:
            return "Bathroom"
        elif 'bedroom' in door_type:
            return "Bedroom"
        elif 'closet' in door_type:
            return "Closet"
        elif 'garage' in door_type:
            return "Garage"
        # Fall back to location-based naming
        elif 'left' in location:
            return "Left Door"
        elif 'right' in location:
            return "Right Door"
        elif 'center' in location:
            return "Center Door"
        else:
            return "Door"
    
    def _fallback_detection(self, frame: np.ndarray) -> List[DoorZone]:
        """Fallback door detection using edge detection."""
        print("Using fallback edge detection for door zones")
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        zones = []
        for idx, contour in enumerate(contours):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by door-like dimensions
            aspect_ratio = h / w if w > 0 else 0
            area = w * h
            
            # Door criteria
            if (1.5 <= aspect_ratio <= 3.5 and 
                area > 5000 and 
                h > frame.shape[0] * 0.3):
                
                zone = DoorZone(
                    id=f"door_{idx+1}",
                    name=f"Door {idx+1}",
                    description="Potential door detected",
                    bbox=(x, y, w, h),
                    confidence=0.6,
                    vlm_context="Edge detection fallback"
                )
                zones.append(zone)
        
        return zones
    
    def calibrate(self, frame: np.ndarray) -> Dict[str, Any]:
        """Calibrate zones using VLM analysis."""
        start_time = time.time()
        
        # Analyze scene
        new_zones = self.analyze_scene(frame)
        
        # Save door regions as images
        for zone in new_zones:
            x, y, w, h = zone.bbox
            # Ensure bbox is within frame bounds
            x = max(0, min(x, frame.shape[1] - 1))
            y = max(0, min(y, frame.shape[0] - 1))
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            
            if w > 0 and h > 0:
                door_region = frame[y:y+h, x:x+w]
                _, buffer = cv2.imencode('.jpg', door_region)
                zone.image_region = base64.b64encode(buffer).decode('utf-8')
        
        # Update zones
        self.zones = new_zones
        
        # Save to file
        self.save_zones()
        
        elapsed = time.time() - start_time
        
        return {
            'success': True,
            'zones_found': len(new_zones),
            'zones': [z.to_dict() for z in new_zones],
            'calibration_time': elapsed,
            'vlm_type': VLM_TYPE or 'fallback'
        }
    
    def get_zone_for_detection(self, bbox: Tuple[int, int, int, int]) -> Optional[DoorZone]:
        """Get zone that matches a detection bbox."""
        if not self.zones:
            return None
        
        det_x, det_y, det_w, det_h = bbox
        det_cx = det_x + det_w // 2
        det_cy = det_y + det_h // 2
        
        # Find zone with highest overlap
        best_zone = None
        best_overlap = 0
        
        for zone in self.zones:
            z_x, z_y, z_w, z_h = zone.bbox
            
            # Calculate intersection
            x1 = max(det_x, z_x)
            y1 = max(det_y, z_y)
            x2 = min(det_x + det_w, z_x + z_w)
            y2 = min(det_y + det_h, z_y + z_h)
            
            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
                det_area = det_w * det_h
                zone_area = z_w * z_h
                union = det_area + zone_area - intersection
                
                overlap = intersection / union if union > 0 else 0
                
                if overlap > best_overlap and overlap > 0.3:  # At least 30% overlap
                    best_overlap = overlap
                    best_zone = zone
        
        return best_zone
    
    def update_zone_name(self, zone_id: str, new_name: str) -> bool:
        """Update the name of a zone."""
        for zone in self.zones:
            if zone.id == zone_id:
                zone.name = new_name
                self.save_zones()
                return True
        return False
    
    def delete_zone(self, zone_id: str) -> bool:
        """Delete a zone."""
        original_count = len(self.zones)
        self.zones = [z for z in self.zones if z.id != zone_id]
        
        if len(self.zones) < original_count:
            self.save_zones()
            return True
        return False
    
    def load_zones(self):
        """Load zones from file."""
        if os.path.exists(self.zones_file):
            try:
                with open(self.zones_file, 'r') as f:
                    data = json.load(f)
                    self.zones = [DoorZone.from_dict(z) for z in data.get('zones', [])]
                    print(f"Loaded {len(self.zones)} door zones")
            except Exception as e:
                print(f"Failed to load zones: {e}")
                self.zones = []
        else:
            self.zones = []
    
    def save_zones(self):
        """Save zones to file."""
        try:
            os.makedirs(os.path.dirname(self.zones_file), exist_ok=True)
            
            data = {
                'zones': [z.to_dict() for z in self.zones],
                'calibrated_at': datetime.now().isoformat(),
                'vlm_model': self.vlm_model,
                'vlm_type': VLM_TYPE or 'fallback'
            }
            
            with open(self.zones_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"Saved {len(self.zones)} door zones")
        except Exception as e:
            print(f"Failed to save zones: {e}")
    
    def draw_zones(self, frame: np.ndarray, active_doors: Dict[str, Any] = None) -> np.ndarray:
        """Draw zones on frame."""
        overlay = frame.copy()
        
        for zone in self.zones:
            x, y, w, h = zone.bbox
            
            # Determine color based on door state
            color = (0, 0, 255)  # Red by default
            thickness = 2
            
            if active_doors:
                # Check if this zone has an active door
                for door_id, door_info in active_doors.items():
                    if door_info.get('zone_id') == zone.id:
                        state = door_info.get('state', 'unknown')
                        if state == 'open':
                            color = (0, 255, 0)  # Green for open
                        elif state == 'closed':
                            color = (255, 0, 0)  # Blue for closed
                        thickness = 3
                        break
            
            # Draw rectangle
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, thickness)
            
            # Draw label
            label = zone.name
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = y - 10 if y > 30 else y + h + 20
            
            cv2.rectangle(overlay, (x, label_y - label_size[1] - 5), 
                         (x + label_size[0] + 5, label_y + 5), color, -1)
            cv2.putText(overlay, label, (x + 2, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Blend overlay with original
        return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    def get_zones(self) -> List[Dict[str, Any]]:
        """Get all zones as dictionaries."""
        return [z.to_dict() for z in self.zones]