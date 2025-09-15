"""
Overlay renderer for video frames with detection visualizations
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime
import structlog

from backend.core.config import Settings

logger = structlog.get_logger()


class OverlayRenderer:
    """
    Renders overlays on video frames for detection visualization
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        
        # Colors (BGR format for OpenCV)
        self.colors = {
            'door_open': (0, 255, 0),      # Green
            'door_closed': (0, 0, 255),    # Red
            'door_unknown': (0, 165, 255),  # Orange
            'person': (255, 0, 0),          # Blue
            'zone': (255, 255, 0),          # Cyan
            'text': (255, 255, 255),        # White
            'background': (0, 0, 0),        # Black
            'alert': (0, 0, 255),           # Red
            'success': (0, 255, 0),         # Green
        }
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        self.line_thickness = 2
        
        # Performance overlay position
        self.perf_overlay_pos = (10, 30)
        
    def render(self, frame: np.ndarray, detections: Dict[str, List[Any]]) -> np.ndarray:
        """
        Render all overlays on frame
        """
        try:
            # Create a copy to avoid modifying original
            display_frame = frame.copy()
            
            # Render zones first (background layer)
            if 'zones' in detections:
                display_frame = self._render_zones(display_frame, detections['zones'])
            
            # Render doors
            if 'doors' in detections:
                display_frame = self._render_doors(display_frame, detections['doors'])
            
            # Render persons
            if 'persons' in detections:
                display_frame = self._render_persons(display_frame, detections['persons'])
            
            # Render performance metrics
            if 'stats' in detections:
                display_frame = self._render_stats(display_frame, detections['stats'])
            
            # Render alerts
            if 'alerts' in detections:
                display_frame = self._render_alerts(display_frame, detections['alerts'])
            
            return display_frame
            
        except Exception as e:
            logger.error(f"Error rendering overlays", error=str(e))
            return frame
    
    def _render_doors(self, frame: np.ndarray, doors: List[Dict]) -> np.ndarray:
        """
        Render door detection overlays
        """
        for door in doors:
            try:
                # Get door properties
                bbox = door.get('bbox', [])
                if len(bbox) != 4:
                    continue
                
                x, y, w, h = bbox
                state = door.get('current_state', 'unknown')
                confidence = door.get('confidence', 0.0)
                name = door.get('name', door.get('id', 'Door'))
                
                # Determine color based on state
                if state == 'open':
                    color = self.colors['door_open']
                elif state == 'closed':
                    color = self.colors['door_closed']
                else:
                    color = self.colors['door_unknown']
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, self.line_thickness)
                
                # Draw filled header for label
                label_bg_height = 25
                cv2.rectangle(frame, (x, y - label_bg_height), (x + w, y), color, -1)
                
                # Draw door label
                label = f"{name} - {state.upper()}"
                if confidence > 0:
                    label += f" ({confidence*100:.0f}%)"
                
                label_size = cv2.getTextSize(label, self.font, 0.5, 1)[0]
                label_y = y - 7
                
                cv2.putText(frame, label, (x + 5, label_y), 
                           self.font, 0.5, self.colors['text'], 1)
                
                # Draw door icon
                self._draw_door_icon(frame, x + w - 25, y - 20, 15, state)
                
                # Draw status indicator
                indicator_x = x + w + 10
                indicator_y = y + h // 2
                self._draw_status_indicator(frame, indicator_x, indicator_y, state)
                
            except Exception as e:
                logger.error(f"Error rendering door", door_id=door.get('id'), error=str(e))
        
        return frame
    
    def _render_persons(self, frame: np.ndarray, persons: List[Dict]) -> np.ndarray:
        """
        Render person tracking overlays
        """
        for person in persons:
            try:
                # Get person properties
                bbox = person.get('bbox', [])
                if len(bbox) != 4:
                    continue
                
                x, y, w, h = bbox
                track_id = person.get('track_id', 'Unknown')
                confidence = person.get('confidence', 0.0)
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), 
                             self.colors['person'], self.line_thickness)
                
                # Draw tracking ID
                label = f"Person {track_id}"
                if confidence > 0:
                    label += f" ({confidence*100:.0f}%)"
                
                label_size = cv2.getTextSize(label, self.font, 0.5, 1)[0]
                
                # Draw label background
                cv2.rectangle(frame, (x, y - 20), (x + label_size[0] + 10, y), 
                             self.colors['person'], -1)
                
                cv2.putText(frame, label, (x + 5, y - 5), 
                           self.font, 0.5, self.colors['text'], 1)
                
                # Draw tracking path if available
                if 'path' in person:
                    self._draw_tracking_path(frame, person['path'])
                
                # Draw center point
                center_x = x + w // 2
                center_y = y + h // 2
                cv2.circle(frame, (center_x, center_y), 3, self.colors['person'], -1)
                
            except Exception as e:
                logger.error(f"Error rendering person", track_id=person.get('track_id'), error=str(e))
        
        return frame
    
    def _render_zones(self, frame: np.ndarray, zones: List[Dict]) -> np.ndarray:
        """
        Render zone overlays
        """
        for zone in zones:
            try:
                # Get zone properties
                coordinates = zone.get('coordinates', [])
                if len(coordinates) < 3:
                    continue
                
                name = zone.get('name', 'Zone')
                occupancy = zone.get('occupancy', 0)
                is_active = zone.get('is_active', False)
                
                # Convert coordinates to numpy array
                points = np.array(coordinates, np.int32)
                
                # Draw zone polygon with transparency
                overlay = frame.copy()
                color = self.colors['zone'] if is_active else (128, 128, 128)
                cv2.fillPoly(overlay, [points], color)
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                
                # Draw zone outline
                cv2.polylines(frame, [points], True, color, 2)
                
                # Draw zone label
                M = cv2.moments(points)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    label = f"{name}"
                    if occupancy > 0:
                        label += f" ({occupancy})"
                    
                    label_size = cv2.getTextSize(label, self.font, 0.6, 2)[0]
                    
                    # Draw label background
                    cv2.rectangle(frame, 
                                 (cx - label_size[0]//2 - 5, cy - label_size[1]//2 - 5),
                                 (cx + label_size[0]//2 + 5, cy + label_size[1]//2 + 5),
                                 (0, 0, 0), -1)
                    
                    cv2.putText(frame, label, 
                               (cx - label_size[0]//2, cy + label_size[1]//2),
                               self.font, 0.6, self.colors['text'], 2)
                
            except Exception as e:
                logger.error(f"Error rendering zone", zone_name=zone.get('name'), error=str(e))
        
        return frame
    
    def _render_stats(self, frame: np.ndarray, stats: Dict) -> np.ndarray:
        """
        Render performance statistics overlay
        """
        try:
            # Create semi-transparent background for stats
            overlay = frame.copy()
            stats_height = 150
            stats_width = 250
            
            cv2.rectangle(overlay, (10, 10), (10 + stats_width, 10 + stats_height), 
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Render stats text
            y_offset = 30
            line_height = 20
            
            # Title
            cv2.putText(frame, "SYSTEM PERFORMANCE", (15, y_offset), 
                       self.font, 0.5, self.colors['success'], 1)
            y_offset += line_height + 5
            
            # FPS
            fps = stats.get('fps', 0)
            fps_color = self.colors['success'] if fps >= 25 else self.colors['alert'] if fps >= 15 else self.colors['alert']
            cv2.putText(frame, f"FPS: {fps:.1f}", (15, y_offset), 
                       self.font, 0.5, fps_color, 1)
            y_offset += line_height
            
            # Latency
            latency = stats.get('latency', 0)
            latency_color = self.colors['success'] if latency < 50 else self.colors['alert']
            cv2.putText(frame, f"Latency: {latency}ms", (15, y_offset), 
                       self.font, 0.5, latency_color, 1)
            y_offset += line_height
            
            # Detection counts
            cv2.putText(frame, f"Doors: {stats.get('doors', 0)}", (15, y_offset), 
                       self.font, 0.5, self.colors['text'], 1)
            y_offset += line_height
            
            cv2.putText(frame, f"People: {stats.get('persons', 0)}", (15, y_offset), 
                       self.font, 0.5, self.colors['text'], 1)
            y_offset += line_height
            
            # Memory usage
            memory = stats.get('memory', 0)
            memory_color = self.colors['success'] if memory < 60 else self.colors['alert'] if memory < 80 else self.colors['alert']
            cv2.putText(frame, f"Memory: {memory:.0f}%", (15, y_offset), 
                       self.font, 0.5, memory_color, 1)
            
            # Draw FPS graph if history available
            if 'fps_history' in stats:
                self._draw_fps_graph(frame, stats['fps_history'], (270, 10))
            
        except Exception as e:
            logger.error(f"Error rendering stats", error=str(e))
        
        return frame
    
    def _render_alerts(self, frame: np.ndarray, alerts: List[Dict]) -> np.ndarray:
        """
        Render alert overlays
        """
        y_offset = frame.shape[0] - 100
        
        for i, alert in enumerate(alerts[:3]):  # Show max 3 alerts
            try:
                alert_type = alert.get('type', 'info')
                message = alert.get('message', '')
                
                # Determine color
                if alert_type == 'error':
                    color = self.colors['alert']
                elif alert_type == 'warning':
                    color = self.colors['door_unknown']
                else:
                    color = self.colors['success']
                
                # Draw alert background
                alert_height = 30
                cv2.rectangle(frame, (10, y_offset - alert_height), 
                             (frame.shape[1] - 10, y_offset), color, -1)
                
                # Draw alert text
                cv2.putText(frame, f"⚠ {message}", (20, y_offset - 10), 
                           self.font, 0.6, self.colors['text'], 2)
                
                y_offset -= alert_height + 5
                
            except Exception as e:
                logger.error(f"Error rendering alert", error=str(e))
        
        return frame
    
    def _draw_door_icon(self, frame: np.ndarray, x: int, y: int, size: int, state: str):
        """
        Draw a simple door icon
        """
        if state == 'open':
            # Draw open door (arc)
            cv2.ellipse(frame, (x, y), (size//2, size//2), 0, -90, 0, 
                       self.colors['door_open'], 2)
        else:
            # Draw closed door (rectangle)
            cv2.rectangle(frame, (x - size//2, y - size//2), 
                         (x + size//2, y + size//2), 
                         self.colors['door_closed'], -1)
    
    def _draw_status_indicator(self, frame: np.ndarray, x: int, y: int, state: str):
        """
        Draw a status indicator dot
        """
        color = self.colors['door_open'] if state == 'open' else self.colors['door_closed']
        cv2.circle(frame, (x, y), 5, color, -1)
        
        # Add pulsing effect for open doors
        if state == 'open':
            cv2.circle(frame, (x, y), 8, color, 1)
    
    def _draw_tracking_path(self, frame: np.ndarray, path: List[Tuple[int, int]]):
        """
        Draw person tracking path
        """
        if len(path) < 2:
            return
        
        for i in range(1, len(path)):
            cv2.line(frame, path[i-1], path[i], self.colors['person'], 1)
    
    def _draw_fps_graph(self, frame: np.ndarray, fps_history: List[float], 
                       position: Tuple[int, int]):
        """
        Draw a simple FPS graph
        """
        if len(fps_history) < 2:
            return
        
        graph_width = 150
        graph_height = 50
        x, y = position
        
        # Draw graph background
        cv2.rectangle(frame, (x, y), (x + graph_width, y + graph_height), 
                     (50, 50, 50), -1)
        
        # Draw graph lines
        max_fps = 60
        points = []
        
        for i, fps in enumerate(fps_history[-30:]):  # Last 30 samples
            px = x + int(i * graph_width / 30)
            py = y + graph_height - int(fps * graph_height / max_fps)
            points.append((px, py))
        
        if len(points) > 1:
            for i in range(1, len(points)):
                cv2.line(frame, points[i-1], points[i], self.colors['success'], 1)
        
        # Draw target line (30 FPS)
        target_y = y + graph_height - int(30 * graph_height / max_fps)
        cv2.line(frame, (x, target_y), (x + graph_width, target_y), 
                (128, 128, 128), 1)
    
    def create_thumbnail(self, frame: np.ndarray, size: Tuple[int, int] = (320, 240)) -> np.ndarray:
        """
        Create a thumbnail of the frame
        """
        return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)