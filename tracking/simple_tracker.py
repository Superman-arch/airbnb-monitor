"""Simple tracker fallback when ByteTracker dependencies are not available."""

import numpy as np
from collections import defaultdict, deque


class SimpleTracker:
    """Simple IoU-based tracker as fallback."""
    
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = {}
        self.next_id = 0
        
    def update(self, detections):
        """Update tracks with new detections.
        
        Args:
            detections: Array of [x1, y1, x2, y2, score] detections
            
        Returns:
            List of tracked objects with IDs
        """
        if len(detections) == 0:
            # Update existing tracks
            to_remove = []
            for track_id, track in self.tracks.items():
                track['age'] += 1
                if track['age'] > self.max_age:
                    to_remove.append(track_id)
            
            for track_id in to_remove:
                del self.tracks[track_id]
            
            return []
        
        # Calculate IoU between existing tracks and new detections
        tracked_objects = []
        used_detections = set()
        
        for track_id, track in self.tracks.items():
            best_iou = 0
            best_det_idx = -1
            
            for det_idx, det in enumerate(detections):
                if det_idx in used_detections:
                    continue
                    
                iou = self._calculate_iou(track['bbox'], det[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx
            
            if best_iou > self.iou_threshold:
                # Update existing track
                track['bbox'] = detections[best_det_idx][:4]
                track['score'] = detections[best_det_idx][4] if len(detections[best_det_idx]) > 4 else 1.0
                track['hits'] += 1
                track['age'] = 0
                used_detections.add(best_det_idx)
                
                if track['hits'] >= self.min_hits:
                    tracked_objects.append({
                        'track_id': track_id,
                        'bbox': track['bbox'].tolist(),
                        'score': track['score']
                    })
            else:
                track['age'] += 1
        
        # Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx not in used_detections:
                self.tracks[self.next_id] = {
                    'bbox': det[:4],
                    'score': det[4] if len(det) > 4 else 1.0,
                    'hits': 1,
                    'age': 0
                }
                
                if self.min_hits <= 1:
                    tracked_objects.append({
                        'track_id': self.next_id,
                        'bbox': det[:4].tolist(),
                        'score': det[4] if len(det) > 4 else 1.0
                    })
                
                self.next_id += 1
        
        # Remove old tracks
        to_remove = []
        for track_id, track in self.tracks.items():
            if track['age'] > self.max_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
        
        return tracked_objects
    
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0