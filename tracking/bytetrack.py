"""ByteTrack implementation for person tracking."""

import numpy as np
from collections import deque

try:
    import lap
    from scipy.spatial.distance import cdist
    BYTETRACK_AVAILABLE = True
except ImportError as e:
    print(f"ByteTrack dependencies not available: {e}")
    BYTETRACK_AVAILABLE = False


class STrack:
    """Single track representation."""
    
    shared_kalman = None
    _count = 0
    
    def __init__(self, tlbr, score, feat=None, feat_history=50):
        """Initialize a single track."""
        # Convert tlbr to tlwh
        self._tlbr = np.asarray(tlbr, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        
        self.score = score
        self.track_id = 0
        
        self.frame_id = 0
        self.start_frame = 0
        
        # Feature for ReID
        self.curr_feat = feat
        self.feat_history = deque(maxlen=feat_history)
        if feat is not None:
            self.feat_history.append(feat)
        
        self.alpha = 0.9
        
    @property
    def tlbr(self):
        """Get current position in tlbr format."""
        return self._tlbr.copy()
    
    @property
    def tlwh(self):
        """Get current position in tlwh format."""
        ret = self._tlbr.copy()
        ret[2:] -= ret[:2]
        return ret
    
    @staticmethod
    def tlbr_to_tlwh(tlbr):
        """Convert tlbr to tlwh."""
        ret = tlbr.copy()
        ret[2:] -= ret[:2]
        return ret
    
    @staticmethod
    def tlwh_to_tlbr(tlwh):
        """Convert tlwh to tlbr."""
        ret = tlwh.copy()
        ret[2:] += ret[:2]
        return ret
    
    def activate(self, frame_id):
        """Start a new track."""
        self.track_id = self.next_id()
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.is_activated = True
        
    @staticmethod
    def next_id():
        """Get next track ID."""
        STrack._count += 1
        return STrack._count
    
    def update(self, new_track, frame_id):
        """Update track with new detection."""
        self.frame_id = frame_id
        self._tlbr = new_track.tlbr
        self.score = new_track.score
        
        if new_track.curr_feat is not None:
            self.curr_feat = new_track.curr_feat
            self.feat_history.append(new_track.curr_feat)
            
    def mark_lost(self):
        """Mark track as lost."""
        pass
    
    def mark_removed(self):
        """Mark track as removed."""
        pass


class ByteTracker:
    """ByteTrack tracker implementation."""
    
    def __init__(self, track_thresh=0.5, match_thresh=0.8, track_buffer=30, min_box_area=100):
        """
        Initialize ByteTracker.
        
        Args:
            track_thresh: Tracking confidence threshold
            match_thresh: Matching threshold for association
            track_buffer: Frames to keep lost tracks
            min_box_area: Minimum box area
        """
        if not BYTETRACK_AVAILABLE:
            raise ImportError("ByteTracker requires lap and scipy. Install with: pip install lap scipy")
        
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.min_box_area = min_box_area
        
        self.frame_id = 0
        self.tracked_stracks = []  # Active tracks
        self.lost_stracks = []     # Lost tracks
        self.removed_stracks = []   # Removed tracks
        
    def update(self, detections):
        """
        Update tracker with new detections.
        
        Args:
            detections: Array of detections [x1, y1, x2, y2, score, class_id]
            
        Returns:
            List of active tracks
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        if len(detections) > 0:
            # Split detections into high and low confidence
            scores = detections[:, 4]
            bboxes = detections[:, :4]
            
            remain_inds = scores > self.track_thresh
            low_inds = np.logical_and(scores > 0.1, scores <= self.track_thresh)
            
            dets_high = bboxes[remain_inds]
            scores_high = scores[remain_inds]
            
            dets_low = bboxes[low_inds]
            scores_low = scores[low_inds]
        else:
            dets_high = np.empty((0, 4))
            scores_high = np.empty((0,))
            dets_low = np.empty((0, 4))
            scores_low = np.empty((0,))
        
        # Create tracks for high confidence detections
        if len(dets_high) > 0:
            detections_high = [STrack(tlbr, s) for tlbr, s in zip(dets_high, scores_high)]
        else:
            detections_high = []
        
        # Add tracks
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        
        # Match with high confidence detections
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        dists = matching_distance(strack_pool, detections_high)
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)
        
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections_high[idet]
            track.update(det, self.frame_id)
            if track.is_activated:
                activated_stracks.append(track)
            else:
                track.activate(self.frame_id)
                activated_stracks.append(track)
        
        # Process unmatched tracks and detections
        for it in u_track:
            track = strack_pool[it]
            track.mark_lost()
            lost_stracks.append(track)
        
        # Create new tracks for unmatched high confidence detections
        for inew in u_detection:
            track = detections_high[inew]
            if track.score < 0.6:
                continue
            track.activate(self.frame_id)
            activated_stracks.append(track)
        
        # Process low confidence detections
        if len(dets_low) > 0:
            detections_low = [STrack(tlbr, s) for tlbr, s in zip(dets_low, scores_low)]
        else:
            detections_low = []
        
        if len(lost_stracks) > 0 and len(detections_low) > 0:
            dists = matching_distance(lost_stracks, detections_low)
            matches, u_track, u_detection = linear_assignment(dists, thresh=0.5)
            
            for itracked, idet in matches:
                track = lost_stracks[itracked]
                det = detections_low[idet]
                track.update(det, self.frame_id)
                track.is_activated = True
                refind_stracks.append(track)
        
        # Remove lost tracks
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.track_buffer:
                track.mark_removed()
                removed_stracks.append(track)
        
        # Update track lists
        self.tracked_stracks = [t for t in self.tracked_stracks if t.is_activated]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, removed_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.removed_stracks.extend(removed_stracks)
        
        # Filter by area
        output_stracks = []
        for track in self.tracked_stracks:
            tlbr = track.tlbr
            area = (tlbr[2] - tlbr[0]) * (tlbr[3] - tlbr[1])
            if area > self.min_box_area:
                output_stracks.append(track)
        
        return output_stracks


def joint_stracks(tlista, tlistb):
    """Combine two track lists."""
    exists = set()
    res = []
    for t in tlista:
        exists.add(t.track_id)
        res.append(t)
    for t in tlistb:
        if t.track_id not in exists:
            exists.add(t.track_id)
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    """Subtract track list B from A."""
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        if t.track_id in stracks:
            del stracks[t.track_id]
    return list(stracks.values())


def matching_distance(tracks, detections):
    """Calculate matching distance matrix."""
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)))
    
    # IoU distance
    track_boxes = np.array([t.tlbr for t in tracks])
    det_boxes = np.array([d.tlbr for d in detections])
    
    # Calculate IoU
    dists = np.zeros((len(tracks), len(detections)))
    for i, track_box in enumerate(track_boxes):
        for j, det_box in enumerate(det_boxes):
            dists[i, j] = 1 - iou(track_box, det_box)
    
    return dists


def iou(box1, box2):
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


def linear_assignment(cost_matrix, thresh=0.8):
    """Solve linear assignment problem."""
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), [], []
    
    matches, unmatched_a, unmatched_b = [], [], []
    
    # Use lap.lapjv for Hungarian algorithm
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
        else:
            unmatched_a.append(ix)
    
    for iy, my in enumerate(y):
        if my < 0:
            unmatched_b.append(iy)
    
    matches = np.array(matches)
    return matches, unmatched_a, unmatched_b