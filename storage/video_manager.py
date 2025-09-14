"""48-hour circular video storage system."""

import os
import cv2
import json
import shutil
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from threading import Thread, Lock
from queue import Queue
import numpy as np


class VideoSegment:
    """Represents a video segment."""
    
    def __init__(self, segment_id: str, start_time: datetime, 
                 duration_minutes: int, file_path: str):
        """Initialize video segment."""
        self.id = segment_id
        self.start_time = start_time
        self.duration_minutes = duration_minutes
        self.file_path = file_path
        self.end_time = start_time + timedelta(minutes=duration_minutes)
        self.size_mb = 0
        self.frame_count = 0
        
    def contains_time(self, timestamp: datetime) -> bool:
        """Check if timestamp is within this segment."""
        return self.start_time <= timestamp < self.end_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration_minutes': self.duration_minutes,
            'file_path': self.file_path,
            'size_mb': self.size_mb,
            'frame_count': self.frame_count
        }


class CircularVideoBuffer:
    """Manages 48-hour circular video storage."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize video buffer."""
        self.config = config
        storage_config = config.get('storage', {})
        
        self.retention_hours = storage_config.get('video_retention_hours', 48)
        self.segment_duration = storage_config.get('segment_duration_minutes', 60)
        self.storage_path = storage_config.get('storage_path', './storage/videos')
        # Use H264 codec for Jetson (better compatibility)
        self.video_codec = storage_config.get('video_codec', 'h264')
        self.video_quality = storage_config.get('video_quality', 23)
        
        # Create storage directory
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Video segments
        self.segments = []  # List of VideoSegment objects
        self.segments_lock = Lock()
        self.current_segment = None
        self.current_writer = None
        
        # Frame buffer for current segment
        self.frame_buffer = Queue(maxsize=300)  # 10 seconds at 30fps
        self.writer_thread = None
        self.running = False
        
        # Load existing segments
        self._load_segments()
        
    def _load_segments(self):
        """Load existing video segments from disk."""
        metadata_file = os.path.join(self.storage_path, 'segments.json')
        
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    segments_data = json.load(f)
                
                for seg_data in segments_data:
                    segment = VideoSegment(
                        seg_data['id'],
                        datetime.fromisoformat(seg_data['start_time']),
                        seg_data['duration_minutes'],
                        seg_data['file_path']
                    )
                    segment.size_mb = seg_data.get('size_mb', 0)
                    segment.frame_count = seg_data.get('frame_count', 0)
                    
                    # Verify file exists
                    if os.path.exists(segment.file_path):
                        self.segments.append(segment)
                        
            except Exception as e:
                print(f"Error loading segments: {e}")
    
    def start(self, camera_resolution: Tuple[int, int], fps: int = 30):
        """Start video recording."""
        self.camera_resolution = camera_resolution
        self.fps = fps
        
        self.running = True
        self.writer_thread = Thread(target=self._writer_worker, daemon=True)
        self.writer_thread.start()
        
        # Start new segment
        self._start_new_segment()
        
        print(f"Video storage started. Retention: {self.retention_hours} hours")
    
    def stop(self):
        """Stop video recording."""
        self.running = False
        
        # Close current segment
        if self.current_writer:
            self.current_writer.release()
        
        # Wait for writer thread
        if self.writer_thread:
            self.writer_thread.join(timeout=5)
        
        # Save metadata
        self._save_segments()
    
    def add_frame(self, frame: np.ndarray, timestamp: Optional[datetime] = None):
        """Add a frame to the video buffer."""
        if not self.running:
            return
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Check if we need a new segment
        if self.current_segment:
            if not self.current_segment.contains_time(timestamp):
                self._start_new_segment()
        
        # Add frame to buffer
        try:
            self.frame_buffer.put((frame, timestamp), timeout=0.1)
        except:
            # Buffer full, skip frame
            pass
    
    def _writer_worker(self):
        """Worker thread for writing video frames."""
        while self.running:
            try:
                # Get frame from buffer
                frame, timestamp = self.frame_buffer.get(timeout=1)
                
                # Write to current segment
                if self.current_writer:
                    self.current_writer.write(frame)
                    self.current_segment.frame_count += 1
                    
            except:
                continue
    
    def _start_new_segment(self):
        """Start a new video segment."""
        with self.segments_lock:
            # Close previous segment
            if self.current_writer:
                self.current_writer.release()
                self._finalize_segment(self.current_segment)
            
            # Create new segment
            now = datetime.now()
            segment_id = f"seg_{now:%Y%m%d_%H%M%S}"
            file_name = f"{segment_id}.mp4"
            file_path = os.path.join(self.storage_path, file_name)
            
            self.current_segment = VideoSegment(
                segment_id, now, self.segment_duration, file_path
            )
            
            # Create video writer with proper codec
            try:
                # Try H264 first (best for Jetson)
                if self.video_codec == 'h264':
                    fourcc = cv2.VideoWriter_fourcc(*'H264')
                elif self.video_codec == 'mp4v':
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                else:
                    # Fallback to MJPEG (always works)
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    file_path = file_path.replace('.mp4', '.avi')
                
                self.current_writer = cv2.VideoWriter(
                    file_path, fourcc, self.fps, self.camera_resolution
                )
                
                if not self.current_writer.isOpened():
                    # Fallback to MJPEG if codec fails
                    print(f"Warning: {self.video_codec} codec failed, using MJPEG")
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    file_path = file_path.replace('.mp4', '.avi')
                    self.current_writer = cv2.VideoWriter(
                        file_path, fourcc, self.fps, self.camera_resolution
                    )
            except Exception as e:
                print(f"Error creating video writer: {e}")
                self.current_writer = None
            
            # Add to segments list
            self.segments.append(self.current_segment)
            
            # Clean up old segments
            self._cleanup_old_segments()
            
            print(f"Started new video segment: {segment_id}")
    
    def _finalize_segment(self, segment: VideoSegment):
        """Finalize a completed segment."""
        if segment and os.path.exists(segment.file_path):
            # Get file size
            segment.size_mb = os.path.getsize(segment.file_path) / (1024 * 1024)
            
            # Save metadata
            self._save_segments()
    
    def _cleanup_old_segments(self):
        """Remove segments older than retention period."""
        cutoff = datetime.now() - timedelta(hours=self.retention_hours)
        
        segments_to_remove = []
        for segment in self.segments:
            if segment.end_time < cutoff:
                segments_to_remove.append(segment)
        
        for segment in segments_to_remove:
            # Delete file
            if os.path.exists(segment.file_path):
                try:
                    os.remove(segment.file_path)
                    print(f"Removed old segment: {segment.id}")
                except Exception as e:
                    print(f"Error removing segment {segment.id}: {e}")
            
            # Remove from list
            self.segments.remove(segment)
    
    def _save_segments(self):
        """Save segment metadata to disk."""
        metadata_file = os.path.join(self.storage_path, 'segments.json')
        
        with self.segments_lock:
            segments_data = [seg.to_dict() for seg in self.segments]
            
            with open(metadata_file, 'w') as f:
                json.dump(segments_data, f, indent=2)
    
    def get_video_at_time(self, timestamp: datetime) -> Optional[str]:
        """Get video file path containing the specified timestamp."""
        with self.segments_lock:
            for segment in self.segments:
                if segment.contains_time(timestamp):
                    return segment.file_path
        return None
    
    def get_video_clip(self, start_time: datetime, duration_seconds: int = 10) -> Optional[str]:
        """Extract a video clip from storage."""
        # Find relevant segments
        end_time = start_time + timedelta(seconds=duration_seconds)
        relevant_segments = []
        
        with self.segments_lock:
            for segment in self.segments:
                if (segment.start_time <= end_time and 
                    segment.end_time >= start_time):
                    relevant_segments.append(segment)
        
        if not relevant_segments:
            return None
        
        # Create clip
        clip_name = f"clip_{start_time:%Y%m%d_%H%M%S}.mp4"
        clip_path = os.path.join(self.storage_path, 'clips', clip_name)
        os.makedirs(os.path.dirname(clip_path), exist_ok=True)
        
        # If single segment, just extract portion
        if len(relevant_segments) == 1:
            segment = relevant_segments[0]
            self._extract_clip(segment.file_path, clip_path, 
                              start_time, duration_seconds)
        else:
            # Multiple segments, need to concatenate
            self._concatenate_clips(relevant_segments, clip_path, 
                                  start_time, end_time)
        
        return clip_path
    
    def _extract_clip(self, video_path: str, output_path: str, 
                     start_time: datetime, duration_seconds: int):
        """Extract a clip from a video file."""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate frame range
        # This is simplified - in production would need precise timestamp mapping
        start_frame = 0  # Would calculate based on segment start time
        end_frame = start_frame + int(fps * duration_seconds)
        
        # Create writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Extract frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for i in range(duration_seconds * int(fps)):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        cap.release()
        out.release()
    
    def _concatenate_clips(self, segments: List[VideoSegment], output_path: str,
                          start_time: datetime, end_time: datetime):
        """Concatenate clips from multiple segments."""
        # Simplified implementation
        # In production, would use ffmpeg for efficient concatenation
        pass
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_size_mb = sum(seg.size_mb for seg in self.segments)
        total_frames = sum(seg.frame_count for seg in self.segments)
        
        oldest_segment = min(self.segments, key=lambda s: s.start_time) if self.segments else None
        newest_segment = max(self.segments, key=lambda s: s.start_time) if self.segments else None
        
        return {
            'total_segments': len(self.segments),
            'total_size_mb': total_size_mb,
            'total_frames': total_frames,
            'oldest_recording': oldest_segment.start_time.isoformat() if oldest_segment else None,
            'newest_recording': newest_segment.start_time.isoformat() if newest_segment else None,
            'retention_hours': self.retention_hours,
            'segment_duration_minutes': self.segment_duration
        }
    
    def save_snapshot(self, frame: np.ndarray, event_id: str) -> str:
        """Save a snapshot for an event."""
        snapshot_dir = os.path.join(self.storage_path, 'snapshots')
        os.makedirs(snapshot_dir, exist_ok=True)
        
        filename = f"{event_id}_{datetime.now():%Y%m%d_%H%M%S}.jpg"
        filepath = os.path.join(snapshot_dir, filename)
        
        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        return filepath