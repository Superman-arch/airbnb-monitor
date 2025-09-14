"""Person journey management across cameras and zones."""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import os
import sqlite3
from threading import Lock


class Person:
    """Represents a tracked person with journey history."""
    
    def __init__(self, person_id: str, track_id: int, first_seen: datetime):
        """Initialize a person."""
        self.person_id = person_id
        self.track_ids = {track_id}  # Can have multiple track IDs across cameras
        self.first_seen = first_seen
        self.last_seen = first_seen
        
        # Journey tracking
        self.journey = []  # List of (timestamp, camera_id, zone_id, action)
        self.current_zone = None
        self.zone_entry_time = None
        
        # Statistics
        self.total_zones_visited = set()
        self.time_in_zones = defaultdict(timedelta)  # zone_id -> total time
        
        # Appearance features for re-identification
        self.appearance_features = deque(maxlen=10)
        
    def update_location(self, camera_id: str, zone_id: Optional[str], 
                       timestamp: datetime, action: str = "move"):
        """Update person's location."""
        self.last_seen = timestamp
        
        # Record journey point
        self.journey.append({
            'timestamp': timestamp.isoformat(),
            'camera_id': camera_id,
            'zone_id': zone_id,
            'action': action
        })
        
        # Update zone statistics
        if zone_id:
            if zone_id != self.current_zone:
                # Exiting previous zone
                if self.current_zone and self.zone_entry_time:
                    duration = timestamp - self.zone_entry_time
                    self.time_in_zones[self.current_zone] += duration
                
                # Entering new zone
                self.current_zone = zone_id
                self.zone_entry_time = timestamp
                self.total_zones_visited.add(zone_id)
    
    def add_track_id(self, track_id: int):
        """Associate a new track ID with this person."""
        self.track_ids.add(track_id)
    
    def get_time_in_zone(self, zone_id: str) -> timedelta:
        """Get total time spent in a specific zone."""
        total = self.time_in_zones.get(zone_id, timedelta())
        
        # Add current session if still in zone
        if self.current_zone == zone_id and self.zone_entry_time:
            total += datetime.now() - self.zone_entry_time
        
        return total
    
    def get_journey_summary(self) -> Dict[str, Any]:
        """Get summary of person's journey."""
        return {
            'person_id': self.person_id,
            'track_ids': list(self.track_ids),
            'first_seen': self.first_seen.isoformat(),
            'last_seen': self.last_seen.isoformat(),
            'duration': str(self.last_seen - self.first_seen),
            'current_zone': self.current_zone,
            'zones_visited': list(self.total_zones_visited),
            'journey_points': len(self.journey),
            'time_in_zones': {
                zone: str(duration) 
                for zone, duration in self.time_in_zones.items()
            }
        }


class JourneyManager:
    """Manages person journeys across cameras and zones."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize journey manager."""
        self.config = config
        
        # Person tracking
        self.persons = {}  # person_id -> Person
        self.track_to_person = {}  # track_id -> person_id
        self.person_counter = 0
        
        # Database
        self.db_path = config.get('storage', {}).get('database_path', './storage/database.db')
        self.db_lock = Lock()
        self._init_database()
        
        # Re-identification threshold
        self.reid_threshold = 0.7
        
    def _init_database(self):
        """Initialize database for journey storage."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create persons table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS persons (
                    person_id TEXT PRIMARY KEY,
                    first_seen TIMESTAMP,
                    last_seen TIMESTAMP,
                    journey_data TEXT
                )
            ''')
            
            # Create events table (for both person and door events)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT,  -- 'person' or 'door'
                    person_id TEXT,
                    door_id TEXT,
                    timestamp TIMESTAMP,
                    camera_id TEXT,
                    zone_id TEXT,
                    zone_name TEXT,
                    action TEXT,  -- entry/exit for person, opened/closed for door
                    confidence REAL,
                    snapshot_path TEXT,
                    metadata TEXT  -- JSON for additional data
                )
            ''')
            
            # Create journey points table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS journey_points (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT,
                    timestamp TIMESTAMP,
                    camera_id TEXT,
                    zone_id TEXT,
                    action TEXT,
                    bbox TEXT,
                    FOREIGN KEY (person_id) REFERENCES persons (person_id)
                )
            ''')
            
            conn.commit()
    
    def store_door_event(self, event: Dict[str, Any]):
        """Store door event in database."""
        with self.db_lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Map door event fields to database columns
                event_type = 'door'
                door_id = event.get('door_id', '')
                timestamp = event.get('timestamp', datetime.now())
                action = event.get('event', '').replace('door_', '')  # Remove 'door_' prefix
                confidence = event.get('confidence', 0.0)
                
                # Store additional metadata as JSON
                import json
                metadata = json.dumps({
                    'previous_state': event.get('previous_state'),
                    'current_state': event.get('current_state'),
                    'bbox': event.get('bbox'),
                    'duration_seconds': event.get('duration_seconds')
                })
                
                cursor.execute('''
                    INSERT INTO events (event_type, door_id, timestamp, action, confidence, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (event_type, door_id, timestamp, action, confidence, metadata))
                
                conn.commit()
    
    def process_tracks(self, tracks: List[Dict[str, Any]], camera_id: str, 
                       zones: List[Any], timestamp: datetime = None) -> List[Dict[str, Any]]:
        """
        Process tracked persons and update journeys.
        
        Returns:
            List of events that occurred
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        events = []
        
        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            center = track['center']
            
            # Get or create person for this track
            person_id = self._get_or_create_person(track_id, timestamp)
            person = self.persons[person_id]
            
            # Check which zone the person is in
            current_zone = None
            zone_name = None
            for zone in zones:
                if zone.contains_bbox(bbox):
                    current_zone = zone.id
                    zone_name = zone.name
                    break
            
            # Detect zone changes
            previous_zone = person.current_zone
            if current_zone != previous_zone:
                # Person changed zones
                if previous_zone and current_zone:
                    # Moved from one zone to another
                    action = "transition"
                elif current_zone:
                    # Entered a zone
                    action = "entry"
                    events.append({
                        'person_id': person_id,
                        'timestamp': timestamp,
                        'camera_id': camera_id,
                        'zone_id': current_zone,
                        'zone_name': zone_name,
                        'action': action,
                        'bbox': bbox,
                        'confidence': track.get('confidence', 0.0)
                    })
                elif previous_zone:
                    # Exited a zone
                    action = "exit"
                    events.append({
                        'person_id': person_id,
                        'timestamp': timestamp,
                        'camera_id': camera_id,
                        'zone_id': previous_zone,
                        'zone_name': self._get_zone_name(previous_zone, zones),
                        'action': action,
                        'bbox': bbox,
                        'confidence': track.get('confidence', 0.0)
                    })
                else:
                    action = "move"
                
                # Update person location
                person.update_location(camera_id, current_zone, timestamp, action)
                
                # Store in database
                self._store_journey_point(person_id, timestamp, camera_id, 
                                        current_zone, action, bbox)
        
        return events
    
    def _get_or_create_person(self, track_id: int, timestamp: datetime) -> str:
        """Get existing person ID or create new one."""
        # Check if track already associated with a person
        if track_id in self.track_to_person:
            return self.track_to_person[track_id]
        
        # Try to re-identify person (simplified for now)
        # In production, would use appearance features
        person_id = None
        
        # If no match found, create new person
        if person_id is None:
            self.person_counter += 1
            person_id = f"P_{timestamp:%Y%m%d}_{self.person_counter:04d}"
            person = Person(person_id, track_id, timestamp)
            self.persons[person_id] = person
        
        # Associate track with person
        self.track_to_person[track_id] = person_id
        
        return person_id
    
    def _get_zone_name(self, zone_id: str, zones: List[Any]) -> str:
        """Get zone name by ID."""
        for zone in zones:
            if zone.id == zone_id:
                return zone.name
        return f"Zone {zone_id}"
    
    def _store_journey_point(self, person_id: str, timestamp: datetime, 
                            camera_id: str, zone_id: Optional[str], 
                            action: str, bbox: List[int]):
        """Store journey point in database."""
        with self.db_lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO journey_points 
                    (person_id, timestamp, camera_id, zone_id, action, bbox)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (person_id, timestamp, camera_id, zone_id, action, json.dumps(bbox)))
                conn.commit()
    
    def store_event(self, event: Dict[str, Any], snapshot_path: Optional[str] = None):
        """Store an event in the database."""
        with self.db_lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO events 
                    (event_type, person_id, timestamp, camera_id, zone_id, zone_name, 
                     action, confidence, snapshot_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    'person',  # event_type
                    event['person_id'],
                    event['timestamp'],
                    event['camera_id'],
                    event['zone_id'],
                    event.get('zone_name'),
                    event['action'],
                    event.get('confidence', 0.0),
                    snapshot_path
                ))
                conn.commit()
    
    def get_person_journey(self, person_id: str) -> Dict[str, Any]:
        """Get complete journey for a person."""
        if person_id in self.persons:
            person = self.persons[person_id]
            return person.get_journey_summary()
        
        # Try to load from database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT person_id, first_seen, last_seen, journey_data
                FROM persons WHERE person_id = ?
            ''', (person_id,))
            
            row = cursor.fetchone()
            if row:
                return json.loads(row[3]) if row[3] else {}
        
        return None
    
    def get_zone_occupancy(self, zone_id: str) -> List[str]:
        """Get list of persons currently in a zone."""
        occupants = []
        for person_id, person in self.persons.items():
            if person.current_zone == zone_id:
                occupants.append(person_id)
        return occupants
    
    def get_recent_events(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get recent events from database (both person and door events)."""
        since = datetime.now() - timedelta(minutes=minutes)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT event_type, person_id, door_id, timestamp, camera_id, 
                       zone_id, zone_name, action, confidence, snapshot_path, metadata
                FROM events 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            ''', (since,))
            
            events = []
            for row in cursor.fetchall():
                event = {
                    'event_type': row[0] or 'person',  # Default to person for backward compat
                    'person_id': row[1],
                    'door_id': row[2],
                    'timestamp': row[3],
                    'camera_id': row[4],
                    'zone_id': row[5],
                    'zone_name': row[6],
                    'action': row[7],
                    'confidence': row[8],
                    'snapshot_path': row[9]
                }
                
                # Parse metadata if present
                if row[10]:
                    try:
                        import json
                        metadata = json.loads(row[10])
                        event.update(metadata)
                    except:
                        pass
                
                events.append(event)
        
        return events
    
    def cleanup_old_data(self, days: int = 7):
        """Remove old journey data from database."""
        cutoff = datetime.now() - timedelta(days=days)
        
        with self.db_lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Remove old events
                cursor.execute('DELETE FROM events WHERE timestamp < ?', (cutoff,))
                
                # Remove old journey points
                cursor.execute('DELETE FROM journey_points WHERE timestamp < ?', (cutoff,))
                
                # Remove old persons
                cursor.execute('DELETE FROM persons WHERE last_seen < ?', (cutoff,))
                
                conn.commit()
        
        # Clean up memory
        to_remove = []
        for person_id, person in self.persons.items():
            if person.last_seen < cutoff:
                to_remove.append(person_id)
        
        for person_id in to_remove:
            del self.persons[person_id]
            # Clean up track associations
            tracks_to_remove = [
                track_id for track_id, pid in self.track_to_person.items() 
                if pid == person_id
            ]
            for track_id in tracks_to_remove:
                del self.track_to_person[track_id]