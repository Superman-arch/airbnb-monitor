"""Unified State Manager for all detection systems."""

import threading
from datetime import datetime
from typing import Dict, Any, List, Optional
from collections import deque
import json

class UnifiedStateManager:
    """Central state management for all detection systems."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, 'initialized'):
            return
        self.initialized = True
        
        # Thread safety
        self.lock = threading.Lock()
        
        # System state
        self.system_ready = False
        self.calibration_complete = False
        
        # Door state
        self.doors = {}  # door_id -> door_info
        self.door_events = deque(maxlen=100)
        
        # Zone state
        self.active_zones = set()
        self.zone_events = deque(maxlen=100)
        
        # Person state
        self.active_persons = {}  # person_id -> person_info
        self.person_count = 0
        self.person_events = deque(maxlen=100)
        
        # Event state
        self.recent_events = deque(maxlen=200)
        self.event_count = 0
        
        # Stats cache
        self._stats_cache = {
            'active_zones': 0,
            'active_persons': 0,
            'total_events': 0,
            'doors_open': 0,
            'doors_closed': 0,
            'last_update': datetime.now()
        }
        
        # Callbacks for real-time updates
        self.update_callbacks = []
        
    def register_callback(self, callback):
        """Register a callback for state updates."""
        with self.lock:
            self.update_callbacks.append(callback)
    
    def _notify_callbacks(self, update_type: str, data: Any):
        """Notify all registered callbacks of state change."""
        for callback in self.update_callbacks:
            try:
                callback(update_type, data)
            except Exception as e:
                print(f"Callback error: {e}")
    
    # Door Management
    def update_door(self, door_id: str, state: str, confidence: float, 
                    bbox: tuple = None, zone: str = None):
        """Update door state."""
        with self.lock:
            if door_id not in self.doors:
                self.doors[door_id] = {
                    'id': door_id,
                    'first_seen': datetime.now(),
                    'state_history': []
                }
            
            door = self.doors[door_id]
            old_state = door.get('state', 'unknown')
            
            # Update door info
            door.update({
                'state': state,
                'confidence': confidence,
                'last_update': datetime.now(),
                'bbox': bbox,
                'zone': zone
            })
            
            # Track state changes
            if old_state != state:
                change_event = {
                    'timestamp': datetime.now(),
                    'door_id': door_id,
                    'old_state': old_state,
                    'new_state': state,
                    'confidence': confidence
                }
                door['state_history'].append(change_event)
                self.door_events.append(change_event)
                self.add_event('door_state_change', change_event)
                
                # Update stats
                self._update_door_stats()
                
                # Notify callbacks
                self._notify_callbacks('door_update', door)
    
    def get_door_states(self) -> Dict[str, Any]:
        """Get all door states."""
        with self.lock:
            return {
                door_id: {
                    'id': door_id,
                    'state': info.get('state', 'unknown'),
                    'confidence': info.get('confidence', 0),
                    'bbox': info.get('bbox'),
                    'zone': info.get('zone'),
                    'last_update': info.get('last_update', datetime.now()).isoformat()
                }
                for door_id, info in self.doors.items()
            }
    
    # Zone Management
    def update_zone(self, zone_id: str, active: bool, motion_level: float = 0):
        """Update zone state."""
        with self.lock:
            was_active = zone_id in self.active_zones
            
            if active:
                self.active_zones.add(zone_id)
            else:
                self.active_zones.discard(zone_id)
            
            if was_active != active:
                event = {
                    'timestamp': datetime.now(),
                    'zone_id': zone_id,
                    'active': active,
                    'motion_level': motion_level
                }
                self.zone_events.append(event)
                self.add_event('zone_activity', event)
                self._notify_callbacks('zone_update', event)
    
    def get_active_zones_count(self) -> int:
        """Get count of active zones."""
        with self.lock:
            return len(self.active_zones)
    
    # Person Management
    def update_person(self, person_id: str, bbox: tuple, confidence: float, 
                     zone: str = None):
        """Update person tracking."""
        with self.lock:
            is_new = person_id not in self.active_persons
            
            self.active_persons[person_id] = {
                'id': person_id,
                'bbox': bbox,
                'confidence': confidence,
                'zone': zone,
                'last_seen': datetime.now()
            }
            
            if is_new:
                event = {
                    'timestamp': datetime.now(),
                    'person_id': person_id,
                    'action': 'detected',
                    'zone': zone
                }
                self.person_events.append(event)
                self.add_event('person_detected', event)
                self._notify_callbacks('person_update', event)
            
            self.person_count = len(self.active_persons)
    
    def remove_stale_persons(self, timeout_seconds: int = 5):
        """Remove persons not seen recently."""
        with self.lock:
            now = datetime.now()
            to_remove = []
            
            for person_id, info in self.active_persons.items():
                if (now - info['last_seen']).total_seconds() > timeout_seconds:
                    to_remove.append(person_id)
            
            for person_id in to_remove:
                del self.active_persons[person_id]
                event = {
                    'timestamp': now,
                    'person_id': person_id,
                    'action': 'lost'
                }
                self.person_events.append(event)
                self.add_event('person_lost', event)
            
            self.person_count = len(self.active_persons)
    
    def get_active_persons_count(self) -> int:
        """Get count of active persons."""
        with self.lock:
            return self.person_count
    
    # Event Management
    def add_event(self, event_type: str, data: Dict[str, Any]):
        """Add a system event."""
        with self.lock:
            event = {
                'id': self.event_count,
                'type': event_type,
                'timestamp': datetime.now().isoformat(),
                'data': data
            }
            self.recent_events.append(event)
            self.event_count += 1
            
            # Update stats
            self._stats_cache['total_events'] = self.event_count
    
    def get_recent_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent events."""
        with self.lock:
            events = list(self.recent_events)
            return events[-limit:] if len(events) > limit else events
    
    # Stats Management
    def _update_door_stats(self):
        """Update door statistics."""
        open_count = sum(1 for d in self.doors.values() if d.get('state') == 'open')
        closed_count = sum(1 for d in self.doors.values() if d.get('state') == 'closed')
        
        self._stats_cache.update({
            'doors_open': open_count,
            'doors_closed': closed_count,
            'last_update': datetime.now()
        })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current system statistics."""
        with self.lock:
            return {
                'active_zones': len(self.active_zones),
                'active_persons': self.person_count,
                'total_events': self.event_count,
                'doors_open': self._stats_cache['doors_open'],
                'doors_closed': self._stats_cache['doors_closed'],
                'doors_total': len(self.doors),
                'system_ready': self.system_ready,
                'calibration_complete': self.calibration_complete,
                'last_update': self._stats_cache['last_update'].isoformat()
            }
    
    # System Management
    def set_system_ready(self, ready: bool = True):
        """Set system ready state."""
        with self.lock:
            self.system_ready = ready
            self._notify_callbacks('system_status', {'ready': ready})
    
    def set_calibration_complete(self, complete: bool = True):
        """Set calibration complete state."""
        with self.lock:
            self.calibration_complete = complete
            self._notify_callbacks('calibration_status', {'complete': complete})
    
    def reset(self):
        """Reset all state."""
        with self.lock:
            self.doors.clear()
            self.active_zones.clear()
            self.active_persons.clear()
            self.recent_events.clear()
            self.person_count = 0
            self.event_count = 0
            self.system_ready = False
            self.calibration_complete = False
            self._notify_callbacks('system_reset', {})

# Global state manager instance
state_manager = UnifiedStateManager()