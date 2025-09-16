"""Centralized logging system for hotel monitoring."""

import logging
import logging.handlers
import json
import os
import glob
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from collections import deque
import threading

class SystemLogger:
    """Production-grade logging system with file and memory storage."""
    
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
        
        # Try to create logs directory, fallback to /tmp if permission denied
        self.log_dir = '/app/logs'
        try:
            os.makedirs(self.log_dir, exist_ok=True)
        except PermissionError:
            # Fallback to /tmp if /app/logs is not writable
            self.log_dir = '/tmp/logs'
            try:
                os.makedirs(self.log_dir, exist_ok=True)
            except:
                # If even /tmp fails, we'll use console-only logging
                self.log_dir = None
        
        # Setup file logging with rotation
        self.setup_file_logging()
        
        # Memory buffer for recent events (for web interface)
        self.recent_events = deque(maxlen=100)
        self.recent_logs = deque(maxlen=500)
        
        # Event types for structured logging
        self.EVENT_DOOR_OPEN = "DOOR_OPEN"
        self.EVENT_DOOR_CLOSED = "DOOR_CLOSED"
        self.EVENT_PERSON_DETECTED = "PERSON_DETECTED"
        self.EVENT_MOTION_DETECTED = "MOTION_DETECTED"
        self.EVENT_ZONE_ENTERED = "ZONE_ENTERED"
        self.EVENT_ZONE_EXITED = "ZONE_EXITED"
        self.EVENT_CALIBRATION = "CALIBRATION"
        self.EVENT_SYSTEM = "SYSTEM"
        
    def setup_file_logging(self):
        """Setup rotating file handlers for different log types."""
        
        # Main application log
        self.app_logger = logging.getLogger('airbnb_monitor')
        self.app_logger.setLevel(logging.DEBUG)
        
        # Add console handler as primary (always works)
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.app_logger.addHandler(console_handler)
        
        # Try to add file handler if we have a writable log directory
        if self.log_dir:
            try:
                app_handler = logging.handlers.RotatingFileHandler(
                    f'{self.log_dir}/monitor.log',
                    maxBytes=10*1024*1024,
                    backupCount=5
                )
                app_handler.setFormatter(console_formatter)
                self.app_logger.addHandler(app_handler)
            except (PermissionError, OSError) as e:
                self.app_logger.warning(f"Could not create file handler: {e}. Using console only.")
        
        # Events log (structured JSON)
        self.event_logger = logging.getLogger('events')
        self.event_logger.setLevel(logging.INFO)
        
        # Add console handler for events
        event_console = logging.StreamHandler()
        self.event_logger.addHandler(event_console)
        
        # Try to add file handler
        if self.log_dir:
            try:
                event_handler = logging.handlers.RotatingFileHandler(
                    f'{self.log_dir}/events.json',
                    maxBytes=10*1024*1024,
                    backupCount=5
                )
                self.event_logger.addHandler(event_handler)
            except (PermissionError, OSError):
                pass  # Silently fallback to console only
        
        # Door state log
        self.door_logger = logging.getLogger('doors')
        self.door_logger.setLevel(logging.INFO)
        
        # Add console handler for doors
        door_console = logging.StreamHandler()
        door_formatter = logging.Formatter(
            '%(asctime)s - DOOR - %(message)s'
        )
        door_console.setFormatter(door_formatter)
        self.door_logger.addHandler(door_console)
        
        # Try to add file handler
        if self.log_dir:
            try:
                door_handler = logging.handlers.RotatingFileHandler(
                    f'{self.log_dir}/doors.log',
                    maxBytes=5*1024*1024,
                    backupCount=3
                )
                door_handler.setFormatter(door_formatter)
                self.door_logger.addHandler(door_handler)
            except (PermissionError, OSError):
                pass  # Silently fallback to console only
        
    def log(self, level: str, message: str, component: str = "SYSTEM"):
        """General logging method."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'component': component,
            'message': message
        }
        
        # Add to memory buffer
        self.recent_logs.append(log_entry)
        
        # Log to file
        if level == 'DEBUG':
            self.app_logger.debug(f"[{component}] {message}")
        elif level == 'INFO':
            self.app_logger.info(f"[{component}] {message}")
        elif level == 'WARNING':
            self.app_logger.warning(f"[{component}] {message}")
        elif level == 'ERROR':
            self.app_logger.error(f"[{component}] {message}")
            
    def log_event(self, event_type: str, data: Dict[str, Any], priority: str = "normal"):
        """Log structured events for analytics and web display."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'priority': priority,
            'data': data
        }
        
        # Add to recent events buffer
        self.recent_events.append(event)
        
        # Log to events file as JSON
        self.event_logger.info(json.dumps(event))
        
        # Also log to main logger for visibility
        self.app_logger.info(f"EVENT: {event_type} - {data}")
        
        return event
        
    def log_door_event(self, door_id: str, state: str, confidence: float, zone: Optional[str] = None):
        """Log door state changes."""
        event_type = self.EVENT_DOOR_OPEN if state == "open" else self.EVENT_DOOR_CLOSED
        
        data = {
            'door_id': door_id,
            'state': state,
            'confidence': confidence,
            'zone': zone
        }
        
        # Log to door-specific log
        self.door_logger.info(f"Door {door_id} is now {state} (confidence: {confidence:.2%})")
        
        # Log as event
        return self.log_event(event_type, data, priority="high" if state == "open" else "normal")
        
    def log_person_detection(self, zone: str, count: int, confidence: float):
        """Log person detection events."""
        data = {
            'zone': zone,
            'count': count,
            'confidence': confidence
        }
        
        return self.log_event(self.EVENT_PERSON_DETECTED, data)
        
    def log_calibration(self, result: Dict[str, Any]):
        """Log calibration results."""
        return self.log_event(self.EVENT_CALIBRATION, result)
        
    def get_recent_events(self, limit: int = 20) -> list:
        """Get recent events for web display."""
        return list(self.recent_events)[-limit:]
        
    def get_recent_logs(self, limit: int = 100) -> list:
        """Get recent log entries for web display."""
        return list(self.recent_logs)[-limit:]
        
    def clear_old_logs(self, days: int = 7):
        """Clean up old log files."""
        import glob
        from datetime import timedelta
        
        cutoff = datetime.now() - timedelta(days=days)
        
        if not self.log_dir:
            return
        
        for log_file in glob.glob(f'{self.log_dir}/*.log.*'):
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(log_file))
                if mtime < cutoff:
                    os.remove(log_file)
                    self.log('INFO', f"Removed old log file: {log_file}")
            except Exception as e:
                self.log('ERROR', f"Failed to remove log file {log_file}: {e}")

# Global logger instance
logger = SystemLogger()