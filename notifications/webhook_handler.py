"""N8N webhook integration for notifications."""

import requests
import json
import base64
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from threading import Thread, Lock
from queue import Queue, Empty
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import os


class WebhookHandler:
    """Handles webhook notifications to n8n."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize webhook handler."""
        self.config = config
        notification_config = config.get('notifications', {})
        
        # Support multiple webhook URLs
        self.webhook_url = notification_config.get('webhook_url', '')
        self.person_webhook_url = notification_config.get('person_webhook_url', '')
        self.door_webhook_url = notification_config.get('door_webhook_url', '')
        
        self.enabled = notification_config.get('enabled', True)
        self.cooldown_seconds = notification_config.get('cooldown_seconds', 10)
        self.include_snapshot = notification_config.get('include_snapshot', True)
        self.include_journey = notification_config.get('include_journey', True)
        
        # Notification queue for async sending
        self.notification_queue = Queue()
        self.worker_thread = None
        self.running = False
        
        # Cooldown tracking
        self.last_notification = {}  # person_id -> last notification time
        self.cooldown_lock = Lock()
        
        # Statistics
        self.notifications_sent = 0
        self.notifications_failed = 0
        
    def start(self):
        """Start the webhook handler."""
        if not self.webhook_url:
            print("Warning: No webhook URL configured")
            return
        
        self.running = True
        self.worker_thread = Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        print(f"Webhook handler started. URL: {self.webhook_url}")
    
    def stop(self):
        """Stop the webhook handler."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
    
    def send_event(self, event: Dict[str, Any], frame: Optional[np.ndarray] = None, 
                   journey: Optional[Dict[str, Any]] = None):
        """
        Queue an event for notification.
        
        Args:
            event: Event data with person_id, zone info, etc.
            frame: Optional frame for snapshot
            journey: Optional journey data
        """
        if not self.enabled:
            return
        
        # Determine which webhook to use based on event type
        webhook_url = self._get_webhook_for_event(event)
        if not webhook_url:
            return
        
        # Check cooldown (use different cooldowns for different event types)
        cooldown_key = event.get('person_id') or event.get('door_id')
        if cooldown_key and not self._check_cooldown(cooldown_key):
            return
        
        # Prepare notification
        notification = self._prepare_notification(event, frame, journey)
        notification['webhook_url'] = webhook_url  # Include target webhook
        
        # Queue for sending
        self.notification_queue.put(notification)
    
    def _get_webhook_for_event(self, event: Dict[str, Any]) -> str:
        """Determine which webhook URL to use based on event type."""
        event_type = event.get('event', event.get('action', ''))
        
        # Door events
        if any(door_word in event_type for door_word in ['door', 'discovered']):
            return self.door_webhook_url or self.webhook_url
        
        # Person events
        elif any(person_word in event_type for person_word in ['entry', 'exit', 'person']):
            return self.person_webhook_url or self.webhook_url
        
        # Default
        return self.webhook_url
    
    def _check_cooldown(self, person_id: str) -> bool:
        """Check if cooldown period has passed for person."""
        with self.cooldown_lock:
            last_time = self.last_notification.get(person_id)
            now = datetime.now()
            
            if last_time:
                elapsed = (now - last_time).total_seconds()
                if elapsed < self.cooldown_seconds:
                    return False
            
            self.last_notification[person_id] = now
            return True
    
    def _prepare_notification(self, event: Dict[str, Any], 
                             frame: Optional[np.ndarray] = None,
                             journey: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Prepare notification payload."""
        # Base notification
        notification = {
            'event_type': event.get('event', event.get('action', 'detection')),
            'timestamp': event.get('timestamp', datetime.now()).isoformat(),
            'camera': {
                'id': event.get('camera_id', 'camera_1')
            }
        }
        
        # Add person data if this is a person event
        if 'person_id' in event:
            notification['person'] = {
                'id': event.get('person_id'),
                'confidence': event.get('confidence', 0.0)
            }
            # Add zone/door info for person events
            if event.get('zone_id'):
                notification['zone'] = {
                    'id': event.get('zone_id'),
                    'name': event.get('zone_name', 'Unknown'),
                    'type': event.get('zone_type', 'unknown')
                }
        
        # Add door data if this is a door event
        if 'door_id' in event:
            notification['door'] = {
                'id': event.get('door_id'),
                'state': event.get('current_state', event.get('state', 'unknown')),
                'previous_state': event.get('previous_state'),
                'confidence': event.get('confidence', 0.0)
            }
            # Add bbox if available
            if 'bbox' in event:
                notification['door']['location'] = event['bbox']
            # Add duration for door_left_open events
            if 'duration_seconds' in event:
                notification['door']['duration_open'] = event['duration_seconds']
        
        # Add snapshot if available
        if self.include_snapshot and frame is not None:
            snapshot_base64 = self._encode_frame(frame, event.get('bbox'))
            notification['snapshot'] = snapshot_base64
        
        # Add journey if available
        if self.include_journey and journey:
            notification['journey'] = {
                'first_seen': journey.get('first_seen'),
                'duration': journey.get('duration'),
                'zones_visited': journey.get('zones_visited', []),
                'current_zone': journey.get('current_zone')
            }
        
        return notification
    
    def _encode_frame(self, frame: np.ndarray, bbox: Optional[List[int]] = None) -> str:
        """Encode frame as base64 string."""
        # Crop to bbox if provided
        if bbox:
            x1, y1, x2, y2 = bbox
            # Add padding
            padding = 20
            h, w = frame.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            cropped = frame[y1:y2, x1:x2]
        else:
            cropped = frame
        
        # Convert to JPEG
        _, buffer = cv2.imencode('.jpg', cropped, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        # Encode to base64
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return f"data:image/jpeg;base64,{img_base64}"
    
    def _worker(self):
        """Worker thread for sending notifications."""
        while self.running:
            try:
                # Get notification from queue (with timeout)
                notification = self.notification_queue.get(timeout=1)
                
                # Send notification
                self._send_notification(notification)
                
            except Empty:
                continue
            except Exception as e:
                print(f"Error in webhook worker: {e}")
    
    def _send_notification(self, notification: Dict[str, Any]):
        """Send notification to webhook."""
        try:
            # Get the target webhook URL from notification
            webhook_url = notification.pop('webhook_url', self.webhook_url)
            
            if not webhook_url:
                print("No webhook URL configured for this event type")
                return
            
            # Send POST request to n8n webhook
            response = requests.post(
                webhook_url,
                json=notification,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                self.notifications_sent += 1
                # Handle both person and door notifications
                if 'person' in notification:
                    print(f"Notification sent for person {notification['person']['id']}")
                elif 'door' in notification:
                    print(f"Notification sent for door {notification['door']['id']}")
                else:
                    print(f"Notification sent: {notification.get('event_type', 'unknown')}")
            else:
                self.notifications_failed += 1
                print(f"Webhook failed with status {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            self.notifications_failed += 1
            print(f"Failed to send webhook: {e}")
    
    def send_test_notification(self) -> bool:
        """Send a test notification to verify webhook is working."""
        test_notification = {
            'event_type': 'test',
            'timestamp': datetime.now().isoformat(),
            'message': 'This is a test notification from Airbnb Monitor',
            'person': {
                'id': 'TEST_001',
                'confidence': 0.99
            },
            'door': {
                'id': 'test_door',
                'name': 'Test Door',
                'type': 'test'
            },
            'camera': {
                'id': 'test_camera'
            }
        }
        
        try:
            response = requests.post(
                self.webhook_url,
                json=test_notification,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                print("Test notification sent successfully")
                return True
            else:
                print(f"Test notification failed with status {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Failed to send test notification: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get notification statistics."""
        return {
            'enabled': self.enabled,
            'webhook_url': self.webhook_url,
            'notifications_sent': self.notifications_sent,
            'notifications_failed': self.notifications_failed,
            'queue_size': self.notification_queue.qsize(),
            'cooldown_seconds': self.cooldown_seconds
        }


class AlertRules:
    """Manages alert rules and filtering."""
    
    def __init__(self):
        """Initialize alert rules."""
        self.rules = []
        
    def add_rule(self, rule: Dict[str, Any]):
        """Add an alert rule."""
        self.rules.append(rule)
    
    def should_alert(self, event: Dict[str, Any]) -> bool:
        """Check if an event should trigger an alert."""
        # If no rules, alert on everything
        if not self.rules:
            return True
        
        # Check each rule
        for rule in self.rules:
            if self._matches_rule(event, rule):
                return True
        
        return False
    
    def _matches_rule(self, event: Dict[str, Any], rule: Dict[str, Any]) -> bool:
        """Check if event matches a rule."""
        # Check action type
        if 'action' in rule:
            if event.get('action') != rule['action']:
                return False
        
        # Check zone
        if 'zone_id' in rule:
            if event.get('zone_id') != rule['zone_id']:
                return False
        
        # Check time range
        if 'time_range' in rule:
            current_time = datetime.now().time()
            start_time = datetime.strptime(rule['time_range']['start'], '%H:%M').time()
            end_time = datetime.strptime(rule['time_range']['end'], '%H:%M').time()
            
            if not (start_time <= current_time <= end_time):
                return False
        
        # Check confidence threshold
        if 'min_confidence' in rule:
            if event.get('confidence', 0) < rule['min_confidence']:
                return False
        
        return True