"""Web interface for Airbnb monitoring system."""

from flask import Flask, render_template, jsonify, request, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import json
import base64
from datetime import datetime, timedelta
import threading
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.motion_detector import MotionZoneDetector
from tracking.journey_manager import JourneyManager


app = Flask(__name__)
app.config['SECRET_KEY'] = 'change-this-in-production'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global references to system components
zone_detector = None
journey_manager = None
monitor_instance = None
current_frame = None
frame_lock = threading.Lock()


def init_app(config, monitor=None):
    """Initialize the web app with system components."""
    global zone_detector, journey_manager, monitor_instance
    
    # Store monitor instance
    monitor_instance = monitor
    if monitor:
        app.config['monitor'] = monitor
        zone_detector = monitor.zone_detector if hasattr(monitor, 'zone_detector') else None
        journey_manager = monitor.journey_manager if hasattr(monitor, 'journey_manager') else None
    else:
        zone_detector = app.config.get('zone_detector') or MotionZoneDetector(config)
        journey_manager = app.config.get('journey_manager') or JourneyManager(config)
        if hasattr(zone_detector, 'initialize'):
            zone_detector.initialize()


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')


@app.route('/test')
def test():
    """Simple test route to verify server is running."""
    return jsonify({
        'status': 'running',
        'message': 'Web server is working!',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/zones', methods=['GET'])
def get_zones():
    """Get all zones."""
    zones = zone_detector.get_zones() if zone_detector else []
    return jsonify([zone.to_dict() for zone in zones])


@app.route('/api/zones', methods=['POST'])
def create_zone():
    """Create a new zone."""
    data = request.json
    zone = zone_detector.add_manual_zone(
        data['coordinates'],
        data['name'],
        data.get('type', 'room_door')
    )
    return jsonify(zone.to_dict())


@app.route('/api/zones/<zone_id>', methods=['PUT'])
def update_zone(zone_id):
    """Update a zone."""
    data = request.json
    success = zone_detector.update_zone(
        zone_id,
        data.get('name'),
        data.get('type')
    )
    return jsonify({'success': success})


@app.route('/api/zones/<zone_id>', methods=['DELETE'])
def delete_zone(zone_id):
    """Delete a zone."""
    success = zone_detector.remove_zone(zone_id)
    return jsonify({'success': success})


@app.route('/api/events')
def get_events():
    """Get recent events."""
    minutes = request.args.get('minutes', 60, type=int)
    events = []
    
    # Try monitor instance first
    if monitor_instance and hasattr(monitor_instance, 'journey_manager'):
        try:
            events = monitor_instance.journey_manager.get_recent_events(minutes)
        except:
            events = []
    # Fallback to global reference
    elif journey_manager:
        try:
            events = journey_manager.get_recent_events(minutes)
        except:
            events = []
    
    return jsonify(events)


@app.route('/api/persons/<person_id>/journey')
def get_person_journey(person_id):
    """Get journey for a specific person."""
    journey = journey_manager.get_person_journey(person_id) if journey_manager else None
    if journey:
        return jsonify(journey)
    return jsonify({'error': 'Person not found'}), 404


@app.route('/api/occupancy')
def get_occupancy():
    """Get current occupancy for all zones."""
    occupancy = {}
    if zone_detector and journey_manager:
        for zone in zone_detector.get_zones():
            occupants = journey_manager.get_zone_occupancy(zone.id)
            occupancy[zone.name] = {
                'zone_id': zone.id,
                'count': len(occupants),
                'person_ids': occupants
            }
    return jsonify(occupancy)


@app.route('/api/stats')
def get_stats():
    """Get system statistics."""
    # Get actual active persons count
    active_persons = 0
    active_zones = 0
    
    # Try monitor instance first
    if monitor_instance:
        if hasattr(monitor_instance, 'journey_manager'):
            active_persons = len(monitor_instance.journey_manager.persons)
        if hasattr(monitor_instance, 'zone_detector'):
            active_zones = len(monitor_instance.zone_detector.get_zones())
    # Fallback to global references
    elif journey_manager:
        active_persons = len(journey_manager.persons)
        active_zones = len(zone_detector.get_zones()) if zone_detector else 0
    
    stats = {
        'zones': active_zones,
        'active_persons': active_persons,
        'timestamp': datetime.now().isoformat()
    }
    return jsonify(stats)


@app.route('/api/doors')
def get_doors():
    """Get current door states."""
    door_states = []
    
    # Try monitor instance
    if monitor_instance and hasattr(monitor_instance, 'door_detector'):
        doors = monitor_instance.door_detector.get_doors() if hasattr(monitor_instance.door_detector, 'get_doors') else monitor_instance.door_detector.doors
        for door in doors:
            if hasattr(door, 'to_dict'):
                door_states.append(door.to_dict())
            else:
                door_states.append({
                    'door_id': door.id,
                    'state': door.current_state,
                    'confidence': getattr(door, 'confidence', 0),
                    'last_change': door.last_change.isoformat() if hasattr(door, 'last_change') else datetime.now().isoformat(),
                    'bbox': getattr(door, 'bbox', [])
                })
    
    return jsonify(door_states)


@app.route('/api/door-zones')
def get_door_zones():
    """Get configured door zones."""
    zones = []
    
    if monitor_instance and hasattr(monitor_instance, 'door_detector'):
        if hasattr(monitor_instance.door_detector, 'get_zones'):
            zones = monitor_instance.door_detector.get_zones()
    
    return jsonify(zones)


@app.route('/api/door-zones/calibrate', methods=['POST'])
def calibrate_door_zones():
    """Calibrate door zones using VLM."""
    if not monitor_instance or not hasattr(monitor_instance, 'door_detector'):
        return jsonify({'success': False, 'error': 'Door detector not available'}), 500
    
    # Get current frame
    frame = None
    with frame_lock:
        if current_frame is not None:
            frame = current_frame.copy()
    
    if frame is None:
        return jsonify({'success': False, 'error': 'No camera frame available'}), 400
    
    # Calibrate zones
    if hasattr(monitor_instance.door_detector, 'calibrate_zones'):
        result = monitor_instance.door_detector.calibrate_zones(frame)
        
        # Broadcast update to all clients
        socketio.emit('zones_updated', result)
        
        return jsonify(result)
    else:
        return jsonify({'success': False, 'error': 'Zone calibration not supported'}), 400


@app.route('/api/door-zones/<zone_id>/name', methods=['PUT'])
def update_zone_name(zone_id):
    """Update the name of a door zone."""
    if not monitor_instance or not hasattr(monitor_instance, 'door_detector'):
        return jsonify({'success': False, 'error': 'Door detector not available'}), 500
    
    data = request.get_json()
    new_name = data.get('name')
    
    if not new_name:
        return jsonify({'success': False, 'error': 'Name is required'}), 400
    
    if hasattr(monitor_instance.door_detector, 'zone_mapper') and monitor_instance.door_detector.zone_mapper:
        success = monitor_instance.door_detector.zone_mapper.update_zone_name(zone_id, new_name)
        if success:
            # Broadcast update
            socketio.emit('zone_name_updated', {'zone_id': zone_id, 'name': new_name})
            return jsonify({'success': True})
    
    return jsonify({'success': False, 'error': 'Failed to update zone name'}), 400


@app.route('/api/door-zones/<zone_id>', methods=['DELETE'])
def delete_door_zone(zone_id):
    """Delete a door zone."""
    if not monitor_instance or not hasattr(monitor_instance, 'door_detector'):
        return jsonify({'success': False, 'error': 'Door detector not available'}), 500
    
    if hasattr(monitor_instance.door_detector, 'zone_mapper') and monitor_instance.door_detector.zone_mapper:
        success = monitor_instance.door_detector.zone_mapper.delete_zone(zone_id)
        if success:
            # Broadcast update
            socketio.emit('zone_deleted', {'zone_id': zone_id})
            return jsonify({'success': True})
    
    return jsonify({'success': False, 'error': 'Failed to delete zone'}), 400


@app.route('/api/frame')
def get_frame():
    """Get current camera frame."""
    global current_frame
    with frame_lock:
        if current_frame is not None:
            _, buffer = cv2.imencode('.jpg', current_frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            return jsonify({'frame': f"data:image/jpeg;base64,{frame_base64}"})
    return jsonify({'error': 'No frame available'}), 404


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    def generate():
        global current_frame
        monitor = app.config.get('monitor')
        
        while True:
            frame = None
            
            # Try to get frame from monitor if available
            if monitor and hasattr(monitor, 'get_current_frame'):
                frame = monitor.get_current_frame()
            elif monitor and hasattr(monitor, 'current_frame'):
                with monitor.frame_lock:
                    if monitor.current_frame is not None:
                        frame = monitor.current_frame.copy()
            else:
                # Fallback to global current_frame
                with frame_lock:
                    if current_frame is not None:
                        frame = current_frame.copy()
            
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                # If no frame available, wait a bit
                import time
                time.sleep(0.1)
    
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print('Client connected')
    emit('connected', {'message': 'Connected to Airbnb Monitor'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print('Client disconnected')


@socketio.on('request_update')
def handle_update_request():
    """Handle request for updated data."""
    # Send current zones
    zones = zone_detector.get_zones() if zone_detector else []
    emit('zones_update', [zone.to_dict() for zone in zones])
    
    # Send recent events
    events = journey_manager.get_recent_events(10) if journey_manager else []
    emit('events_update', events)


def update_frame(frame):
    """Update the current frame for web display."""
    global current_frame
    with frame_lock:
        current_frame = frame.copy()


def broadcast_event(event):
    """Broadcast an event to all connected clients."""
    # Ensure event has event_type for proper frontend handling
    if 'event_type' not in event:
        if any(key in event for key in ['door_id', 'door_opened', 'door_closed', 'door_discovered']):
            event['event_type'] = 'door'
        elif 'person_id' in event:
            event['event_type'] = 'person'
    
    socketio.emit('new_event', event, broadcast=True)


if __name__ == '__main__':
    # For testing only
    import yaml
    with open('../config/settings.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    init_app(config)
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)