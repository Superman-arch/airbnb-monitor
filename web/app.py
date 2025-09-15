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
    """Get enhanced system statistics."""
    # Get actual active persons count
    active_persons = 0
    active_zones = 0
    active_doors = 0
    doors_open = 0
    doors_closed = 0
    
    # Try monitor instance first
    if monitor_instance:
        if hasattr(monitor_instance, 'journey_manager'):
            active_persons = len(monitor_instance.journey_manager.persons)
        if hasattr(monitor_instance, 'zone_detector'):
            active_zones = len(monitor_instance.zone_detector.get_zones())
        if hasattr(monitor_instance, 'door_detector'):
            if hasattr(monitor_instance.door_detector, 'doors'):
                doors_dict = monitor_instance.door_detector.doors
                if isinstance(doors_dict, dict):
                    active_doors = len(doors_dict)
                    for door in doors_dict.values():
                        if door.current_state == 'open':
                            doors_open += 1
                        elif door.current_state == 'closed':
                            doors_closed += 1
    # Fallback to global references
    elif journey_manager:
        active_persons = len(journey_manager.persons)
        active_zones = len(zone_detector.get_zones()) if zone_detector else 0
    
    stats = {
        'zones': active_zones,
        'active_persons': active_persons,
        'active_doors': active_doors,
        'doors_open': doors_open,
        'doors_closed': doors_closed,
        'timestamp': datetime.now().isoformat()
    }
    return jsonify(stats)


@app.route('/api/doors')
def get_doors():
    """Get current door states with enhanced information."""
    door_states = []
    
    # Try monitor instance
    if monitor_instance and hasattr(monitor_instance, 'door_detector'):
        # First try to get configured zones to show even without active detections
        zones = []
        if hasattr(monitor_instance.door_detector, 'get_zones'):
            zones = monitor_instance.door_detector.get_zones()
        
        if hasattr(monitor_instance.door_detector, 'get_doors'):
            doors = monitor_instance.door_detector.get_doors()
        else:
            # Handle dictionary of doors
            doors = monitor_instance.door_detector.doors.values() if isinstance(monitor_instance.door_detector.doors, dict) else monitor_instance.door_detector.doors
        
        # If we have zones but no active doors, show zones as configured doors
        if zones and len(list(doors)) == 0:
            for zone in zones:
                zone_dict = zone if isinstance(zone, dict) else zone.to_dict() if hasattr(zone, 'to_dict') else {'id': str(zone)}
                door_state = {
                    'door_id': zone_dict.get('id', 'unknown'),
                    'door_name': zone_dict.get('name', 'Unknown Door'),
                    'state': 'configured',
                    'confidence': 0,
                    'bbox': zone_dict.get('bbox', []),
                    'zone_id': zone_dict.get('id'),
                    'spatial_hash': None,
                    'metadata': {'configured': True}
                }
                door_states.append(door_state)
        
        for door in doors:
            if hasattr(door, 'to_dict'):
                door_state = door.to_dict()
            else:
                door_state = {
                    'door_id': door.id,
                    'state': door.current_state,
                    'confidence': getattr(door, 'confidence', 0),
                    'last_change': door.last_change.isoformat() if hasattr(door, 'last_change') else datetime.now().isoformat(),
                    'bbox': getattr(door, 'bbox', [])
                }
            
            # Add additional useful information
            door_state['door_name'] = getattr(door, 'zone_name', None) or door_state.get('door_id', 'Unknown')
            door_state['spatial_hash'] = getattr(door, 'spatial_hash', None)
            door_state['zone_id'] = getattr(door, 'zone_id', None)
            door_state['metadata'] = getattr(door, 'metadata', {})
            
            door_states.append(door_state)
    
    return jsonify(door_states)


@app.route('/api/door-zones')
def get_door_zones():
    """Get configured door zones with full details."""
    zones = []
    
    if monitor_instance and hasattr(monitor_instance, 'door_detector'):
        if hasattr(monitor_instance.door_detector, 'get_zones'):
            zones = monitor_instance.door_detector.get_zones()
            
            # Add door state information to zones
            if hasattr(monitor_instance.door_detector, 'doors'):
                doors_dict = monitor_instance.door_detector.doors
                for zone in zones:
                    # Find matching door for this zone
                    matching_door = None
                    if isinstance(doors_dict, dict):
                        for door in doors_dict.values():
                            if getattr(door, 'zone_id', None) == zone.get('id'):
                                matching_door = door
                                break
                    
                    if matching_door:
                        zone['door_state'] = matching_door.current_state
                        zone['door_confidence'] = getattr(matching_door, 'confidence', 0)
                        zone['door_id'] = matching_door.id
    
    return jsonify(zones)


@app.route('/api/door-zones/calibrate', methods=['POST'])
def calibrate_door_zones():
    """Calibrate door zones using VLM with enhanced feedback."""
    if not monitor_instance or not hasattr(monitor_instance, 'door_detector'):
        return jsonify({'success': False, 'error': 'Door detector not available'}), 500
    
    # Get current frame from monitor
    frame = None
    
    # Try to get frame from monitor first
    if monitor_instance:
        if hasattr(monitor_instance, 'current_frame'):
            with monitor_instance.frame_lock:
                if monitor_instance.current_frame is not None:
                    frame = monitor_instance.current_frame.copy()
    
    # Fallback to global frame
    if frame is None:
        with frame_lock:
            if current_frame is not None:
                frame = current_frame.copy()
    
    if frame is None:
        return jsonify({'success': False, 'error': 'No camera frame available. Please wait for camera to initialize.'}), 400
    
    # Calibrate zones
    if hasattr(monitor_instance.door_detector, 'calibrate_zones'):
        result = monitor_instance.door_detector.calibrate_zones(frame)
        
        # Save door states after calibration
        if result.get('success') and hasattr(monitor_instance.door_detector, 'save_door_states'):
            monitor_instance.door_detector.save_door_states()
        
        # Broadcast update to all clients
        socketio.emit('zones_updated', result)
        
        # Also broadcast door update
        if result.get('success'):
            doors = monitor_instance.door_detector.get_doors() if hasattr(monitor_instance.door_detector, 'get_doors') else []
            door_data = [door.to_dict() if hasattr(door, 'to_dict') else {'id': door.id} for door in doors]
            socketio.emit('doors_updated', {'doors': door_data})
        
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


@app.route('/api/doors/<door_id>', methods=['GET'])
def get_door_details(door_id):
    """Get detailed information about a specific door."""
    if not monitor_instance or not hasattr(monitor_instance, 'door_detector'):
        return jsonify({'error': 'Door detector not available'}), 500
    
    if hasattr(monitor_instance.door_detector, 'doors'):
        doors_dict = monitor_instance.door_detector.doors
        
        # Find door by ID or spatial hash
        door = None
        if isinstance(doors_dict, dict):
            door = doors_dict.get(door_id)
            if not door:
                # Try to find by matching door.id
                for d in doors_dict.values():
                    if d.id == door_id or getattr(d, 'spatial_hash', None) == door_id:
                        door = d
                        break
        
        if door:
            door_info = door.to_dict() if hasattr(door, 'to_dict') else {
                'id': door.id,
                'state': door.current_state,
                'confidence': getattr(door, 'confidence', 0)
            }
            
            # Add history if available
            if hasattr(door, 'change_history'):
                door_info['history'] = list(door.change_history)[-10:]  # Last 10 changes
            
            return jsonify(door_info)
    
    return jsonify({'error': 'Door not found'}), 404


@app.route('/api/doors/<door_id>/history')
def get_door_history(door_id):
    """Get history for a specific door."""
    if not monitor_instance or not hasattr(monitor_instance, 'door_detector'):
        return jsonify([])
    
    if hasattr(monitor_instance.door_detector, 'doors'):
        doors_dict = monitor_instance.door_detector.doors
        
        # Find door
        door = None
        if isinstance(doors_dict, dict):
            door = doors_dict.get(door_id)
            if not door:
                for d in doors_dict.values():
                    if d.id == door_id or getattr(d, 'spatial_hash', None) == door_id:
                        door = d
                        break
        
        if door and hasattr(door, 'change_history'):
            # Return last 20 changes
            history = list(door.change_history)[-20:]
            return jsonify(history)
    
    return jsonify([])


@app.route('/api/doors/<door_id>', methods=['DELETE'])
def delete_door(door_id):
    """Delete a door."""
    if not monitor_instance or not hasattr(monitor_instance, 'door_detector'):
        return jsonify({'success': False, 'error': 'Door detector not available'}), 500
    
    if hasattr(monitor_instance.door_detector, 'doors'):
        doors_dict = monitor_instance.door_detector.doors
        
        # Find and delete door
        if isinstance(doors_dict, dict):
            if door_id in doors_dict:
                del doors_dict[door_id]
                # Save state
                if hasattr(monitor_instance.door_detector, 'save_door_states'):
                    monitor_instance.door_detector.save_door_states()
                
                # Broadcast update
                socketio.emit('door_deleted', {'door_id': door_id})
                
                return jsonify({'success': True})
            else:
                # Try to find by door.id
                for key, d in list(doors_dict.items()):
                    if d.id == door_id or getattr(d, 'spatial_hash', None) == door_id:
                        del doors_dict[key]
                        # Save state
                        if hasattr(monitor_instance.door_detector, 'save_door_states'):
                            monitor_instance.door_detector.save_door_states()
                        
                        # Broadcast update
                        socketio.emit('door_deleted', {'door_id': door_id})
                        
                        return jsonify({'success': True})
    
    return jsonify({'success': False, 'error': 'Door not found'}), 404


@app.route('/api/doors/<door_id>/metadata', methods=['PUT'])
def update_door_metadata(door_id):
    """Update door metadata (name, type, etc)."""
    if not monitor_instance or not hasattr(monitor_instance, 'door_detector'):
        return jsonify({'success': False, 'error': 'Door detector not available'}), 500
    
    data = request.get_json()
    
    if hasattr(monitor_instance.door_detector, 'doors'):
        doors_dict = monitor_instance.door_detector.doors
        
        # Find door
        door = None
        if isinstance(doors_dict, dict):
            door = doors_dict.get(door_id)
            if not door:
                for d in doors_dict.values():
                    if d.id == door_id:
                        door = d
                        break
        
        if door:
            # Update metadata
            if hasattr(door, 'metadata'):
                door.metadata.update(data)
            
            # Update specific fields
            if 'name' in data:
                door.zone_name = data['name']
                if hasattr(door, 'metadata'):
                    door.metadata['name'] = data['name']
            
            if 'type' in data and hasattr(door, 'metadata'):
                door.metadata['type'] = data['type']
            
            # Save state
            if hasattr(monitor_instance.door_detector, 'save_door_states'):
                monitor_instance.door_detector.save_door_states()
            
            # Broadcast update
            socketio.emit('door_metadata_updated', {
                'door_id': door_id,
                'metadata': data
            })
            
            return jsonify({'success': True})
    
    return jsonify({'success': False, 'error': 'Door not found'}), 404


@app.route('/api/doors/reset', methods=['POST'])
def reset_doors():
    """Reset all door detections and start fresh."""
    if not monitor_instance or not hasattr(monitor_instance, 'door_detector'):
        return jsonify({'success': False, 'error': 'Door detector not available'}), 500
    
    if hasattr(monitor_instance.door_detector, 'reset'):
        monitor_instance.door_detector.reset()
        
        # Start learning if edge detection
        if hasattr(monitor_instance.door_detector, 'start_learning'):
            monitor_instance.door_detector.start_learning()
        
        # Broadcast reset
        socketio.emit('doors_reset', {})
        
        return jsonify({'success': True, 'message': 'Door detection reset successfully'})
    
    return jsonify({'success': False, 'error': 'Reset not supported'}), 400


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
            
            # Try to get display frame (with overlays) first
            if monitor:
                if hasattr(monitor, 'display_frame'):
                    with monitor.frame_lock:
                        if monitor.display_frame is not None:
                            frame = monitor.display_frame.copy()
                # Fallback to raw frame if display frame not available
                if frame is None and hasattr(monitor, 'current_frame'):
                    with monitor.frame_lock:
                        if monitor.current_frame is not None:
                            frame = monitor.current_frame.copy()
            else:
                # Fallback to global current_frame
                with frame_lock:
                    if current_frame is not None:
                        frame = current_frame.copy()
            
            if frame is not None:
                # Reduce quality slightly for better streaming performance
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
                _, buffer = cv2.imencode('.jpg', frame, encode_params)
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
    """Broadcast an event to all connected clients with enhanced data."""
    # Ensure event has event_type for proper frontend handling
    if 'event_type' not in event:
        if any(key in event for key in ['door_id', 'door_opened', 'door_closed', 'door_discovered']):
            event['event_type'] = 'door'
        elif 'person_id' in event:
            event['event_type'] = 'person'
    
    # Add additional context for door events
    if event.get('event_type') == 'door' and monitor_instance:
        if hasattr(monitor_instance, 'door_detector'):
            # Try to get door details
            door_id = event.get('door_id')
            if door_id and hasattr(monitor_instance.door_detector, 'doors'):
                doors_dict = monitor_instance.door_detector.doors
                # Check if door exists in dictionary
                door = None
                if isinstance(doors_dict, dict):
                    # Could be keyed by spatial hash or door_id
                    door = doors_dict.get(door_id)
                    if not door:
                        # Try to find by matching door.id
                        for d in doors_dict.values():
                            if d.id == door_id:
                                door = d
                                break
                
                if door:
                    event['spatial_hash'] = getattr(door, 'spatial_hash', None)
                    event['zone_name'] = getattr(door, 'zone_name', None)
                    event['zone_id'] = getattr(door, 'zone_id', None)
    
    socketio.emit('new_event', event, to='/')


def broadcast_log(message, level='info'):
    """Broadcast a log message to all connected clients."""
    from datetime import datetime
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'level': level,
        'message': message
    }
    socketio.emit('log_message', log_entry, to='/')


def broadcast_stats(stats):
    """Broadcast system statistics to all connected clients."""
    socketio.emit('stats_update', stats, to='/')


if __name__ == '__main__':
    # For testing only
    import yaml
    with open('../config/settings.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    init_app(config)
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)