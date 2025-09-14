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
current_frame = None
frame_lock = threading.Lock()


def init_app(config):
    """Initialize the web app with system components."""
    global zone_detector, journey_manager
    zone_detector = MotionZoneDetector(config)
    journey_manager = JourneyManager(config)
    zone_detector.initialize()


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')


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
    events = journey_manager.get_recent_events(minutes) if journey_manager else []
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
    stats = {
        'zones': len(zone_detector.get_zones()) if zone_detector else 0,
        'active_persons': len(journey_manager.persons) if journey_manager else 0,
        'timestamp': datetime.now().isoformat()
    }
    return jsonify(stats)


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
        while True:
            with frame_lock:
                if current_frame is not None:
                    _, buffer = cv2.imencode('.jpg', current_frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
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
    socketio.emit('new_event', event, broadcast=True)


if __name__ == '__main__':
    # For testing only
    import yaml
    with open('../config/settings.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    init_app(config)
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)