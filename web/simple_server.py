#!/usr/bin/env python3
"""
Simple HTTP server alternative to Flask.
Uses only Python standard library for maximum compatibility.
"""

import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.state_manager import UnifiedStateManager

class SimpleAPIHandler(BaseHTTPRequestHandler):
    """Simple HTTP request handler for the monitoring API."""
    
    def log_message(self, format, *args):
        """Override to suppress default logging."""
        pass
    
    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        # CORS headers
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        
        try:
            if path == '/api/stats':
                # Get current stats
                state = UnifiedStateManager()
                stats = {
                    'active_zones': state.get_active_zones_count(),
                    'active_persons': state.get_active_persons_count(),
                    'events': state.get_event_count(),
                    'timestamp': time.time()
                }
                self.wfile.write(json.dumps(stats).encode())
                
            elif path == '/api/doors':
                # Get door states
                state = UnifiedStateManager()
                doors = []
                for door_id, door_data in state.door_states.items():
                    doors.append({
                        'id': door_id,
                        'state': door_data.get('state', 'unknown'),
                        'confidence': door_data.get('confidence', 0),
                        'zone': door_data.get('zone', ''),
                        'last_update': door_data.get('last_update', 0)
                    })
                self.wfile.write(json.dumps(doors).encode())
                
            elif path == '/api/persons':
                # Get person detections
                state = UnifiedStateManager()
                persons = []
                for person_id, person_data in state.person_states.items():
                    persons.append({
                        'id': person_id,
                        'zone': person_data.get('zone', 'unknown'),
                        'confidence': person_data.get('confidence', 0),
                        'bbox': person_data.get('bbox', []),
                        'last_update': person_data.get('last_update', 0)
                    })
                self.wfile.write(json.dumps(persons).encode())
                
            elif path == '/api/events':
                # Get recent events
                state = UnifiedStateManager()
                events = state.get_recent_events(limit=50)
                self.wfile.write(json.dumps(events).encode())
                
            elif path == '/api/logs':
                # Get recent logs
                state = UnifiedStateManager()
                logs = state.get_recent_logs(limit=100)
                self.wfile.write(json.dumps(logs).encode())
                
            elif path == '/health':
                # Health check
                health = {
                    'status': 'ok',
                    'server': 'simple',
                    'timestamp': time.time()
                }
                self.wfile.write(json.dumps(health).encode())
                
            elif path == '/' or path.startswith('/static'):
                # Serve the dashboard HTML
                self.serve_dashboard()
                
            else:
                # 404 for unknown paths
                self.send_response(404)
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Not found'}).encode())
                
        except Exception as e:
            error_response = {'error': str(e)}
            self.wfile.write(json.dumps(error_response).encode())
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def serve_dashboard(self):
        """Serve the dashboard HTML."""
        html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hotel Security Monitor (Simple Server)</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        h1 {
            color: #667eea;
            font-size: 28px;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #666;
            font-size: 14px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        .stat-card:hover {
            transform: translateY(-5px);
        }
        .stat-value {
            font-size: 36px;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }
        .stat-label {
            color: #666;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .content-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .panel {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .panel h2 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 20px;
        }
        .door-item, .person-item, .event-item {
            padding: 10px;
            margin-bottom: 10px;
            background: #f7f9fc;
            border-radius: 8px;
            border-left: 3px solid #667eea;
        }
        .door-open { border-left-color: #ef4444; }
        .door-closed { border-left-color: #10b981; }
        .status {
            font-weight: bold;
            text-transform: uppercase;
            font-size: 12px;
        }
        .status.open { color: #ef4444; }
        .status.closed { color: #10b981; }
        .timestamp {
            color: #999;
            font-size: 12px;
        }
        .loading {
            text-align: center;
            padding: 20px;
            color: #999;
        }
        @media (max-width: 768px) {
            .content-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏨 Hotel Security Monitor</h1>
            <div class="subtitle">Simple Server Mode - Real-time Monitoring Dashboard</div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value" id="active-zones">0</div>
                <div class="stat-label">Active Zones</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="active-persons">0</div>
                <div class="stat-label">Active Persons</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="events">0</div>
                <div class="stat-label">Total Events</div>
            </div>
        </div>
        
        <div class="content-grid">
            <div class="panel">
                <h2>🚪 Door Status</h2>
                <div id="doors-list" class="loading">Loading...</div>
            </div>
            
            <div class="panel">
                <h2>👤 Person Detections</h2>
                <div id="persons-list" class="loading">Loading...</div>
            </div>
            
            <div class="panel">
                <h2>📋 Recent Events</h2>
                <div id="events-list" class="loading">Loading...</div>
            </div>
            
            <div class="panel">
                <h2>📝 System Logs</h2>
                <div id="logs-list" class="loading">Loading...</div>
            </div>
        </div>
    </div>
    
    <script>
        // Update functions
        async function updateStats() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();
                document.getElementById('active-zones').textContent = data.active_zones || 0;
                document.getElementById('active-persons').textContent = data.active_persons || 0;
                document.getElementById('events').textContent = data.events || 0;
            } catch (error) {
                console.error('Error fetching stats:', error);
            }
        }
        
        async function updateDoors() {
            try {
                const response = await fetch('/api/doors');
                const doors = await response.json();
                const container = document.getElementById('doors-list');
                
                if (doors.length === 0) {
                    container.innerHTML = '<div class="loading">No doors configured</div>';
                    return;
                }
                
                container.innerHTML = doors.map(door => `
                    <div class="door-item door-${door.state}">
                        <div><strong>${door.id}</strong> - Zone: ${door.zone || 'N/A'}</div>
                        <div class="status ${door.state}">${door.state}</div>
                        <div class="timestamp">Confidence: ${(door.confidence * 100).toFixed(1)}%</div>
                    </div>
                `).join('');
            } catch (error) {
                console.error('Error fetching doors:', error);
            }
        }
        
        async function updatePersons() {
            try {
                const response = await fetch('/api/persons');
                const persons = await response.json();
                const container = document.getElementById('persons-list');
                
                if (persons.length === 0) {
                    container.innerHTML = '<div class="loading">No persons detected</div>';
                    return;
                }
                
                container.innerHTML = persons.map(person => `
                    <div class="person-item">
                        <div><strong>Person ${person.id}</strong></div>
                        <div>Zone: ${person.zone}</div>
                        <div class="timestamp">Confidence: ${(person.confidence * 100).toFixed(1)}%</div>
                    </div>
                `).join('');
            } catch (error) {
                console.error('Error fetching persons:', error);
            }
        }
        
        async function updateEvents() {
            try {
                const response = await fetch('/api/events');
                const events = await response.json();
                const container = document.getElementById('events-list');
                
                if (events.length === 0) {
                    container.innerHTML = '<div class="loading">No recent events</div>';
                    return;
                }
                
                container.innerHTML = events.slice(0, 10).map(event => `
                    <div class="event-item">
                        <div><strong>${event.type}</strong></div>
                        <div>${event.message}</div>
                        <div class="timestamp">${new Date(event.timestamp * 1000).toLocaleTimeString()}</div>
                    </div>
                `).join('');
            } catch (error) {
                console.error('Error fetching events:', error);
            }
        }
        
        async function updateLogs() {
            try {
                const response = await fetch('/api/logs');
                const logs = await response.json();
                const container = document.getElementById('logs-list');
                
                if (logs.length === 0) {
                    container.innerHTML = '<div class="loading">No recent logs</div>';
                    return;
                }
                
                container.innerHTML = logs.slice(0, 10).map(log => `
                    <div class="event-item">
                        <div class="timestamp">${new Date(log.timestamp * 1000).toLocaleTimeString()}</div>
                        <div>${log.message}</div>
                    </div>
                `).join('');
            } catch (error) {
                console.error('Error fetching logs:', error);
            }
        }
        
        // Update all data
        function updateAll() {
            updateStats();
            updateDoors();
            updatePersons();
            updateEvents();
            updateLogs();
        }
        
        // Initial load and periodic updates
        updateAll();
        setInterval(updateAll, 2000);  // Update every 2 seconds
    </script>
</body>
</html>'''
        
        # Send HTML response
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())


def run_simple_server(host='0.0.0.0', port=5000):
    """Run the simple HTTP server."""
    print(f"[SIMPLE-SERVER] Starting simple HTTP server on {host}:{port}")
    
    try:
        server = HTTPServer((host, port), SimpleAPIHandler)
        print(f"[SIMPLE-SERVER] Server running at http://{host}:{port}")
        print("[SIMPLE-SERVER] Dashboard available at http://localhost:5000")
        print("[SIMPLE-SERVER] Press Ctrl+C to stop")
        
        # Run server in thread
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        
        return server, server_thread
        
    except Exception as e:
        print(f"[SIMPLE-SERVER] Failed to start: {e}")
        return None, None


if __name__ == "__main__":
    # Test the simple server on port 5001 to avoid conflicts
    import sys
    port = 5001 if len(sys.argv) < 2 else int(sys.argv[1])
    
    server, thread = run_simple_server(port=port)
    if server:
        try:
            # Keep running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[SIMPLE-SERVER] Shutting down...")
            server.shutdown()