#!/usr/bin/env python3
"""
Diagnostic script to test Flask independently.
This helps isolate Flask issues from the main application.
"""

import sys
import time

print("=" * 60)
print("FLASK DIAGNOSTIC TEST")
print("=" * 60)

# Test 1: Check if Flask is installed
print("\n[TEST 1] Checking Flask installation...")
try:
    import flask
    print(f"✓ Flask version: {flask.__version__}")
except ImportError as e:
    print(f"✗ Flask not installed: {e}")
    print("Install with: pip3 install flask")
    sys.exit(1)

# Test 2: Check Flask-CORS
print("\n[TEST 2] Checking Flask-CORS...")
try:
    import flask_cors
    print(f"✓ Flask-CORS installed")
except ImportError as e:
    print(f"✗ Flask-CORS not installed: {e}")
    print("Install with: pip3 install flask-cors")
    sys.exit(1)

# Test 3: Check Flask-SocketIO (optional)
print("\n[TEST 3] Checking Flask-SocketIO (optional)...")
socketio_available = False
try:
    import flask_socketio
    # Don't check __version__ as it might not exist
    socketio_available = True
    print(f"✓ Flask-SocketIO installed")
except ImportError as e:
    print(f"⚠ Flask-SocketIO not installed: {e}")
    print("The web interface will work without WebSocket support")
    print("To add WebSocket support: pip3 install flask-socketio")
except AttributeError as e:
    print(f"⚠ Flask-SocketIO is broken: {e}")
    print("The web interface will work without WebSocket support")

# Test 4: Create minimal Flask app
print("\n[TEST 4] Creating minimal Flask app...")
try:
    from flask import Flask, jsonify
    app = Flask(__name__)
    
    @app.route('/')
    def home():
        return "Flask is working!"
    
    @app.route('/test')
    def test():
        return jsonify({'status': 'ok', 'message': 'Test endpoint working'})
    
    print("✓ Flask app created successfully")
except Exception as e:
    print(f"✗ Failed to create Flask app: {e}")
    sys.exit(1)

# Test 5: Try to start Flask server
print("\n[TEST 5] Starting Flask server on port 5001...")
print("(This will run for 10 seconds then stop)")
print("-" * 40)

from threading import Thread
import socket

def run_flask():
    """Run Flask in a thread."""
    try:
        # Use a different port to avoid conflicts
        app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
    except Exception as e:
        print(f"Flask error: {e}")

# Start Flask in a thread
thread = Thread(target=run_flask, daemon=True)
thread.start()

# Give Flask time to start
time.sleep(2)

# Test 6: Check if we can connect
print("\n[TEST 6] Testing connection to Flask...")
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    result = sock.connect_ex(('localhost', 5001))
    sock.close()
    
    if result == 0:
        print("✓ Successfully connected to Flask server!")
        print("✓ Flask is working correctly on this system")
    else:
        print("✗ Could not connect to Flask server")
        print("Flask thread might have crashed")
except Exception as e:
    print(f"✗ Connection test failed: {e}")

# Test 7: Check network interfaces
print("\n[TEST 7] Checking network interfaces...")
try:
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"Hostname: {hostname}")
    print(f"Local IP: {local_ip}")
    
    # Try to get actual network IP
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    network_ip = s.getsockname()[0]
    s.close()
    print(f"Network IP: {network_ip}")
except Exception as e:
    print(f"Network check error: {e}")

# Summary
print("\n" + "=" * 60)
print("DIAGNOSTIC SUMMARY")
print("=" * 60)

if thread.is_alive():
    print("✓ Flask server thread is still running")
    print("✓ All tests passed - Flask should work in the main app")
    print("\nIf the main app still fails, the issue might be:")
    print("  1. Port 5000 is blocked or in use")
    print("  2. Circular import issues in the main app")
    print("  3. Resource constraints on the Jetson Nano")
else:
    print("✗ Flask server thread died")
    print("There's an issue with Flask on this system")

print("\n[Waiting 5 seconds before exit...]")
time.sleep(5)