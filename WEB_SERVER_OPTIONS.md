# Web Server Options for Airbnb Monitor

The monitoring system provides multiple web server options to handle different environments and compatibility issues.

## Available Options

### 1. **Default Flask Server** (Recommended)
```bash
python3 run_optimized.py
```
- Full-featured web interface with WebSocket support (if Flask-SocketIO is installed)
- Real-time updates and live video streaming
- Requires: `flask`, `flask-cors`, and optionally `flask-socketio`

### 2. **Simple HTTP Server** (Fallback Option)
```bash
python3 run_optimized.py --simple-server
```
- Basic web interface using only Python standard library
- No external dependencies required
- Provides dashboard with polling-based updates (refreshes every 2 seconds)
- Good option when Flask has compatibility issues

### 3. **No Web Interface** (Monitoring Only)
```bash
python3 run_optimized.py --no-web
```
- Runs the monitoring system without any web interface
- Console output only
- Lowest resource usage
- Good for headless deployments or debugging

## Troubleshooting

### Flask Import Errors
If you see errors like:
- `ImportError: cannot import name 'BaseRequest' from 'werkzeug.wrappers'`
- `AttributeError: module 'flask_socketio' has no attribute '__version__'`

**Solutions:**
1. The system will automatically try to fix Werkzeug compatibility issues
2. Use `--simple-server` flag for a working web interface without Flask
3. Use `--no-web` flag to run without any web interface

### Port Already in Use
If port 5000 is already in use:
1. Check what's using it: `lsof -i :5000`
2. Kill the process: `kill -9 <PID>`
3. Or change the port in `config/settings_optimized.yaml`:
```yaml
web:
  port: 5001  # Change to any available port
```

### Installation Commands

**For full Flask support:**
```bash
pip3 install flask flask-cors flask-socketio
```

**For minimal setup (simple server only):**
No additional packages needed - uses Python standard library

## Web Interface Access

Once running, access the dashboard at:
- Local machine: `http://localhost:5000`
- From network: `http://<device-ip>:5000`

Replace `<device-ip>` with your device's actual IP address (shown in console output when server starts).

## Performance Considerations

- **Flask with SocketIO**: Best real-time performance, higher resource usage
- **Flask without SocketIO**: Good performance, moderate resource usage
- **Simple Server**: Basic performance, lowest resource usage
- **No Web**: Best monitoring performance, no web overhead

Choose based on your specific needs and hardware capabilities.