# Quick Start Guide - Airbnb Monitoring System

## 🚀 Start Everything with One Command

```bash
./start_monitoring.sh
```

This will:
1. Start VLM server for intelligent door detection
2. Start Roboflow inference server (if enabled)
3. Launch the main monitoring system
4. Open web interface on port 5000

## 📱 Access the Web Dashboard

Open in your browser:
- `http://<jetson-ip>:5000`
- `http://localhost:5000` (on Jetson)

## 🎯 First-Time Setup

When you first access the dashboard:

1. **Click "Calibrate Doors"**
   - VLM will analyze your camera view
   - Automatically detect all doors
   - Name them intelligently (Main Entrance, Bathroom, etc.)
   - Save zones for future use

2. **Verify Detection**
   - Check that all doors are highlighted
   - Edit names if needed
   - Test by opening/closing doors

## 📊 What You'll See

### Live Dashboard Shows:
- **Camera Feed** - Real-time video with overlays
- **Door Zones** - Color-coded boxes around each door
- **Door States** - Green (open) / Red (closed)
- **Person Tracking** - Numbered IDs for each person
- **Room Occupancy** - Count of people in each zone
- **Event Log** - Timeline of door events and movements
- **Journey Paths** - Visual tracking of person movements

### Status Indicators:
- 🟢 Door Open
- 🔴 Door Closed
- 🟡 Motion Detected
- 👤 Person Detected
- 📹 Recording Active

## 🛠️ Service Management

### Using Service Manager:
```bash
# Start all services
python3 service_manager.py start

# Check status
python3 service_manager.py status

# Stop all services
python3 service_manager.py stop

# Restart everything
python3 service_manager.py restart
```

### Manual Control:
```bash
# Stop everything
./stop_monitoring.sh

# Start individual services
python3 vlm_pipeline.py           # VLM server only
python3 run_optimized.py          # Monitor only
```

## 📁 System Components

### Services Running:
1. **VLM Server** (Port 8080)
   - Analyzes images to detect doors
   - Names zones intelligently
   - Provides calibration

2. **Inference Server** (Port 9001) *Optional*
   - Roboflow model for 96.7% accuracy
   - Real-time door state detection

3. **Main Monitor** (Port 5000)
   - Camera capture
   - Person tracking
   - Door monitoring
   - Web interface
   - Video recording

## 🔧 Configuration

Edit `config/settings.yaml` to customize:

```yaml
# Enable/disable features
vlm:
  enabled: true
  auto_calibrate_on_start: true

door_detection:
  use_inference: true  # Use Roboflow model

storage:
  video_retention_hours: 48
```

## 📊 Monitoring Features

### Automatic Actions:
- **Door Zone Calibration** - On first run
- **Person Tracking** - ByteTrack algorithm
- **Journey Recording** - Track movement patterns
- **Event Logging** - All door/person events
- **Video Recording** - 48-hour circular buffer
- **Real-time Updates** - WebSocket streaming

### Data Storage:
- **Video**: `storage/videos/` (48-hour retention)
- **Snapshots**: `storage/snapshots/`
- **Door Zones**: `config/door_zones.json`
- **Events**: `storage/database.db`
- **Logs**: `logs/`

## 🐛 Troubleshooting

### VLM Not Working:
```bash
# Test VLM directly
python3 vlm_pipeline.py

# Check CUDA
python3 check_cuda.py

# Use CPU mode
python3 vlm_server_cpu.py
```

### Web Interface Not Loading:
```bash
# Check if port 5000 is in use
netstat -tln | grep 5000

# Restart monitor
pkill -f run_optimized.py
python3 run_optimized.py
```

### No Door Detection:
```bash
# Disable VLM and use edge detection
sed -i 's/enabled: true/enabled: false/' config/settings.yaml
./start_monitoring.sh
```

## 📈 Performance Tips

1. **Reduce Camera Resolution** for better FPS
2. **Disable Inference Server** if not needed
3. **Use Edge Detection** as fallback
4. **Adjust Detection Thresholds** in settings

## 🔄 Auto-Start on Boot

Add to `/etc/rc.local` or create systemd service:
```bash
cd /home/amadeo/Desktop/airbnb-monitor
./start_monitoring.sh &
```

## 📱 Mobile Access

The web interface is mobile-responsive. Access from any device on your network:
`http://<jetson-ip>:5000`

---

## Need Help?

- Logs: Check `logs/` directory
- Config: Edit `config/settings.yaml`
- Stop: Run `./stop_monitoring.sh`
- Restart: Run `./start_monitoring.sh`