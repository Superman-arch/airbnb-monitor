# Airbnb Monitoring System

A comprehensive monitoring system for Airbnb/apartment complexes that automatically detects doors, tracks people movement, and sends real-time notifications.

## Features

### Core Functionality
- **Automatic Door Detection**: Uses motion-based zone detection to identify high-traffic areas (doors)
- **Person Tracking**: Unique ID generation for each person without facial recognition
- **Journey Tracking**: Tracks person movement across cameras and zones
- **Real-time Notifications**: N8N webhook integration for instant alerts
- **48-Hour Video Storage**: Circular buffer system with automatic cleanup
- **Web Interface**: Configure zones, view live feed, manage settings

### Technical Features
- **YOLOv8** for person detection (pre-trained, no custom training needed)
- **ByteTrack** for real-time multi-person tracking
- **Motion-based zone detection** (ready for ML upgrade later)
- **Modular architecture** for easy extension
- **Jetson Nano optimized** with USB/CSI camera support

## Installation

### Prerequisites
- Jetson Nano (or Ubuntu 18.04+ system)
- USB webcam or CSI camera
- Python 3.8+
- 32GB+ storage for video retention

### Quick Setup

1. Clone and enter the directory:
```bash
cd airbnb-monitor
```

2. Run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

3. Configure your settings:
```bash
nano config/settings.yaml
```

Key settings to configure:
- `notifications.webhook_url`: Your n8n webhook URL
- `camera.source`: 0 for USB camera, "csi" for CSI camera
- `storage.video_retention_hours`: Hours of video to keep (default 48)

## Usage

### Start the System

Basic usage with display:
```bash
python3 main.py
```

Headless operation (no display):
```bash
python3 main.py --no-display
```

Test webhook configuration:
```bash
python3 main.py --test-webhook
```

### Keyboard Controls
- `q` - Quit application
- `s` - Save current zones
- `t` - Send test webhook
- `h` - Show help
- `m` - Display motion heatmap

### Web Interface

Access the web interface at `http://<jetson-ip>:5000`

Features:
- Draw and name door zones
- View live camera feed
- Monitor recent events
- Track current occupancy
- Configure zone types (main entrance, room door, common area)

## System Architecture

### Detection Flow
1. **Motion Detection** → Identifies high-traffic areas
2. **Zone Creation** → Auto or manual door zone definition
3. **Person Detection** → YOLOv8 detects people in frame
4. **Tracking** → ByteTrack assigns unique IDs
5. **Journey Management** → Tracks movement across zones
6. **Event Generation** → Entry/exit events created
7. **Notifications** → Webhook sent to n8n

### Data Flow
```
Camera → Frame Processing → Detection & Tracking → Event Generation
                ↓                                         ↓
          Video Storage                            Webhook/Database
```

## N8N Webhook Integration

The system sends webhooks with the following payload:

```json
{
  "event_type": "entry",
  "timestamp": "2024-01-14T10:30:00Z",
  "person": {
    "id": "P_20240114_0042",
    "confidence": 0.89
  },
  "door": {
    "id": "zone_001",
    "name": "Room 3B",
    "type": "room_door"
  },
  "journey": {
    "first_seen": "2024-01-14T09:15:00Z",
    "zones_visited": ["main_entrance", "hallway", "room_3b"],
    "current_zone": "room_3b"
  },
  "snapshot": "base64_encoded_image"
}
```

### N8N Workflow Setup

1. Create a webhook node in n8n
2. Copy the webhook URL
3. Add to `config/settings.yaml`
4. Process the data (send to Telegram, Email, database, etc.)

## Configuration

### Zone Types
- `main_entrance`: High traffic, many unique persons
- `room_door`: Low traffic, few unique persons
- `common_area`: Medium traffic shared spaces

### Performance Tuning

For Jetson Nano optimization:
```yaml
detection:
  person_model: "yolov8n.pt"  # Nano model for speed
  confidence_threshold: 0.45
  
tracking:
  track_thresh: 0.25
  min_box_area: 100  # Adjust based on camera distance
```

## Adding Multi-Camera Support

The system is designed to scale with Raspberry Pi Zero cameras:

1. Set up Pi Zero with streaming:
```bash
# On Pi Zero
cd ../common
python3 stream_video.py
```

2. Add camera to configuration:
```yaml
additional_cameras:
  - url: "http://pi1.local:8000/stream.mjpg"
  - url: "http://pi2.local:8000/stream.mjpg"
```

## Troubleshooting

### Low FPS
- Use YOLOv8n (nano) model instead of larger variants
- Reduce camera resolution in settings
- Disable display with `--no-display`

### Webhook Not Working
- Test with `python3 main.py --test-webhook`
- Check n8n webhook is active
- Verify network connectivity

### Zone Detection Issues
- Let system learn for 24 hours for better motion patterns
- Manually draw zones via web interface
- Adjust `motion_threshold` in settings

## Future Enhancements

The system is designed for easy upgrades:

1. **Custom Door Detection Model**
   - Train YOLO on door dataset
   - Replace motion detection with ML model
   - Simply change `detection.mode` to "ml"

2. **Supabase Integration**
   - Ready for cloud database
   - Replace SQLite with Supabase client

3. **Smart Lock Integration**
   - API ready for lock control
   - Automated entry management

4. **Advanced Analytics**
   - Occupancy patterns
   - Traffic analysis
   - Behavior insights

## System Requirements

### Minimum
- Jetson Nano 4GB / Ubuntu PC
- 720p USB camera
- 32GB storage
- 4GB RAM

### Recommended
- Jetson TX2/Xavier
- Multiple cameras
- 128GB+ storage
- Ethernet connection

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the logs in `logs/`
3. Test webhook connectivity
4. Verify camera is working with `cv2.VideoCapture`

## License

This project is built on top of the open-source home security system.
Please respect the original license and attribution requirements.