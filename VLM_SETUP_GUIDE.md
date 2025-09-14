# VLM Setup Guide for Jetson Nano

## Quick Start (Recommended)

Run these commands on your Jetson Nano:

```bash
# 1. Navigate to project directory
cd ~/airbnb-monitor

# 2. Run the VLM setup script
./setup_vlm_jetson.sh

# 3. When prompted, select option 1 (Phi 3.5 Vision)
# This is optimized for Jetson Nano's 4GB RAM

# 4. Start the VLM server
./start_phi_vision.sh

# 5. Run the monitoring system
python3 run_optimized.py

# 6. Open web interface to calibrate doors
# http://<jetson-ip>:5000
# Click "Calibrate Doors" button
```

## What the VLM Does

The Vision Language Model (VLM) provides intelligent environment analysis:

1. **Automatic Door Detection** - Identifies all doors in camera view
2. **Smart Naming** - Names doors based on visual context (Main Entrance, Bathroom, etc.)
3. **Zone Creation** - Creates tracking zones for each door
4. **Persistent Storage** - Saves configuration for future use

## System Requirements

- Jetson Nano with 4GB RAM
- JetPack 4.6 or later (you have Jetson 6 - perfect!)
- Docker installed (comes with JetPack)
- Active camera connected

## Verify Installation

Check if VLM is running:

```bash
# Check if VLM container is running
docker ps | grep vlm-server

# Test VLM endpoint
curl http://localhost:8080/health

# Check logs
docker logs vlm-server
```

## Troubleshooting

### VLM Server Won't Start
```bash
# Check available memory
free -h

# Stop unnecessary services
sudo systemctl stop cups
sudo systemctl stop bluetooth

# Restart Docker
sudo systemctl restart docker
```

### Camera Frame Not Available
```bash
# Ensure camera is connected
ls /dev/video*

# Test camera
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera ERROR')"
```

### Low Memory Issues
If you get out-of-memory errors, use Option 3 (llamafile) instead:
```bash
./setup_vlm_jetson.sh
# Select option 3
```

## Alternative: llamafile (Lower Memory)

If Phi 3.5 Vision uses too much memory:

```bash
# Download smaller llamafile model
wget -O phi-vision-mini.llamafile https://huggingface.co/Mozilla/Phi-3.5-vision-instruct-llamafile/resolve/main/Phi-3.5-vision-instruct.Q4_K_M.llamafile

chmod +x phi-vision-mini.llamafile

# Run llamafile server
./phi-vision-mini.llamafile --server --host 0.0.0.0 --port 8080 --nobrowser
```

## Manual Zone Calibration

If VLM is unavailable, the system will fall back to edge detection. You can still manually calibrate:

1. Open web interface
2. Click on the video feed to mark door corners
3. Name each zone manually
4. Save configuration

## Performance Tips

1. **Reduce Camera Resolution** - Already optimized in settings.yaml (640x480)
2. **Process Every N Frames** - VLM processes every 2nd frame by default
3. **Use Hardware Acceleration** - Phi 3.5 Vision uses CUDA automatically
4. **Close Unnecessary Apps** - Free up RAM before starting

## Monitoring VLM Performance

View real-time logs in web interface:
- Green logs = VLM working correctly
- Yellow logs = Using fallback detection
- Red logs = Errors to investigate

## Next Steps

After VLM is running:

1. **Calibrate Doors** - Click button in web interface
2. **Verify Zones** - Check that all doors are detected
3. **Adjust Names** - Edit door names if needed
4. **Monitor Events** - System will track door open/close events

The VLM significantly improves accuracy compared to edge detection alone!