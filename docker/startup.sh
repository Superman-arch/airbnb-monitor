#!/bin/bash

# Startup script for backend container with camera initialization

echo "Starting Security Monitoring Backend..."

# Check for video devices
echo "Checking for video devices..."
ls -la /dev/video* 2>/dev/null || echo "No video devices found in /dev/"

# Check v4l2 devices
if command -v v4l2-ctl &> /dev/null; then
    echo "Available video devices:"
    v4l2-ctl --list-devices
else
    echo "v4l2-ctl not available"
fi

# Set permissions for video devices
echo "Setting permissions for video devices..."
for device in /dev/video*; do
    if [ -e "$device" ]; then
        echo "Setting permissions for $device"
        chmod 666 "$device" 2>/dev/null || echo "Could not set permissions for $device"
    fi
done

# Test camera access
echo "Testing camera access..."
if [ -e "/dev/video0" ]; then
    echo "Testing /dev/video0..."
    timeout 2 dd if=/dev/video0 of=/dev/null bs=1M count=1 2>/dev/null && echo "Camera test successful" || echo "Camera test failed"
fi

# Export camera device based on what's available
if [ -e "/dev/video0" ]; then
    export CAMERA_DEVICE="/dev/video0"
    echo "Using camera device: $CAMERA_DEVICE"
elif [ -e "/dev/video1" ]; then
    export CAMERA_DEVICE="/dev/video1"
    echo "Using camera device: $CAMERA_DEVICE"
else
    echo "WARNING: No camera device found!"
    export CAMERA_DEVICE="0"  # Try index 0 as fallback
fi

# Enable Jetson clocks if available
if command -v jetson_clocks &> /dev/null; then
    echo "Enabling Jetson clocks for maximum performance..."
    jetson_clocks
fi

# Start the application
echo "Starting application..."
cd /app
exec python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1 --loop uvloop