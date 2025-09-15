# Jetson Nano Deployment Guide for Door Detection

This guide will help you deploy the Roboflow inference server on your Jetson Nano for accurate door detection (96.7% accuracy).

## Prerequisites

1. **Jetson Nano** with JetPack installed (4.5, 4.6, 5.x, or 6.x)
2. **Docker** installed with NVIDIA runtime
3. **At least 10GB free disk space** for the inference container
4. **Network connection** for initial model download

## Quick Setup

### Step 1: Transfer Files to Jetson Nano

Copy the entire `airbnb-monitor` directory to your Jetson Nano:

```bash
# From your Mac (replace with your Jetson's IP)
scp -r /Users/amadeobonde/Downloads/homesecurity-master/airbnb-monitor jetson@<JETSON_IP>:~/
```

### Step 2: SSH into your Jetson Nano

```bash
ssh jetson@<JETSON_IP>
cd ~/airbnb-monitor
```

### Step 3: Run the Setup Script

```bash
# Make scripts executable
chmod +x setup_jetson_inference.sh
chmod +x test_inference.py
chmod +x run_optimized.py

# Run the setup script
./setup_jetson_inference.sh
```

The script will:
- Detect your JetPack version automatically
- Download the appropriate Roboflow container
- Start the inference server on port 9001
- Optionally enable TensorRT for better performance

**Note**: When asked about TensorRT:
- Choose **Yes** for better performance (2-3x faster)
- Choose **No** for faster initial setup (TensorRT compilation takes ~15 minutes)

### Step 4: Test the Inference Server

```bash
# Install Python dependencies
pip3 install inference-sdk requests opencv-python-headless

# Test the server
python3 test_inference.py
```

You should see:
```
✓ Server is healthy
✓ Inference successful!
```

### Step 5: Run the Monitor Application

```bash
# Run the optimized monitor with door detection
python3 run_optimized.py
```

The application will:
1. Connect to the inference server
2. Start processing camera frames
3. Detect doors using the Roboflow model
4. Display results in the web dashboard

## Accessing the Dashboard

Open a web browser and navigate to:
```
http://<JETSON_IP>:5000
```

You should see:
- Live camera feed
- **Active Doors** section showing detected doors
- Real-time door open/closed status

## Performance Optimization

### For Jetson Nano (4GB RAM)

The default settings are optimized for Jetson Nano:
- Processes every 2nd frame
- 640x480 resolution
- Target: 10-15 FPS

### Enable TensorRT (Recommended)

If you didn't enable TensorRT during setup, you can do it later:

```bash
# Stop current container
sudo docker stop inference-server
sudo docker rm inference-server

# Start with TensorRT
sudo docker run -d \
    --name inference-server \
    --runtime nvidia \
    --restart unless-stopped \
    --read-only \
    -p 9001:9001 \
    --volume ~/.inference/cache:/tmp:rw \
    --security-opt="no-new-privileges" \
    --cap-drop="ALL" \
    --cap-add="NET_BIND_SERVICE" \
    -e ONNXRUNTIME_EXECUTION_PROVIDERS="[TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider]" \
    roboflow/roboflow-inference-server-jetson-4.6.1:latest
```

**Note**: First model load with TensorRT takes 15+ minutes for optimization. Subsequent runs are much faster.

## Troubleshooting

### 1. Server Not Starting

Check Docker status:
```bash
sudo docker ps -a
sudo docker logs inference-server
```

### 2. Out of Memory

Reduce resolution in `config/settings.yaml`:
```yaml
camera:
  resolution: [320, 240]  # Lower resolution
```

### 3. Low FPS

Increase frame skipping in `config/settings.yaml`:
```yaml
door_detection:
  process_every_n_frames: 3  # Process every 3rd frame
```

### 4. Model Download Issues

The door model downloads on first use. If it fails:
```bash
# Clear cache and restart
rm -rf ~/.inference/cache
sudo docker restart inference-server
```

## Auto-Start on Boot

To start the inference server automatically on boot:

```bash
# Create systemd service
sudo tee /etc/systemd/system/roboflow-inference.service > /dev/null <<EOF
[Unit]
Description=Roboflow Inference Server
After=docker.service
Requires=docker.service

[Service]
Type=simple
Restart=always
RestartSec=10
ExecStart=/usr/bin/docker start -a inference-server
ExecStop=/usr/bin/docker stop inference-server

[Install]
WantedBy=multi-user.target
EOF

# Enable service
sudo systemctl enable roboflow-inference.service
sudo systemctl start roboflow-inference.service
```

## Monitor Service

Similarly, create a service for the monitor application:

```bash
sudo tee /etc/systemd/system/airbnb-monitor.service > /dev/null <<EOF
[Unit]
Description=Airbnb Monitor with Door Detection
After=roboflow-inference.service
Requires=roboflow-inference.service

[Service]
Type=simple
Restart=always
RestartSec=10
User=jetson
WorkingDirectory=/home/jetson/airbnb-monitor
ExecStart=/usr/bin/python3 /home/jetson/airbnb-monitor/run_optimized.py
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable airbnb-monitor.service
sudo systemctl start airbnb-monitor.service
```

## Monitoring Logs

View real-time logs:
```bash
# Inference server logs
sudo docker logs -f inference-server

# Monitor application logs
sudo journalctl -u airbnb-monitor -f
```

## Expected Results

Once everything is running correctly:

1. **Inference Server**: Running on port 9001
2. **Web Dashboard**: Accessible on port 5000
3. **Door Detection**: 
   - Doors detected immediately (no learning phase)
   - 96.7% accuracy with Roboflow model
   - Real-time open/closed status
   - Visual bounding boxes on detected doors

## Support

- Check server health: `curl http://localhost:9001/health`
- View server info: `curl http://localhost:9001/info`
- Test detection: `python3 test_inference.py`
- Monitor logs: `sudo docker logs -f inference-server`

## Additional Notes

- The Roboflow model `is-my-door-open/2` is specifically trained for door detection
- First model load downloads ~50MB (cached for future use)
- The model can detect multiple doors simultaneously
- Door states: open, closed, moving, unknown
- Webhook notifications supported for door events