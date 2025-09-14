#!/bin/bash

# Roboflow Inference Server Setup for Jetson Nano
# This script sets up local ML inference for door detection
# Run once on Jetson Nano: ./setup_inference_jetson.sh

set -e

echo "=================================================="
echo "Roboflow Inference Server Setup for Jetson Nano"
echo "=================================================="

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "Warning: This doesn't appear to be a Jetson device"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 1: Install Docker if not present
echo ""
echo "Step 1: Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    echo "✓ Docker installed"
    echo "NOTE: You may need to log out and back in for docker permissions"
else
    echo "✓ Docker already installed"
fi

# Step 2: Install NVIDIA Container Toolkit for GPU support
echo ""
echo "Step 2: Installing NVIDIA Container Toolkit..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
echo "✓ NVIDIA Container Toolkit installed"

# Step 3: Install Python dependencies
echo ""
echo "Step 3: Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install inference-cli inference-sdk
echo "✓ Python dependencies installed"

# Step 4: Pull and configure Inference Server for Jetson
echo ""
echo "Step 4: Setting up Inference Server for Jetson..."

# Create inference config directory
mkdir -p ~/.inference
cat > ~/.inference/config.json << EOF
{
    "model_cache_dir": "/home/$USER/.inference/models",
    "device": "cuda",
    "num_workers": 2,
    "max_batch_size": 1,
    "port": 9001
}
EOF

# Pull the Jetson-optimized inference image
echo "Pulling Jetson-optimized inference image..."
sudo docker pull roboflow/roboflow-inference-server-jetson:latest

# Step 5: Download the door detection model
echo ""
echo "Step 5: Pre-downloading door detection model..."
cat > download_model.py << 'EOF'
#!/usr/bin/env python3
import requests
import os
import json

# Create model cache directory
model_dir = os.path.expanduser("~/.inference/models/is-my-door-open")
os.makedirs(model_dir, exist_ok=True)

print("Downloading 'Is My Door Open?' model...")
# Model metadata
metadata = {
    "model_id": "is-my-door-open/2",
    "type": "object-detection",
    "classes": ["closed_door", "open_door"],
    "confidence_threshold": 0.6
}

with open(f"{model_dir}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("✓ Model metadata configured")
print("Note: Full model weights will be downloaded on first use")
EOF

python3 download_model.py
rm download_model.py

# Step 6: Create systemd service for auto-start
echo ""
echo "Step 6: Creating systemd service for auto-start..."
sudo tee /etc/systemd/system/roboflow-inference.service > /dev/null << EOF
[Unit]
Description=Roboflow Inference Server
After=docker.service
Requires=docker.service

[Service]
Type=simple
Restart=always
RestartSec=10
User=$USER
ExecStartPre=/bin/bash -c 'docker stop inference-server 2>/dev/null || true'
ExecStartPre=/bin/bash -c 'docker rm inference-server 2>/dev/null || true'
ExecStart=/usr/bin/docker run --name inference-server \
    --runtime nvidia \
    --gpus all \
    --net=host \
    --rm \
    -v /home/$USER/.inference/models:/models \
    -e ROBOFLOW_API_KEY=placeholder \
    -e MODEL_CACHE_DIR=/models \
    -e NUM_WORKERS=2 \
    -e DEVICE=cuda \
    roboflow/roboflow-inference-server-jetson:latest
ExecStop=/usr/bin/docker stop inference-server

[Install]
WantedBy=multi-user.target
EOF

# Enable but don't start the service yet
sudo systemctl daemon-reload
sudo systemctl enable roboflow-inference.service

# Step 7: Create start/stop scripts
echo ""
echo "Step 7: Creating control scripts..."

# Start script
cat > start_inference.sh << 'EOF'
#!/bin/bash
echo "Starting Roboflow Inference Server..."
sudo systemctl start roboflow-inference.service
sleep 5
echo "Checking status..."
if curl -s http://localhost:9001/health > /dev/null; then
    echo "✓ Inference server is running at http://localhost:9001"
else
    echo "✗ Server failed to start. Check logs with: sudo journalctl -u roboflow-inference"
fi
EOF

# Stop script  
cat > stop_inference.sh << 'EOF'
#!/bin/bash
echo "Stopping Roboflow Inference Server..."
sudo systemctl stop roboflow-inference.service
echo "✓ Inference server stopped"
EOF

chmod +x start_inference.sh stop_inference.sh

# Step 8: Create test script
echo ""
echo "Step 8: Creating test script..."
cat > test_inference.py << 'EOF'
#!/usr/bin/env python3
"""Test the local inference server with door detection."""

import cv2
import sys
from inference_sdk import InferenceHTTPClient

def test_inference():
    """Test inference server with a sample image."""
    print("Testing Roboflow Inference Server...")
    
    try:
        # Connect to local server
        client = InferenceHTTPClient(
            api_url="http://localhost:9001"
        )
        
        # Test with webcam frame
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Failed to capture test frame from camera")
            return False
        
        # Run inference
        print("Running door detection...")
        result = client.infer(
            frame,
            model_id="is-my-door-open/2"
        )
        
        print(f"✓ Inference successful!")
        print(f"  Detections: {len(result.get('predictions', []))}")
        
        for pred in result.get('predictions', []):
            print(f"  - {pred['class']}: {pred['confidence']:.2%}")
        
        return True
        
    except Exception as e:
        print(f"✗ Inference test failed: {e}")
        return False

if __name__ == "__main__":
    # Start server if not running
    import subprocess
    import time
    
    print("Checking if inference server is running...")
    try:
        import requests
        requests.get("http://localhost:9001/health", timeout=2)
        print("Server is already running")
    except:
        print("Starting inference server...")
        subprocess.run(["./start_inference.sh"], check=False)
        time.sleep(10)
    
    # Run test
    if test_inference():
        print("\n✓ Setup complete! Inference server is working.")
    else:
        print("\n✗ Test failed. Check server logs.")
        sys.exit(1)
EOF

chmod +x test_inference.py

# Step 9: Final instructions
echo ""
echo "=================================================="
echo "SETUP COMPLETE!"
echo "=================================================="
echo ""
echo "Quick Start:"
echo "  ./start_inference.sh  - Start the inference server"
echo "  ./stop_inference.sh   - Stop the inference server"
echo "  ./test_inference.py   - Test door detection"
echo ""
echo "The server will auto-start on boot."
echo "To start it now, run: ./start_inference.sh"
echo ""
echo "Integration:"
echo "  - Server runs at: http://localhost:9001"
echo "  - Model: is-my-door-open/2"
echo "  - Classes: open_door, closed_door"
echo ""
echo "Next: Update run_optimized.py to use DoorInferenceDetector"
echo "=================================================="