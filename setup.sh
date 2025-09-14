#!/bin/bash

echo "==================================="
echo "Airbnb Monitor Setup Script"
echo "For Jetson Nano / Ubuntu Systems"
echo "==================================="

# Check if running on Jetson
if [ -f /etc/nv_tegra_release ]; then
    echo "Detected Jetson device"
    IS_JETSON=true
else
    echo "Not running on Jetson, assuming standard Linux"
    IS_JETSON=false
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p storage/{videos,snapshots,clips}
mkdir -p logs
mkdir -p config

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev python3-opencv
sudo apt-get install -y libopencv-dev
sudo apt-get install -y ffmpeg

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Download YOLOv8 model
echo "Downloading YOLOv8 model..."
python3 -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt')"

# Set up configuration
if [ ! -f config/settings.yaml ]; then
    echo "Configuration file already exists at config/settings.yaml"
    echo "Please edit it to set your webhook URL and other settings"
else
    echo "Configuration file exists"
fi

# Create systemd service (optional)
echo "Do you want to install as a system service? (y/n)"
read -r response
if [[ "$response" == "y" ]]; then
    sudo tee /etc/systemd/system/airbnb-monitor.service > /dev/null <<EOF
[Unit]
Description=Airbnb Monitor Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
ExecStart=/usr/bin/python3 $(pwd)/main.py --no-display
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    echo "Service installed. You can start it with:"
    echo "  sudo systemctl start airbnb-monitor"
    echo "  sudo systemctl enable airbnb-monitor  # To start on boot"
fi

echo ""
echo "==================================="
echo "Setup complete!"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. Edit config/settings.yaml to set your n8n webhook URL"
echo "2. Run the application:"
echo "   python3 main.py"
echo ""
echo "For headless operation (no display):"
echo "   python3 main.py --no-display"
echo ""
echo "To access the web interface:"
echo "   Open http://localhost:5000 in your browser"
echo ""
echo "To test webhook:"
echo "   python3 main.py --test-webhook"