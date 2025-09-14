#!/bin/bash

echo "======================================"
echo "Fixing YOLOv8 Compatibility"
echo "======================================"
echo ""

# Option 1: Update ultralytics to latest version
echo "Option 1: Updating ultralytics to latest version..."
pip3 install --upgrade ultralytics

# Option 2: If that doesn't work, try specific version known to work with PyTorch 2.6
echo ""
echo "If the above doesn't work, you can try:"
echo "pip3 install ultralytics==8.3.52"
echo ""

# Option 3: Clear any cached models
echo "Clearing cached YOLO models..."
rm -rf ~/.cache/torch/hub/ultralytics_*
rm -f yolov8n.pt

echo ""
echo "Now test the monitoring system:"
echo "./start_monitoring.sh"
echo ""
echo "======================================"