#!/bin/bash

echo "====================================="
echo "Native VLM Setup for Jetson Nano"
echo "====================================="
echo ""
echo "This will set up Phi-3.5 Vision using HuggingFace transformers"
echo "No Docker or llamafile needed - runs directly in Python!"
echo ""

# Check Python version
python3 --version

echo ""
echo "Installing required Python packages..."
echo "This may take a few minutes..."

# Install packages
pip3 install --upgrade pip
pip3 install transformers==4.43.0
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip3 install flask pillow accelerate

echo ""
echo "✓ Dependencies installed!"

# Create start script
cat > start_vlm_native.sh << 'EOF'
#!/bin/bash

echo "====================================="
echo "Starting Native VLM Server"
echo "====================================="
echo ""

# Kill any existing VLM server
pkill -f "vlm_server_native.py" 2>/dev/null

echo "Starting Phi-3.5 Vision server..."
echo "First run will download the model (~4GB)"
echo "This may take 10-15 minutes on first run..."
echo ""

# Start the server
python3 vlm_server_native.py &

# Save PID
echo $! > vlm_native.pid

echo ""
echo "VLM server starting on port 8080..."
echo "Check status: curl http://localhost:8080/health"
echo ""
echo "To stop: ./stop_vlm_native.sh"
echo "====================================="
EOF

chmod +x start_vlm_native.sh

# Create stop script
cat > stop_vlm_native.sh << 'EOF'
#!/bin/bash
echo "Stopping Native VLM server..."
pkill -f "vlm_server_native.py"
if [ -f vlm_native.pid ]; then
    kill $(cat vlm_native.pid) 2>/dev/null
    rm vlm_native.pid
fi
echo "VLM server stopped"
EOF

chmod +x stop_vlm_native.sh

echo ""
echo "====================================="
echo "Setup Complete!"
echo "====================================="
echo ""
echo "To start the VLM server:"
echo "  ./start_vlm_native.sh"
echo ""
echo "Note: First run will download the model from HuggingFace"
echo "      This requires internet and ~4GB disk space"
echo ""
echo "The server will:"
echo "- Download Phi-3.5 Vision automatically"
echo "- Run on your Jetson's GPU"
echo "- Provide door detection API on port 8080"
echo "====================================="