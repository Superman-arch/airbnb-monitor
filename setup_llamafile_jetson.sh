#!/bin/bash

echo "====================================="
echo "LlamaFile VLM Setup for Jetson Nano"
echo "====================================="
echo ""
echo "This will set up Phi-3.5 Vision using llamafile"
echo "No Docker required - runs directly on your Jetson!"
echo ""

# Check if llamafile already exists
if [ -f "Phi-3.5-vision-instruct.Q4_K_M.llamafile" ]; then
    echo "✓ Llamafile already downloaded"
else
    echo "Downloading Phi-3.5 Vision llamafile..."
    echo "This is about 2.5GB and may take a few minutes..."
    
    wget -O Phi-3.5-vision-instruct.Q4_K_M.llamafile \
        https://huggingface.co/Mozilla/Phi-3.5-vision-instruct-llamafile/resolve/main/Phi-3.5-vision-instruct.Q4_K_M.llamafile
    
    if [ $? -ne 0 ]; then
        echo "❌ Download failed. Trying alternative URL..."
        # Try alternative download
        curl -L -o Phi-3.5-vision-instruct.Q4_K_M.llamafile \
            https://huggingface.co/Mozilla/Phi-3.5-vision-instruct-llamafile/resolve/main/Phi-3.5-vision-instruct.Q4_K_M.llamafile
    fi
    
    # Make executable
    chmod +x Phi-3.5-vision-instruct.Q4_K_M.llamafile
    echo "✓ Download complete!"
fi

# Create start script
cat > start_llamafile_vlm.sh << 'EOF'
#!/bin/bash

echo "====================================="
echo "Starting LlamaFile VLM Server"
echo "====================================="
echo ""

# Kill any existing llamafile process
pkill -f "Phi-3.5-vision-instruct" 2>/dev/null

echo "Starting Phi-3.5 Vision server on port 8080..."
echo "This provides local vision-language analysis"
echo ""

# Start llamafile server
# Using conservative settings for Jetson Nano
./Phi-3.5-vision-instruct.Q4_K_M.llamafile \
    --server \
    --host 0.0.0.0 \
    --port 8080 \
    --nobrowser \
    --ctx-size 2048 \
    --n-gpu-layers 999 \
    --threads 4 &

# Save PID
echo $! > llamafile.pid

sleep 5

# Check if server is running
if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo ""
    echo "✓ VLM server is running!"
    echo "  API endpoint: http://localhost:8080/v1/chat/completions"
    echo "  Web UI: http://localhost:8080"
    echo ""
    echo "You can now run: python3 run_optimized.py"
else
    echo "⚠ Server may still be starting. Check in a moment."
fi

echo ""
echo "To stop the server: pkill -f Phi-3.5-vision-instruct"
echo "====================================="
EOF

chmod +x start_llamafile_vlm.sh

# Create stop script
cat > stop_llamafile_vlm.sh << 'EOF'
#!/bin/bash
echo "Stopping LlamaFile VLM server..."
pkill -f "Phi-3.5-vision-instruct"
if [ -f llamafile.pid ]; then
    kill $(cat llamafile.pid) 2>/dev/null
    rm llamafile.pid
fi
echo "VLM server stopped"
EOF

chmod +x stop_llamafile_vlm.sh

# Update settings.yaml if needed
if [ -f "config/settings.yaml" ]; then
    echo ""
    echo "Updating configuration..."
    
    # Check if VLM config exists
    if grep -q "^vlm:" config/settings.yaml; then
        # Update existing config using sed
        sed -i.bak 's|endpoint:.*|endpoint: "http://localhost:8080/v1/chat/completions"|' config/settings.yaml
        sed -i 's|model:.*|model: "Phi-3.5-vision-instruct"|' config/settings.yaml
        echo "✓ Updated VLM configuration"
    else
        # Add VLM config
        cat >> config/settings.yaml << 'EOF'

# VLM Configuration for llamafile
vlm:
  enabled: true
  endpoint: "http://localhost:8080/v1/chat/completions"
  model: "Phi-3.5-vision-instruct"
  timeout: 30
  calibration:
    auto_calibrate_on_start: true
    recalibrate_interval: 86400
    save_zone_images: true
EOF
        echo "✓ Added VLM configuration"
    fi
fi

echo ""
echo "====================================="
echo "Setup Complete!"
echo "====================================="
echo ""
echo "To start the VLM server:"
echo "  ./start_llamafile_vlm.sh"
echo ""
echo "To stop the VLM server:"
echo "  ./stop_llamafile_vlm.sh"
echo ""
echo "The VLM will:"
echo "- Analyze your camera view"
echo "- Identify all doors automatically"
echo "- Name them intelligently"
echo "- Create tracking zones"
echo ""
echo "All processing is 100% local on your Jetson!"
echo "====================================="