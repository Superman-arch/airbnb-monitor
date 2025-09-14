#!/bin/bash

# VLM Setup for Jetson - Door Zone Mapping
echo "===================================="
echo "VLM Setup for Door Zone Mapping"
echo "===================================="
echo ""
echo "This script sets up a Vision Language Model (VLM) for intelligent door zone mapping."
echo "The VLM will analyze your camera view to identify and name doors automatically."
echo ""

# Function to check Jetson platform
check_jetson() {
    if [ -f /etc/nv_tegra_release ]; then
        echo "✓ Jetson platform detected"
        # Get L4T version
        L4T_VERSION=$(head -n 1 /etc/nv_tegra_release | sed 's/.*R\([0-9]*\).*/\1/')
        echo "  L4T Version: R${L4T_VERSION}"
        return 0
    else
        echo "⚠ Not running on Jetson - continuing anyway"
        return 1
    fi
}

# Function to check available memory
check_memory() {
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    echo "Total memory: ${TOTAL_MEM}GB"
    
    if [ "$TOTAL_MEM" -lt 8 ]; then
        echo "⚠ Warning: Less than 8GB RAM detected"
        echo "  Phi 3.5 Vision (3.8GB) recommended for low memory"
    else
        echo "✓ Sufficient memory for larger models"
    fi
}

# Main menu
echo "Select VLM deployment method:"
echo ""
echo "1) Phi 3.5 Vision (Recommended - 3.8GB, fastest)"
echo "2) LLaVA 1.5 7B (Better accuracy - 7GB, slower)"
echo "3) llamafile Server (Easy setup, various models)"
echo "4) Skip VLM setup (use fallback edge detection)"
echo ""
read -p "Enter choice (1-4): " CHOICE

case $CHOICE in
    1)
        echo ""
        echo "Setting up Phi 3.5 Vision..."
        echo ""
        
        # Option 1: Using NVIDIA Jetson containers
        echo "Pulling Phi Vision container..."
        sudo docker pull dustynv/transformers:r36.3.0
        
        # Create runner script
        cat > run_phi_vision.py << 'EOF'
#!/usr/bin/env python3
"""Phi Vision VLM server for door zone mapping."""

import os
import json
import base64
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

app = Flask(__name__)

# Load model
print("Loading Phi-3.5-vision model...")
model_id = "microsoft/Phi-3.5-vision-instruct"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True
)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model": "phi-3.5-vision"})

@app.route('/completion', methods=['POST'])
def completion():
    data = request.json
    prompt = data.get('prompt', '')
    image_b64 = data.get('image', '')
    
    # Decode image
    image = Image.open(BytesIO(base64.b64decode(image_b64)))
    
    # Process with model
    inputs = processor(prompt, image, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=500, temperature=0.1)
    
    response = processor.decode(output[0], skip_special_tokens=True)
    
    return jsonify({"content": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
EOF
        
        # Create service script
        cat > start_phi_vision.sh << 'EOF'
#!/bin/bash
echo "Starting Phi Vision VLM server..."
sudo docker run -d \
    --name vlm-server \
    --runtime nvidia \
    --restart unless-stopped \
    -p 8080:8080 \
    -v $(pwd)/run_phi_vision.py:/app/server.py:ro \
    dustynv/transformers:r36.3.0 \
    python3 /app/server.py
EOF
        chmod +x start_phi_vision.sh
        
        echo ""
        echo "✓ Phi Vision setup complete!"
        echo "Start server with: ./start_phi_vision.sh"
        ;;
        
    2)
        echo ""
        echo "Setting up LLaVA 1.5..."
        echo ""
        
        # Pull LLaVA container
        sudo docker pull dustynv/llava:r36.2.0
        
        # Create LLaVA runner
        cat > start_llava.sh << 'EOF'
#!/bin/bash
echo "Starting LLaVA VLM server..."
sudo docker run -d \
    --name vlm-server \
    --runtime nvidia \
    --restart unless-stopped \
    -p 8080:8080 \
    -v /tmp/vlm:/tmp/vlm \
    dustynv/llava:r36.2.0 \
    python3 -m llava.serve.api --model-path liuhaotian/llava-v1.5-7b --port 8080
EOF
        chmod +x start_llava.sh
        
        echo "✓ LLaVA setup complete!"
        echo "Start server with: ./start_llava.sh"
        ;;
        
    3)
        echo ""
        echo "Setting up llamafile server..."
        echo ""
        
        # Download llamafile
        echo "Downloading llamafile with Phi-3.5-vision..."
        wget -O phi-3.5-vision.llamafile https://huggingface.co/Mozilla/Phi-3.5-vision-instruct-llamafile/resolve/main/Phi-3.5-vision-instruct.Q4_K_M.llamafile
        chmod +x phi-3.5-vision.llamafile
        
        # Create runner script
        cat > start_llamafile.sh << 'EOF'
#!/bin/bash
echo "Starting llamafile VLM server..."
./phi-3.5-vision.llamafile --server --host 0.0.0.0 --port 8080 --nobrowser
EOF
        chmod +x start_llamafile.sh
        
        echo "✓ llamafile setup complete!"
        echo "Start server with: ./start_llamafile.sh"
        ;;
        
    4)
        echo ""
        echo "Skipping VLM setup."
        echo "The system will use edge detection for door zones."
        echo "This is less accurate but doesn't require additional resources."
        ;;
        
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

# Add VLM config to settings
if [ "$CHOICE" != "4" ]; then
    echo ""
    echo "Updating configuration..."
    
    # Check if config exists
    if [ -f "config/settings.yaml" ]; then
        # Add VLM config if not exists
        if ! grep -q "^vlm:" config/settings.yaml; then
            cat >> config/settings.yaml << 'EOF'

# VLM Configuration for zone mapping
vlm:
  enabled: true
  endpoint: "http://localhost:8080/completion"
  model: "phi-3.5-vision"
  timeout: 30
EOF
            echo "✓ Added VLM configuration to settings.yaml"
        else
            echo "✓ VLM configuration already exists"
        fi
    fi
fi

echo ""
echo "===================================="
echo "Setup Complete!"
echo "===================================="
echo ""

if [ "$CHOICE" != "4" ]; then
    echo "Next steps:"
    echo "1. Start the VLM server (if not using llamafile):"
    echo "   ./start_*.sh"
    echo ""
    echo "2. Start the monitor:"
    echo "   python3 run_optimized.py"
    echo ""
    echo "3. Open web interface and click 'Calibrate Doors'"
    echo "   http://<jetson-ip>:5000"
    echo ""
    echo "The VLM will analyze your camera view and automatically:"
    echo "- Identify all visible doors"
    echo "- Name them based on context (Main Entrance, Bathroom, etc.)"
    echo "- Create persistent zones for tracking"
else
    echo "VLM setup skipped. Using edge detection fallback."
fi