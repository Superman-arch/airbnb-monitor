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
echo "1) Phi-4 Multimodal (Recommended - 7.7GB Docker, best accuracy)"
echo "2) llamafile Server (Easy setup, no Docker needed)"
echo "3) Skip VLM setup (use fallback edge detection)"
echo ""
read -p "Enter choice (1-3): " CHOICE

case $CHOICE in
    1)
        echo ""
        echo "Setting up Phi-4 Multimodal..."
        echo ""
        
        # Pull Phi-4 Multimodal container
        echo "Pulling Phi-4 Multimodal container (7.7GB)..."
        echo "This may take a while depending on your internet speed."
        sudo docker pull bhimrazy/phi-4-multimodal:latest
        
        # Check if container already exists
        if docker ps -a | grep -q vlm-server; then
            echo "Removing existing VLM container..."
            docker stop vlm-server 2>/dev/null
            docker rm vlm-server 2>/dev/null
        fi
        
        # Create service script
        cat > start_phi4_vision.sh << 'EOF'
#!/bin/bash
echo "Starting Phi-4 Multimodal VLM server..."

# Check if container is already running
if docker ps | grep -q vlm-server; then
    echo "VLM server is already running. Stopping it first..."
    docker stop vlm-server
    docker rm vlm-server
fi

# Start Phi-4 container
docker run -d \
    --name vlm-server \
    --runtime nvidia \
    --restart unless-stopped \
    -p 8000:8000 \
    bhimrazy/phi-4-multimodal:latest

echo "VLM server started on port 8000"
echo "Test with: curl http://localhost:8000/v1/models"
EOF
        chmod +x start_phi4_vision.sh
        
        echo ""
        echo "✓ Phi-4 Multimodal setup complete!"
        echo "Start server with: ./start_phi4_vision.sh"
        ;;
        
    2)
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
        
    3)
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
if [ "$CHOICE" != "3" ]; then
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
  endpoint: "http://localhost:8000/v1/chat/completions"
  model: "microsoft/Phi-4-multimodal-instruct"
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

if [ "$CHOICE" != "3" ]; then
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