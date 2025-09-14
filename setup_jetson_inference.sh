#!/bin/bash

# Roboflow Inference Server Setup for Jetson Nano
# This script sets up the Roboflow inference server on Jetson devices

echo "==================================="
echo "Roboflow Inference Server Setup"
echo "==================================="

# Function to detect JetPack version
detect_jetpack_version() {
    if [ -f /etc/nv_tegra_release ]; then
        # Parse L4T version from tegra release file
        L4T_VERSION=$(head -n 1 /etc/nv_tegra_release | sed 's/.*R\([0-9]*\).*/\1/')
        L4T_REVISION=$(head -n 1 /etc/nv_tegra_release | sed 's/.*R[0-9]* (release), REVISION: \([0-9.]*\).*/\1/')
        
        echo "Detected L4T Version: R${L4T_VERSION}.${L4T_REVISION}"
        
        # Map L4T to JetPack version
        if [ "$L4T_VERSION" = "32" ]; then
            if [[ "$L4T_REVISION" == "5"* ]]; then
                JETPACK="4.5"
            elif [[ "$L4T_REVISION" == "6"* ]] || [[ "$L4T_REVISION" == "7"* ]]; then
                JETPACK="4.6"
            fi
        elif [ "$L4T_VERSION" = "34" ] || [ "$L4T_VERSION" = "35" ]; then
            JETPACK="5"
        elif [ "$L4T_VERSION" = "36" ]; then
            JETPACK="6"
        fi
        
        echo "JetPack Version: ${JETPACK}"
    else
        echo "Unable to detect JetPack version. Please specify manually."
        echo "Options: 4.5, 4.6, 5, 6"
        read -p "Enter JetPack version: " JETPACK
    fi
}

# Stop existing container if running
echo "Stopping existing inference server if running..."
sudo docker stop inference-server 2>/dev/null
sudo docker rm inference-server 2>/dev/null

# Detect JetPack version
detect_jetpack_version

# Select appropriate Docker image based on JetPack version
case "$JETPACK" in
    "4.5")
        IMAGE="roboflow/roboflow-inference-server-jetson-4.5.0:latest"
        ;;
    "4.6")
        IMAGE="roboflow/roboflow-inference-server-jetson-4.6.1:latest"
        ;;
    "5")
        IMAGE="roboflow/roboflow-inference-server-jetson-5.1.1:latest"
        ;;
    "6")
        IMAGE="roboflow/roboflow-inference-server-jetson-6.0.0:latest"
        ;;
    *)
        echo "Unsupported JetPack version: $JETPACK"
        echo "Using JetPack 4.6 image as fallback (common for Jetson Nano)"
        IMAGE="roboflow/roboflow-inference-server-jetson-4.6.1:latest"
        ;;
esac

echo ""
echo "Using Docker image: $IMAGE"
echo ""

# Ask about TensorRT optimization
echo "TensorRT can significantly improve performance but requires ~15 minutes"
echo "for initial model compilation."
read -p "Enable TensorRT optimization? (y/n): " USE_TENSORRT

# Start the Docker container
echo ""
echo "Starting Roboflow Inference Server..."

if [ "$USE_TENSORRT" = "y" ] || [ "$USE_TENSORRT" = "Y" ]; then
    echo "Starting with TensorRT enabled..."
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
        $IMAGE
else
    echo "Starting without TensorRT..."
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
        $IMAGE
fi

# Check if container started successfully
sleep 5
if sudo docker ps | grep -q inference-server; then
    echo ""
    echo "✓ Inference server started successfully!"
    echo ""
    echo "Waiting for server to be ready..."
    
    # Wait for server to be ready
    MAX_ATTEMPTS=30
    ATTEMPT=0
    while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
        if curl -s http://localhost:9001/info > /dev/null 2>&1; then
            echo "✓ Server is ready!"
            echo ""
            echo "Server info:"
            curl -s http://localhost:9001/info | python3 -m json.tool
            break
        fi
        ATTEMPT=$((ATTEMPT + 1))
        echo -n "."
        sleep 2
    done
    
    if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
        echo ""
        echo "⚠ Server took too long to start. Check logs with:"
        echo "sudo docker logs inference-server"
    fi
    
    echo ""
    echo "==================================="
    echo "Setup Complete!"
    echo "==================================="
    echo ""
    echo "The Roboflow Inference Server is now running on port 9001"
    echo ""
    echo "To test door detection, run:"
    echo "  python3 test_inference.py"
    echo ""
    echo "To view logs:"
    echo "  sudo docker logs -f inference-server"
    echo ""
    echo "To stop the server:"
    echo "  sudo docker stop inference-server"
    echo ""
else
    echo ""
    echo "✗ Failed to start inference server"
    echo "Check logs with: sudo docker logs inference-server"
    exit 1
fi