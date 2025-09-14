#!/bin/bash

# Start Roboflow Inference Server on Jetson

echo "Starting Roboflow Inference Server..."

# Method 1: Try inference-cli first (recommended for Jetson)
if command -v inference &> /dev/null; then
    echo "Using inference-cli to start server..."
    
    # Kill any existing inference process
    pkill -f "inference server" 2>/dev/null || true
    
    # Start inference server in background
    # It automatically detects Jetson and uses correct image
    nohup inference server start > inference.log 2>&1 &
    
    echo "Waiting for server to start (this may take 30-60 seconds on first run)..."
    
    # Wait for server with timeout
    for i in {1..60}; do
        if curl -s http://localhost:9001/health > /dev/null 2>&1; then
            echo "✓ Inference server is running at http://localhost:9001"
            echo ""
            echo "To test: python3 test_inference.py"
            echo "To stop: ./stop_inference.sh"
            echo "To view logs: tail -f inference.log"
            exit 0
        fi
        echo -n "."
        sleep 1
    done
    
    echo ""
    echo "✗ Server failed to start after 60 seconds"
    echo "Check logs: cat inference.log"
    exit 1
    
else
    echo "Error: inference-cli not found"
    echo "Install with: pip3 install inference-cli"
    echo ""
    echo "Or trying Docker method..."
    
    # Method 2: Try Docker with Jetson-specific image
    echo "Attempting Docker with Jetson image..."
    
    # Stop any existing containers
    sudo docker stop inference-server 2>/dev/null || true
    sudo docker rm inference-server 2>/dev/null || true
    
    # Try Jetson-specific image
    echo "Starting Jetson-optimized container..."
    sudo docker run --rm -d \
        --name inference-server \
        --runtime nvidia \
        --net=host \
        -e ROBOFLOW_API_KEY=placeholder \
        roboflow/roboflow-inference-server-jetson-5.1.1:latest
    
    # Wait and check
    sleep 10
    
    if curl -s http://localhost:9001/health > /dev/null 2>&1; then
        echo "✓ Inference server is running at http://localhost:9001"
        exit 0
    else
        echo "✗ Docker method also failed"
        echo "Please install inference-cli: pip3 install inference-cli"
        exit 1
    fi
fi