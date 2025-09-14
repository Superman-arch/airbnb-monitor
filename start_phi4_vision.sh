#!/bin/bash

echo "====================================="
echo "Starting Phi-4 Multimodal VLM Server"
echo "====================================="
echo ""

# Check if container is already running
if docker ps | grep -q vlm-server; then
    echo "VLM server is already running. Stopping it first..."
    docker stop vlm-server
    docker rm vlm-server
fi

# Check if old container exists but stopped
if docker ps -a | grep -q vlm-server; then
    echo "Removing old VLM container..."
    docker rm vlm-server
fi

echo "Starting Phi-4 Multimodal container..."
echo "This provides local vision-language analysis for door detection"
echo ""

# Run Phi-4 container
docker run -d \
    --name vlm-server \
    --runtime nvidia \
    --restart unless-stopped \
    -p 8000:8000 \
    bhimrazy/phi-4-multimodal:latest

# Check if container started successfully
sleep 3
if docker ps | grep -q vlm-server; then
    echo ""
    echo "✓ VLM server started successfully!"
    echo ""
    echo "Testing endpoint..."
    
    # Test the endpoint
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "✓ VLM API is responding"
        echo ""
        echo "Server is ready at: http://localhost:8000"
        echo "You can now run: python3 run_optimized.py"
    else
        echo "⚠ API not responding yet. It may take a minute to initialize."
        echo "Check status with: docker logs vlm-server"
    fi
else
    echo "❌ Failed to start VLM server"
    echo "Check logs with: docker logs vlm-server"
    exit 1
fi

echo ""
echo "====================================="
echo "VLM Setup Complete!"
echo "====================================="