#!/bin/bash

# Start Roboflow Inference Server on Jetson

echo "Starting Roboflow Inference Server..."

# Stop any existing inference containers
docker stop inference-server 2>/dev/null || true
docker rm inference-server 2>/dev/null || true

# Pull the image if not present
echo "Checking for Docker image..."
if ! docker images | grep -q "roboflow/roboflow-inference-server-gpu"; then
    echo "Pulling inference server image..."
    docker pull roboflow/roboflow-inference-server-gpu:latest
fi

# Start the inference server
echo "Starting inference container..."
docker run --rm -d \
    --name inference-server \
    --runtime nvidia \
    --net=host \
    -e ROBOFLOW_API_KEY=placeholder \
    -e MODEL_CACHE_DIR=/tmp/cache \
    -e NUM_WORKERS=1 \
    roboflow/roboflow-inference-server-gpu:latest

# Wait for server to start
echo "Waiting for server to start..."
sleep 5

# Check if running
if curl -s http://localhost:9001/health > /dev/null 2>&1; then
    echo "✓ Inference server is running at http://localhost:9001"
    echo ""
    echo "To test: python3 test_inference.py"
    echo "To stop: ./stop_inference.sh"
else
    echo "✗ Server failed to start"
    echo "Check logs with: docker logs inference-server"
    exit 1
fi