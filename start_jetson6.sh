#!/bin/bash

# Quick start for JetPack 6 with Roboflow Inference Server
echo "Starting Roboflow Inference Server for JetPack 6..."

# Stop and remove existing container if running
sudo docker stop inference-server 2>/dev/null
sudo docker rm inference-server 2>/dev/null

# Start the JetPack 6 optimized container
echo "Starting container with TensorRT optimization..."
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
    roboflow/roboflow-inference-server-jetson-6.0.0:latest

echo "Waiting for server to start..."
sleep 5

# Check if server is running
if sudo docker ps | grep -q inference-server; then
    echo "✓ Server started successfully!"
    echo ""
    echo "Testing server health..."
    for i in {1..10}; do
        if curl -s http://localhost:9001/health > /dev/null 2>&1; then
            echo "✓ Server is ready!"
            echo ""
            echo "Server is running at: http://localhost:9001"
            echo ""
            echo "Next steps:"
            echo "1. Test door detection: python3 test_inference.py"
            echo "2. Run monitor: python3 run_optimized.py"
            echo ""
            echo "View logs: sudo docker logs -f inference-server"
            exit 0
        fi
        echo -n "."
        sleep 2
    done
    echo ""
    echo "Server is starting up. Check logs: sudo docker logs inference-server"
else
    echo "✗ Failed to start server"
    echo "Check logs: sudo docker logs inference-server"
    exit 1
fi