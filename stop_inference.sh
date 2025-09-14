#!/bin/bash

# Stop Roboflow Inference Server

echo "Stopping Roboflow Inference Server..."

# Method 1: Stop inference-cli process
pkill -f "inference server" 2>/dev/null && echo "✓ Stopped inference-cli server"

# Method 2: Stop Docker container (if running)
if docker ps | grep -q inference-server; then
    sudo docker stop inference-server 2>/dev/null
    sudo docker rm inference-server 2>/dev/null
    echo "✓ Stopped Docker container"
fi

echo "✓ Inference server stopped"