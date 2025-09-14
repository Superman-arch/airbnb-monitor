#!/bin/bash

# Stop Roboflow Inference Server

echo "Stopping Roboflow Inference Server..."

# Stop the container
docker stop inference-server 2>/dev/null

# Remove the container
docker rm inference-server 2>/dev/null

echo "✓ Inference server stopped"