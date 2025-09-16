#!/bin/bash

# Start script for docker compose (with space) 

echo "========================================="
echo "Security Monitoring System - Docker Compose"
echo "========================================="

# Check if docker compose is available
if ! docker compose version &> /dev/null; then
    echo "Error: 'docker compose' command not found!"
    echo "Please ensure Docker Desktop or Docker Engine with compose plugin is installed."
    exit 1
fi

# Check for camera devices on host
echo ""
echo "Checking for camera devices on host..."
if [ -e "/dev/video0" ]; then
    echo "✓ Found /dev/video0"
    ls -la /dev/video0
elif [ -e "/dev/video1" ]; then
    echo "✓ Found /dev/video1"
    ls -la /dev/video1
else
    echo "⚠ Warning: No /dev/video* devices found on host"
    echo "The system will attempt to use camera index 0"
fi

# Stop any running containers
echo ""
echo "Stopping any existing containers..."
docker compose -f docker/docker-compose.yml down

# Remove old images to ensure fresh build
echo ""
echo "Removing old images..."
docker rmi security-backend:latest security-frontend:latest 2>/dev/null || true

# Build and start containers
echo ""
echo "Building and starting containers..."
docker compose -f docker/docker-compose.yml up --build -d

# Wait for services to start
echo ""
echo "Waiting for services to start..."
sleep 10

# Check container status
echo ""
echo "Container status:"
docker compose -f docker/docker-compose.yml ps

# Test camera endpoint
echo ""
echo "Testing camera endpoint..."
curl -s http://localhost:8000/api/camera/test | python3 -m json.tool 2>/dev/null || echo "Camera test endpoint not ready yet"

# Show logs
echo ""
echo "========================================="
echo "System is starting up!"
echo ""
echo "Access points:"
echo "  - Frontend: http://localhost:3000"
echo "  - Backend API: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo "  - Camera Test: http://localhost:8000/api/camera/test"
echo ""
echo "To view logs:"
echo "  docker compose -f docker/docker-compose.yml logs -f backend"
echo ""
echo "To stop:"
echo "  docker compose -f docker/docker-compose.yml down"
echo "========================================="

# Follow backend logs
echo ""
echo "Following backend logs (Ctrl+C to exit)..."
docker compose -f docker/docker-compose.yml logs -f backend