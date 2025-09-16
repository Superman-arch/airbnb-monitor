# Security Monitoring System - Jetson Nano Setup Guide

## Important Context
This system is designed to run on **NVIDIA Jetson Nano** (not on a development computer). All setup and deployment happens directly on the Jetson Nano device.

## Target Hardware
- **Device**: NVIDIA Jetson Orin Nano Super (or compatible Jetson device)
- **Camera**: IC800 1080P HD USB Camera at `/dev/video0`
- **Platform**: ARM64/aarch64 architecture
- **OS**: JetPack/L4T (Linux for Tegra)

## Repository Information
- **GitHub URL**: https://github.com/Superman-arch/airbnb-monitor
- **Purpose**: Home security monitoring with AI-powered door and person detection
- **Stack**: FastAPI (backend), React (frontend), Docker Compose deployment

## Setup on Jetson Nano

### Prerequisites Check
```bash
# Check camera is connected
v4l2-ctl --list-devices
# Should show: IC800 1080P HD at /dev/video0

# Check Docker is installed
docker --version
docker compose version  # Note: use 'docker compose' with space, NOT 'docker-compose'
```

### Initial Setup Commands
```bash
# Clone repository
git clone https://github.com/Superman-arch/airbnb-monitor.git ~/airbnb-monitor
cd ~/airbnb-monitor

# Make start script executable
chmod +x start.sh

# Run the system
./start.sh
```

### Updating from GitHub
When changes are made and pushed to GitHub, pull them on the Jetson:
```bash
cd ~/airbnb-monitor
git pull origin main
./start.sh
```

## Development Workflow

### 1. Making Changes (on development machine)
```bash
# Make your code changes
# Test locally if possible
# Stage and commit
git add -A
git commit -m "Description of changes"
git push origin main
```

### 2. Deploying to Jetson (on Jetson Nano)
```bash
cd ~/airbnb-monitor
git pull origin main
docker compose -f docker/docker-compose.yml down
docker compose -f docker/docker-compose.yml up --build
```

## Common Issues and Solutions

### Camera Not Working
1. **Check device exists**: `ls -la /dev/video*`
2. **Check permissions**: Device should be accessible
3. **Test camera**: `curl http://localhost:8000/api/camera/test`
4. **Check logs**: `docker compose -f docker/docker-compose.yml logs backend`

### Docker Compose Issues
- Always use `docker compose` (with space), NOT `docker-compose`
- The system uses modern Docker Compose v2 syntax
- No `version:` field needed in docker-compose.yml

### Build Failures
```bash
# Clean rebuild
docker compose -f docker/docker-compose.yml down
docker system prune -a  # Warning: removes all unused images
docker compose -f docker/docker-compose.yml build --no-cache
docker compose -f docker/docker-compose.yml up
```

## Key Configuration Files

### docker-compose.yml
- **Backend service**: Requires `privileged: true` for camera access
- **Volumes**: Must mount `/dev/video0` and `/dev/video1`
- **Runtime**: Uses `nvidia` runtime for GPU acceleration

### Backend Configuration
- **Camera fallback**: Tries `/dev/video0`, `/dev/video1`, then indices 0, 1
- **Retry logic**: 5 attempts with exponential backoff
- **Startup validation**: Tests camera before marking service as ready

### Environment Variables (.env)
```bash
CAMERA_DEVICE=/dev/video0
CAMERA_INIT_TIMEOUT=30
CAMERA_RETRY_COUNT=5
JETSON_MODE=true
USE_TENSORRT=true
USE_GPU=true
```

## Monitoring and Access

### Access Points
- **Frontend**: http://<jetson-ip>:3000
- **Backend API**: http://<jetson-ip>:8000
- **API Documentation**: http://<jetson-ip>:8000/docs
- **Camera Test**: http://<jetson-ip>:8000/api/camera/test

### Health Checks
```bash
# From Jetson
curl http://localhost:8000/api/health
curl http://localhost:8000/api/camera/test

# From remote machine
curl http://<jetson-ip>:8000/api/health
```

### Viewing Logs
```bash
# All services
docker compose -f docker/docker-compose.yml logs

# Backend only (with follow)
docker compose -f docker/docker-compose.yml logs -f backend

# Last 100 lines
docker compose -f docker/docker-compose.yml logs --tail=100 backend
```

## Architecture Notes

### Why Privileged Mode?
The backend container runs in privileged mode to access `/dev/video*` devices. This is required for:
- Direct hardware access to USB cameras
- V4L2 (Video4Linux2) operations
- Setting device permissions

### Camera Initialization Flow
1. Startup script (`docker/startup.sh`) checks for video devices
2. Sets permissions on `/dev/video*`
3. Exports CAMERA_DEVICE environment variable
4. VideoProcessor tries multiple devices with retry logic
5. Validates frames are actually captured
6. Frontend auto-falls back between MJPEG and WebSocket

### Streaming Methods
1. **MJPEG**: Primary method at `/api/streams/video/live`
2. **WebSocket**: Fallback at `/api/streams/ws/video`
3. Frontend automatically switches based on connectivity

## Commands Reference

### Start/Stop
```bash
# Start
./start.sh
# OR
docker compose -f docker/docker-compose.yml up

# Stop
docker compose -f docker/docker-compose.yml down

# Restart
docker compose -f docker/docker-compose.yml restart
```

### Debugging
```bash
# Check running containers
docker ps

# Enter backend container
docker exec -it security-backend /bin/bash

# Check camera inside container
docker exec security-backend ls -la /dev/video*
docker exec security-backend v4l2-ctl --list-devices
```

### Updates
```bash
# Pull latest code
git pull origin main

# Rebuild specific service
docker compose -f docker/docker-compose.yml build backend

# Full rebuild
docker compose -f docker/docker-compose.yml build --no-cache
```

## Important Reminders

1. **This runs on Jetson Nano**, not development computer
2. **Use `docker compose`** (space), not `docker-compose` (hyphen)
3. **Camera at `/dev/video0`** - IC800 1080P HD USB camera
4. **Privileged mode required** for camera hardware access
5. **Pull from GitHub** to update Jetson after pushing changes

## Troubleshooting Checklist

- [ ] Camera physically connected to Jetson USB port
- [ ] `/dev/video0` exists on host system
- [ ] Docker Compose v2 installed (not v1)
- [ ] Using `docker compose` command (with space)
- [ ] Backend container running in privileged mode
- [ ] Volumes mounted correctly in docker-compose.yml
- [ ] Frontend accessing correct API endpoints
- [ ] Firewall/network allows ports 3000, 8000

## Contact
Repository: https://github.com/Superman-arch/airbnb-monitor