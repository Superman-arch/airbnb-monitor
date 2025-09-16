# Deployment Instructions - Detection System Fixes

## Issues Fixed

1. **Detection Method Errors**
   - Fixed `monitoring.py` calling wrong methods on detector objects
   - Changed `door_detector.detect()` to `door_detector.process_frame()`
   - Changed `person_tracker.update()` to `person_tracker.detect()`
   - Changed `zone_detector.detect_motion_zones()` to `zone_detector.detect()`

2. **Redis Connection Error**
   - Fixed URL parsing in `redis_client.py` to properly extract host from Redis URL
   - Now uses `urllib.parse` instead of string splitting

3. **WebSocket Connection Issues**
   - Added `/ws` location to main nginx.conf for WebSocket proxy
   - This fixes the 502 errors when frontend tries to connect

## Deployment Steps

### On Your Development Machine

1. **Commit the changes:**
```bash
git add -A
git commit -m "Fix detection system errors and WebSocket connections

- Fix method calls in monitoring.py for all detectors
- Fix Redis URL parsing to properly extract host
- Add WebSocket proxy configuration to nginx"
git push origin main
```

### On Your Jetson Nano

1. **Pull the latest changes:**
```bash
cd ~/airbnb-monitor
git pull origin main
```

2. **Stop the current containers:**
```bash
docker compose -f docker/docker-compose.yml down
```

3. **Rebuild the backend container with the fixes:**
```bash
docker compose -f docker/docker-compose.yml build backend
```

4. **Restart all services:**
```bash
docker compose -f docker/docker-compose.yml up -d
```

5. **Check the logs to verify fixes:**
```bash
# Watch backend logs for errors
docker compose -f docker/docker-compose.yml logs -f backend

# In another terminal, check if detection errors are gone
docker compose -f docker/docker-compose.yml logs backend | grep -i error
```

## Verification

After deployment, you should see:
- ✅ No more "object has no attribute" errors
- ✅ Redis connects successfully to `redis:6379`
- ✅ WebSocket connections work (no 502 errors)
- ✅ Detection systems processing frames properly

## Access Points

- Frontend: http://<jetson-ip>
- Backend API: http://<jetson-ip>:8000
- API Docs: http://<jetson-ip>:8000/docs
- Grafana: http://<jetson-ip>:3001 (admin/admin123)

## Troubleshooting

If issues persist:

1. **Check container status:**
```bash
docker ps -a
```

2. **Verify Redis is running:**
```bash
docker exec security-backend redis-cli -h redis -a redis123 ping
```

3. **Test WebSocket connection:**
```bash
# From Jetson
curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" http://localhost/ws
```

4. **Full rebuild if needed:**
```bash
docker compose -f docker/docker-compose.yml down
docker system prune -a  # Warning: removes all unused images
docker compose -f docker/docker-compose.yml build --no-cache
docker compose -f docker/docker-compose.yml up
```