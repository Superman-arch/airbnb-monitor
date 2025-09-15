#!/bin/bash

echo "====================================="
echo "Stopping Airbnb Monitoring System"
echo "====================================="
echo ""

# Stop VLM server
if [ -f .vlm.pid ]; then
    VLM_PID=$(cat .vlm.pid)
    echo "Stopping VLM server (PID: $VLM_PID)..."
    kill $VLM_PID 2>/dev/null
    rm .vlm.pid
else
    echo "Stopping any VLM processes..."
    pkill -f vlm_pipeline.py 2>/dev/null
    pkill -f vlm_server 2>/dev/null
fi

# Stop monitor
if [ -f .monitor.pid ]; then
    MONITOR_PID=$(cat .monitor.pid)
    echo "Stopping monitor (PID: $MONITOR_PID)..."
    kill $MONITOR_PID 2>/dev/null
    rm .monitor.pid
else
    echo "Stopping any monitor processes..."
    pkill -f run_optimized.py 2>/dev/null
fi

# Stop Roboflow inference
echo "Stopping Roboflow Inference Server..."
docker stop roboflow_inference 2>/dev/null
docker rm roboflow_inference 2>/dev/null

# Clean up any remaining processes
echo "Cleaning up remaining processes..."
pkill -f "python3.*airbnb" 2>/dev/null
fuser -k 5000/tcp 2>/dev/null
fuser -k 8080/tcp 2>/dev/null
fuser -k 9001/tcp 2>/dev/null

echo ""
echo "All services stopped."
echo ""
echo "To restart: ./start_monitoring.sh"
echo "====================================="