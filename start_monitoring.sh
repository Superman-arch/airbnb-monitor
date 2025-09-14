#!/bin/bash

echo "====================================="
echo "Airbnb Monitoring System Startup"
echo "====================================="
echo ""
echo "Starting all services for complete monitoring..."
echo ""

# Function to check if port is in use
check_port() {
    netstat -tln | grep -q ":$1 "
    return $?
}

# Function to kill process on port
kill_port() {
    echo "Stopping any process on port $1..."
    fuser -k $1/tcp 2>/dev/null || true
}

# Clean up any existing services
echo "Cleaning up existing services..."
pkill -f vlm_pipeline.py 2>/dev/null
pkill -f run_optimized.py 2>/dev/null
docker stop roboflow_inference 2>/dev/null
docker rm roboflow_inference 2>/dev/null

echo ""
echo "Step 1: Starting VLM Server"
echo "----------------------------"
if check_port 8080; then
    echo "Port 8080 in use, cleaning up..."
    kill_port 8080
    sleep 2
fi

echo "Starting VLM pipeline server..."
nohup python3 vlm_pipeline.py > logs/vlm.log 2>&1 &
VLM_PID=$!
echo "VLM PID: $VLM_PID"

# Wait for VLM to be ready
echo "Waiting for VLM to initialize..."
for i in {1..30}; do
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo "✓ VLM server ready!"
        break
    fi
    echo -n "."
    sleep 1
done

if ! curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo "⚠ VLM server not responding, but continuing..."
fi

echo ""
echo "Step 2: Starting Roboflow Inference Server"
echo "-------------------------------------------"
echo "Checking for Roboflow Inference Server..."

# Check if inference server is needed
USE_INFERENCE=$(grep "use_inference:" config/settings.yaml | awk '{print $2}')
if [ "$USE_INFERENCE" = "true" ]; then
    echo "Starting Roboflow Inference Server..."
    docker run -d --rm \
        --name roboflow_inference \
        -p 9001:9001 \
        roboflow/roboflow-inference-server-cpu:latest 2>/dev/null || \
    echo "Inference server already running or not available"
    
    # Wait for inference server
    echo "Waiting for Inference Server..."
    for i in {1..20}; do
        if curl -s http://localhost:9001/health > /dev/null 2>&1; then
            echo "✓ Inference server ready!"
            break
        fi
        echo -n "."
        sleep 1
    done
else
    echo "Inference server disabled in config"
fi

echo ""
echo "Step 3: Starting Main Monitoring System"
echo "----------------------------------------"
echo "This will:"
echo "  - Auto-calibrate door zones (first run)"
echo "  - Start person tracking"
echo "  - Begin door monitoring"
echo "  - Launch web interface on port 5000"
echo ""

# Create logs directory if not exists
mkdir -p logs
mkdir -p storage/videos
mkdir -p storage/snapshots
mkdir -p config

# Start the main monitoring system
echo "Starting Airbnb Monitor..."
python3 run_optimized.py 2>&1 | tee logs/monitor.log &
MONITOR_PID=$!
echo "Monitor PID: $MONITOR_PID"

# Wait for web interface to be ready
echo ""
echo "Waiting for web interface..."
for i in {1..30}; do
    if curl -s http://localhost:5000/test > /dev/null 2>&1; then
        echo "✓ Web interface ready!"
        break
    fi
    echo -n "."
    sleep 1
done

echo ""
echo "====================================="
echo "System Status"
echo "====================================="
echo ""

# Check all services
echo "Service Status:"
if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo "  ✓ VLM Server: Running (port 8080)"
else
    echo "  ✗ VLM Server: Not responding"
fi

if [ "$USE_INFERENCE" = "true" ]; then
    if curl -s http://localhost:9001/health > /dev/null 2>&1; then
        echo "  ✓ Inference Server: Running (port 9001)"
    else
        echo "  ⚠ Inference Server: Not responding (using fallback)"
    fi
fi

if curl -s http://localhost:5000/test > /dev/null 2>&1; then
    echo "  ✓ Web Interface: Running (port 5000)"
else
    echo "  ✗ Web Interface: Not responding"
fi

# Get IP address
IP_ADDR=$(hostname -I | awk '{print $1}')

echo ""
echo "====================================="
echo "Access Your Monitoring System"
echo "====================================="
echo ""
echo "Web Dashboard: http://$IP_ADDR:5000"
echo "              http://localhost:5000"
echo ""
echo "Features:"
echo "  • Live camera feed with door zones"
echo "  • Real-time door open/close detection"
echo "  • Person tracking and room occupancy"
echo "  • Event history and journey tracking"
echo "  • 48-hour video recording"
echo ""
echo "First Run:"
echo "  Click 'Calibrate Doors' to map your environment"
echo ""
echo "Logs:"
echo "  VLM: logs/vlm.log"
echo "  Monitor: logs/monitor.log"
echo ""
echo "To stop all services:"
echo "  ./stop_monitoring.sh"
echo ""
echo "====================================="
echo "Monitoring Active!"
echo "====================================="

# Save PIDs for stop script
echo "$VLM_PID" > .vlm.pid
echo "$MONITOR_PID" > .monitor.pid

# Keep script running
echo ""
echo "Press Ctrl+C to stop all services..."
trap 'echo "Stopping services..."; kill $VLM_PID $MONITOR_PID 2>/dev/null; docker stop roboflow_inference 2>/dev/null; exit' INT
wait