# Roboflow Inference Setup for Door Detection

This system uses Roboflow's local inference server for accurate ML-based door detection. It runs completely offline on your Jetson Nano after initial setup.

## Features
- **96.7% accuracy** door open/closed detection
- **No internet required** after setup (runs locally)
- **No API keys needed** for runtime
- **Auto-starts on boot**
- **GPU accelerated** on Jetson Nano

## Quick Setup (Jetson Nano)

1. **Run the setup script:**
   ```bash
   chmod +x setup_inference_jetson.sh
   ./setup_inference_jetson.sh
   ```

2. **Start the inference server:**
   ```bash
   ./start_inference.sh
   ```

3. **Test door detection:**
   ```bash
   ./test_inference.py
   ```

4. **Run the monitoring system:**
   ```bash
   python run_optimized.py
   ```

## How It Works

1. **Inference Server** runs in Docker on port 9001
2. **Door model** ("Is My Door Open?") detects open/closed doors
3. **Python client** sends frames to local server
4. **Results** include bounding boxes and open/closed classification

## System Architecture

```
Camera → run_optimized.py → Inference Server (localhost:9001)
                ↓                    ↓
         DoorInferenceDetector   ML Model (is-my-door-open/2)
                ↓                    ↓
            Door Events         open_door / closed_door
                ↓
           Web Interface / Webhooks
```

## Configuration

In `config/settings.yaml`:
```yaml
door_detection:
  use_inference: true
  inference_url: "http://localhost:9001"
  model_id: "is-my-door-open/2"
  confidence_threshold: 0.6
```

## Advantages Over Edge Detection

| Feature | Edge Detection | ML Inference |
|---------|---------------|--------------|
| Accuracy | ~60% | 96.7% |
| Learning Phase | 30 seconds | None |
| False Alarms | Many | Rare |
| Shadow Sensitivity | High | None |
| Works Across Properties | No | Yes |

## Troubleshooting

### Server not starting
```bash
# Check Docker
docker ps
sudo systemctl status roboflow-inference

# Restart
sudo systemctl restart roboflow-inference
```

### Low FPS
```bash
# Adjust frame skipping in config
process_every_n_frames: 3  # Process every 3rd frame
```

### Connection refused
```bash
# Check if server is running
curl http://localhost:9001/health

# Check logs
sudo journalctl -u roboflow-inference -f
```

## Manual Docker Commands

```bash
# Start server manually
docker run --rm -it \
  --runtime nvidia \
  --gpus all \
  --net=host \
  roboflow/roboflow-inference-server-jetson:latest

# Stop all inference containers
docker stop $(docker ps -q --filter ancestor=roboflow/roboflow-inference-server-jetson)
```

## Performance

- **Jetson Nano**: 10-15 FPS with door detection
- **Model size**: ~25MB
- **Inference time**: ~100ms per frame
- **Memory usage**: ~500MB

## Auto-Start on Boot

The service is already configured to auto-start:
```bash
# Enable auto-start
sudo systemctl enable roboflow-inference

# Disable auto-start
sudo systemctl disable roboflow-inference
```