#!/bin/bash

echo "====================================="
echo "Install PyTorch with CUDA for Jetson Orin"
echo "JetPack 6 / CUDA 12.6 Compatible"
echo "====================================="
echo ""

# Check current PyTorch
echo "Current PyTorch installation:"
python3 -c "import torch; print(f'Version: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch not installed"
echo ""

# Set CUDA environment first
echo "Setting CUDA environment variables..."
export CUDA_HOME=/usr/local/cuda-12.6
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.6/bin:$PATH

# Verify CUDA
if [ -d "/usr/local/cuda-12.6" ]; then
    echo "✓ CUDA 12.6 found at /usr/local/cuda-12.6"
elif [ -d "/usr/local/cuda" ]; then
    echo "✓ CUDA found at /usr/local/cuda"
    export CUDA_HOME=/usr/local/cuda
else
    echo "⚠ CUDA directory not found. Please check installation."
fi

echo ""
echo "Removing old PyTorch..."
pip3 uninstall torch torchvision torchaudio -y

echo ""
echo "Installing PyTorch with CUDA support for Jetson Orin..."
echo "This will download ~150MB and may take a few minutes..."

# For JetPack 6.0 with Python 3.10
TORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.3.0-cp310-cp310-linux_aarch64.whl"

# Download if not exists
if [ ! -f "torch-2.3.0-cp310-cp310-linux_aarch64.whl" ]; then
    echo "Downloading PyTorch 2.3.0 for Jetson..."
    wget $TORCH_URL
else
    echo "Using existing PyTorch wheel..."
fi

# Install PyTorch
echo "Installing PyTorch..."
pip3 install torch-2.3.0-cp310-cp310-linux_aarch64.whl

# Install torchvision
echo "Installing torchvision..."
pip3 install torchvision

# Install other dependencies
echo "Installing additional dependencies..."
pip3 install numpy pillow

echo ""
echo "Verifying installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
else:
    print('ERROR: CUDA still not detected!')
    print('Try: source ~/.bashrc and run again')
"

echo ""
echo "====================================="
echo "Installation Complete!"
echo "====================================="
echo ""
echo "If CUDA is still not detected:"
echo "1. Add to ~/.bashrc:"
echo "   export CUDA_HOME=/usr/local/cuda-12.6"
echo "   export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:\$LD_LIBRARY_PATH"
echo "   export PATH=/usr/local/cuda-12.6/bin:\$PATH"
echo ""
echo "2. Run: source ~/.bashrc"
echo "3. Try: python3 vlm_server_simple.py"
echo "====================================="