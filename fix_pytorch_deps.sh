#!/bin/bash

echo "====================================="
echo "Fix PyTorch Dependencies for Jetson"
echo "====================================="
echo ""

# Function to clean install
clean_install() {
    echo "Cleaning up PyTorch installations..."
    
    # Uninstall all torch-related packages
    pip3 uninstall torch torchvision torchaudio bitsandbytes transformers -y
    
    # Clear pip cache
    pip3 cache purge
    
    echo "✓ Cleanup complete"
    echo ""
    
    echo "Installing compatible versions..."
    
    # Install PyTorch 2.0.1 (known to work on Jetson)
    pip3 install torch==2.0.1 torchvision==0.15.2
    
    # Install transformers
    pip3 install transformers==4.43.0
    
    # Install other dependencies
    pip3 install flask pillow accelerate psutil
    
    # Try to install bitsandbytes without upgrading torch
    pip3 install bitsandbytes --no-deps 2>/dev/null || echo "Bitsandbytes skipped (optional)"
    
    echo "✓ Installation complete"
}

# Function to verify installation
verify_install() {
    echo ""
    echo "Verifying installation..."
    python3 -c "
import torch
import torchvision
import transformers
print(f'PyTorch: {torch.__version__}')
print(f'Torchvision: {torchvision.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
}

# Main menu
echo "This will fix PyTorch dependency conflicts"
echo ""
echo "1) Clean install (recommended)"
echo "2) Just verify current installation"
echo "3) Exit"
echo ""
read -p "Enter choice (1-3): " CHOICE

case $CHOICE in
    1)
        clean_install
        verify_install
        ;;
    2)
        verify_install
        ;;
    3)
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "====================================="
echo "Done! Now try running:"
echo "  python3 vlm_server_simple.py"
echo "====================================="