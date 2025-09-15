#!/bin/bash

echo "====================================="
echo "PyTorch Fix for Jetson Nano"
echo "====================================="
echo ""

# Check current PyTorch version
echo "Current PyTorch version:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
echo ""

echo "Choose an option:"
echo "1) Try to run VLM with compatibility patch (recommended)"
echo "2) Install PyTorch 2.1 for Jetson (has float8 support)"
echo "3) Install latest PyTorch from pip"
echo ""
read -p "Enter choice (1-3): " CHOICE

case $CHOICE in
    1)
        echo ""
        echo "Testing VLM with compatibility patch..."
        python3 vlm_server_native.py
        ;;
        
    2)
        echo ""
        echo "Installing PyTorch 2.1 for Jetson..."
        echo "This may take a while..."
        
        # Uninstall old versions
        pip3 uninstall torch torchvision torchaudio -y
        
        # Download and install PyTorch 2.1 for Jetson
        # For Python 3.10 on JetPack 5.x
        wget https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp310-cp310-linux_aarch64.whl
        pip3 install torch-2.1.0a0+41361538.nv23.06-cp310-cp310-linux_aarch64.whl
        
        # Install torchvision
        pip3 install torchvision
        
        echo ""
        echo "New PyTorch version:"
        python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
        ;;
        
    3)
        echo ""
        echo "Installing latest PyTorch from pip..."
        pip3 install --upgrade torch torchvision
        
        echo ""
        echo "New PyTorch version:"
        python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
        ;;
        
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "====================================="
echo "Done! Now try running:"
echo "  python3 vlm_server_native.py"
echo "====================================="