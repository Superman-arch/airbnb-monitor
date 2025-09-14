#!/bin/bash

echo "====================================="
echo "Jetson Nano Super Optimization"
echo "For 8GB RAM - Phi-3.5 Vision Setup"
echo "====================================="
echo ""

# Check current memory
echo "Current memory status:"
free -h
echo ""

# Function to setup swap
setup_swap() {
    echo "Setting up 8GB swap space..."
    
    # Check if swap already exists
    if [ -f /swapfile ]; then
        echo "Swap file already exists. Checking size..."
        SWAP_SIZE=$(du -h /swapfile | cut -f1)
        echo "Current swap size: $SWAP_SIZE"
        
        read -p "Do you want to recreate swap? (y/n): " RECREATE
        if [ "$RECREATE" = "y" ]; then
            sudo swapoff /swapfile
            sudo rm /swapfile
        else
            return
        fi
    fi
    
    # Create 8GB swap
    echo "Creating 8GB swap file..."
    sudo fallocate -l 8G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    
    # Make permanent
    if ! grep -q "/swapfile" /etc/fstab; then
        echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    fi
    
    echo "✓ Swap configured"
}

# Function to optimize system
optimize_system() {
    echo "Optimizing system settings..."
    
    # Memory overcommit - allows more aggressive memory allocation
    sudo sysctl -w vm.overcommit_memory=1
    sudo sysctl -w vm.overcommit_ratio=80
    
    # Swappiness - use swap more aggressively
    sudo sysctl -w vm.swappiness=60
    
    # Clear cache
    sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"
    
    echo "✓ System optimized"
}

# Function to stop unnecessary services
stop_services() {
    echo "Stopping unnecessary services to free RAM..."
    
    # List of services to stop
    SERVICES="cups bluetooth avahi-daemon snapd"
    
    for service in $SERVICES; do
        if systemctl is-active --quiet $service; then
            echo "Stopping $service..."
            sudo systemctl stop $service
        fi
    done
    
    # Disable GUI if running
    if [ "$XDG_CURRENT_DESKTOP" != "" ]; then
        echo "Note: GUI is running. For maximum performance, consider:"
        echo "  sudo systemctl stop gdm3"
        echo "  (This will close the desktop environment)"
    fi
    
    echo "✓ Services stopped"
}

# Function to set GPU memory split
set_gpu_memory() {
    echo "Setting optimal GPU/CPU memory split..."
    
    # Set maximum GPU memory growth
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    
    # Limit GPU memory to leave room for CPU
    export CUDA_VISIBLE_DEVICES=0
    export CUDA_LAUNCH_BLOCKING=1
    
    echo "✓ GPU memory configured"
}

# Main menu
echo "Select optimization level:"
echo "1) Quick optimize (recommended)"
echo "2) Full optimize with swap"
echo "3) Maximum performance (stops GUI)"
echo "4) Check current status only"
echo ""
read -p "Enter choice (1-4): " CHOICE

case $CHOICE in
    1)
        echo ""
        optimize_system
        stop_services
        set_gpu_memory
        ;;
        
    2)
        echo ""
        setup_swap
        optimize_system
        stop_services
        set_gpu_memory
        ;;
        
    3)
        echo ""
        echo "WARNING: This will stop the GUI!"
        read -p "Continue? (y/n): " CONFIRM
        if [ "$CONFIRM" = "y" ]; then
            setup_swap
            optimize_system
            stop_services
            sudo systemctl stop gdm3 2>/dev/null
            sudo systemctl stop lightdm 2>/dev/null
            set_gpu_memory
        fi
        ;;
        
    4)
        echo ""
        echo "System Status:"
        echo "--------------"
        free -h
        echo ""
        echo "Swap status:"
        swapon --show
        echo ""
        echo "GPU status:"
        nvidia-smi 2>/dev/null || echo "nvidia-smi not available"
        exit 0
        ;;
        
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "====================================="
echo "Optimization Complete!"
echo "====================================="
echo ""
echo "Memory after optimization:"
free -h
echo ""
echo "You can now run:"
echo "  python3 vlm_server_optimized.py"
echo ""
echo "To restore services later:"
echo "  sudo systemctl start cups bluetooth"
echo "====================================="