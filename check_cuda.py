#!/usr/bin/env python3
"""Check CUDA availability and fix common issues on Jetson."""

import sys
import os

print("Checking CUDA setup on Jetson...")
print("-" * 40)

# Check PyTorch
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        print(f"  GPU name: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("\n⚠ CUDA not detected by PyTorch!")
        print("\nPossible fixes:")
        print("1. Check CUDA_HOME:")
        print(f"   Current: {os.environ.get('CUDA_HOME', 'Not set')}")
        print("   Try: export CUDA_HOME=/usr/local/cuda")
        print("\n2. Check LD_LIBRARY_PATH:")
        print(f"   Current: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
        print("   Try: export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH")
        print("\n3. Reinstall PyTorch for Jetson:")
        print("   pip3 uninstall torch torchvision -y")
        print("   pip3 install torch==2.0.1 torchvision==0.15.2")
        
except ImportError as e:
    print(f"✗ PyTorch not installed: {e}")
    sys.exit(1)

# Check nvidia-smi
print("\n" + "-" * 40)
print("Checking nvidia-smi...")
result = os.system("nvidia-smi > /dev/null 2>&1")
if result == 0:
    print("✓ nvidia-smi available")
    os.system("nvidia-smi")
else:
    print("✗ nvidia-smi not found")
    print("  This is normal on Jetson. Use tegrastats instead:")
    os.system("tegrastats --interval 1000 | head -1")

# Check Jetson stats
print("\n" + "-" * 40)
print("Checking Jetson configuration...")
if os.path.exists("/etc/nv_tegra_release"):
    with open("/etc/nv_tegra_release", "r") as f:
        print(f"✓ Jetson info: {f.read().strip()}")
else:
    print("✗ Not running on Jetson")

# Suggest environment setup
print("\n" + "-" * 40)
print("Recommended environment setup for Jetson:")
print("")
print("Add to ~/.bashrc:")
print("export CUDA_HOME=/usr/local/cuda")
print("export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH")
print("export PATH=/usr/local/cuda/bin:$PATH")
print("")
print("Then run: source ~/.bashrc")