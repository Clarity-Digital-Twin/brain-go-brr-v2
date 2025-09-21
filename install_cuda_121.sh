#!/bin/bash
# Install CUDA 12.1 toolkit to match PyTorch cu121

echo "Installing CUDA 12.1 toolkit..."

# Add NVIDIA package repositories
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA 12.1 toolkit (just the toolkit, not the driver)
sudo apt-get install -y cuda-toolkit-12-1

# Verify installation
if [ -d "/usr/local/cuda-12.1" ]; then
    echo "✅ CUDA 12.1 installed successfully"
    /usr/local/cuda-12.1/bin/nvcc --version
else
    echo "❌ CUDA 12.1 installation failed"
    exit 1
fi

echo ""
echo "Add these to your ~/.bashrc:"
echo "export CUDA_HOME=/usr/local/cuda-12.1"
echo "export PATH=\$CUDA_HOME/bin:\$PATH"
echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"