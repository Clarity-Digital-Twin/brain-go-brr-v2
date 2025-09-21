#!/bin/bash
# Fix mamba-ssm by rebuilding with correct CUDA 12.1

echo "=== Fixing mamba-ssm with CUDA 12.1 ==="

# Set environment to use CUDA 12.1
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Verify CUDA 12.1 is available
if [ ! -d "$CUDA_HOME" ]; then
    echo "❌ CUDA 12.1 not found at $CUDA_HOME"
    echo "Please run: sudo ./install_cuda_121.sh first"
    exit 1
fi

echo "Using CUDA from: $CUDA_HOME"
nvcc --version

# Clean uninstall old mamba-ssm and causal-conv1d
echo ""
echo "1. Uninstalling broken packages..."
uv pip uninstall mamba-ssm causal-conv1d

# Ensure build dependencies
echo ""
echo "2. Installing build dependencies..."
uv pip install --upgrade pip setuptools wheel ninja

# Check PyTorch version and CUDA
echo ""
echo "3. Checking PyTorch configuration..."
uv run python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'ABI: {\"TRUE\" if torch._C._GLIBCXX_USE_CXX11_ABI else \"FALSE\"}')"

# Build causal-conv1d first (mamba dependency)
echo ""
echo "4. Building causal-conv1d..."
uv pip install --no-build-isolation --no-cache-dir causal-conv1d

# Build mamba-ssm
echo ""
echo "5. Building mamba-ssm..."
uv pip install --no-build-isolation --no-cache-dir mamba-ssm

# Verify it works
echo ""
echo "6. Verifying installation..."
uv run python -c "
import torch
from mamba_ssm import Mamba2
print('✅ Mamba-SSM loaded successfully!')

# Test on GPU
device = torch.device('cuda')
model = Mamba2(d_model=512, d_state=16, d_conv=4, expand=2).to(device)
x = torch.randn(1, 1024, 512).to(device)
y = model(x)
print(f'✅ GPU test passed! Output shape: {y.shape}')
print('✅ Mamba-SSM is working with GPU acceleration!')
"

echo ""
echo "=== Fix complete! ==="
echo ""
echo "You can now run training with real Bi-Mamba-2:"
echo "python -m src train configs/local/train.yaml"