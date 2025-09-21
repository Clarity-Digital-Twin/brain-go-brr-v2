# Mamba-SSM GPU Fix Documentation

## The Problem
**mamba-ssm 2.2.5 fails with undefined symbol error:**
```
ImportError: selective_scan_cuda.cpython-311-x86_64-linux-gnu.so:
undefined symbol: _ZN3c104cuda14ExchangeDeviceEa
```

## Root Cause
Binary incompatibility between prebuilt mamba-ssm wheel and our PyTorch version.

## Our Current Stack
- **Python**: 3.11.13
- **PyTorch**: 2.2.2+cu121 (CUDA 12.1)
- **System CUDA**: 12.6 (nvcc)
- **GPU**: RTX 4090 (Capability 8.9)
- **mamba-ssm**: 2.2.5 (broken)

## The Issue
mamba-ssm 2.2.5 was compiled against a different PyTorch version, causing ABI mismatch.

## Solution Options

### Option 1: Downgrade mamba-ssm (QUICKEST)
```bash
# Uninstall broken version
uv pip uninstall mamba-ssm -y

# Install working version
uv pip install mamba-ssm==2.2.4
```

### Option 2: Build from Source (MOST RELIABLE)
```bash
# Uninstall broken version
uv pip uninstall mamba-ssm causal-conv1d -y

# Set CUDA environment
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Build from source
git clone https://github.com/state-spaces/mamba.git /tmp/mamba
cd /tmp/mamba
git checkout v2.2.2  # Use stable version matching PyTorch 2.2.x
uv pip install .
cd -
```

### Option 3: Use Specific Prebuilt Wheel
```bash
# Check ABI version
uv run python -c "import torch; print('ABI:', 'TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')"

# Download matching wheel from:
# https://github.com/state-spaces/mamba/releases
# Look for: mamba_ssm-2.2.2+cu121torch2.2cxx11abi[TRUE/FALSE]-cp311-cp311-linux_x86_64.whl

# Install specific wheel
uv pip install [downloaded_wheel_file]
```

## Verification
```python
from mamba_ssm import Mamba2
print("✅ Mamba-SSM loaded successfully!")

# Test CUDA kernels
import torch
device = torch.device("cuda")
model = Mamba2(d_model=512).to(device)
x = torch.randn(1, 1024, 512).to(device)
y = model(x)
print(f"✅ Output shape: {y.shape}")
```

## Key Insights
1. **mamba-ssm 2.2.5 has known issues** with undefined symbols
2. **Version 2.2.4 works** for most users
3. **Building from source** ensures compatibility but takes time
4. **PyTorch 2.2.x + CUDA 12.1** is a supported combination

## Recommended Fix
For immediate results: **Downgrade to mamba-ssm 2.2.4**

This is what most users reported working, and it's the fastest solution.