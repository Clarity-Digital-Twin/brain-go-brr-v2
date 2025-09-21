# GPU Setup Requirements

## Prerequisites
- NVIDIA GPU with CUDA capability >= 7.0
- PyTorch 2.2.2 with CUDA 12.1 (`torch==2.2.2+cu121`)
- **CUDA Toolkit 12.1** for building extensions

## Critical: CUDA Version Must Match PyTorch

If you see this error:
```
ImportError: undefined symbol: _ZN3c104cuda14ExchangeDeviceEa
```

Your CUDA toolkit version doesn't match PyTorch's CUDA version!

### Check your versions:
```bash
# PyTorch CUDA version
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"

# System CUDA version
nvcc --version
```

These MUST match (e.g., both 12.1)!

## Installation Steps

### 1. Install correct CUDA Toolkit
```bash
# For PyTorch cu121, install CUDA 12.1:
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-1
```

### 2. Set environment variables
Add to `~/.bashrc`:
```bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### 3. Install GPU dependencies
```bash
# Install with GPU extras
uv sync --extra gpu

# If mamba-ssm fails, rebuild from source:
export CUDA_HOME=/usr/local/cuda-12.1
uv pip uninstall mamba-ssm causal-conv1d
uv pip install --no-build-isolation causal-conv1d
uv pip install --no-build-isolation mamba-ssm
```

## Verification
```python
from mamba_ssm import Mamba2
print("✅ Mamba-SSM loaded successfully!")
```

## Common Issues

### Issue: "Mamba-SSM not available, using fallback"
**Cause**: mamba-ssm not installed or failed to build
**Fix**: Follow installation steps above

### Issue: Undefined symbol errors
**Cause**: CUDA version mismatch
**Fix**: Install matching CUDA toolkit version

### Issue: Build takes forever
**Normal**: Building CUDA extensions takes 10-15 minutes

## For Contributors
- Always check CUDA compatibility before training
- The Conv1d fallback is NOT functionally equivalent to Mamba
- Training without proper Mamba-SSM defeats the purpose of this architecture