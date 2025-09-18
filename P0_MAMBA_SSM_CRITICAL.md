# P0: MAMBA-SSM INSTALLATION CRITICAL BLOCKER

## Current Status: CRITICAL - Bi-Mamba-2 is CORE to our architecture!
Date: 2025-01-18
Priority: P0 - Our entire competitive advantage depends on Bi-Mamba-2

## The Problem
mamba-ssm fails with undefined symbol error:
```
ImportError: undefined symbol: _ZN2at4_ops10zeros_like4callERKNS_6TensorE...
```

This means our Bi-Mamba-2 layers are falling back to Conv1d which:
- Is NOT functionally equivalent
- Loses the O(N) sequence modeling advantage
- Defeats the entire purpose of our architecture!

## Root Cause
PyTorch version mismatch! mamba-ssm compiled extensions are incompatible with:
- PyTorch 2.0.1 (too old)
- PyTorch 2.1.0 (still broken)
- PyTorch 2.3+ (too new, breaks ABI)

## THE SOLUTION

### Option 1: PyTorch 2.2.2 (RECOMMENDED)
```bash
# Uninstall everything
uv pip uninstall torch torchvision torchaudio mamba-ssm causal-conv1d

# Install PyTorch 2.2.2 with CUDA 12.1
uv pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# Install mamba-ssm fresh
uv pip install mamba-ssm --no-cache-dir

# Test it
python -c "from mamba_ssm import Mamba; print('MAMBA WORKING!')"
```

### Option 2: Build from source (if Option 1 fails)
```bash
# Install build dependencies
sudo apt-get install ninja-build

# Clone and build
git clone https://github.com/state-spaces/mamba.git
cd mamba
pip install -e .
```

## Environment Details
- **GPU**: RTX 4090 (CUDA 12.8 available)
- **WSL2**: Linux 5.15.167.4-microsoft-standard-WSL2
- **Python**: 3.10
- **Current PyTorch**: 2.1.0+cu118 (WRONG VERSION!)

## Why This Matters
Our ENTIRE architecture advantage is Bi-Mamba-2:
- Transformers: O(N²) complexity → unusable for long EEG
- Pure CNNs: No global context → miss seizure patterns
- **Bi-Mamba-2: O(N) with global context → THE GAME CHANGER**

Without mamba-ssm working:
- We're just another CNN architecture
- No competitive advantage
- Training will be WRONG

## Testing After Fix
```python
import torch
from mamba_ssm import Mamba

# Should work without errors
model = Mamba(
    d_model=512,
    d_state=16,
    d_conv=4,
    expand=2
).cuda()

x = torch.randn(1, 100, 512).cuda()
y = model(x)
print(f"Mamba output: {y.shape}")  # Should be [1, 100, 512]
```

## STATUS: FIXING NOW...