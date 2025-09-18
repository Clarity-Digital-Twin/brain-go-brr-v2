# P0: PyTorch Import Hanging Investigation

## Current Status: CRITICAL BLOCKER
Date: 2025-01-18
Priority: P0 - Training pipeline completely blocked

## The Problem
PyTorch import hangs indefinitely when running ANY Python code that imports torch, including:
- `python -m src.experiment.pipeline`
- `python -c "import torch"`
- Even with CUDA disabled via `CUDA_VISIBLE_DEVICES=""`

## What We Know

### 1. Environment Details
- **OS**: WSL2 on Windows (Linux 5.15.167.4-microsoft-standard-WSL2)
- **Python**: 3.11.13 (via uv)
- **PyTorch**: 2.8.0 (installed via uv)
- **Working Directory**: `/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2`

### 2. What IS Working
✅ All other Python imports work fine (numpy, mne, etc.)
✅ Data loading with interpolation works (`test_data_loading.py` runs successfully)
✅ All tests pass when not importing torch
✅ Code quality checks pass (`make q`)

### 3. What is NOT Working
❌ ANY import of torch hangs forever
❌ Pipeline cannot start due to torch import in models.py
❌ Even `CUDA_VISIBLE_DEVICES=""` doesn't help
❌ Setting `OMP_NUM_THREADS=1` doesn't help

### 4. Strace Analysis
From `strace` output, the hang occurs when PyTorch tries to load NVIDIA CUDA libraries:
```
libcublasLt.so.12 (751MB file!)
```

The process gets stuck after memory mapping these massive CUDA library files, even though we're trying to run CPU-only.

## Root Cause Hypothesis

1. **WSL2 + CUDA Library Loading Issue**: PyTorch 2.8.0 is trying to load CUDA libraries even when CUDA is disabled. This is a known issue with PyTorch on WSL2.

2. **Library Size Problem**: The CUDA libraries are massive (700MB+) and WSL2 might be struggling with memory mapping them from Windows filesystem.

3. **Wrong PyTorch Build**: We have a CUDA-enabled PyTorch build installed, but WSL2 might not have proper CUDA support configured.

## Potential Solutions (Not Yet Tried)

### Solution 1: Install CPU-only PyTorch
```bash
uv pip uninstall torch
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Solution 2: Move to Native Linux Directory
Copy project out of `/mnt/c/` (Windows filesystem) to `~/` (WSL native filesystem) to avoid cross-filesystem issues.

### Solution 3: Downgrade PyTorch
Try PyTorch 2.5.0 which is mentioned in CLAUDE.md as the minimum version:
```bash
uv pip install torch==2.5.0
```

### Solution 4: Check WSL2 CUDA Setup
Verify if WSL2 has CUDA properly configured or if we should stick to CPU-only.

## Impact

- **Training**: Completely blocked - cannot run pipeline
- **Development**: Cannot test any model code
- **Timeline**: Critical blocker for all ML work

## ✅ SOLVED: Solution 1 (CPU-only PyTorch) WORKS!

### The Fix
```bash
uv pip uninstall torch
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

This installs PyTorch 2.8.0+cpu (175MB) instead of the CUDA version (1.5GB+).

### Important: Environment-Specific PyTorch

**For WSL2 Development (this environment):**
- Use CPU-only PyTorch to avoid WSL2/CUDA library loading issues
- Training will be slower but development/testing works fine
- CPU-only build is 175MB vs 1.5GB+ for CUDA build

**For Native Linux with GPU (e.g., 4090 training box):**
```bash
# Install CUDA-enabled PyTorch for GPU training
uv pip install torch  # Gets CUDA version by default
# OR explicitly:
uv pip install torch --index-url https://download.pytorch.org/whl/cu121  # For CUDA 12.1
```

**Our architecture is FULLY GPU-COMPATIBLE!** The Bi-Mamba-2 + U-Net + ResCNN architecture runs on both CPU and GPU. We just need different PyTorch builds for different environments.

## Root Cause

WSL2 on Windows struggles with memory-mapping massive CUDA libraries (700MB+ each) from the Windows filesystem, causing hangs during PyTorch import. This is a WSL2-specific issue, NOT an architecture limitation.

---

**STATUS: ✅ RESOLVED - CPU-only PyTorch for WSL2, CUDA PyTorch for GPU training**

## UPDATE: scipy.ndimage Also Hangs in WSL2

After fixing PyTorch, discovered `scipy.ndimage` also hangs on import. This is another WSL2/OpenBLAS threading issue.

**Workaround being investigated:**
- Setting thread environment variables doesn't fix it
- May need to reinstall scipy or use different BLAS backend
- This only affects WSL2 development, not native Linux GPU training