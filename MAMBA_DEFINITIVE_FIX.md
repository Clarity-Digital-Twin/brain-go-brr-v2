# The DEFINITIVE Mamba-SSM Fix

## The Problem
**ABI mismatch**: PyTorch built with CUDA 12.1, but mamba-ssm compiled with system CUDA 12.6

## The Solution
**Install CUDA 12.1 toolkit and rebuild mamba-ssm**

## Step 1: Install CUDA 12.1 Toolkit
```bash
sudo ./install_cuda_121.sh
```
This installs CUDA 12.1 toolkit alongside your existing CUDA 12.6.

## Step 2: Fix mamba-ssm
```bash
./fix_mamba_ssm.sh
```
This will:
1. Set CUDA_HOME to 12.1
2. Uninstall broken mamba-ssm
3. Rebuild with correct CUDA version
4. Verify GPU acceleration works

## Why This Works
- **PyTorch 2.2.2+cu121** expects extensions built with CUDA 12.1
- **System has CUDA 12.6** → mismatch!
- **Solution**: Install CUDA 12.1 toolkit for building extensions
- **Key flag**: `--no-build-isolation` ensures using correct PyTorch headers

## After Fix
Training will show:
- ✅ No more "Mamba-SSM not available, using fallback"
- ✅ Real Bi-Mamba-2 with O(N) complexity
- ✅ GPU-accelerated selective scan

## The Scripts
1. **install_cuda_121.sh** - Installs CUDA 12.1 toolkit
2. **fix_mamba_ssm.sh** - Rebuilds mamba-ssm with correct CUDA

Run them in order. The whole process takes ~5-10 minutes.