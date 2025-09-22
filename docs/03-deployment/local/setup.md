# Complete Setup Guide

## System Requirements

### Hardware
- **GPU**: NVIDIA RTX 4090 or equivalent (24GB VRAM minimum)
- **RAM**: 32GB minimum, 64GB recommended
- **Storage**: 200GB+ for datasets and caches

### Software
- **OS**: Ubuntu 20.04+ or WSL2
- **Python**: 3.11 or 3.12 (tested with 3.11.13)
- **CUDA Driver**: 525.60+ (for RTX 4090)
- **CUDA Toolkit**: 12.1 (MUST match PyTorch's CUDA version)

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/clarity-digital-twin/brain-go-brr-v2.git
cd brain-go-brr-v2

# 2. Install base dependencies
make setup

# 3. Install CUDA 12.1 toolkit (if needed)
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-1

# 4. Install GPU extensions
make setup-gpu

# 5. Start training
make train-local
```

## Critical Version Requirements

**These exact versions are required - DO NOT CHANGE:**

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.11.13 | 3.12 works but less tested |
| PyTorch | 2.2.2+cu121 | MUST use CUDA 12.1 build |
| CUDA Toolkit | 12.1 | MUST match PyTorch |
| mamba-ssm | 2.2.2 | 2.2.4/2.2.5 have bugs |
| causal-conv1d | 1.4.0 | 1.5+ requires PyTorch 2.4+ |
| numpy | <2.0 | 2.x breaks mamba-ssm |

## Why UV Can't Install GPU Packages

**This is a known limitation, not a bug:**

1. **mamba-ssm and causal-conv1d require PyTorch at BUILD time**
2. **UV's build isolation prevents access to installed PyTorch**
3. **Solution**: Use `--no-build-isolation` flag

This is standard for PyTorch CUDA extensions (Flash-Attention, Apex, DeepSpeed, etc.)

## Installation Details

### Step 1: Base Setup
```bash
make setup  # Installs PyTorch and core dependencies
```

### Step 2: Verify CUDA
```bash
# Check PyTorch CUDA version
.venv/bin/python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"
# Should show: 12.1

# Check system CUDA toolkit
nvcc --version
# Should show: release 12.1
```

### Step 3: Install GPU Extensions
```bash
# Set environment
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install with --no-build-isolation
uv pip install --no-build-isolation causal-conv1d==1.4.0
uv pip install --no-build-isolation mamba-ssm==2.2.2
```

Or simply: `make setup-gpu`

### Step 4: Verify Installation
```bash
.venv/bin/python -c "
from mamba_ssm import Mamba2
import torch
print('✅ Mamba-SSM working!')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
"
```

## Running Training

### ✅ CORRECT Methods

```bash
# Direct Python
.venv/bin/python -m src train configs/local/train.yaml

# Via Makefile
make train-local

# In tmux
tmux new -d -s train_full '.venv/bin/python -m src train configs/local/train.yaml'
```

### ❌ INCORRECT Methods

```bash
# DON'T USE - UV tries to rebuild GPU packages
uv run python -m src train configs/local/train.yaml
```

## Dataset Setup

### Download TUSZ
1. Create account at physionet.org
2. Download TUH EEG Seizure Corpus v2.0.0
3. Place files in `data_ext4/tusz/edf/train/`

### Cache Building
First run builds cache (4-6 hours):
- **Train**: `cache/tusz/train/` (3734 files + manifest.json)
- **Val**: `cache/tusz/val/` (933 files, no manifest)

## Troubleshooting

### "Mamba-SSM not available, using fallback"
- **Cause**: mamba-ssm not installed or wrong version
- **Fix**: Run `make setup-gpu`

### ImportError: undefined symbol
- **Cause**: CUDA toolkit version mismatch
- **Fix**: Install CUDA 12.1 toolkit, rebuild packages

### UV build fails
- **Expected**: UV can't build GPU packages
- **Fix**: Use `make setup-gpu` instead of `uv sync --extra gpu`

### WSL2 multiprocessing hangs
- **Fix**: Configs already set `num_workers: 0`

## Environment Variables

Add to `~/.bashrc`:
```bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## Performance Expectations

### RTX 4090 (24GB VRAM)
- Batch size 8: ~0.5s per batch
- Smoke test: ~5 minutes
- Full training: 16-20 hours

### CPU Only
- **NOT SUPPORTED** - Would take ~4 years