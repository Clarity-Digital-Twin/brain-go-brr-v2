# Complete Setup Guide for Brain-Go-Brr v2

## System Requirements

### Hardware
- **GPU**: NVIDIA RTX 4090 or equivalent (24GB VRAM minimum)
- **RAM**: 32GB minimum, 64GB recommended
- **Storage**: 200GB+ for datasets and caches

### Software
- **OS**: Ubuntu 20.04+ or WSL2
- **Python**: 3.11 or 3.12 (verified with 3.11.13)
- **CUDA Driver**: 525.60+ (for RTX 4090)
- **CUDA Toolkit**: 12.1 (MUST match PyTorch's CUDA version)

## Critical Version Requirements

### Core Dependencies (MUST match exactly)
```toml
torch = "2.2.2+cu121"  # PyTorch with CUDA 12.1
numpy = "1.26.4"       # numpy 2.x breaks mamba-ssm
mamba-ssm = "2.2.4"    # 2.2.5 has ABI issues
causal-conv1d = "1.5.2"
```

### Why These Versions Matter
- **PyTorch 2.2.2+cu121**: Latest stable with CUDA 12.1 support
- **numpy < 2.0**: mamba-ssm incompatible with numpy 2.x
- **mamba-ssm 2.2.4**: Version 2.2.5 has undefined symbol errors
- **CUDA Toolkit 12.1**: MUST match PyTorch's cu121 for building extensions

## Installation Steps

### 1. Clone Repository
```bash
git clone https://github.com/clarity-digital-twin/brain-go-brr-v2.git
cd brain-go-brr-v2
```

### 2. Install UV Package Manager
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Basic Setup
```bash
make setup  # Installs Python dependencies
```

### 4. GPU Setup (CRITICAL for training)

#### 4a. Check CUDA Compatibility
```bash
# Check PyTorch CUDA version
uv run python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"
# Should show: PyTorch CUDA: 12.1

# Check system CUDA toolkit
nvcc --version
# Should show: release 12.1
```

#### 4b. If Versions Don't Match
```bash
# Install CUDA 12.1 toolkit
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-1

# Add to ~/.bashrc
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

#### 4c. Install GPU Dependencies
```bash
make setup-gpu  # Installs mamba-ssm and verifies GPU
```

#### 4d. If mamba-ssm Fails
```bash
# Manual rebuild with correct CUDA
export CUDA_HOME=/usr/local/cuda-12.1
uv pip uninstall mamba-ssm causal-conv1d
uv pip install --no-build-isolation causal-conv1d==1.5.2
uv pip install --no-build-isolation mamba-ssm==2.2.4
```

### 5. Verify Installation
```bash
# Check all components
uv run python -c "
import torch
from mamba_ssm import Mamba2
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
print(f'✅ Mamba-SSM loaded successfully!')
"
```

## Configuration Files

### Local Configs (configs/local/)
- **smoke.yaml**: Quick 1-epoch test (5 min on RTX 4090)
- **dev.yaml**: Development testing (10 epochs)
- **train.yaml**: Full training (100 epochs, ~16-20 hours)
- **eval.yaml**: Evaluation only

### Modal/Cloud Configs (configs/modal/)
- Same as above but with cloud-specific paths
- Uses `/results/` instead of local paths
- Higher batch sizes for A100 GPUs

### Critical Config Settings
```yaml
data:
  num_workers: 0           # WSL2-critical: avoid hangs
  pin_memory: false        # WSL2-critical: /dev/shm issues
  use_balanced_sampling: true  # CRITICAL for class imbalance

model:
  mamba:
    conv_kernel: 5         # Note: CUDA kernels coerce to 4
```

## Dataset Setup

### Download TUSZ Dataset
```bash
# Create data directory
mkdir -p data_ext4/tusz/edf/train

# Download from physionet.org (requires account)
# Place .edf and .csv files in data_ext4/tusz/edf/train/
```

### Cache Building
First run will build cache (~4-6 hours):
```
cache/tusz/
├── train/         # 3734 files + manifest.json
└── val/           # 933 files (no manifest)
```

## Training

### Local Training
```bash
# Smoke test (5 min)
python -m src train configs/local/smoke.yaml

# Full training (16-20 hours)
python -m src train configs/local/train.yaml
```

### Monitor Training
```bash
# In another terminal
tensorboard --logdir results/tensorboard
```

## Common Issues

### Issue: "Mamba-SSM not available, using fallback"
**Cause**: mamba-ssm not installed or ABI mismatch
**Fix**: Follow GPU Setup step 4d above

### Issue: ImportError undefined symbol
**Cause**: CUDA version mismatch
**Fix**: Ensure CUDA toolkit matches PyTorch (both 12.1)

### Issue: OOM on WSL2
**Cause**: WSL2 memory limits
**Fix**: Create `.wslconfig` with:
```ini
[wsl2]
memory=24GB
swap=8GB
```

### Issue: Multiprocessing hangs
**Cause**: WSL2 /dev/shm issues
**Fix**: Set `num_workers: 0` in configs

## Development Workflow

### Before Making Changes
```bash
git checkout -b feature/your-feature
make test  # Ensure tests pass
```

### After Making Changes
```bash
make q     # Run quality checks (lint, format, type)
make test  # Run tests
git commit -m "feat: your changes"
```

### Code Quality Commands
```bash
make lint   # Check code style
make format # Auto-format code
make type   # Type checking
make q      # All quality checks
```

## Performance Expectations

### RTX 4090 (24GB)
- Batch size 8: ~0.5s per batch
- Smoke test: ~5 minutes
- Full training: 16-20 hours

### CPU Only (NOT RECOMMENDED)
- ~40s per batch (80x slower)
- Full training: ~4 years

## Deployment

### Modal Cloud
```bash
cd deploy/modal
modal deploy app.py
modal run app.py --config configs/modal/train_a100.yaml
```

## Final Checklist

- [ ] Python 3.11+ installed
- [ ] UV package manager installed
- [ ] CUDA 12.1 toolkit installed
- [ ] PyTorch 2.2.2+cu121 installed
- [ ] mamba-ssm 2.2.4 working (no fallback)
- [ ] TUSZ dataset downloaded
- [ ] Can run smoke test successfully

## Support

Issues: https://github.com/clarity-digital-twin/brain-go-brr-v2/issues

Remember: The Conv1d fallback is NOT functionally equivalent to Mamba!