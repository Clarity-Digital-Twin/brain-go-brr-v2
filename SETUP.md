# Installation Guide - Brain-Go-Brr V3.4.1

**Last Updated**: 2025-10-02
**Stack**: PyTorch 2.5.0+cu124, CUDA 12.4, mamba-ssm 2.2.5

---

## ⚠️ CRITICAL: Exact Version Requirements

The following versions are **LOCKED** and **MUST NOT BE CHANGED**:

```python
PyTorch==2.5.0+cu124      # EXACT version for Mamba+PyG compatibility
CUDA Toolkit==12.4        # Must match PyTorch CUDA version
mamba-ssm==2.2.5          # Includes A100 int64 indexing fix + PR #708 patch
causal-conv1d==1.5.2      # Latest stable for PyTorch 2.5+
torch-geometric==2.6.1    # Latest compatible with torch 2.5.0
numpy==1.26.4             # 2.x breaks mamba-ssm
```

**Why these versions?**
- PyTorch 2.5.0 required for torch-geometric 2.6.1 and latest CUDA optimizations
- CUDA 12.4 matches PyTorch build target (+cu124)
- mamba-ssm 2.2.5 includes critical A100 fixes + manually applied PR #708 patch
- numpy 1.x required (2.x has breaking changes for mamba-ssm)

---

## 📋 Installation Order (CRITICAL)

**PREREQUISITE**: Install CUDA 12.4 toolkit BEFORE running make commands

### Step 1: Install CUDA 12.4 Toolkit

```bash
# Ubuntu/WSL2
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-4

# Verify installation
/usr/local/cuda-12.4/bin/nvcc --version

# Add to ~/.bashrc or ~/.zshrc
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Reload shell
source ~/.bashrc  # or source ~/.zshrc
```

**Why toolkit is required?**
PyTorch 2.5.0+cu124 includes CUDA 12.4 **runtime** but NOT the **toolkit**. The toolkit is required to compile mamba-ssm and causal-conv1d from source.

### Step 2: Clone Repository

```bash
git clone <repo-url>
cd brain-go-brr-v2
```

### Step 3: Base Environment Setup

```bash
# Install base dependencies (PyTorch, core packages)
make setup
```

This installs:
- Python 3.11+ virtual environment (via uv)
- PyTorch 2.5.0+cu124
- Core dependencies (numpy, scipy, sklearn, etc.)

### Step 4: GPU Extensions Setup

```bash
# Install mamba-ssm, causal-conv1d, PyG, and TCN (all-in-one)
make setup-gpu
```

**What this does:**
- Installs mamba-ssm and causal-conv1d with `--no-build-isolation` (required for PyTorch access)
- Installs PyG from pre-built wheels (torch-scatter, torch-sparse, torch-geometric)
- Installs pytorch-tcn for temporal modeling
- Verifies all components working

**Why not `uv sync -E graph`?**
- Would try to build from source and fail
- Must use pre-built wheels from https://data.pyg.org/whl/torch-2.5.0+cu124.html
- `make setup-gpu` handles this correctly

### Step 5: Verify Installation

```bash
# Verify mamba-ssm CUDA
.venv/bin/python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('✅ Mamba CUDA OK')"

# Verify PyG
.venv/bin/python -c "import torch_geometric; print('✅ PyG OK')"

# Verify full stack
make test
```

---

## 🚀 Quick Start Guide

### Local Training (RTX 4090)

```bash
# 🚨 CRITICAL: Set NaN protection flags (REQUIRED for PyTorch 2.5.0+)
export BGB_SANITIZE_GRADS=1  # Prevents gradient explosion
export BGB_NAN_DEBUG=1       # Shows NaN warnings

# Smoke test (1 epoch, 3 files)
make s

# Full training in tmux (recommended)
tmux new -s train
export BGB_SANITIZE_GRADS=1 BGB_NAN_DEBUG=1
make train-local  # or: .venv/bin/python -m src train configs/local/train.yaml
# Detach: Ctrl+B then D
# Reattach: tmux attach -t train
```

**Expected Performance:**
- VRAM: 12-20GB (24GB total)
- Speed: ~2-3 hours/epoch
- Total: ~200-300 hours for 100 epochs

### Modal Cloud Training (A100-80GB)

```bash
# Test Mamba CUDA before training
modal run deploy/modal/app.py --action test-mamba

# Smoke test (50 files)
modal run deploy/modal/app.py --action train --config configs/modal/smoke.yaml

# Full training (detached for long runs)
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml

# Monitor training
modal app list                    # List running apps
modal app logs <app-id>           # Stream logs
modal app stop <app-id>          # Stop training
```

**Expected Performance:**
- VRAM: 40-60GB (80GB total)
- Speed: ~1 hour/epoch
- Total: ~100 hours for 100 epochs (~$319)

**Note**: Modal automatically sets `BGB_SANITIZE_GRADS=1` and `BGB_NAN_DEBUG=1`

---

## ❌ Common Installation Issues

### Issue 1: Symbol mismatch error
```
Symbol mismatch: _ZN3c104cuda9SetDeviceEab
```

**Cause**: mamba-ssm compiled against wrong PyTorch version

**Fix**:
```bash
# Rebuild mamba-ssm from source
make setup-gpu  # This clears caches and rebuilds
```

### Issue 2: CUDA toolkit not found
```
nvcc: not found
```

**Cause**: CUDA 12.4 toolkit not installed

**Fix**: Install CUDA 12.4 toolkit (see Step 1 above)

### Issue 3: PyG installation fails
```
Could not find a version that satisfies the requirement torch-geometric
```

**Cause**: Trying to install from PyPI instead of pre-built wheels

**Fix**: Use `make setup-gpu` which includes PyG pre-built wheels (NOT `uv sync -E graph`)

### Issue 4: Mamba CUDA errors during training
```
RuntimeError: CUDA error: an illegal memory access was encountered
```

**Cause**: Stale mamba-ssm installation or missing gradient sanitization

**Fix**:
```bash
# Rebuild mamba-ssm
make setup-gpu

# Set gradient sanitization flags
export BGB_SANITIZE_GRADS=1
export BGB_NAN_DEBUG=1
```

---

## 🛠️ Development Tools

### Code Quality

```bash
# Lint + format + type check (RUN AFTER EVERY CHANGE)
make q

# Quick tests (no coverage)
make t

# Full test suite with coverage
make test

# Training-safe tests (CPU only, for concurrent testing)
make ts
```

### GPU Testing

```bash
# Performance tests (requires GPU)
make test-performance

# All GPU tests (stop training first!)
make test-gpu
```

**CRITICAL**: Performance tests require significant GPU memory. Use `make ts` or `BGB_SKIP_GPU_TESTS=1` during training.

---

## 🔧 Environment Variables

### Required for Training
```bash
export BGB_SANITIZE_GRADS=1      # RECOMMENDED - prevents gradient corruption
export BGB_NAN_DEBUG=1           # Shows NaN warnings for debugging
```

### Data Configuration
```bash
export BGB_SMOKE_TEST=1          # Limit to 3 files (smoke test)
export BGB_LIMIT_FILES=50        # Custom file limit
export BGB_FORCE_MANIFEST_REBUILD=1  # Rebuild cache manifest
```

### Testing
```bash
export BGB_SKIP_GPU_TESTS=1      # Skip GPU tests during training
export BGB_TEST_GPU_FRACTION=0.4 # GPU memory fraction for tests (default: 0.4)
```

### WSL2 Fixes
```bash
export UV_LINK_MODE=copy         # Prevent permission issues
```

---

## 📦 What Gets Installed

### Base (`make setup`)
- Python 3.11+ (uv virtual environment)
- PyTorch 2.5.0+cu124
- Core ML: numpy 1.26.4, scipy, scikit-learn
- Data: pandas, h5py, pyarrow
- Config: pydantic, pyyaml
- CLI: click, rich
- Monitoring: wandb, tensorboard

### GPU Extensions (`make setup-gpu`)
- mamba-ssm 2.2.5 (with PR #708 patch)
- causal-conv1d 1.5.2
- Built with `--no-build-isolation` flag

### PyTorch Geometric (`make setup-pyg`)
- torch-geometric 2.6.1
- torch-scatter, torch-sparse, torch-cluster
- Pre-built wheels for torch 2.5.0+cu124

### Development Tools (included in base)
- Testing: pytest, pytest-cov
- Linting: ruff
- Type checking: mypy + type stubs
- Profiling: psutil

---

## 🚨 Critical DO NOTs

### ❌ DON'T change package versions
Every version is locked for compatibility. Changing any version will break the stack.

### ❌ DON'T use `uv run` for training
UV tries to rebuild GPU packages and fails. Use `.venv/bin/python` directly.

### ❌ DON'T use different CUDA versions
CUDA 12.4 is required to match PyTorch 2.5.0+cu124 build.

### ❌ DON'T install PyG with `uv sync -E graph`
Must use pre-built wheels via `make setup-pyg`.

### ❌ DON'T skip CUDA toolkit installation
PyTorch includes runtime but NOT toolkit. Toolkit required for mamba-ssm compilation.

---

## ✅ Verification Checklist

After installation, verify:

- [ ] CUDA toolkit 12.4 installed: `/usr/local/cuda-12.4/bin/nvcc --version`
- [ ] PyTorch CUDA available: `.venv/bin/python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Mamba CUDA working: `.venv/bin/python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('OK')"`
- [ ] PyG installed: `.venv/bin/python -c "import torch_geometric; print('OK')"`
- [ ] Quality checks pass: `make q`
- [ ] Tests pass: `make t`
- [ ] Smoke test runs: `make s`

---

## 📚 Additional Resources

- **Project Overview**: `CLAUDE.md` (quick commands and architecture)
- **Installation Details**: `INSTALLATION.md` (comprehensive guide)
- **Configuration**: `configs/README.md` (training configs)
- **Architecture**: `docs/04-model/v3-architecture.md`
- **Stability Guide**: `docs/04-model/v3-stability-evolution.md`

---

## 🆘 Getting Help

If installation fails:

1. Check error message against Common Issues above
2. Verify all versions match exactly (especially CUDA 12.4)
3. Try rebuilding GPU packages: `make setup-gpu`
4. Run verification checklist
5. Check GPU with `nvidia-smi`

**Remember**: The stack is sensitive to version mismatches. When in doubt, start fresh with exact versions specified.
