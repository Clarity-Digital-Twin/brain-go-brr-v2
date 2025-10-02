# Installation Guide for Brain-Go-Brr V3

## Stack Overview

The stack uses **TCN + BiMamba + GNN + LPE** for O(N) seizure detection.
V3 is the only supported architecture; the legacy V2 heuristic graph path has been removed.
- **PyTorch 2.5.0** with CUDA 12.4 (EXACT version required)
- **Mamba-SSM 2.2.5** (bidirectional state-space model, includes A100 int64 fix)
- **PyTorch Geometric 2.6.1** (graph neural networks with Laplacian PE)
- **pytorch-tcn 1.2.3** (temporal convolutional networks)

## Local Installation (WSL2/Linux with GPU)

### Prerequisites

**CRITICAL**: CUDA 12.4 toolkit is **required** to build mamba-ssm from source.

#### 1. Install CUDA 12.4 Toolkit (Ubuntu/WSL2)
```bash
# Check current CUDA version
nvcc --version  # Should show "release 12.4" after installation

# If CUDA 12.4 not installed, install it:
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-4

# Verify installation
/usr/local/cuda-12.4/bin/nvcc --version
# Should output: "Cuda compilation tools, release 12.4, V12.4.131"
```

**Note**: PyTorch 2.5.0+cu124 includes CUDA 12.4 **runtime** but not the **toolkit**. The toolkit is needed to compile CUDA extensions like mamba-ssm.

#### 2. Install UV Package Manager
```bash
# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Quick Setup
```bash
# Clone and setup
git clone https://github.com/clarity-digital-twin/brain-go-brr-v2.git
cd brain-go-brr-v2

# Base environment
make setup

# GPU components (CRITICAL ORDER)
make setup-gpu  # or make g

# Verify installation
.venv/bin/python -c "
import torch, torch_geometric, mamba_ssm, pytorch_tcn
print(f'✅ Torch {torch.__version__} (CUDA {torch.version.cuda})')
print(f'✅ PyG {torch_geometric.__version__}')
print('✅ Mamba-SSM imported')
print('✅ TCN imported')
"
```

### Manual Installation (if make fails)
```bash
# 1. Create venv with uv
uv sync

# 2. Set CUDA environment for building extensions
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"  # A100/RTX 3000/4000 series

# 3. Install Mamba-SSM (FORCE SOURCE BUILD to avoid cached wrong-CUDA wheels)
rm -rf ~/.cache/uv ~/.cache/pip  # Clear caches to avoid stale wheels
uv pip install --no-build-isolation --no-binary causal-conv1d causal-conv1d==1.5.2
uv pip install --no-build-isolation --no-binary mamba-ssm mamba-ssm==2.2.5

# 4. Install PyG with pre-built wheels (AVOID COMPILATION)
.venv/bin/pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
.venv/bin/pip install torch-geometric==2.6.1

# 5. Install TCN
uv pip install pytorch-tcn==1.2.3

# 6. Verify CUDA kernels are working
python -c "import torch; from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('✅ Mamba CUDA kernels working!')"
```

## Modal Cloud Installation

Modal uses a custom container system (not Docker). The image is built in `deploy/modal/app.py`:

```python
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["build-essential", "git", "wget"])

    # CUDA 12.4 runtime
    .run_commands("wget https://developer.download.nvidia.com/...")

    # PyTorch 2.5.0 + CUDA 12.4
    .run_commands("pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124")

    # Mamba-SSM (compile from source)
    .run_commands("""
        export CUDA_HOME=/usr/local/cuda-12.4
        pip install --no-build-isolation causal-conv1d==1.5.2 mamba-ssm==2.2.5
    """)

    # PyG with pre-built wheels
    .run_commands("""
        pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
          -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
        pip install torch-geometric==2.6.1
    """)

    # TCN and project
    .run_commands("pip install pytorch-tcn==1.2.3")
    .pip_install_from_pyproject("./pyproject.toml")
)
```

## Version Compatibility Matrix

| Component | Version | Why This Version |
|-----------|---------|------------------|
| Python | 3.11+ | Required for modern type hints |
| PyTorch | 2.5.0 | Latest stable with CUDA 12.4 support |
| CUDA | 12.4 | PyTorch 2.5.0 build target |
| numpy | 1.26.4 | numpy 2.x breaks mamba-ssm |
| mamba-ssm | 2.2.5 | Latest, includes A100 int64 indexing fix |
| causal-conv1d | 1.5.2 | Latest stable for PyTorch 2.5+ |
| torch-geometric | 2.6.1 | Latest (Sep 2024) stable for torch 2.5.0 |
| pytorch-tcn | 1.2.3 | Pure PyTorch, any version works |

## Common Issues

### 1. Symbol Mismatch Error (Most Common)
**Error**: `undefined symbol: _ZN3c104cuda9SetDeviceEab` when importing mamba_ssm

**Root Cause**: mamba-ssm was compiled against wrong CUDA version (UV cached a wheel built with wrong toolkit)

**Solution**: Force rebuild from source with CUDA 12.4
```bash
# Set CUDA 12.4 environment
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"

# Purge all caches and stale artifacts
uv pip uninstall mamba-ssm causal-conv1d
rm -rf ~/.cache/uv ~/.cache/pip
rm -rf .venv/lib/python3.11/site-packages/mamba_ssm*
rm -rf .venv/lib/python3.11/site-packages/causal_conv1d*
rm -rf .venv/lib/python3.11/site-packages/selective_scan*

# Force source build (--no-binary prevents using cached wheels)
uv pip install --no-build-isolation --no-binary causal-conv1d causal-conv1d==1.5.2
uv pip install --no-build-isolation --no-binary mamba-ssm mamba-ssm==2.2.5

# Verify CUDA kernels work
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('✅ OK')"
```

### 2. CUDA 12.4 Toolkit Not Installed
**Error**: Makefile warns "CUDA 12.4 toolkit required!" or nvcc not found

**Solution**: Install CUDA 12.4 toolkit (see Prerequisites section above)
```bash
# Ubuntu/WSL2
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-4

# Verify
/usr/local/cuda-12.4/bin/nvcc --version
```

### 3. PyG Installation Fails with uv
**Error**: `ModuleNotFoundError: No module named 'torch'`

**Solution**: PyG extensions need PyTorch at build time. Use pre-built wheels:
```bash
.venv/bin/pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

### 4. Mamba-SSM CUDA Runtime Errors
**Error**: `RuntimeError: CUDA error: no kernel image is available`

**Solution**: Verify CUDA 12.4 in PATH and rebuild:
```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
# Rebuild mamba-ssm (see issue #1 for full steps)
```

### 5. WSL2 Permission Issues
**Error**: `OSError: [Errno 1] Operation not permitted`

**Solution**: Use copy mode for uv:
```bash
export UV_LINK_MODE=copy
```

### 6. Modal CPU Bottlenecks
**Symptom**: Training stuck at epoch boundaries

**Solution**: Increase CPU/RAM allocation in `deploy/modal/app.py`:
```python
@app.function(
    gpu="A100-80GB",
    memory=98304,   # 96GB RAM (default: 32GB)
    cpu=24,         # 24 cores (default: 0.125!)
)
```

## Testing Installation

### Quick Smoke Test
```bash
# Local (1 epoch, 3 files)
make smoke-local  # or: make s

# Modal (1 epoch, 50 files)
modal run deploy/modal/app.py --action train --config configs/modal/smoke.yaml
```

### Verify Components
```python
# Test each component
python -c "from src.brain_brr.models.tcn import TCNEncoder; print('✅ TCN')"
python -c "from src.brain_brr.models.mamba import BiMambaBlock; print('✅ BiMamba')"
python -c "from src.brain_brr.models.gnn_pyg import GNNBlock; print('✅ GNN+LPE')"
python -c "from src.brain_brr.models.detector import SeizureDetector; print('✅ Detector')"
```

## Cache Directories

### Local
```yaml
# configs/local/train.yaml
data:
  cache_dir: cache/tusz  # Has train/dev NPZ caches (official splits)
```

### Modal
```yaml
# configs/modal/train.yaml
data:
  cache_dir: /results/cache/tusz  # Persistent SSD volume
```

## Running Training

### Local (RTX 4090)
```bash
# Full training in tmux
tmux new -s train
make train-local

# Watch progress
tmux attach -t train
```

### Modal (A100)
```bash
# Test Mamba CUDA first
modal run deploy/modal/app.py --action test-mamba

# Full training (detached)
modal run --detach deploy/modal/app.py \
  --action train --config configs/modal/train.yaml

# Monitor
modal app logs <app-id>
```

## Environment Variables

```bash
# Smoke tests
export BGB_SMOKE_TEST=1     # Limit to 3 files
export BGB_LIMIT_FILES=50   # Custom file limit

# Debugging
export BGB_NAN_DEBUG=1      # Debug NaN losses
export BGB_DISABLE_TQDM=1   # Disable progress bars (auto on Modal)

# Mamba fallback (if CUDA issues)
export SEIZURE_MAMBA_FORCE_FALLBACK=1
```

## Next Steps

After installation:
1. Run smoke test to verify setup
2. Check cache has expected files (train ≈4667, dev ≈1832 for full training)
3. Start with conservative batch sizes (12 for RTX 4090, 64 for A100)
4. Monitor first epoch carefully for NaN losses
5. Use focal loss for class imbalance (12:1 ratio)
