# Installation Guide for Brain-Go-Brr v2.6/V3

## Stack Overview

The stack uses **TCN + BiMamba + GNN + LPE** for O(N) seizure detection (V3 path = learned adjacency; legacy V2 heuristic graphs have been removed):
- **PyTorch 2.2.2** with CUDA 12.1 (EXACT version required)
- **Mamba-SSM 2.2.2** (bidirectional state-space model)
- **PyTorch Geometric 2.6.1** (graph neural networks with Laplacian PE)
- **pytorch-tcn 1.2.3** (temporal convolutional networks)

## Local Installation (WSL2/Linux with GPU)

### Prerequisites
```bash
# Check CUDA version (need 12.1)
nvcc --version

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

# 2. Install Mamba-SSM (requires build tools)
export CUDA_HOME=/usr/local/cuda-12.1
uv pip install --no-build-isolation causal-conv1d==1.4.0
uv pip install --no-build-isolation mamba-ssm==2.2.2

# 3. Install PyG with pre-built wheels (AVOID COMPILATION)
.venv/bin/pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
.venv/bin/pip install torch-geometric==2.6.1

# 4. Install TCN
uv pip install pytorch-tcn==1.2.3
```

## Modal Cloud Installation

Modal uses a custom container system (not Docker). The image is built in `deploy/modal/app.py`:

```python
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["build-essential", "git", "wget"])

    # CUDA 12.1 runtime
    .run_commands("wget https://developer.download.nvidia.com/...")

    # PyTorch 2.2.2 + CUDA 12.1
    .run_commands("pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121")

    # Mamba-SSM (compile from source)
    .run_commands("""
        export CUDA_HOME=/usr/local/cuda-12.1
        pip install --no-build-isolation causal-conv1d==1.4.0 mamba-ssm==2.2.2
    """)

    # PyG with pre-built wheels
    .run_commands("""
        pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
          -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
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
| PyTorch | 2.2.2 | Mamba-SSM + PyG compatibility |
| CUDA | 12.1 | PyTorch 2.2.2 build target |
| numpy | 1.26.4 | numpy 2.x breaks mamba-ssm |
| mamba-ssm | 2.2.2 | 2.2.5 has bugs, 2.2.4 has issues |
| causal-conv1d | 1.4.0 | 1.5+ requires PyTorch 2.4+ |
| torch-geometric | 2.6.1 | Latest stable for torch 2.2.2 |
| pytorch-tcn | 1.2.3 | Pure PyTorch, any version works |

## Common Issues

### 1. PyG Installation Fails with uv
**Error**: `ModuleNotFoundError: No module named 'torch'`

**Solution**: PyG extensions need PyTorch at build time. Use pre-built wheels:
```bash
.venv/bin/pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
```

### 2. Mamba-SSM CUDA Errors
**Error**: `RuntimeError: CUDA error: no kernel image is available`

**Solution**: Ensure CUDA 12.1 toolkit installed and:
```bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
# Rebuild mamba-ssm
uv pip uninstall mamba-ssm causal-conv1d
uv pip install --no-build-isolation causal-conv1d==1.4.0 mamba-ssm==2.2.2
```

### 3. WSL2 Permission Issues
**Error**: `OSError: [Errno 1] Operation not permitted`

**Solution**: Use copy mode for uv:
```bash
export UV_LINK_MODE=copy
```

### 4. Modal CPU Bottlenecks
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
  cache_dir: cache/tusz  # Has 3734 pre-processed NPZ files
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
2. Check cache has expected files (~3734 for full training)
3. Start with conservative batch sizes (12 for RTX 4090, 64 for A100)
4. Monitor first epoch carefully for NaN losses
5. Use focal loss for class imbalance (12:1 ratio)
