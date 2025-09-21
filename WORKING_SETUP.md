# THE ACTUAL WORKING SETUP

## The Problem
UV can't build mamba-ssm and causal-conv1d because they require PyTorch at BUILD TIME.
These packages must be installed with `--no-build-isolation` AFTER PyTorch is installed.

## The CORRECT Setup Process

### 1. Base Install (without GPU extras)
```bash
uv sync  # Installs PyTorch and base deps
```

### 2. GPU Extensions (manual install)
```bash
# Set CUDA environment
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install GPU extensions with --no-build-isolation
uv pip install --no-build-isolation causal-conv1d==1.4.0
uv pip install --no-build-isolation mamba-ssm==2.2.2
```

## Working Versions (DO NOT CHANGE)
- **Python**: 3.11.13
- **PyTorch**: 2.2.2+cu121
- **CUDA Toolkit**: 12.1
- **mamba-ssm**: 2.2.2
- **causal-conv1d**: 1.4.0

## Why These Specific Versions
- **causal-conv1d 1.5.x** requires PyTorch 2.4+ (we have 2.2.2)
- **mamba-ssm 2.2.5** has undefined symbol bugs
- **mamba-ssm 2.2.4** also has issues
- **mamba-ssm 2.2.2** works perfectly with our stack

## Running Training

### Direct Python (WORKS)
```bash
.venv/bin/python -m src train configs/local/train.yaml
```

### With tmux (WORKS)
```bash
tmux new -d -s train_full '.venv/bin/python -m src train configs/local/train.yaml'
```

### With UV (BROKEN - tries to rebuild)
```bash
# DON'T USE: uv run python -m src train
# UV tries to rebuild GPU packages and fails
```

## Monitoring Training
```bash
tmux attach -t train_full  # View live
tmux capture-pane -t train_full -p | tail -20  # Check progress
```

## DO NOT USE
- `requirements-exact.txt` - We use UV/pyproject.toml
- `uv run` for training - It tries to rebuild GPU packages
- `uv sync --extra gpu` - Can't build without PyTorch

## Environment Variables
Add to ~/.bashrc:
```bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## Verification
```bash
.venv/bin/python -c "
from mamba_ssm import Mamba2
import torch
print('✅ Mamba working!')
print(f'✅ CUDA: {torch.cuda.is_available()}')
"
```