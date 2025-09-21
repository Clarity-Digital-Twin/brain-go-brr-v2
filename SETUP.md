# FINAL WORKING SETUP - WHAT ACTUALLY WORKS

## Current Status
- ✅ **Training running** in tmux session `train_full`
- ✅ **GPU Mamba working** (no fallback)
- ✅ **Processing val data** (361/933 files)

## What Works vs What Doesn't

### ✅ WORKS
```bash
# Base setup
make setup                    # Installs base deps
make setup-gpu               # Installs GPU packages manually

# Training commands
.venv/bin/python -m src train configs/local/train.yaml     # Direct python
make train-local             # Uses .venv/bin/python directly

# Tmux commands
tmux new -d -s train_full '.venv/bin/python -m src train configs/local/train.yaml'
tmux attach -t train_full    # View training
tmux capture-pane -t train_full -p | tail -20  # Check progress
```

### ❌ BROKEN
```bash
uv run python -m src train   # UV tries to rebuild GPU packages and fails
uv sync --extra gpu          # Can't build mamba-ssm/causal-conv1d
```

## The Core Problem
**UV cannot build mamba-ssm and causal-conv1d** because:
1. They require PyTorch at BUILD time
2. UV's build isolation prevents access to installed PyTorch
3. Must use `--no-build-isolation` flag

## Correct Installation Process

### Step 1: Clone & Basic Setup
```bash
git clone <repo>
cd brain-go-brr-v2
make setup           # Installs PyTorch and base deps
```

### Step 2: Install CUDA 12.1 (if not present)
```bash
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-1
```

### Step 3: Install GPU Extensions
```bash
make setup-gpu       # Manually installs mamba-ssm & causal-conv1d
```

### Step 4: Start Training
```bash
# Option A: Direct
make train-local

# Option B: Tmux
tmux new -d -s train_full '.venv/bin/python -m src train configs/local/train.yaml'
```

## Critical Versions (DO NOT CHANGE)
```python
Python: 3.11.13
PyTorch: 2.2.2+cu121
CUDA Toolkit: 12.1
mamba-ssm: 2.2.2         # NOT 2.2.4 or 2.2.5!
causal-conv1d: 1.4.0     # NOT 1.5.x!
```

## What We Changed

### Makefile
- `train-local` now uses `.venv/bin/python` instead of `uv run`
- `setup-gpu` manually installs with `--no-build-isolation`

### pyproject.toml
- Removed GPU packages from `[project.optional-dependencies]`
- GPU packages must be installed manually

## Quick Commands Reference

| Task | Command |
|------|---------|
| Setup | `make setup && make setup-gpu` |
| Train | `make train-local` |
| Train in tmux | `tmux new -d -s train_full '.venv/bin/python -m src train configs/local/train.yaml'` |
| Check training | `tmux attach -t train_full` |
| Kill training | `tmux kill-session -t train_full` |
| Check GPU | `nvidia-smi` |
| Verify Mamba | `.venv/bin/python -c "from mamba_ssm import Mamba2; print('OK')"` |

## For OSS Contributors

### Will Work
1. Clone repo
2. `make setup`
3. Install CUDA 12.1 toolkit
4. `make setup-gpu`
5. `make train-local`

### Won't Work
- Using `uv sync --extra gpu`
- Using `uv run` for training
- Different CUDA versions (MUST be 12.1)
- Different package versions

## Environment Variables
Add to ~/.bashrc:
```bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## The Golden Rule
**NEVER use `uv run` for training** - it tries to rebuild GPU packages and fails.
**ALWAYS use `.venv/bin/python` directly** for training.