# WSL2 Troubleshooting Guide

## Critical Issue: Pipeline Hangs on /mnt/c

### Problem
Training pipeline hangs indefinitely when data is on Windows filesystem (/mnt/c).

### Root Cause
WSL2 uses 9P protocol for /mnt/c mounts. Each file operation takes 10-100ms, making bulk I/O operations on large datasets (80GB with thousands of files) extremely slow.

### Solution
Copy data to native Linux ext4 filesystem:

```bash
# Create data directory on Linux filesystem
mkdir -p data_ext4

# Copy data (run in tmux for large transfers)
tmux new -s datacopy
cp -r /mnt/c/path/to/data/* data_ext4/
# Detach: Ctrl+B, D
# Reattach: tmux attach -t datacopy
```

### Performance Impact
- /mnt/c: ~10-100ms per file operation
- ext4: <1ms per file operation
- 100-1000x speedup for data loading

## DataLoader Multiprocessing Issues

### Problem
DataLoader with num_workers > 0 causes hangs or crashes.

### Solution
Set WSL2-safe DataLoader configuration:

```yaml
# In config files
data:
  num_workers: 0          # Avoid multiprocessing hangs
  pin_memory: false        # Prevent /dev/shm issues
  persistent_workers: false # Critical for stability
```

### Code-level Fix
```python
# In pipeline.py
import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)
```

## Python Output Not Visible

### Problem
Python output doesn't appear in terminal during execution.

### Solution
Set environment variables:

```bash
export PYTHONUNBUFFERED=1      # Force unbuffered output
export PYTHONFAULTHANDLER=1    # Dump tracebacks on deadlock
```

Consider exporting these in your shell profile or a small wrapper script when needed.

## Mamba-SSM Installation Issues

### Problem
mamba-ssm requires specific CUDA setup and fails to build.

### Solution
1. Ensure correct environment and install order (PyTorch → numpy → mamba-ssm).
2. When troubleshooting, set:
   - Dynamic LD_LIBRARY_PATH detection
   - CUDA library paths
   - Architecture-specific settings

## Git Performance on Large Datasets

### Problem
Git operations become slow with large data directories.

### Solution
1. Always add data directories to `.gitignore` BEFORE copying data:
```gitignore
data/
data_ext4/
```

2. If files accidentally get staged:
```bash
git reset --hard HEAD
git clean -fd
```

## Memory Management

### Problem
Large dataset processing can exhaust memory.

### Solution
1. Monitor memory usage:
```bash
free -h
watch -n 1 free -h
```

2. Limit batch size in config:
```yaml
training:
  batch_size: 8  # Reduce for memory constraints
```

3. Use environment variable to limit files:
```bash
export BGB_LIMIT_FILES=100
python -m src train configs/smoke_test.yaml
```

## Thread Contention

### Problem
Multiple libraries spawn threads causing contention.

### Solution
Limit thread usage (set in `run_gpu.sh`):

```bash
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

## Quick Debugging Checklist

1. **Pipeline hanging?**
   - Check data location: Is it on /mnt/c? → Move to ext4
   - Check num_workers: Should be 0 for WSL2

2. **No output visible?**
   - Set PYTHONUNBUFFERED=1
   - Use tmux for long-running processes

3. **Memory issues?**
   - Reduce batch_size
   - Set BGB_LIMIT_FILES for testing

4. **Git slow?**
   - Ensure data directories are in .gitignore
   - Work with data on ext4, not /mnt/c

5. **Import errors?**
   - Use `uv sync` to ensure dependencies
   - For GPU: `uv sync -E gpu`

## Environment Variables Reference

```bash
# Core WSL2 fixes
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export LIBTORCH_USE_RTLD_GLOBAL=YES

# Thread limiting
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Optional controls
export BGB_LIMIT_FILES=100      # Limit dataset size for testing
export BGB_DISABLE_TQDM=1       # Disable progress bars
export BGB_DISABLE_TB=1         # Disable TensorBoard
export UV_LINK_MODE=copy        # WSL2 filesystem compatibility
```

## Recommended Workflow

1. **Initial Setup**:
   ```bash
   make setup
   uv sync -E gpu
   ```

2. **Data Preparation**:
   ```bash
   # Copy to ext4 (not /mnt/c!)
   mkdir -p data_ext4
   cp -r /mnt/c/data/* data_ext4/
   ```

3. **Configuration**:
   - Use `configs/smoke_test.yaml` for quick tests
   - Ensure `num_workers: 0` in all configs

4. **Training**:
```bash
# Smoke test
python -m src train configs/smoke_test.yaml

# In tmux for long runs (WSL2-safe)
tmux new -s train
python -m src train configs/tusz_train_wsl2.yaml
```

5. **Monitoring**:
   ```bash
   tmux attach -t train
   htop  # CPU/memory usage
   nvidia-smi  # GPU usage
   ```

## Key Discoveries

1. **9P Protocol**: The root cause of most WSL2 performance issues is the 9P filesystem protocol used for /mnt/c mounts.

2. **Multiprocessing**: WSL2 has issues with fork-based multiprocessing. Always use spawn method.

3. **Shared Memory**: /dev/shm is limited in WSL2. Avoid pin_memory=True.

4. **Thread Spawning**: Multiple libraries creating threads causes severe contention. Limit to 1 thread per library.

5. **Data Location**: ALWAYS work with data on native Linux filesystem (ext4), never on Windows mounts.
