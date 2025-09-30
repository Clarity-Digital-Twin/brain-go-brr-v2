# Full Stack Upgrade Plan (mamba-ssm 2.2.2 → 2.2.5)

**Branch**: `fix/upgrade-mamba`
**Created**: 2025-09-29 21:35 UTC
**Context**: Test 1A failed (AMP not the cause). If Test 1B passes, confirms Mamba CUDA kernels are broken on A100.

---

## Upgrade Scope

### Current Stack (v3.2.1)
```
Python: 3.11.13
PyTorch: 2.2.2+cu121
CUDA: 12.1
mamba-ssm: 2.2.2
causal-conv1d: 1.4.0
torch-geometric: 2.6.1
numpy: 1.26.4
```

### Target Stack (v3.3.0)
```
Python: 3.11.13 (no change)
PyTorch: 2.4.0+cu121 OR 2.5.0+cu124
CUDA: 12.4 (if PyTorch 2.5)
mamba-ssm: 2.2.5
causal-conv1d: 1.5.0+
torch-geometric: 2.6.1+
numpy: 1.26.4 (no change - mamba-ssm still needs <2.0)
```

---

## Phase 1: Research & Verification (2-4 hours)

### 1.1 PyTorch Breaking Changes
- [ ] Check PyTorch 2.4 release notes: https://github.com/pytorch/pytorch/releases/tag/v2.4.0
- [ ] Check PyTorch 2.5 release notes: https://github.com/pytorch/pytorch/releases/tag/v2.5.0
- [ ] Search for API deprecations affecting:
  - `torch.amp.autocast`
  - `torch.nn` module APIs
  - Optimizer APIs (AdamW)
  - Scheduler APIs (CosineAnnealingLR)

### 1.2 mamba-ssm 2.2.5 Compatibility
- [ ] Check mamba-ssm 2.2.5 release notes: https://github.com/state-spaces/mamba/releases/tag/v2.2.5
- [ ] Verify PyTorch version requirements
- [ ] Check if A100 CUDA kernel bugs were fixed
- [ ] Look for issues mentioning "illegal memory access" or "XID 31"

### 1.3 torch-geometric Compatibility
- [ ] Check PyG compatibility matrix: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
- [ ] Find latest PyG version for PyTorch 2.4/2.5
- [ ] Check if pre-built wheels exist for target PyTorch+CUDA combo

### 1.4 causal-conv1d Compatibility
- [ ] Verify causal-conv1d 1.5+ works with mamba-ssm 2.2.5
- [ ] Check if 1.5+ requires PyTorch 2.4+ (expected)

---

## Phase 2: Local Upgrade (4-8 hours)

### 2.1 Backup Current Environment
```bash
# Save current requirements
.venv/bin/pip freeze > requirements_backup_v3.2.1.txt

# Backup cache manifests
cp cache/tusz/train/manifest.json cache_manifests_v3.2.1_train.json
cp cache/tusz/dev/manifest.json cache_manifests_v3.2.1_dev.json
```

### 2.2 Install CUDA 12.4 (if needed)
```bash
# Check current CUDA version
nvcc --version

# If upgrading to PyTorch 2.5, install CUDA 12.4
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4

export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### 2.3 Update pyproject.toml
```toml
[project]
dependencies = [
    "torch==2.4.0",  # Updated from 2.2.2
    "numpy==1.26.4",  # Keep <2.0 for mamba-ssm
    "scipy==1.11.4",
    "scikit-learn==1.3.2",
    # ... other deps
]

# GPU packages (manual install)
# mamba-ssm==2.2.5 (updated from 2.2.2)
# causal-conv1d==1.5.0 (updated from 1.4.0)
```

### 2.4 Upgrade PyTorch
```bash
# Uninstall old PyTorch
uv pip uninstall torch torchvision

# Install PyTorch 2.4.0 with CUDA 12.1
uv pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Verify
.venv/bin/python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
```

### 2.5 Upgrade mamba-ssm Stack
```bash
# Uninstall old versions
uv pip uninstall mamba-ssm causal-conv1d

# Install new versions with CUDA 12.4 (if applicable)
export CUDA_HOME=/usr/local/cuda-12.4
uv pip install --no-build-isolation causal-conv1d==1.5.0
uv pip install --no-build-isolation mamba-ssm==2.2.5

# Verify
.venv/bin/python -c "from mamba_ssm import Mamba2; print('✅ Mamba-SSM 2.2.5 imported')"
```

### 2.6 Upgrade torch-geometric
```bash
# Uninstall old PyG
uv pip uninstall torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv

# Find correct PyG version for PyTorch 2.4
# Check: https://data.pyg.org/whl/torch-2.4.0+cu121.html

# Install new PyG
.venv/bin/pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
.venv/bin/pip install torch-geometric==2.6.1

# Verify
.venv/bin/python -c "import torch_geometric as tg; print(f'✅ PyG {tg.__version__}')"
```

### 2.7 Reinstall Project
```bash
uv sync
```

---

## Phase 3: Testing (2-4 hours)

### 3.1 Unit Tests
```bash
# Run full test suite
make test

# Check for failures related to:
# - PyTorch API changes
# - Mamba forward pass
# - GNN/PyG operations
```

### 3.2 Integration Tests
```bash
# Run integration tests
make test-gpu

# Focus on:
# - Mamba integration tests
# - GNN integration tests
# - Full detector forward/backward pass
```

### 3.3 Cache Rebuild
```bash
# CRITICAL: Cache must be rebuilt with new PyTorch version
# Old cache may be incompatible

# Backup old cache
mv cache/tusz cache/tusz_v3.2.1_backup

# Rebuild cache (takes ~1-2 hours)
# This requires access to raw TUSZ data
# If cache rebuild fails, we cannot proceed with upgrade

# Option A: Rebuild from raw data
python -m src populate_cache configs/local/train.yaml

# Option B: Download pre-built cache (if available from team)
# aws s3 cp s3://bucket/cache_v3.3.0.tar.gz .
# tar -xzf cache_v3.3.0.tar.gz
```

### 3.4 Local Smoke Test
```bash
# Run smoke test (1 epoch, 3 files)
make s

# Should complete without errors
# Check for:
# - No NaN losses
# - No CUDA errors
# - Model converges (loss decreases)
```

### 3.5 Local Short Training Run
```bash
# Run 5 epochs on full dataset
.venv/bin/python -m src train configs/local/train.yaml --epochs 5

# Verify:
# - Training stable (no crashes)
# - Validation metrics reasonable
# - No memory leaks
```

---

## Phase 4: Modal Upgrade (2-4 hours)

### 4.1 Update Modal Image
Edit `deploy/modal/app.py`:

```python
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")  # Updated CUDA
    .env({
        "CUDA_HOME": "/usr/local/cuda-12.4",  # Updated
        "TORCH_CUDA_ARCH_LIST": "8.0",  # A100 only (learned from diagnostics)
        "FORCE_REBUILD": "2025-09-29-v3.3.0",  # Force cache bust
    })
    .apt_install([
        "build-essential",
        "git",
        "wget",
        "ninja-build",
    ])
    # PyTorch 2.4.0
    .run_commands(
        "pip install torch==2.4.0 torchvision==0.19.0 'numpy<2.0' "
        "--index-url https://download.pytorch.org/whl/cu121"
    )
    # mamba-ssm 2.2.5
    .run_commands(
        "pip install --no-build-isolation --no-cache-dir causal-conv1d==1.5.0"
    )
    .run_commands(
        "pip install --no-build-isolation --no-cache-dir mamba-ssm==2.2.5"
    )
    # PyG 2.6.1
    .run_commands(
        "pip install torch-scatter torch-sparse torch-cluster torch-spline-conv "
        "-f https://data.pyg.org/whl/torch-2.4.0+cu121.html"
    )
    .run_commands("pip install torch-geometric==2.6.1")
    .run_commands("pip install pytorch-tcn==1.2.3")
    .pip_install_from_pyproject("./pyproject.toml")
)
```

### 4.2 Modal Smoke Test
```bash
# Test Mamba CUDA first
modal run deploy/modal/app.py --action test-mamba

# Run smoke test (1 epoch, 50 files)
modal run deploy/modal/app.py --action train --config configs/modal/smoke.yaml

# Expected: Preflight PASSES (no XID 31)
```

### 4.3 Modal Full Training (if smoke passes)
```bash
# Launch full training (detached)
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml

# Monitor first hour for crashes
modal app list
modal app logs <app-id>

# If stable after 1 hour, let it run
```

---

## Phase 5: Validation (24-48 hours)

### 5.1 Training Stability
- [ ] Monitor training for 24 hours
- [ ] Check for XID 31 crashes
- [ ] Verify no NaN losses
- [ ] Check memory usage (should be similar to v3.2.1)

### 5.2 Performance Comparison
Compare to baseline (if available):
- [ ] Training speed (time per epoch)
- [ ] Memory usage (peak VRAM)
- [ ] Model metrics (AUROC, sensitivity@10FA)

### 5.3 Documentation Updates
- [ ] Update INSTALLATION.md with new versions
- [ ] Update CLAUDE.md with new stack
- [ ] Update pyproject.toml comments
- [ ] Document any API changes encountered

---

## Rollback Plan

If upgrade fails or introduces issues:

### Option A: Revert Local
```bash
# Restore old environment
rm -rf .venv
uv sync

# Reinstall old GPU stack
export CUDA_HOME=/usr/local/cuda-12.1
uv pip install --no-build-isolation causal-conv1d==1.4.0
uv pip install --no-build-isolation mamba-ssm==2.2.2
.venv/bin/pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
.venv/bin/pip install torch-geometric==2.6.1

# Restore old cache
rm -rf cache/tusz
mv cache/tusz_v3.2.1_backup cache/tusz
```

### Option B: Delete Branch
```bash
git checkout main
git branch -D fix/upgrade-mamba
git push origin --delete fix/upgrade-mamba
```

---

## Decision Criteria

### Proceed with Upgrade If:
1. ✅ Test 1B passes (confirms Mamba CUDA is the issue)
2. ✅ mamba-ssm 2.2.5 has fixes for A100 CUDA bugs (research confirms)
3. ✅ Local smoke test passes with new stack
4. ✅ Modal smoke test passes (no XID 31)
5. ✅ Team has 1-2 weeks for careful testing

### Abort Upgrade If:
1. ❌ Test 1B fails (Mamba CUDA not the issue)
2. ❌ Local tests fail with new stack
3. ❌ Modal smoke test crashes with XID 31 (upgrade didn't fix it)
4. ❌ Cache rebuild fails
5. ❌ Breaking API changes require extensive refactoring

---

## Timeline Estimate

| Phase | Time | Risk |
|-------|------|------|
| Research | 2-4 hours | Low |
| Local upgrade | 4-8 hours | Medium |
| Testing | 2-4 hours | Medium |
| Modal upgrade | 2-4 hours | Medium |
| Validation | 24-48 hours | High |
| **Total** | **3-7 days** | **Medium-High** |

---

## Next Steps

1. ⏸️ **WAIT for Test 1B** (~40 more minutes)
2. **If Test 1B passes**: Start Phase 1 (Research)
3. **If Test 1B fails**: Abort upgrade, investigate other causes

---

**Status**: Prepared, waiting for Test 1B results
**Branch**: `fix/upgrade-mamba`
**Can rollback**: YES (delete branch)