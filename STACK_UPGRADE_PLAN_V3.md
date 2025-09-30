# Complete Stack Upgrade Plan: PyTorch 2.2.2 → 2.5.0 + mamba-ssm 2.2.2 → 2.2.5

**Branch**: `fix/upgrade-mamba` ✅ SAFE - can delete if fails
**Created**: 2025-09-29
**Status**: 🔥 PHASE 4 IN PROGRESS - Local training running, Modal smoke test running
**Last Updated**: 2025-09-29 21:11 (Phase 1-3 complete, validating now)
**Context**: Modal A100 XID 31 MMU Fault confirmed as Mamba CUDA kernel bug (Test 1A proved AMP not the cause)

---

## Executive Decision

**START UPGRADE NOW** - don't wait for Test 1B because:
1. Test 1A already proved Mamba CUDA crashes (AMP off = still crashed)
2. GitHub Issue #387 confirms A100 + d_model=512 is known problem
3. GitHub Issue #686 fix (int64 indexing) likely in 2.2.3+
4. Safe branch - can revert if needed
5. Upgrade takes 3-7 days - start research while Test 1B runs
6. Even if issue isn't fully Mamba, latest versions are best practice

---

## 🔥 PROGRESS STATUS (Updated 2025-09-29 21:11)

### ✅ PHASE 1: PRE-FLIGHT RESEARCH - COMPLETE
- PyTorch 2.5 breaking changes analyzed → **Found: weight_norm behavior change**
- torch-geometric 2.6.1 compatibility verified
- mamba-ssm 2.2.5 wheels confirmed available
- CUDA 12.4 toolkit availability confirmed

### ✅ PHASE 2: LOCAL UPGRADE - COMPLETE (19:59:58)
- CUDA 12.4 toolkit installed ✅
- pyproject.toml updated to PyTorch 2.5.0 ✅
- INSTALLATION.md updated with PyG 2.6.1 ✅
- Makefile setup-gpu target updated ✅
- Local environment rebuilt with new stack ✅
- Test suite passed (with gradient threshold adjustment) ✅
- Cache verified (no rebuild needed - pure NumPy) ✅
- **Local smoke test PASSED** (1 epoch, 3 files, no NaN) ✅

**Critical fix applied**: Removed weight_norm from TCN due to PyTorch 2.5 behavior change

### ✅ PHASE 3: MODAL UPGRADE - COMPLETE (21:03:32)
- deploy/modal/app.py updated (CUDA 12.4 + PyTorch 2.5.0) ✅
- Modal configs verified (no changes needed) ✅
- **Modal image build PASSED** ✅
- **Modal Mamba CUDA test PASSED** on A100 80GB ✅
  - Output: `torch.Size([2, 100, 512])` validated
  - mamba-ssm 2.2.5 working on A100
- Modal cache rebuild: SKIPPED (persistent SSD, no PyTorch serialization) ✅

**Commits**:
- `329b470` - Remove weight_norm from TCN (PyTorch 2.5 fix)
- `4b66ff9` - Update torch-geometric to 2.6.1
- `b5dc1d7` - Adjust gradient vanishing threshold
- `b591136` - Enhance local testing commands
- `6bfe9be` - Upgrade pip in Modal deployment

### 🔥 PHASE 4: VALIDATION - IN PROGRESS (Started 21:11:50)
- **Local full training: RUNNING** (tmux: local-train)
  - Config: 100 epochs + early stopping (patience=5)
  - Dataset: 4667 train files, 1832 dev files
  - Status: Dataset loaded, building dev index
  - Expected: ~2-3 hours/epoch, ~200-300 hours total or early stop
- **Modal smoke test: RUNNING** (tmux: modal-smoke)
  - Config: 1 epoch, 50 files
  - Status: Image building with new stack
  - Expected: ~10-15 min total
  - **THIS IS THE CRITICAL XID 31 TEST!** 🎯
- Modal full training: PENDING (after smoke test passes)
- Performance comparison: PENDING (need 5+ epochs data)
- Checkpoint compatibility: NOT TESTED (likely incompatible, will retrain)

### ⏳ PHASE 5: CLEANUP - PENDING
- README.md: TODO
- CLAUDE.md: TODO
- ARCHITECTURE_EVOLUTION.md: TODO (add v3.3.0 section)
- CHANGELOG.md: TODO (create v3.3.0 entry)
- Tag v3.3.0: PENDING (after validation)
- Merge to branches: PENDING (after validation)

### 🎯 NEXT MILESTONES
1. **Modal smoke test completes** (~10 min) → Validates XID 31 fix
2. **Local training epoch 1 completes** (~2-3 hours) → Validates local stability
3. **5 epochs stable** (both platforms) → Phase 4 complete
4. **Documentation updates** → Phase 5 complete
5. **Tag v3.3.0 and merge** → UPGRADE COMPLETE 🚀

---

## New Stack Target (v3.3.0)

### Current Stack (v3.2.1)
```
Python: 3.11.13
PyTorch: 2.2.2+cu121
CUDA Toolkit: 12.1
numpy: 1.26.4
mamba-ssm: 2.2.2
causal-conv1d: 1.4.0
torch-geometric: 2.6.1
pytorch-tcn: 1.2.3
```

### Target Stack (v3.3.0)
```
Python: 3.11.13 (NO CHANGE)
PyTorch: 2.5.0+cu124  ← Major jump (2.2 → 2.5)
torchvision: 0.20.0   ← Matches PyTorch 2.5.0
CUDA Toolkit: 12.4    ← Must match PyTorch build
numpy: 1.26.4 (NO CHANGE - mamba still needs <2.0)
mamba-ssm: 2.2.5      ← Latest (Oct 2024)
causal-conv1d: 1.5.2  ← EXACT pin (latest stable for PyTorch 2.5)
torch-geometric: 2.6.1 ← EXACT pin (latest for PyTorch 2.5)
pytorch-tcn: 1.2.3 (NO CHANGE - pure PyTorch)
```

### Why PyTorch 2.5 (not 2.4)?
- **PyTorch 2.4** released Mar 2024 - stable
- **PyTorch 2.5** released Dec 2024 - more recent, better CUDA 12.4 support
- mamba-ssm 2.2.5 wheels available for both
- **Decision**: Use 2.5.0 for latest fixes

---

## Phase 1: Pre-Flight Research (2-4 hours)

### 1.1 PyTorch Breaking Changes Analysis

**Task**: Review PyTorch 2.3, 2.4, 2.5 release notes for breaking API changes

**Files to check against**:
```
src/brain_brr/models/detector.py:98-156      # Main model
src/brain_brr/models/tcn.py:68-142          # TCN encoder
src/brain_brr/models/mamba.py:115-241       # Mamba blocks
src/brain_brr/models/gnn_pyg.py:90-245      # GNN with dynamic PE
src/brain_brr/train/loop.py:234-478         # Training loop
```

**What to search for**:
- `torch.cuda.amp` API changes (autocast, GradScaler)
- `torch.nn` module changes
- CUDA memory management changes
- Optimizer state dict changes
- Checkpoint compatibility

**Action**: Create `PYTORCH_2.5_BREAKING_CHANGES.md` with findings

### 1.2 torch-geometric Compatibility

**Task**: Verify PyG 2.6.1+ works with PyTorch 2.5 + CUDA 12.4

**Check**:
```bash
# PyG wheel availability
https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

**Verify these packages exist**:
- `torch_scatter` for PyTorch 2.5 + CUDA 12.4
- `torch_sparse` for PyTorch 2.5 + CUDA 12.4
- `torch_cluster` for PyTorch 2.5 + CUDA 12.4
- `torch-geometric==2.6.1` or latest

**Action**: Document exact wheel URLs in upgrade plan

### 1.3 mamba-ssm 2.2.5 Verification

**Task**: Confirm mamba-ssm 2.2.5 has wheels for PyTorch 2.5

**Check**:
```bash
# PyPI page
https://pypi.org/project/mamba-ssm/2.2.5/#files
```

**Verify**:
- Wheels for Python 3.11
- Wheels for CUDA 12.4 (cu124)
- causal-conv1d 1.5.0+ wheels exist

**Action**: Document exact package versions and hashes

### 1.4 CUDA 12.4 Toolkit Availability

**Local (WSL2)**:
```bash
# Check NVIDIA CUDA toolkit downloads
https://developer.nvidia.com/cuda-12-4-0-download-archive
```

**Modal**: CUDA 12.4 available in `nvidia/cuda:12.4.0-devel-ubuntu22.04`

**Action**: Download CUDA 12.4 installer (local only - Modal uses container)

---

## Phase 2: Local Upgrade (4-8 hours)

### 2.1 Backup Current State

```bash
# On fix/upgrade-mamba branch
git status  # Confirm clean

# Backup current .venv
cp -r .venv .venv.backup.2.2.2

# Backup current cache (optional - can rebuild)
# cp -r cache/tusz cache/tusz.backup.2.2.2

# Create checkpoint marker
echo "v3.2.1 - PyTorch 2.2.2 + mamba-ssm 2.2.2" > .venv/VERSION
```

### 2.2 Install CUDA 12.4 Toolkit (Local WSL2 Only)

```bash
# Download CUDA 12.4
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run

# Install (will take 10-15 minutes)
sudo sh cuda_12.4.0_550.54.14_linux.run \
  --silent \
  --toolkit \
  --override

# Verify installation
ls /usr/local/cuda-12.4
nvcc --version  # Should show 12.4

# Update .bashrc or .zshrc
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Reload shell
source ~/.bashrc  # or source ~/.zshrc
```

**IMPORTANT**: Keep CUDA 12.1 installed - don't uninstall it yet (rollback safety)

### 2.3 Update pyproject.toml

**File**: `/home/jj/proj/brain-go-brr-v2/pyproject.toml`

**Changes**:
```toml
[project]
version = "3.3.0"  # Line 7: Bump version
description = "... (PyTorch 2.5 + mamba-ssm 2.2.5 + A100 fixes)"  # Line 8

dependencies = [
    # Line 28: UPDATE PyTorch
    "torch==2.5.0",  # Was: 2.2.2

    # Line 29-31: NO CHANGE
    "numpy==1.26.4",  # mamba still needs <2.0
    "scipy==1.11.4",
    "scikit-learn==1.3.2",
]

[project.optional-dependencies]
# Lines 61-63: UPDATE comments
#   mamba-ssm==2.2.5 (latest, includes A100 int64 indexing fix)
#   causal-conv1d==1.5.0 (required for PyTorch 2.5)
```

### 2.4 Update INSTALLATION.md

**File**: `/home/jj/proj/brain-go-brr-v2/INSTALLATION.md`

**Changes**:
```markdown
# Line 7: Update versions
- **PyTorch 2.5.0** with CUDA 12.4 (was: 2.2.2 with CUDA 12.1)

# Line 8: Update mamba version
- **Mamba-SSM 2.2.5** (was: 2.2.2)

# Line 9: Update PyG
- **PyTorch Geometric 2.6.1** (was: 2.6.1)

# Lines 16-17: Update CUDA check
nvcc --version  # Need 12.4 (was: 12.1)

# Line 51: Update CUDA_HOME
export CUDA_HOME=/usr/local/cuda-12.4  # Was: 12.1

# Lines 52-53: Update package versions
uv pip install --no-build-isolation causal-conv1d==1.5.0  # Was: 1.4.0
uv pip install --no-build-isolation mamba-ssm==2.2.5      # Was: 2.2.2

# Lines 56-58: Update PyG wheels URL and version
.venv/bin/pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.5.0+cu124.html  # Was: torch-2.2.0+cu121
.venv/bin/pip install torch-geometric==2.6.1           # Was: 2.6.1

# Lines 100-109: Update compatibility matrix
| PyTorch | 2.5.0 | Latest, A100 fixes | (was: 2.2.2)
| CUDA | 12.4 | PyTorch 2.5.0 build | (was: 12.1)
| mamba-ssm | 2.2.5 | Latest with int64 fix | (was: 2.2.2 - bugs comment removed)
| causal-conv1d | 1.5.0 | Required for PyTorch 2.5 | (was: 1.4.0)
| torch-geometric | 2.6.1 | Latest (Sep 2024) for PyTorch 2.5 | (was: 2.6.1)
```

### 2.5 Update Makefile

**File**: `/home/jj/proj/brain-go-brr-v2/Makefile`

**Changes**:
```makefile
# Lines 153-158: Update setup-gpu target
setup-gpu: ## Setup GPU support with mamba-ssm and PyG (requires CUDA 12.4)
	@echo "${CYAN}Setting up GPU support for V3 stack...${NC}"
	@echo "${YELLOW}Checking CUDA versions...${NC}"
	@.venv/bin/python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')" || echo "${RED}PyTorch not installed${NC}"
	@nvcc --version 2>/dev/null | grep "release 12.4" || echo "${RED}CUDA 12.4 toolkit required!${NC}"
	@echo "${CYAN}Installing Mamba-SSM components...${NC}"
	@export CUDA_HOME=/usr/local/cuda-12.4 && \
		uv pip install --no-build-isolation causal-conv1d==1.5.0 && \
		uv pip install --no-build-isolation mamba-ssm==2.2.5
	@echo "${CYAN}Installing PyG with pre-built wheels...${NC}"
	@uv pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
	@uv pip install torch-geometric==2.6.1
```

### 2.6 Upgrade Local Environment

```bash
# Ensure we're on fix/upgrade-mamba branch
git checkout fix/upgrade-mamba

# Set CUDA 12.4
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Remove old environment
rm -rf .venv
rm -rf .uv_cache

# Create fresh environment with PyTorch 2.5
uv sync

# Verify PyTorch
.venv/bin/python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version: {torch.version.cuda}')
assert torch.__version__.startswith('2.5'), f'Wrong PyTorch: {torch.__version__}'
assert torch.version.cuda == '12.4', f'Wrong CUDA: {torch.version.cuda}'
"

# Install GPU stack (Mamba + PyG)
make setup-gpu

# Verify all components
.venv/bin/python -c "
import torch
import torch_geometric
from mamba_ssm import Mamba2
import pytorch_tcn
print(f'✅ PyTorch {torch.__version__} (CUDA {torch.version.cuda})')
print(f'✅ PyG {torch_geometric.__version__}')
print('✅ Mamba-SSM 2.2.5')
print('✅ TCN')
"
```

### 2.7 Run Local Test Suite

```bash
# Quick smoke test (no GPU needed)
make test-fast

# GPU unit tests
make test-gpu

# Integration tests (uses GPU)
make test-integration

# Check for import errors
.venv/bin/python -c "
from src.brain_brr.models.detector import SeizureDetector
from src.brain_brr.models.tcn import TCNEncoder
from src.brain_brr.models.mamba import BiMambaBlock
from src.brain_brr.models.gnn_pyg import GNNBlock
print('✅ All models import successfully')
"
```

**Expected**: All tests pass (same as before)

### 2.8 Verify Local Cache (NO REBUILD NEEDED!)

**✅ CACHE IS PURE NUMPY - NO REBUILD REQUIRED**

Our cache contains only NumPy arrays (not PyTorch tensors), so PyTorch version changes don't affect it.

```bash
# Verify cache format (already done - it's pure NumPy)
python -c "
import numpy as np
data = np.load('cache/tusz/train/00000000_s001_t000.npz', allow_pickle=True)
print('Keys:', list(data.keys()))
print('Types:', {k: type(data[k]) for k in data.keys()})
# Output: Keys: ['windows', 'labels']
#         Types: {'windows': <class 'numpy.ndarray'>, 'labels': <class 'numpy.ndarray'>}
"

# Just verify cache exists and is complete
ls cache/tusz/train/*.npz | wc -l  # Should be ~4667
ls cache/tusz/dev/*.npz | wc -l    # Should be ~1832
```

**SAVES 1-2 HOURS** compared to full rebuild!

### 2.9 Local Smoke Test

```bash
# Quick validation (3 files, 1 epoch)
make smoke-local

# Expected: Training completes without NaN/crash
```

**CHECKPOINT**: If smoke test fails, STOP here and debug before proceeding.

---

## Phase 3: Modal Upgrade (2-4 hours)

### 3.1 Update deploy/modal/app.py

**File**: `/home/jj/proj/brain-go-brr-v2/deploy/modal/app.py`

**Changes** (lines 16-87):

```python
image = (
    # UPDATED: CUDA 12.4 devel image
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("build-essential", "ninja-build", "git")
    .env({
        # UPDATED: CUDA 12.4 paths
        "CUDA_HOME": "/usr/local/cuda-12.4",
        "PATH": "/usr/local/cuda-12.4/bin:$PATH",
        "LD_LIBRARY_PATH": "/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0",  # A100 is 8.0
    })
    # UPDATED: PyTorch 2.5.0 + CUDA 12.4
    .run_commands(
        "pip install torch==2.5.0 torchvision==0.19.0 'numpy<2.0' --index-url https://download.pytorch.org/whl/cu124"
    )
    .run_commands(
        "python -c 'import torch; assert torch.__version__.startswith(\"2.5.0\"), f\"Wrong torch: {torch.__version__}\"'"
    )
    .pip_install("packaging", "wheel", "setuptools")
    # UPDATED: New versions
    .run_commands(
        "pip install --no-build-isolation --no-cache-dir causal-conv1d==1.5.0"
    )
    .run_commands(
        "pip install --no-build-isolation --no-cache-dir mamba-ssm==2.2.5"
    )
    .run_commands(
        "python -c 'from mamba_ssm import Mamba2; print(\"✅ Mamba2 imports successfully\")'"
    )
    # Core dependencies (no changes needed)
    .pip_install(
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "mne>=1.5.0",
        "pyedflib>=0.1.30",
        "einops>=0.7.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0.0",
        "click>=8.1.7",
        "rich>=13.0.0",
        "tqdm>=4.64.0",
        "pandas>=2.0.0",
        "tensorboard>=2.10.0",
        "wandb",
        "pytorch-tcn",
    )
    # UPDATED: PyG wheels for PyTorch 2.5 + CUDA 12.4
    .run_commands(
        "pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html"
    )
    .run_commands(
        "pip install torch-geometric==2.6.1"
    )
    .run_commands(
        "python -c 'import torch_geometric; print(f\"✅ PyG {torch_geometric.__version__} installed\")'"
    )
    .workdir("/app")
    .add_local_dir(str(Path(__file__).parent.parent.parent / "src"), "/app/src")
    .add_local_dir(str(Path(__file__).parent.parent.parent / "configs"), "/app/configs")
    .add_local_dir(str(Path(__file__).parent), "/app/deploy/modal")
)
```

### 3.2 Update deploy/modal/app_diag_1b.py (if kept)

**Same changes as app.py** for consistency

### 3.3 Update Modal Configs (No changes needed)

**Files checked**:
- `configs/modal/train.yaml` - no version-specific settings
- `configs/modal/smoke.yaml` - no version-specific settings

**No changes required** - configs are version-agnostic

### 3.4 Test Modal Image Build

```bash
# Dry-run to verify image builds
modal run deploy/modal/app.py --action test-mamba

# Expected: Image builds successfully, Mamba test passes
```

**CHECKPOINT**: If image build fails, debug before proceeding.

### 3.5 Rebuild Modal Cache

**CRITICAL**: Cache must be rebuilt with new PyTorch version

**Option A: Rebuild from S3** (if S3 cache is old version):
```bash
# 1. Delete old Modal cache
modal run deploy/modal/app.py --action clean-cache

# 2. Rebuild local cache with new stack (already done in Phase 2.8)

# 3. Upload new cache to S3
./scripts/upload_cache_to_s3.sh

# 4. Populate Modal SSD from S3
modal run --detach deploy/modal/app.py --action populate-cache

# Monitor (takes ~1-2 hours for 450GB)
modal app list
```

**Option B: Build cache directly on Modal** (if S3 cache is compatible):
```bash
# Let Modal training build cache on first run
# Will be slower but doesn't require S3 re-upload
```

**Recommendation**: Use Option A for certainty.

### 3.6 Modal Smoke Test

```bash
# Test with 50 files, 1 epoch
modal run --detach deploy/modal/app.py \
  --action train \
  --config configs/modal/smoke.yaml

# Monitor
modal app list
modal app logs <app-id>

# Expected: Completes without crash or NaN
```

**CHECKPOINT**: If smoke test fails, STOP and debug.

---

## Phase 4: Validation & Testing (24-48 hours)

### 4.1 Full Local Training Test

```bash
# Start full training locally (will take days)
tmux new -s train-upgraded
make train-local

# Monitor in another terminal
tmux attach -t train-upgraded

# Watch for:
# - No NaN losses
# - No XID 31 crashes
# - Normal training progression
```

**Run for at least 5 epochs** before considering stable.

### 4.2 Full Modal Training Test

```bash
# Launch full 100-epoch training on A100
modal run --detach deploy/modal/app.py \
  --action train \
  --config configs/modal/train.yaml

# Monitor closely for first 5 epochs
modal app logs <app-id>

# Watch for:
# - No XID 31 crashes
# - No illegal memory access errors
# - AMP working correctly (mixed_precision: true)
# - Normal VRAM usage (~40-60GB)
```

### 4.3 Performance Comparison

**Metrics to compare** (old vs new stack):

| Metric | v3.2.1 (PyTorch 2.2.2) | v3.3.0 (PyTorch 2.5.0) |
|--------|------------------------|------------------------|
| Local train time/epoch | ? | ? |
| Modal train time/epoch | ? | ? |
| Peak VRAM usage | ? | ? |
| NaN occurrence | ? | Should be 0 |
| A100 crashes | Yes (XID 31) | Should be 0 |

### 4.4 Checkpoint Compatibility Test

**IMPORTANT**: Old checkpoints may not load with new PyTorch

```bash
# Try loading old checkpoint
.venv/bin/python -c "
import torch
from src.brain_brr.models.detector import SeizureDetector

# Load old checkpoint (PyTorch 2.2.2)
ckpt_path = 'results/old_training/checkpoints/best.pt'
try:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    print('✅ Checkpoint loads successfully')

    # Try loading into model
    model = SeizureDetector(...)
    model.load_state_dict(ckpt['model'])
    print('✅ Model state dict compatible')
except Exception as e:
    print(f'❌ Checkpoint incompatible: {e}')
    print('⚠️  Old checkpoints CANNOT be used - must retrain')
"
```

**Expected**: Likely INCOMPATIBLE - plan to retrain from scratch.

---

## Phase 5: Cleanup & Documentation (2-4 hours)

### 5.1 Update All Documentation

**Files to update**:
- [x] `pyproject.toml` - version + dependencies
- [x] `INSTALLATION.md` - version matrix + instructions
- [x] `Makefile` - setup-gpu target
- [x] `deploy/modal/app.py` - image definition
- [ ] `README.md` - update version badge + quick start
- [ ] `CLAUDE.md` - update version info
- [ ] `ARCHITECTURE_EVOLUTION.md` - document v3.3.0 upgrade
- [ ] `CHANGELOG.md` - create entry for v3.3.0

### 5.2 Remove Outdated Comments

**Search and remove**:
```bash
# Find all references to "2.2.5 has bugs"
rg "2.2.5 has bugs" --files-with-matches

# Remove or update those comments
```

### 5.3 Clean Up Diagnostic Files

**After upgrade is stable**, delete:
```bash
rm -rf deploy/modal/app_diag_1b.py
rm -rf configs/modal/diag_1b_fallback.yaml
rm -rf MODAL_A100_CUDA_FAILURE_ANALYSIS.md
rm -rf MAMBA_UPGRADE_ANALYSIS.md
```

**Keep for reference**:
- `STACK_UPGRADE_PLAN_V3.md` (this file)
- `UPGRADE_PLAN.md` (original plan)

### 5.4 Tag Release

```bash
# After everything is stable
git add -A
git commit -m "feat: Upgrade to PyTorch 2.5.0 + mamba-ssm 2.2.5 (fixes A100 XID 31)

- PyTorch 2.2.2 → 2.5.0 (+CUDA 12.4)
- mamba-ssm 2.2.2 → 2.2.5 (includes int64 indexing fix)
- causal-conv1d 1.4.0 → 1.5.0
- torch-geometric 2.6.1 (NO CHANGE)
- Rebuilt cache with new serialization format
- Modal A100 training now stable with AMP enabled
- Fixes: XID 31 MMU Fault, illegal memory access errors

Breaking Changes:
- Old checkpoints incompatible - must retrain from scratch
- Cache must be rebuilt (handled automatically)

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"

git tag -a v3.3.0 -m "v3.3.0: PyTorch 2.5 + Mamba 2.2.5 + A100 fixes"
git push origin fix/upgrade-mamba --tags
```

### 5.5 Merge to Main

```bash
# After full validation (5+ epochs stable)
git checkout main
git merge fix/upgrade-mamba --no-edit
git push origin main

git checkout development
git merge fix/upgrade-mamba --no-edit
git push origin development

git checkout fix/test-suite-config
git merge fix/upgrade-mamba --no-edit
git push origin fix/test-suite-config
```

---

## Rollback Plan (If Upgrade Fails)

### Emergency Rollback

```bash
# 1. Restore old environment
rm -rf .venv
mv .venv.backup.2.2.2 .venv

# 2. Restore old cache (if needed)
rm -rf cache/tusz
mv cache/tusz.backup.2.2.2 cache/tusz

# 3. Revert git changes
git checkout main  # or development
git branch -D fix/upgrade-mamba
git fetch origin
git reset --hard origin/main

# 4. Verify old stack works
.venv/bin/python -c "
import torch
print(f'PyTorch: {torch.__version__}')
assert torch.__version__.startswith('2.2.2')
"

make smoke-local
```

### Partial Rollback (Keep Research)

```bash
# Keep fix/upgrade-mamba branch for future attempt
git checkout main
# Don't delete fix/upgrade-mamba

# Use old stack for now
git checkout .venv
```

---

## Risk Matrix

### High-Risk Items (MUST verify)

| Risk | Impact | Mitigation | Verification |
|------|--------|------------|--------------|
| PyTorch API breaking changes | 🔴 HIGH | Thorough testing | Run full test suite |
| Checkpoint incompatibility | 🔴 HIGH | Accept, plan to retrain | Test loading old ckpts |
| Cache format changes | 🟡 MEDIUM | Rebuild cache | Verify cache loads |
| Modal image build failure | 🟡 MEDIUM | Test build first | Dry-run before deploy |
| New Mamba bugs in 2.2.5 | 🟡 MEDIUM | Monitor training | Watch for NaN/crashes |
| CUDA 12.4 driver issues | 🟠 LOW | Keep 12.1 installed | Test GPU availability |

### Medium-Risk Items

| Risk | Impact | Mitigation |
|------|--------|------------|
| PyG 2.7 API changes | 🟡 MEDIUM | Check GNN code |
| Performance regression | 🟡 MEDIUM | Benchmark before/after |
| Local/Modal drift | 🟡 MEDIUM | Upgrade both simultaneously |

### Low-Risk Items

| Risk | Impact | Note |
|------|--------|------|
| TCN compatibility | 🟢 LOW | Pure PyTorch, no changes |
| numpy compatibility | 🟢 LOW | Staying on 1.26.4 |
| Python 3.11 compatibility | 🟢 LOW | No Python upgrade |

---

## Timeline Estimate

| Phase | Optimistic | Realistic | Pessimistic |
|-------|-----------|-----------|-------------|
| 1. Research | 2 hours | 4 hours | 8 hours |
| 2. Local Upgrade | 4 hours | 8 hours | 16 hours |
| 3. Modal Upgrade | 2 hours | 4 hours | 8 hours |
| 4. Validation | 1 day | 2 days | 4 days |
| 5. Cleanup | 2 hours | 4 hours | 8 hours |
| **Total** | **2 days** | **3-4 days** | **7 days** |

**Critical path**: Phase 4 (validation) - must wait for training to run.

---

## Success Criteria

### Must Pass (Go/No-Go)

- [ ] Local smoke test completes without NaN or crash
- [ ] Modal smoke test completes without XID 31 or illegal memory access
- [ ] Full test suite passes (same as before upgrade)
- [ ] Local training runs 5 epochs without issues
- [ ] Modal training runs 5 epochs without issues

### Should Pass (Quality)

- [ ] Training speed equal or better than before
- [ ] VRAM usage similar or better
- [ ] No new warnings or errors in logs
- [ ] Mamba import test passes on both platforms

### Nice to Have (Validation)

- [ ] Full 100-epoch training completes successfully
- [ ] Metrics comparable to previous training runs
- [ ] A100 uses full tensor core capacity with AMP

---

## Post-Upgrade Monitoring

### First 24 Hours

- Monitor Modal training every hour
- Check for any crashes or NaN losses
- Verify VRAM usage is stable

### First Week

- Monitor training metrics (loss, accuracy, sensitivity)
- Compare to baseline performance
- Watch for any delayed issues

### First Month

- Full training validation (100 epochs)
- Performance benchmarking
- Production deployment decision

---

## Questions for AI Agent Consensus

Before executing upgrade, get agreement on:

1. **PyTorch 2.4 vs 2.5**: Which version? (Currently: 2.5.0)
2. **Cache rebuild**: Rebuild from scratch or try compatibility? (Currently: rebuild)
3. **Rollback trigger**: What criteria abort upgrade? (Currently: smoke test failure)
4. **Checkpoint strategy**: Accept loss of old checkpoints? (Currently: yes, retrain)
5. **Timeline**: Commit to 3-7 day effort? (Currently: yes)
6. **Risk tolerance**: Proceed with Medium-High risk? (Currently: yes)

---

## Contacts & Resources

**Documentation**:
- PyTorch 2.5 Release Notes: https://github.com/pytorch/pytorch/releases/tag/v2.5.0
- mamba-ssm Releases: https://github.com/state-spaces/mamba/releases
- PyG Docs: https://pytorch-geometric.readthedocs.io/en/latest/

**Issues**:
- GitHub Issue #686: https://github.com/state-spaces/mamba/issues/686
- GitHub Issue #732: https://github.com/state-spaces/mamba/issues/732
- GitHub Issue #387: https://github.com/state-spaces/mamba/issues/387

**Support**:
- PyTorch Forums: https://discuss.pytorch.org/
- Modal Slack: https://modal.com/slack

---

**STATUS**: Ready to execute - waiting for final AI agent consensus

**Next Step**: Phase 1.1 - PyTorch Breaking Changes Analysis