# Docker Implementation Plan for Brain-Go-Brr v3.6.1

**Status:** 🚧 IN PROGRESS
**Last Updated:** October 4, 2025
**Version:** v3.6.1-gradient-logging-enhancement

---

## Purpose

This document is our **step-by-step implementation plan** to containerize Brain-Go-Brr v3.6.1 for reproducible training. This is NOT user documentation - it's our internal roadmap to build Docker support from scratch.

---

## Current State Assessment

### ❌ What Doesn't Exist Yet
- [ ] `Dockerfile` in repo root
- [ ] `docker-compose.yml` in repo root
- [ ] `.dockerignore` file
- [ ] Pre-built images on GHCR
- [ ] Cache tarball for download
- [ ] Docker-specific configs
- [ ] Health check validation

### ✅ What We Already Have
- [x] Working local training (RTX 4090, batch_size=8)
- [x] Working Modal training (A100, batch_size=48)
- [x] Cache built at `cache/tusz/{train,dev}/` (449GB)
- [x] Modal's proven approach (`deploy/modal/app.py`)
- [x] mamba-ssm PR #708 patch (`deploy/modal/patch_mamba_pr708.py`)
- [x] PyG wheel URLs for torch 2.5.0+cu124
- [x] Mid-epoch checkpointing (every 30 min)

---

## Implementation Phases

### Phase 1: Dockerfile Creation (Multi-Stage Build)

**Goal:** Create production-ready Dockerfile that replicates Modal's proven setup

#### Step 1.1: Create Base Dockerfile Structure

```dockerfile
# Dockerfile (repo root)
# ============================================================================
# Stage 1: Builder - Compile GPU libraries
# ============================================================================
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda-12.4 \
    PATH=/usr/local/cuda-12.4/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9" \
    FORCE_CUDA=1

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ninja-build \
    git \
    python3.11 \
    python3.11-dev \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3.11 -m pip install --upgrade pip setuptools wheel packaging

# CRITICAL: Install numpy FIRST to prevent PyTorch from pulling 2.x (breaks mamba-ssm)
RUN pip install numpy==1.26.4

# Install PyTorch (EXACT version from Modal)
RUN pip install \
    torch==2.5.0 \
    torchvision==0.20.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Verify CUDA
RUN python3.11 -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"

# ============================================================================
# Stage 2: Runtime - Lightweight production
# ============================================================================
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda-12.4 \
    PATH=/usr/local/cuda-12.4/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install runtime Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled packages from builder
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy source
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY pyproject.toml /app/

# Install editable
RUN pip install -e .

CMD ["python3.11", "-m", "src", "train", "/app/configs/local/train.yaml"]
```

**Validation:**
```bash
# Test build (without GPU libraries yet)
docker build --target builder -t bgb-builder:test .
docker build -t bgb:test .
```

---

#### Step 1.2: Add mamba-ssm Compilation

**Add to builder stage:**
```dockerfile
# After PyTorch installation

# Compile causal-conv1d (5-10 min)
RUN pip install --no-build-isolation --no-cache-dir causal-conv1d==1.5.2

# Download mamba-ssm source
RUN mkdir -p /tmp/mamba_src && \
    cd /tmp/mamba_src && \
    wget https://files.pythonhosted.org/packages/ba/2d/fbd909f6e6d48c491a9ed7ae68e8a890d8409aba4a6356741e2a9c6adad5/mamba_ssm-2.2.5.tar.gz && \
    tar -xzf mamba_ssm-2.2.5.tar.gz

# Compile mamba-ssm (10-15 min)
RUN pip install --no-build-isolation --no-cache-dir /tmp/mamba_src/mamba_ssm-2.2.5
```

**Validation:**
```bash
docker build --target builder -t bgb-builder:mamba .
docker run --rm bgb-builder:mamba python3.11 -c "from mamba_ssm import Mamba2; print('✅ Mamba imported')"
```

---

#### Step 1.3: Apply PR #708 Patch

**Add to builder stage:**
```dockerfile
# After mamba-ssm compilation

# Copy patch script
COPY deploy/modal/patch_mamba_pr708.py /tmp/

# Apply patch
RUN python3.11 /tmp/patch_mamba_pr708.py

# Verify patch
RUN python3.11 -c "
from pathlib import Path
tri_dir = Path('/usr/local/lib/python3.11/dist-packages/mamba_ssm/ops/triton')
files = ['ssd_chunk_scan.py', 'ssd_chunk_state.py', 'ssd_state_passing.py', 'ssd_combined.py']
assert all('.to(tl.int64)' in (tri_dir / f).read_text() for f in files), 'Patch failed'
print('✅ PR #708 patch verified')
"
```

**Validation:**
```bash
docker build --target builder -t bgb-builder:patched .
docker run --rm bgb-builder:patched python3.11 -c "
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
print('✅ Mamba with PR #708 patch works')
"
```

---

#### Step 1.4: Install PyTorch Geometric

**Add to builder stage:**
```dockerfile
# After mamba-ssm patch

# Install PyG with pre-built wheels (EXACT URLs from Modal)
RUN pip install \
    torch_scatter \
    torch_sparse \
    torch_cluster \
    torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.5.0+cu124.html && \
    pip install torch-geometric==2.6.1

# Verify PyG
RUN python3.11 -c "import torch_geometric; print(f'✅ PyG {torch_geometric.__version__}')"
```

**Validation:**
```bash
docker build --target builder -t bgb-builder:pyg .
docker run --rm bgb-builder:pyg python3.11 -c "
from torch_geometric.nn import SSGConv
print('✅ PyG imported successfully')
"
```

---

#### Step 1.5: Install Remaining Dependencies

**Add to builder stage:**
```dockerfile
# After PyG installation

# Install remaining dependencies (EXACT versions from pyproject.toml)
RUN pip install \
    pytorch-tcn==1.2.3 \
    scipy==1.11.4 \
    scikit-learn==1.3.2 \
    matplotlib>=3.5.0 \
    seaborn>=0.11.0 \
    pandas>=2.0.0 \
    wandb>=0.16.0 \
    einops>=0.7.0 \
    mne>=1.5.0 \
    pydantic>=2.0.0 \
    pyedflib>=0.1.30 \
    pyyaml>=6.0.0 \
    click>=8.1.7 \
    rich>=13.0.0 \
    tqdm>=4.64.0

# NOTE: tensorboard is dev-only, not needed for runtime
```

**Validation:**
```bash
docker build -t bgb:complete .

# Test all imports (CORRECT package name: src.brain_brr)
docker run --rm bgb:complete python3.11 -c "
import torch
import mamba_ssm
import torch_geometric
from src.brain_brr.models.detector import SeizureDetector
print('✅ All imports successful')
"
```

---

### Phase 2: docker-compose.yml Creation

#### Step 2.1: Base Service Definition

```yaml
# docker-compose.yml (repo root)
version: '3.8'

services:
  train-base:
    build:
      context: .
      dockerfile: Dockerfile
    image: brain-go-brr:v3.6.1

    # GPU access (requires nvidia-container-toolkit)
    runtime: nvidia

    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      - BGB_NAN_DEBUG=1

    volumes:
      # Cache (read-only) - MUST use absolute path, no ~
      - ${CACHE_DIR}:/cache/tusz:ro
      # Results (read-write)
      - ./results:/results:rw
      # Configs (read-only)
      - ./configs:/app/configs:ro

    shm_size: 8gb
```

**NOTE:** `runtime: nvidia` requires nvidia-container-toolkit. Alternative: run with `docker compose run --gpus all <service>`

**Validation:**
```bash
# Create .env file first (MUST expand $USER manually, Docker doesn't do it)
cat > .env << EOF
CACHE_DIR=$HOME/eeg-cache/tusz
EOF

# Test config parsing
docker compose config | head -20
```

---

#### Step 2.2: Add Smoke Test Service

```yaml
services:
  # ... train-base ...

  smoke-test:
    extends: train-base
    container_name: bgb-smoke

    environment:
      - BGB_SMOKE_TEST=1
      - BGB_LIMIT_FILES=50

    command: python3.11 -m src train /app/configs/local/smoke.yaml
```

**Validation:**
```bash
# Dry run (don't actually train)
# NOTE: Must use --gpus all if runtime: nvidia doesn't work
docker compose run --gpus all --rm smoke-test python3.11 -c "
import torch
assert torch.cuda.is_available(), 'No CUDA'
print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
"
```

---

#### Step 2.3: Add Full Training Service

```yaml
services:
  # ... smoke-test ...

  train:
    extends: train-base
    container_name: bgb-train

    command: python3.11 -m src train /app/configs/local/train.yaml

    restart: unless-stopped
```

**Validation:**
```bash
# Test resume flag (CORRECT: boolean only, no path argument)
docker-compose run --rm train \
  python3.11 -m src train /app/configs/local/train.yaml --resume

# Should auto-detect newest checkpoint in /results/checkpoints/
```

---

#### Step 2.4: Add Development Service

```yaml
services:
  # ... train ...

  dev:
    extends: train-base
    container_name: bgb-dev

    volumes:
      # Override: editable source code
      - ./src:/app/src:rw
      - ${CACHE_DIR}:/cache/tusz:ro
      - ./results:/results:rw
      - ./configs:/app/configs:ro

    command: bash
    stdin_open: true
    tty: true
```

**Validation:**
```bash
# Drop into shell
docker compose run --rm dev

# Inside container, verify editable mode (CORRECT package name):
root@bgb-dev:/app# python3.11 -c "import src.brain_brr; print(src.brain_brr.__file__)"
# Should show: /app/src/brain_brr/__init__.py (mounted volume, not copied)
```

---

### Phase 3: .dockerignore Configuration

```bash
# .dockerignore (repo root)
# Prevent unnecessary files from slowing build

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg-info/
dist/
build/

# Virtual environments
.venv/
venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Data (mount as volume instead)
cache/
data_ext4/
results/

# Git
.git/
.gitignore

# Docs (exclude most, keep critical ones)
docs/
*.md
!README.md
!DOCKER_IMPLEMENTATION_PLAN.md
!CLAUDE.md
!INSTALLATION.md
!CHANGELOG.md

# Tests
tests/

# Secrets
.env
*.key
*.pem
```

**Validation:**
```bash
# Check what gets sent to build context
docker build --no-cache -t bgb:test . 2>&1 | grep "Sending build context"
# Should be < 10MB (just src/ and configs/)
```

---

### Phase 4: Testing & Validation

#### Step 4.1: Build Validation

```bash
# Full build with timing
time docker build -t brain-go-brr:v3.6.1 .

# Expected time: ~20-30 minutes first build
# Expected size: ~7GB final image

# Check image size
docker images brain-go-brr:v3.6.1
```

---

#### Step 4.2: Smoke Test Validation

```bash
# Prerequisites check
ls ~/eeg-cache/tusz/train/manifest.json  # Must exist
ls ~/eeg-cache/tusz/dev/manifest.json    # Must exist

# Run smoke test (1 epoch, 50 files, ~5 min)
docker compose up smoke-test

# Expected output (KEY LINES):
# [CACHE] ✅ Using valid cache: 4667 NPZ files
# [MODEL] Model parameters: 31,474,186
# [GRADIENTS] Last 100 batches: P50=X.XX | IQR=X.XX | P95=X.XX | Max=X.XX
# [TRAIN] ✅ Epoch 1/1 complete
```

---

#### Step 4.3: Full Training Test (1 Epoch Only)

```bash
# CLI has NO --max-epochs flag! Use smoke config (already 1 epoch)
docker compose run --rm train \
  python3.11 -m src train /app/configs/local/smoke.yaml

# Verify mid-epoch checkpointing works:
ls -lh ./results/checkpoints/mid_epoch_*.pt
# Should see files if epoch > 30 minutes
```

---

#### Step 4.4: Resume Test

```bash
# Run 1 epoch first (use smoke config)
docker compose run --rm train \
  python3.11 -m src train /app/configs/local/smoke.yaml

# Resume from checkpoint (auto-detects last.pt or mid_epoch_*.pt)
# NOTE: --resume is a boolean flag, no path argument!
docker compose run --rm train \
  python3.11 -m src train /app/configs/local/train.yaml \
  --resume

# Verify it loads checkpoint:
# [RESUME] Loading checkpoint from /results/checkpoints/last.pt
# [RESUME] Resuming from epoch 2, batch 0
```

---

### Phase 5: Known Limitations & Future Work

#### ❌ Not Implemented (Out of Scope)
1. **Multi-GPU / DDP Training**
   - Reason: No distributed code in `src/brain_brr/train/loop.py`
   - Future: Add DDP support, then update docker-compose.yml

2. **Pre-built GHCR Images**
   - Reason: Need GitHub Actions CI/CD pipeline
   - Future: Add `.github/workflows/docker-build.yml`

3. **Cache Tarball Downloads**
   - Reason: Need hosting (S3/GCS/GHCR packages)
   - Future: Upload cache, update docs with real URL

4. **Health Checks with psutil**
   - Reason: psutil not in dependencies
   - Fix: Either add `psutil` to builder or use simpler health check:
     ```yaml
     healthcheck:
       test: ["CMD", "python3.11", "-c", "import torch; torch.cuda.is_available()"]
     ```

5. **Jupyter Integration**
   - Reason: jupyterlab not installed in image
   - Future: Add `RUN pip install jupyterlab` to builder if needed

---

### Phase 6: Performance Validation

#### Step 6.1: Compare Docker vs Native Speed

```bash
# Native training (baseline)
time make s  # Smoke test natively
# Record: Time for 1 epoch

# Docker training (comparison)
time docker compose up smoke-test
# Record: Time for 1 epoch

# Expected: Docker should be 99-99.5% of native speed
# Overhead: <1% from containerization
```

---

#### Step 6.2: Memory Usage Validation

```bash
# Monitor Docker stats during training
docker stats bgb-train

# Expected for batch_size=8 (RTX 4090):
# GPU: ~20GB VRAM
# RAM: ~8-12GB
# CPU: 0-100% (num_workers=0 in WSL2)
```

---

## Implementation Checklist

### Prerequisites
- [ ] NVIDIA Container Toolkit installed (`nvidia-ctk --version`)
- [ ] Docker Compose v2 installed (`docker compose version`)
- [ ] Cache exists at `~/eeg-cache/tusz/{train,dev}/`
- [ ] GPU verified (`nvidia-smi`)

### Phase 1: Dockerfile
- [ ] Step 1.1: Base structure builds
- [ ] Step 1.2: mamba-ssm compiles
- [ ] Step 1.3: PR #708 patch applies
- [ ] Step 1.4: PyG installs
- [ ] Step 1.5: All imports work

### Phase 2: docker-compose.yml
- [ ] Step 2.1: Base service defined
- [ ] Step 2.2: Smoke test service works
- [ ] Step 2.3: Full training service works
- [ ] Step 2.4: Dev service works

### Phase 3: .dockerignore
- [ ] Created and build context < 10MB

### Phase 4: Testing
- [ ] Step 4.1: Build completes (~20 min)
- [ ] Step 4.2: Smoke test passes (1 epoch)
- [ ] Step 4.3: Full training runs (1 epoch)
- [ ] Step 4.4: Resume works

### Phase 5: Documentation
- [ ] Update README.md with Docker quick start
- [ ] Create DOCKER_USER_GUIDE.md for end users
- [ ] Update CLAUDE.md with Docker commands

### Phase 6: Performance
- [ ] Docker vs native speed comparison (<1% overhead)
- [ ] Memory usage validated

---

## Success Criteria

✅ **Docker implementation is complete when:**

1. `docker build -t brain-go-brr:v3.6.1 .` succeeds in <30 min
2. `docker-compose up smoke-test` completes in ~5 min with no errors
3. `docker-compose up train` runs full training with mid-epoch checkpointing
4. `docker-compose run --rm train ... --resume` loads checkpoints correctly
5. Docker training speed is 99%+ of native speed
6. All imports work: torch, mamba_ssm, torch_geometric, src modules

---

## Next Steps

1. **Create Dockerfile** (Phase 1) - Start with Step 1.1
2. **Test each stage** - Validate after every step
3. **Create docker-compose.yml** (Phase 2) - Build incrementally
4. **Run smoke test** (Phase 4.2) - First full validation
5. **Document for users** (Phase 5) - After everything works

---

**Owner:** @jj
**Timeline:** 1-2 days implementation + testing
**Priority:** Medium (nice-to-have for reproducibility, not blocking)
