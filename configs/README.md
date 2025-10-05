# Brain-Go-Brr V3 Configuration Files

## 🧠 Architecture: V3 Dual-Stream (TCN + BiMamba + GNN)

All configs use the V3 dual-stream architecture:
- **TCN**: Multi-scale temporal feature extraction (8 layers, stride=16)
- **Node Stream**: Per-electrode BiMamba (d_model=64, 6 layers, headdim=8)
- **Edge Stream**: Per-edge BiMamba (d_model=16, 2 layers, headdim=4)
- **GNN**: Vectorized SSGConv with **DYNAMIC** Laplacian PE (α=0.05, k=16)
  - **Dynamic PE**: Recomputed per timestep from evolving adjacency (EvoBrain approach)
  - **Vectorized**: All 960 timesteps computed in parallel (100-1000x faster than loops)
  - **Numerical Stability**: FP32 eigendecomposition with sign consistency
- **Total Parameters**: ~31.5M

## 📁 Directory Structure

```
configs/
├── local/                    # Local WSL2/Linux configs (RTX 4090 optimized)
│   ├── smoke.yaml           # Quick test (1 epoch, 3 files via BGB_LIMIT_FILES=3)
│   └── train.yaml           # Full training (100 epochs; train/dev official splits)
│
└── modal/                    # Modal cloud GPU configs (A100-80GB optimized)
    ├── smoke.yaml           # Quick cloud test (1 epoch, 50 files)
    └── train.yaml           # Full cloud training (100 epochs; train/dev official splits)
```

## ⚡ Critical Cache Configuration

### Local (RTX 4090)
```yaml
data:
  cache_dir: cache/tusz     # MUST use existing cache for train/dev splits
```
- **Location**: `cache/tusz/train/` (≈4667 NPZ) + `cache/tusz/dev/` (≈1832 NPZ)
- **Warning**: Do NOT use `cache/v2.6_full/` - it's empty!

### Modal (A100)
```yaml
data:
  cache_dir: /results/cache/tusz  # Persistent SSD volume
```
- **Location**: `/results/cache/tusz/train/` + `/results/cache/tusz/dev/`
- **Built once**: First run builds cache, all subsequent runs reuse
- **NOT on S3**: Cache is on fast Modal SSD, never touches S3 after build

## 🚀 Usage Examples

### Local Training (RTX 4090)
```bash
# Smoke test (requires environment variables)
BGB_LIMIT_FILES=3 BGB_SMOKE_TEST=1 python -m src train configs/local/smoke.yaml
# Or use the helper script:
./run_smoke_test.sh

# Full training (watch in tmux recommended)
tmux new -s train
python -m src train configs/local/train.yaml
```

### Modal Cloud Training (A100)
```bash
# Test Mamba CUDA first
modal run deploy/modal/app.py --action test-mamba

# Smoke test (app.py sets BGB_LIMIT_FILES=50 automatically)
modal run --detach deploy/modal/app.py --action train --config configs/modal/smoke.yaml

# Full training (detached)
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml

# Monitor
modal app list
modal app logs <app-id>
```

**Note**: Modal resource allocation (24 CPU cores, 96GB RAM, A100-80GB GPU) is configured in `deploy/modal/app.py`, NOT in the YAML configs.

## 🔑 Key Configuration Differences

| Setting | Local (RTX 4090) | Modal (A100-80GB) | Why Different |
|---------|------------------|-------------------|---------------|
| **Batch Size** | **8** | **48** | 24GB vs 80GB VRAM |
| **Gradient Accumulation** | **1** | **1** | No accumulation needed |
| **Effective Batch** | **8** | **48** | Local: 2× faster, Modal: 6× faster |
| Mixed Precision | false | true | RTX 4090 FP16 can cause NaNs |
| Learning Rate | 1.0e-4 | 8.0e-5 | Stability vs. large batch scaling |
| Workers | 0 | 4 | WSL2 fix vs parallel IO |
| Prefetch Factor | 2 | 2 | Conservative for memory |
| Persistent Workers | false | false | Prevents spawn delay + memory leaks |
| **Mid-Epoch Checkpoints** | **30 min** | **30 min** | Crash recovery for long epochs |
| **Mid-Epoch Keep** | **3** | **3** | Rolling window of snapshots |
| Cache Location | `cache/tusz/` | `/results/cache/tusz/` | Filesystem differences |

## ⚡ Oct 2025 Speed Optimizations

### Local (RTX 4090): batch_size 4 → 8 ✅
**Discovered**: Actual VRAM usage was only 10GB @ batch_size=4 (not the expected 16GB)
**Change**: Doubled batch_size to 8 for 2× speed improvement
**Result**:
- Epoch time: **~3 hours** (2× faster than batch=4)
- Full training (100 epochs): **~300 hours** (~12.5 days)
- VRAM usage: **~20GB** (4GB safety buffer)
- **Status**: ✅ Production stable

### Modal (A100-80GB): batch_size 32×2 → 48×1 ✅
**Previous safe config**: batch_size=32, gradient_accumulation_steps=2
**Current config**: batch_size=48, gradient_accumulation_steps=1
**Result**:
- Batches/epoch: **~1283** (33% fewer than 32×2)
- Epoch time: **~1 hour** (verified in smoke tests)
- Full training (100 epochs): **~100 hours**, **~$319 @ $1.50/hr**
- Peak memory: **~58GB** (smoke test verified)
- **Status**: ✅ Production ready

### Mid-Epoch Checkpointing (v3.6.1) ✅
**Problem**: Long epochs (1-7 hours) meant crashes could waste significant progress
**Solution**: Save checkpoints every 30 minutes during training
**Configuration** (both local + modal):
```yaml
checkpoint_interval: 1                # Every epoch
mid_checkpoint_interval_s: 1800       # Every 30 minutes
mid_epoch_keep: 3                     # Keep last 3 snapshots
```
**Impact**:
- **Max loss on crash**: 30 minutes (vs 1-7 hours)
- **Storage**: ~375MB (3 × 125MB checkpoints)
- **Resume**: Automatic from newest mid-epoch or last.pt

## 🚨 CRITICAL: A100 OOM Lessons Learned (Original Crash)

### The Oct 2025 Crash (batch_size=64 + grad_accum=1)

**What Happened**: Modal training OOM'd at batch 0 backward pass
**Error**: `CUDA out of memory. Tried to allocate 10.69 GiB. GPU 0 has total capacity of 79.25 GiB of which 2.04 GiB is free. Process 1 has 77.20 GiB memory in use.`

**Root Cause**:
- `batch_size=64` + `gradient_accumulation_steps=1` → forward+backward processes **64 samples at once**
- Peak memory during backward: **~77GB** (exceeds A100-80GB capacity)

**Why batch_size Matters More Than You Think**:
```
batch_size=64, grad_accum=1:  Peak = 77GB (CRASH ❌)
batch_size=32, grad_accum=2:  Peak = 50GB (SAFE ✅)
```
Both have **same effective batch** (64) and **same learning dynamics**, but:
- **batch_size** controls **peak memory** (forward+backward activations)
- **grad_accum** splits backward into smaller chunks, reducing peak

**The Fix**:
```yaml
# BROKEN (causes OOM):
batch_size: 64
gradient_accumulation_steps: 1

# SAFE (proven stable):
batch_size: 32
gradient_accumulation_steps: 2
# Effective batch still 64, peak memory reduced by ~35%
```

**Key Takeaway**: On A100-80GB with V3 architecture (31M params, deep TCN+Mamba+GNN):
- ✅ **batch_size ≤ 32** is SAFE
- ❌ **batch_size = 64** causes OOM during backward
- ✅ Use **gradient_accumulation** to increase effective batch without OOM

See `docs/05-training/modal.md` for full memory profiling.

## ⚠️ Common Pitfalls

1. **Wrong Cache Directory**:
   - ❌ Local: `cache/v2.6_full/` (empty)
   - ✅ Local: `cache/tusz/{train,dev}/` (4667 + 1832 files) - Using TUSZ's 'dev' naming!

2. **Modal Cache Misconception**:
   - ❌ "Cache is on S3 causing slowdowns"
   - ✅ Cache is on Modal SSD from first run

3. **A100 OOM from Large Batch Size**:
   - ❌ `batch_size=64` + `gradient_accumulation_steps=1` → 77GB peak (CRASH)
   - ✅ `batch_size=32` + `gradient_accumulation_steps=2` → 50GB peak (SAFE)

3. **Mixed Precision on RTX 4090**:
   - ❌ `mixed_precision: true` causes NaN losses
   - ✅ Keep `mixed_precision: false` for stability

4. **PyG Not Installed**:
   - Run `make setup-gpu` locally (installs PyG from prebuilt wheels for torch 2.2.2+cu121)
   - Modal image includes PyG automatically

## 🏗️ Model Configuration (All Configs)

```yaml
model:
  architecture: v3  # V3 dual-stream architecture

  tcn:
    num_layers: 8
    kernel_size: 7
    stride_down: 16

  mamba:
    n_layers: 6
    d_model: 512
    d_state: 16
    conv_kernel: 4  # CUDA constraint
    # Node/Edge streams use different params (see detector.py)

  graph:
    enabled: true
    # PyG is required; no separate toggle needed
    alpha: 0.05    # SSGConv mixing parameter
    k_eigenvectors: 16  # Laplacian PE dimension
    use_dynamic_pe: true  # Dynamic PE (recomputed per timestep)

    # V3-specific edge stream config:
    edge_mamba_layers: 2
    edge_mamba_d_state: 8
    edge_mamba_d_model: 16  # Must be multiple of 8
    edge_similarity_margin: 0.01  # v3.2.0: Safety margin from ±1 boundaries
```

## 📊 Expected Training Times

| Config | Platform | Time/Epoch | Total Time | Cost |
|--------|----------|------------|------------|------|
| Local Train | RTX 4090 | ~3-4 hours | ~300-400 hours | Electricity |
| Modal Train | A100-80GB | ~1 hour | ~100 hours | ~$250 @ $2.50/hr |
| Smoke Test | Both | ~5 mins | 5 mins | Minimal |

**Note**: Local is slower due to smaller batch size (4 vs 32) but more stable on 24GB VRAM.

## 🔧 Environment Variables

| Variable | Purpose | When to Use |
|----------|---------|-------------|
| `BGB_SMOKE_TEST=1` | Skip seizure sampling | Local smoke tests |
| `BGB_LIMIT_FILES=3` | Limit to 3 files | Local smoke (required!) |
| `BGB_LIMIT_FILES=N` | Limit to N files | Testing |
| `BGB_DISABLE_TQDM=1` | Disable progress bars | Modal (automatic) |
| `BGB_NAN_DEBUG=1` | Debug NaN losses | If training fails |
| `BGB_FORCE_MANIFEST_REBUILD=1` | Rebuild cache manifest | If cache corrupted |

## 📈 Post-Processing (All Configs)

```yaml
postprocessing:
  hysteresis:
    tau_on: 0.86   # Seizure onset threshold
    tau_off: 0.78  # Seizure offset threshold
  morphology:
    opening_kernel: 11
    closing_kernel: 31
  duration:
    min_duration_s: 3.0
    max_duration_s: 600.0
```

## 🎯 Training Strategy

1. **Focal Loss**: Essential for 12:1 class imbalance
2. **Balanced Sampling** (`use_balanced_sampling: true`):
   - **Training**: Uses manifest to oversample seizures (8% → ~30% in batches)
   - **Validation**: Always uses natural distribution (~8% seizures) for real metrics
   - **Why different**: Train on balanced data to learn, validate on real distribution
3. **Cosine Schedule**: Smooth learning rate decay
4. **Early Stopping**: Patience=5 on sensitivity@10FA/24h

## 🚨 Critical Notes

- **V3 Architecture**: Dual-stream with learned edge dynamics
- **BiMamba2 headdim**: Node=8, Edge=4 (prevents CUDA fallback)
- **Cache Reuse**: Both platforms reuse existing preprocessed cache
- **Smoke tests**: Local needs manual env vars, Modal sets automatically
- **Full state-space modeling**: No Conv1d fallbacks with proper headdim
