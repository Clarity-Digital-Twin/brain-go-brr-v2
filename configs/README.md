# Brain-Go-Brr V4 Configuration Files

## 🧠 Architectures: BiMamba2 & FLA Dual-Stream Stacks

All configs implement the V3 dual-stream detector (TCN → temporal streams → GNN →
fusion → decoder). Two temporal stacks are supported:

- **BiMamba2** (production baseline)
  - Node stream: BiMamba2 (6 layers, d_model=64)
  - Edge stream: BiMamba2 (2 layers, d_model=16)
- **FLA – Gated DeltaNet** (research candidate)
  - Node/edge streams: Gated DeltaNet with delta-rule gating
  - Edge stream uses `d_model=32` (FLA causal_conv1d requirement)

Both share:
- **TCN** front-end (8 layers, stride=16, dropout=0.15)
- **GNN** back-end (SSGConv ×2, dynamic Laplacian PE with k=16)
- **Dynamic PE** recomputed per timestep with sign consistency safeguards
- **Focal loss** + hysteresis/morphology post-processing

## 📁 Directory Structure

```
configs/
├── local/                      # RTX 4090 optimized configs
│   ├── smoke_bimamba.yaml      # BiMamba2 smoke (3 files, 1 epoch)
│   ├── train_bimamba.yaml      # BiMamba2 full training (100 epochs)
│   ├── smoke_fla.yaml          # FLA smoke
│   └── train_fla.yaml          # FLA full training
│
└── modal/                      # Modal A100-80GB configs
    ├── smoke_bimamba.yaml      # BiMamba2 smoke (50 files, 1 epoch)
    ├── train_bimamba.yaml      # BiMamba2 full training
    ├── smoke_fla.yaml          # FLA smoke
    └── train_fla.yaml          # FLA full training
```

## ⚡ Critical Cache Configuration

### Local (RTX 4090)
```yaml
data:
  cache_dir: cache/tusz_mmap  # Memory-mapped NPY cache for train/dev splits
```
- **Location**: `cache/tusz_mmap/train/` (9334 NPY files: 4667 × 2 for data+labels) + `cache/tusz_mmap/dev/` (3664 NPY files: 1832 × 2)
- **Format**: Uncompressed NPY (mmap-enabled) replaces old compressed NPZ

### Modal (A100)
```yaml
data:
  cache_dir: /results/cache/tusz_mmap  # Persistent SSD volume (mmap NPY)
```
- **Location**: `/results/cache/tusz_mmap/train/` + `/results/cache/tusz_mmap/dev/`
- **Built once**: populate-cache copies from S3, subsequent runs reuse
- **Format**: Memory-mapped NPY for minimal RAM usage (<1 GB vs 387 GB)

## 🚀 Usage Examples

### Local Training (RTX 4090)
```bash
# BiMamba2 smoke test (requires environment variables)
BGB_LIMIT_FILES=3 BGB_SMOKE_TEST=1 python -m src train configs/local/smoke_bimamba.yaml

# FLA smoke test
BGB_LIMIT_FILES=3 BGB_SMOKE_TEST=1 python -m src train configs/local/smoke_fla.yaml

# BiMamba2 full training (watch in tmux recommended)
tmux new -s train-bimamba
python -m src train configs/local/train_bimamba.yaml

# FLA full training
tmux new -s train-fla
python -m src train configs/local/train_fla.yaml
```

### Modal Cloud Training (A100)
```bash
# Test Mamba CUDA first
modal run deploy/modal/app.py --action test-mamba

# BiMamba2 smoke test (app.py sets BGB_LIMIT_FILES=50 automatically)
modal run --detach deploy/modal/app.py --action train --config configs/modal/smoke_bimamba.yaml

# FLA smoke test
modal run --detach deploy/modal/app.py --action train --config configs/modal/smoke_fla.yaml

# BiMamba2 full training (detached)
modal run --detach deploy/modal/app.py --action train --config configs/modal/train_bimamba.yaml

# FLA full training (detached)
modal run --detach deploy/modal/app.py --action train --config configs/modal/train_fla.yaml

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
| Cache Location | `cache/tusz_mmap/` | `/results/cache/tusz_mmap/` | Mmap NPY format |

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
- Epoch time: **~7-12 hours** (training 1-2h + validation 5.8h documented; smoke tests don't include full validation)
- Full training (100 epochs): **~700-1200 hours**, **$3,400-$5,300+** @ $4.40/hr (GPU+CPU+RAM)
- Peak memory: **~58GB** (smoke test verified)
- **Status**: ⚠️ **EXPENSIVE** - validation overhead makes Modal training cost-prohibitive

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
   - ❌ `cache/v2.6_full/` (old path, no longer exists)
   - ✅ Local: `cache/tusz_mmap/{train,dev}/` (9334 + 3664 NPY files) - Using TUSZ's 'dev' naming!

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
   - Run `make setup-gpu` locally (installs PyG from prebuilt wheels for PyTorch 2.5.0+cu124)
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

| Config | Platform | Time/Epoch | Total Time |
|--------|----------|------------|------------|
| Local Train | RTX 4090 | ~3 hours | ~300 hours (12.5 days) |
| Modal Train | A100-80GB | ~7-12 hours | ~700-1200 hours |
| Smoke Test | Both | ~5 mins | 5 mins |

**Note**: Modal training is EXPENSIVE due to validation overhead (5.8h documented per epoch). BiMamba2 training PAUSED at Epoch 6 due to high costs ($1,118 spent). Modal cost: ~$4.40/hr (GPU+CPU+RAM) = $3,400-$5,300+ for 100 epochs. Local training preferred for cost-effectiveness.

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
