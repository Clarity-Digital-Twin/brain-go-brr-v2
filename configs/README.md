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
├── local/                          # RTX 4090 optimized configs
│   ├── smoke_bimamba.yaml          # BiMamba2 smoke (3 files, 1 epoch)
│   ├── train_bimamba.yaml          # BiMamba2 full training (100 epochs)
│   ├── smoke_fla.yaml              # FLA smoke
│   ├── train_fla.yaml              # FLA full training (baseline)
│   ├── train_fla_exp1_reg.yaml     # FLA Exp1: Stronger regularization
│   ├── train_fla_exp2_lr.yaml      # FLA Exp2: Lower learning rate (standby)
│   ├── train_fla_exp3_smaller.yaml # FLA Exp3: Smaller model (RETIRED)
│   └── train_fla_exp4_cyclic.yaml  # FLA Exp4: Cyclic LR restarts (COMPLETE)
│
└── modal/                          # Modal A100-80GB configs
    ├── smoke_bimamba.yaml          # BiMamba2 smoke (50 files, 1 epoch)
    ├── train_bimamba.yaml          # BiMamba2 full training
    ├── smoke_fla.yaml              # FLA smoke
    └── train_fla.yaml              # FLA full training
```

## 🧪 Hyperparameter Experiment Configs (FLA Stack)

Following the research methodology in `TRAINING_METHODOLOGY.md`, three targeted experiments test overfitting hypotheses:

```
configs/local/
  train_fla.yaml              # Baseline FLA (31M params)
  train_fla_exp1_reg.yaml     # Exp1: Stronger regularization (HIGH priority)
  train_fla_exp2_lr.yaml      # Exp2: Lower learning rate (available, lower priority)
  train_fla_exp3_smaller.yaml # Exp3: Smaller model 17M params (rejected)
  train_fla_exp4_cyclic.yaml  # Exp4: Cyclic LR restarts (COMPLETE)
```

**Experiment Details:**

| Experiment | Hypothesis | Changes | Output Dir |
|------------|-----------|---------|------------|
| **Baseline** | N/A | N/A | `results/local_fla_training` |
| **Exp1: Regularization** | Insufficient regularization causing overfitting | dropout 0.1→0.2, weight_decay 0.01→0.05 | `results/local_fla_exp1_reg` |
| **Exp2: Lower LR** | Learning rate too high, late instability | lr 1e-4→5e-5, warmup 0.03→0.05 | `results/local_fla_exp2_lr` |
| **Exp3: Smaller Model** | Model too large for dataset (4,667 files) | 6→4 layers, 512→384 dim (31M→17M) | `results/local_fla_exp3_smaller` |
| **Exp4: Cyclic LR** | Cyclic LR restarts improve sensitivity@FA targets | cosine→cosine_restarts, t_initial=10, t_mult=2, eta_min=1e-6, patience 20→15 | `results/local_fla_exp4_cyclic` |

**Status Snapshot (Dec 2025):**
- Exp4: ✅ COMPLETE — TUSZ eval 35.9% sensitivity @ 10 FA/24h (AUROC 0.8654); SSOT: `results/local_fla_exp4_cyclic/eval_results_v2.json`
- Baseline: Historical baseline run notes (superseded by Exp4)
- Exp1: Completed (negative result, model not overfitting)
- Exp2: Optional follow-up (not executed)
- Exp3: Archived (capacity reduction no longer considered)

**Critical Constraints:**
- Each experiment uses a unique `output_dir` to protect checkpoints
- `edge_mamba_d_model` remains 32 to satisfy FLA kernel requirements
- Refer to `EXPERIMENTAL_PLAN.md` for current sequencing and rationale

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

# FLA full training (baseline)
tmux new -s train-fla
python -m src train configs/local/train_fla.yaml

# FLA Experiment 1 (Stronger Regularization)
tmux new -s exp1-reg
export BGB_NAN_DEBUG=1
python -m src train configs/local/train_fla_exp1_reg.yaml

# FLA Experiment 2 (Lower Learning Rate)
tmux new -s exp2-lr
export BGB_NAN_DEBUG=1
python -m src train configs/local/train_fla_exp2_lr.yaml

# FLA Experiment 4 (Cyclic LR restarts)
tmux new -s exp4-cyclic
export BGB_NAN_DEBUG=1
python -m src train configs/local/train_fla_exp4_cyclic.yaml
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
| Persistent Workers | false | true | Local: num_workers=0 incompatible; Modal: keeps mmap pages warm |
| **Mid-Epoch Checkpoints** | **30 min** | **30 min** | Crash recovery for long epochs |
| **Mid-Epoch Keep** | **3** | **3** | Rolling window of snapshots |
| Cache Location | `cache/tusz_mmap/` | `/results/cache/tusz_mmap/` | Mmap NPY format |

## ⚡ Oct 2025 Speed Optimizations

### Local (RTX 4090): batch_size 4 → 8 ✅
**Discovered**: Actual VRAM usage was only 10GB @ batch_size=4 (not the expected 16GB)
**Change**: Doubled batch_size to 8 for 2× speed improvement
**Result** (MEASURED from FLA training logs):
- **Training time**: **~4.1h** per epoch (7702 batches @ ~2.1s/batch)
- **Validation time**: **~5.5h** per epoch (18528 batches, disk-backed)
- **Total epoch time**: **~9.6h** (training 42.8% + validation 57.2%)
- **Full training (100 epochs)**: **~960 hours** (40 days)
- **VRAM usage**: ~20GB (4GB safety buffer)
- **Status**: ✅ Production stable (Epoch 7/100 in progress)
- **Note**: BiMamba2 may be faster than FLA; exact timing TBD

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

## 📊 Actual Training Times (MEASURED from logs)

| Config | Platform | Training | Validation | Total/Epoch | 100 Epochs |
|--------|----------|----------|------------|-------------|------------|
| **Local FLA** | **RTX 4090** | **~4.1h** | **~5.5h** | **~9.6h** | **960h (40 days)** |
| Modal BiMamba2 | A100-80GB | ~1-2h (docs) | ~5.8h (docs) | ~7-12h (docs) | ~700-1200h |
| Smoke Test | Both | N/A | N/A | ~5 mins | 5 mins |

**CRITICAL FINDINGS (Local FLA on RTX 4090):**
- **Validation is the bottleneck**: Takes 1.3× longer than training (5.5h vs 4.1h)
- **Validation overhead**: 57.2% of total epoch time
- **Training overhead**: 42.8% of total epoch time
- **Epoch 1 anomaly**: 7.2h total (faster due to warmup/cache effects)
- **Epochs 2-6 average**: 10.1h (consistent performance)

**Modal BiMamba2 Status:**
- Training PAUSED at Epoch 6 due to high costs
- Cost so far: $1,118 (6 epochs) = **$186/epoch**
- Estimated 100 epochs: **$18,600** at $4.40/hr (GPU+CPU+RAM)
- **Local training saves $18,600!**

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
4. **Early Stopping**: See "Two-Tier Early Stopping Strategy" below

## 🛑 Two-Tier Early Stopping Strategy

**IMPORTANT**: Configs use different `patience` and `min_epochs` based on purpose:

### Full Training Configs (train_*.yaml)
```yaml
early_stopping:
  patience: 20        # Tolerant of plateaus
  min_epochs: 30      # Don't stop before 30% complete
  metric: sensitivity_at_10fa
```

**Why patience=20?**
- Medical imaging literature: Best clinical checkpoints often come 10-20 epochs AFTER minimum val_loss
- "Second peak" phenomenon: Sensitivity can improve at epochs 30-50 even as val_loss rises
- Original patience=5 stopped baseline FLA at epoch 13 (13% complete) - too aggressive

**Files using patience=20:**
- `train_bimamba.yaml` (local + modal)
- `train_fla.yaml` (local + modal)

### Experiment Configs (*_exp*.yaml)
```yaml
early_stopping:
  patience: 5         # Match baseline training
  min_epochs: 0       # Allow early stopping anytime
  metric: sensitivity_at_10fa
```

**Why keep patience=5 for experiments?**
- **Fair comparison**: Baseline trained with patience=5, experiments must too
- **Scientific method**: Change one variable (hyperparameters), not training procedure
- **Isolate effect**: Test if regularization/LR/size helps, not if longer training helps

**Files using patience=5:**
- `train_fla_exp1_reg.yaml` (stronger regularization)
- `train_fla_exp2_lr.yaml` (lower learning rate)
- `train_fla_exp3_smaller.yaml` (smaller model)

### Smoke Test Configs (smoke_*.yaml)
```yaml
early_stopping:
  patience: 5         # Appropriate for short runs
  min_epochs: 0       # Allow immediate stopping
  metric: sensitivity_at_10fa
```

**Why patience=5 for smoke tests?**
- Smoke tests run 1-3 epochs max (not 100)
- Early stopping rarely triggers anyway
- Fast iteration for debugging

**Decision Tree (Post-Exp4):**
- Exp4 is the current best held-out benchmark (TUSZ eval 35.9% @ 10 FA/24h); SSOT: `results/local_fla_exp4_cyclic/eval_results_v2.json`
- Future experiments should compare against Exp4 (not the old baseline) using the same dev/eval protocol

See `STATUS.md` "Lessons Learned - Early Stopping" for full history and rationale.

## 🚨 Critical Notes

- **V3 Architecture**: Dual-stream with learned edge dynamics
- **BiMamba2 headdim**: Node=8, Edge=4 (prevents CUDA fallback)
- **Cache Reuse**: Both platforms reuse existing preprocessed cache
- **Smoke tests**: Local needs manual env vars, Modal sets automatically
- **Full state-space modeling**: No Conv1d fallbacks with proper headdim
