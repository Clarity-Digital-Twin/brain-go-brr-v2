# Release Notes

## v3.0.1 - CRITICAL Patient Leakage Fix (2025-09-24)

### 🚨 EMERGENCY RELEASE - ALL PREVIOUS MODELS INVALID

**Type**: Critical Bug Fix
**Severity**: P0 BLOCKER

**WARNING: ALL PREVIOUS TRAINING RESULTS ARE INVALID**

This emergency release fixes a **CRITICAL BUG** where patients appeared in both training and validation splits, completely invalidating all previous validation metrics.

### What Happened

During a critical code review, we discovered that patient `aaaaagxr` (and potentially hundreds of others) appeared in BOTH training and validation splits with different recording sessions. This means:

1. **All validation metrics were artificially inflated**
2. **Models learned patient-specific patterns rather than generalizable seizure patterns**
3. **Any published results using these models are invalid**

### Critical Fixes

- **PATIENT LEAKAGE ELIMINATED** (P0 Blocker):
  - Previous file-level alphabetical splitting mixed patient data across train/val splits
  - Now using TUSZ official train/dev/eval splits with enforced patient disjointness
  - Runtime validation fails fast if any patient appears in multiple splits
  - Files: `src/brain_brr/data/tusz_splits.py` (new), `src/brain_brr/train/loop.py`

- **FA Curve Threshold Bug** (P0 Blocker):
  - `sensitivity_at_fa_rates()` was passing ignored threshold parameter
  - Now properly clones post_cfg and sets tau_on/off for each FA target
  - File: `src/brain_brr/eval/metrics.py`

### Also Fixed
- **TensorBoard Import**: Now optional with try/except pattern
- **TCN Channels Config**: Removed unused field that was ignored by implementation
- **Manifest Strictness**: NPZ files without labels now excluded with warnings
- **CLI Threshold Export**: Robust key coercion for "10", 10, or 10.0

### Verification
```
[SPLIT STATS] OFFICIAL TUSZ SPLITS:
  Train: 579 patients, 4667 files
  Val:   53 patients, 1832 files
  ✅ PATIENT DISJOINTNESS VERIFIED - No leakage!
```

### Required Migration Steps

1. **Stop All Training Immediately**
   ```bash
   tmux kill-session -t train
   modal app stop <app-id>
   ```

2. **Delete Contaminated Cache**
   ```bash
   rm -rf cache/tusz/train_windows/
   rm -rf cache/tusz/val_windows/
   rm -rf /results/cache/tusz/  # Modal
   ```

3. **Update Configuration**
   ```yaml
   data:
     data_dir: data_ext4/tusz/edf  # Parent directory
     split_policy: official_tusz    # REQUIRED
   ```

4. **Rebuild Cache with Proper Splits**
   ```bash
   python -m src build-cache \
     --data-dir data_ext4/tusz/edf \
     --cache-dir cache/tusz
   ```

5. **Restart Training from Scratch**
   ```bash
   # Local
   make train-local

   # Modal
   modal run --detach deploy/modal/app.py \
     --action train --config configs/modal/train.yaml
   ```

### Impact Assessment

- **Research**: Any results must be re-run with proper splits
- **Production**: Models in production are unreliable and must be replaced
- **Publications**: Consider retracting or updating any published results

### Lessons Learned

1. **ALWAYS use patient-level splits for medical data**
2. **NEVER split by files when files belong to patients**
3. **ALWAYS validate disjointness before training**
4. **Official splits exist for a reason - use them!**

---

## v3.0.0 - V3 Dual-Stream Architecture with Dynamic LPE (2025-09-24)

### 🎉 Major Release: Production-Ready V3 Architecture

Complete implementation of dual-stream processing with dynamic Laplacian positional encoding, representing the culmination of 6 months of research and development.

### ✨ Key Highlights

#### Dual-Stream Innovation
- **Node Stream**: 19× parallel BiMamba2 for electrode features
- **Edge Stream**: 171× BiMamba2 learning adjacency from data
- **Dynamic LPE**: Time-evolving positional encoding (k=16 eigenvectors)
- **Vectorized GNN**: 10× speedup processing all timesteps at once

#### Production Hardening
- Comprehensive NaN protection throughout model
- Memory-optimized for both RTX 4090 and A100
- Numerical stability fixes in eigendecomposition
- Training currently running on both platforms

#### Performance Metrics
- **Model**: 31,475,722 parameters
- **RTX 4090**: 16GB VRAM (batch_size=4, interval=5)
- **A100**: 60GB VRAM (batch_size=64, full dynamic)
- **Speedup**: 10× faster GNN operations

### 🔄 Breaking Changes
- V2 heuristic graphs → V3 learned adjacency
- Static PE → Dynamic PE with configurable intervals
- Sequential GNN → Vectorized parallel processing
- Batch sizes optimized per platform

### 📦 Installation
```bash
git checkout v3.0.0
make setup && make setup-gpu
```

### 🚀 Quick Start
```bash
# Local (RTX 4090)
tmux new -s v3_full
make train-local

# Modal (A100)
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml
```

### 📚 Documentation
- Architecture: `docs/V3_ARCHITECTURE_AS_IMPLEMENTED.md`
- Changelog: `CHANGELOG.md`
- Configuration: `configs/README.md`

---

## v2.3.0 - TCN Architecture + Training Robustness (2025-09-23)

### 🚀 Major Architecture Change
**Replaced U-Net + ResCNN with Temporal Convolutional Networks (TCN)**

Complete architectural refactor with TCN for superior temporal modeling + massive training stability improvements.

### ✨ Key Highlights

#### Architecture Revolution
- **NEW**: TCN encoder (8 layers, dilated convolutions)
- **KEPT**: Bidirectional Mamba-2 (6 layers, O(N) complexity)
- **RESULT**: ~34M parameters, faster training, better gradients

#### Training Robustness 🛡️
- **NaN Protection**: Comprehensive handling with isolation and diagnostics
- **Focal Loss Fix**: Numerical stability (clamped logits, bounded p_t)
- **Gradient Monitoring**: Enhanced tracking and intelligent clipping
- **Recovery**: Can now continue training through intermittent NaN losses

#### Critical Fixes 🔧
- **NaN Accumulator**: Fixed bug where one NaN contaminated all future losses
- **Focal Underflow**: Prevented (1-p_t)^gamma → 0 with high confidence
- **Performance Tests**: Hardware-aware thresholds (RTX: 125ms, A100: 110ms)
- **Mixed Precision**: Better FP16 stability with optional sanitization

### 🔧 Configuration

```yaml
model:
  architecture: tcn  # TCN + Mamba hybrid

  tcn:
    num_layers: 8
    channels: [64, 128, 256, 512]
    kernel_size: 7
    dropout: 0.15
    stride_down: 16
    use_cuda_optimizations: true

  mamba:
    n_layers: 6
    d_model: 512
    d_state: 16
    conv_kernel: 4  # CUDA constraint
    # v2.6 preview: Dynamic GNN + LPE will use learned adjacency from an edge Mamba stream (no heuristic cosine/correlation graphs). PyG SSGConv (alpha=0.05) + Laplacian PE (k=16) is the canonical backend.
```

### 📊 Training Progress
- Local: Loss converging healthily (~2.5-3.0)
- Modal A100: 100-epoch training in progress
- Expected: ~100 hours, ~$319 total cost

### ⚠️ Breaking Changes
- Model checkpoints from v2.2.x incompatible
- Config requires `tcn:` section (not `unet:`/`rescnn:`)

---

## v2.1.0 - Modal Optimized: 10x Faster, 90% Cheaper (2025-09-22)

### 🚀 Major Performance Breakthrough

This release delivers **10x training speedup** and **90% cost reduction** for Modal cloud training through critical optimizations and bug fixes.

### Key Improvements

#### ⚡ Performance Optimizations
- **Mixed Precision (FP16)**: Leverages A100 tensor cores - 3.8x faster
- **Batch Size 128**: Full 80GB VRAM utilization - 2x throughput
- **Result**: ~5s/batch (was ~48s/batch)
- **Cost**: $319 for 100 epochs (was $3,190 for same)

#### 📊 W&B Integration Fixed
- WandBLogger properly wired into training loop
- Team entity configuration corrected
- Full cloud experiment tracking working

#### 💾 Critical Discovery
- **Cache was ALWAYS on Modal SSD** - never on S3!
- Removed unnecessary "cache optimizer"
- Real bottleneck was FP32 + small batch size

#### 📚 Documentation Overhaul
- Complete reorganization into logical sections
- Balanced sampling optimization documented (7200x speedup)
- Removed all outdated/incorrect documentation

### Quick Upgrade

```bash
git pull origin main
git checkout v2.1.0

# Verify your Modal configs have:
# - mixed_precision: true
# - batch_size: 128
# - entity: your-wandb-team-name

# Launch optimized training
modal run --detach deploy/modal/app.py \
  --action train \
  --config configs/modal/train.yaml
```

### Performance Metrics

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Batch Time | 48s | 5s | **10x faster** |
| Total Time | 1000hr | 100hr | **10x faster** |
| Cost | $3,190 | $319 | **90% cheaper** |

### Breaking Changes
None - pure performance improvements

### Known Issues
- First epoch: 30-60min cache build (one-time)
- Mamba CUDA: d_conv coerced 5→4

---

**Full Changelog**: https://github.com/Clarity-Digital-Twin/brain-go-brr-v2/compare/v0.2.0...v2.1.0

## v0.2.0 - Critical Bug Fixes (2025-09-21)

### 🚨 Critical Fixes Required

This release fixes **P0 blockers** that prevented seizure detection in training. If you're using v0.1.0, **upgrade immediately**.

### What's Fixed

#### CSV Parser (CRITICAL)
- **Before**: Training detected 0% seizures due to broken TUSZ CSV_BI parser
- **After**: Parser correctly reads all seizure annotations
- **Impact**: Training now finds 313 partial and 55 full seizure windows in test cache

#### Seizure Type Detection
- **Before**: Only looked for "seiz" label (doesn't exist in TUSZ)
- **After**: Detects all TUSZ types: gnsz, fnsz, cpsz, absz, spsz, tcsz, tnsz, mysz
- **Impact**: Complete seizure coverage in training data

#### Training Stability
- Implemented BalancedSeizureDataset with SeizureTransformer's formula
- Added hard guards to prevent training with 0 seizures
- Fixed Modal pipeline limiting to 50 files instead of 3734

#### Configuration Cleanup
- Reorganized configs into clean `local/` and `modal/` structure
- Fixed WSL2 compatibility issues
- Verified A100 optimizations for cloud training

### Quick Upgrade

```bash
git pull
git checkout v0.2.0

# For local training
python -m src train configs/local/train.yaml

# For Modal cloud
modal run --detach deploy/modal/app.py::train
```

### Verification

After cache build, you should see:
```
✅ Cache build complete + manifest: partial=XXX, full=XX, none=XXXX
```

If `partial > 0`, the fixes are working correctly.

### Documentation

- See `configs/README.md` for new config structure
- Check `CHANGELOG.md` for complete fix details
- Review `FIX_SUMMARY_20250921.md` for technical details

---

**Full Changelog**: https://github.com/Clarity-Digital-Twin/brain-go-brr-v2/compare/v0.1.0...v0.2.0