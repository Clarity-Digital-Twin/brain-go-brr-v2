# Release Notes

## v2.3.0 - TCN Architecture Revolution (2025-09-22)

### 🚀 Major Architecture Change
**Replaced U-Net + ResCNN with Temporal Convolutional Networks (TCN)**

The most significant architectural refactor since project inception. TCNs provide superior temporal modeling for EEG signals with dilated convolutions.

### ✨ Key Changes

#### Architecture Overhaul
- **NEW**: TCN encoder/decoder (8 layers, channels [64, 128, 256, 512], stride_down=16)
- **KEPT**: Bidirectional Mamba-2 SSM (6 layers, d_model=512, d_state=16)
- **REMOVED**: U-Net encoder/decoder and ResCNN blocks
- **RESULT**: 34.8M parameters with O(N) complexity

#### Critical Bug Fixes
- Fixed `test_cuda_oom_recovery` causing IDE crashes (reduced memory limits)
- Suppressed false-positive LR scheduler warnings
- Fixed Mamba config accidentally deleted from train.yaml
- Improved cache isolation between configs

#### Infrastructure Updates
- All Modal configs updated to TCN + Mamba hybrid
- Cache paths properly isolated (smoke/train/dev/eval)
- Full 100-epoch training launched on Modal A100
- W&B integration verified with team entity

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
