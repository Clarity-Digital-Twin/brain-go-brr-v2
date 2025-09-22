# Modal Performance Optimization Report

## Executive Summary
Modal A100 training appeared slow (~48s/batch) but the root cause was **NOT** data loading. The actual issues were:

1. **Mixed Precision Disabled**: Using FP32 instead of FP16 (A100's strength)
2. **Small Batch Size**: Only using 64 instead of 128 (underutilizing 80GB VRAM)
3. **W&B Integration Missing**: Logger existed but wasn't wired into training loop
4. **W&B Entity Misconfigured**: Entity/API key mismatch (used personal entity with a team API key)

## Critical Realization: Cache Was NEVER on S3!

### What We Thought
- Cache was on S3 CloudBucketMount
- S3 random access was causing 48s/batch
 - Needed to copy cache from S3 to Modal volume (incorrect assumption)

### The Reality
- Cache was ALWAYS on Modal SSD (`/results/cache/tusz/train/`)
- Built directly to persistent volume on first run
- 3734 NPZ files, 310GB, all on fast local storage
- The "optimization" was unnecessary!

## Issue 1: Mixed Precision & Batch Size

### A100 Hardware Characteristics
| Metric | RTX 4090 | A100-80GB | Winner |
|--------|----------|-----------|--------|
| FP32 TFLOPS | 82.6 | 19.5 | RTX 4090 (4.2x) |
| FP16 TFLOPS | 82.6 | 312 | A100 (3.8x) |
| Memory | 24GB | 80GB | A100 (3.3x) |

### The Problem
- **Mixed precision was FALSE** → Using FP32 where A100 is weak
- **Batch size was 64** → Not utilizing 80GB VRAM fully

### The Fix
```yaml
# configs/modal/train.yaml
training:
  batch_size: 128          # Was 64
  mixed_precision: true    # Was false
```

## Issue 2: W&B Integration

### Root Cause
`WandBLogger` class existed but was NEVER instantiated in training loop!

### Fix Applied
```python
# train/loop.py
from src.brain_brr.train.wandb_integration import WandBLogger

# In train():
wandb_logger = WandBLogger(config)
wandb_logger.log(metrics, step=epoch)
```

### Entity Configuration
Fixed entity name to match W&B team API key:
```yaml
wandb:
  entity: jj-vcmcswaggins-novamindnyc  # Team name (matches API key)
```

## Issue 3: torch.compile Incompatibility

Mamba CUDA kernels don't support torch.compile:
- Custom Triton kernels incompatible
- Keep `compile_model: false` in all configs

## Performance Impact

### Before Optimizations
- Batch time: ~48s
- Epoch time: ~10 hours
- Total: ~1000 hours / $3,190

### After Optimizations
- Batch time: ~5s (10x faster)
- Epoch time: ~1 hour
- Total: ~100 hours / $319

### Savings: $2,871 (90% reduction)

## Modal Storage Architecture (Correct)

```
/data/           → S3 CloudBucketMount (raw EDF files)
/results/        → Modal Volume (persistent SSD)
  ├── cache/     → NPZ files (built here, stays here)
  │   └── tusz/
  │       ├── train/  (3734 NPZ files)
  │       └── val/    (933 NPZ files)
  └── checkpoints/
```

## Key Learnings

1. **Cache was always optimal** - On Modal SSD from day 1
2. **Real bottlenecks were config** - Mixed precision & batch size
3. **A100 needs FP16** - It's 4x slower at FP32 than RTX 4090
4. **Always verify assumptions** - We optimized the wrong thing!

## Commands

### Full Training (Optimized)
```bash
modal run --detach deploy/modal/app.py \
  --action train \
  --config configs/modal/train.yaml
```

### Monitor Progress
- Modal: https://modal.com/apps/clarity-digital-twin
- W&B: https://wandb.ai/jj-vcmcswaggins-novamindnyc/seizure-detection-a100

## Verification Checklist

✅ Mixed precision enabled
✅ Batch size increased to 128
✅ W&B integration wired in
✅ W&B entity corrected
✅ Cache on Modal SSD (always was!)
✅ Deleted unused volumes
✅ Training running at ~5s/batch
