# Configuration Status - FINAL CHECK ✅

> Archived note: Canonical configuration lives under `docs/03-configuration/` and `docs/05-training/`.
> See `docs/ARCHIVE_MAPPING.md` for where this content was integrated.

## Local Configs (RTX 4090 - 24GB VRAM)

### ✅ train.yaml - OPTIMIZED
```yaml
batch_size: 4                    # Uses 16GB (safe margin)
use_dynamic_pe: true             # V3 enabled
semi_dynamic_interval: 5         # PE every 19.5ms (192 eigendecomps)
mixed_precision: false           # Disabled (causes NaNs)
warmup_ratio: 0.10              # Increased for stability
gradient_clip: 0.5              # Strong clipping
```
**Memory**: 16GB/24GB used ✅
**Status**: RUNNING NOW in tmux session `v3_full`

### ✅ smoke.yaml - CORRECT
```yaml
batch_size: 1                    # Minimal for quick test
use_dynamic_pe: true             # V3 enabled
semi_dynamic_interval: 10        # Reduced memory (96 eigendecomps)
mixed_precision: false           # Disabled for safety
```
**Memory**: Safe for smoke tests ✅
**Status**: READY

## Modal Configs (A100 - 80GB VRAM)

### ✅ train.yaml - PERFECT
```yaml
batch_size: 64                   # A100 can handle large batches
use_dynamic_pe: true             # V3 enabled
semi_dynamic_interval: 1         # FULL dynamic (every timestep)
mixed_precision: true            # A100 tensor cores safe
```
**Memory**: ~60GB/80GB used ✅
**Resources**: 24 CPUs, 96GB RAM (critical!)
**Status**: NO CHANGES NEEDED

### ✅ smoke.yaml - PERFECT
```yaml
batch_size: 16                   # Quick validation
use_dynamic_pe: true             # V3 enabled
semi_dynamic_interval: 1         # Full dynamic (A100 has memory)
mixed_precision: true            # Safe on A100
```
**Memory**: ~15GB/80GB used ✅
**Status**: NO CHANGES NEEDED

## Summary

| Config | Dynamic PE | Interval | Batch | Memory | Status |
|--------|-----------|----------|-------|---------|--------|
| **local/train** | ✅ | 5 | 4 | 16/24GB | **RUNNING** |
| **local/smoke** | ✅ | 10 | 1 | <5GB | READY |
| **modal/train** | ✅ | 1 | 64 | 60/80GB | PERFECT |
| **modal/smoke** | ✅ | 1 | 16 | 15/80GB | PERFECT |

## Critical Settings Applied

### All Configs Have:
- ✅ `use_balanced_sampling: true` (prevents zero-seizure batches)
- ✅ `loss: focal` with proper alpha/gamma
- ✅ `architecture: v3` (dual-stream enabled)
- ✅ `graph.enabled: true` with k=3 sparsity

### NaN Protection (Applied):
- ✅ Decoder clamping before logits
- ✅ Focal loss probability clamping
- ✅ Training loop NaN sanitization
- ✅ Debug waypoints throughout model

## Next Steps

1. **Local**: Monitor `tmux attach -t v3_full` for stability
2. **Modal**: Ready to deploy with `modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml`

**ALL CONFIGS ARE CORRECT AND OPTIMIZED** 🚀
