# Modal Performance Optimization Report

## Executive Summary
Modal A100 performance hinges on three levers:

1) FP16 mixed precision on A100 (tensor cores)  
2) Adequate batch size given v3 dual‑stream memory needs  
3) Vectorized GNN + static Laplacian PE (v3), not per‑timestep loops

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
  batch_size: 48           # v3 dual‑stream; raise if headroom remains
  mixed_precision: true    # A100 FP16 acceleration
```

## Issue 2: Vectorized GNN + Static PE (v3)

The v3 stack eliminates the per‑timestep CPU loop by batching all `(B×T)` graphs into a single disjoint PyG batch, and uses a static Laplacian PE buffer computed once from the 10–20 structural graph. This removes thousands of tiny `Data` allocations per step and repeated eigendecomposition.

Edge temporal stream uses a learned 1→D→1 lift (default D=16) around Bi‑Mamba2 to keep fused CUDA kernels active and improve capacity.

## Issue 3: torch.compile Incompatibility

Mamba CUDA kernels don't support torch.compile:
- Custom Triton kernels incompatible
- Keep `compile_model: false` in all configs

## Performance Impact

Observed outcomes vary by dataset/cache and worker settings. With FP16, vectorized GNN, and proper caching on Modal SSD, expect sub‑second to a few‑seconds per batch on A100‑80GB. Always validate with a smoke run before full training.

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

## CPU and Memory Sizing

- Prefer 16–32 CPU cores to keep 8–12 DataLoader workers fed (2–4 cores per worker).
- Use 64–128 GB RAM to accommodate large validation sets (often larger than training).
- Keep caches on Modal SSD volume (`/results/cache/...`) — never on S3.

## Validation Throughput and Logging

- Validation can have more batches than training (e.g., ~810 vs ~778) and used to appear idle.
- The training loop now logs validation start, 2‑minute heartbeats, and completion with average loss.
- Use `modal app logs <app-id>` to monitor these messages during long validations.

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

✅ Mixed precision enabled (FP16)  
✅ Vectorized GNN path active (no per‑timestep Data churn)  
✅ Static PE buffer computed once  
✅ Cache on Modal SSD (`/results/cache/tusz`)  
✅ Batch size set for v3 memory (e.g., 48); raise if headroom allows  
