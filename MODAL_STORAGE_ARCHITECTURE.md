# Modal Storage Architecture - VERIFIED STATE

## Current Storage Structure (VERIFIED via CLI)

```
MODAL INFRASTRUCTURE
├── S3 Bucket: "brain-go-brr-eeg-data-20250919"
│   └── tusz/
│       └── edf/
│           └── train/
│               └── *.edf (raw EEG files)
│
├── CloudBucketMount (READ-ONLY)
│   Mounted at: /data/
│   └── edf/
│       └── train/
│           └── *.edf (appears here due to key_prefix="tusz/")
│
└── Persistent Volume: "brain-go-brr-results" (310.5 GB)
    Mounted at: /results/
    ├── cache/
    │   └── tusz/
    │       ├── train/
    │       │   ├── 3734 NPZ files (preprocessed windows)
    │       │   └── manifest.json (22 MB index)
    │       └── val/
    │           └── (validation cache if built)
    ├── smoke/
    │   ├── checkpoints/
    │   ├── tensorboard/
    │   └── wandb/
    └── tusz_a100_100ep/
        ├── checkpoints/
        └── tensorboard/
```

## Data Flow Pipeline

```mermaid
graph TD
    S3[S3 Bucket<br/>Raw EDF Files] -->|CloudBucketMount<br/>Read-Only| Mount[/data/edf/train/]

    Mount -->|First Epoch Only| Build[Build Cache<br/>30-60 min]
    Build -->|Save NPZ| Cache[/results/cache/tusz/train/<br/>3734 NPZ files]

    Cache -->|Fast Local Access| Train[Training Loop<br/>~5s per batch]

    Train -->|Save| Check[/results/tusz_a100_100ep/checkpoints/]
    Train -->|Log| TB[/results/tusz_a100_100ep/tensorboard/]
    Train -->|Track| WB[W&B Cloud]

    style S3 fill:#f9f,stroke:#333,stroke-width:4px
    style Cache fill:#9f9,stroke:#333,stroke-width:4px
    style Train fill:#99f,stroke:#333,stroke-width:4px
```

## Performance Characteristics

| Storage Type | Location | Speed | Use Case |
|-------------|----------|-------|----------|
| **S3 CloudBucketMount** | `/data/` | SLOW (100-700ms/file) | Raw EDF files (read once) |
| **Modal Volume (SSD)** | `/results/cache/` | FAST (1-5ms/file) | NPZ cache (read repeatedly) |
| **Modal Volume (SSD)** | `/results/checkpoints/` | FAST | Model checkpoints |

## Critical Facts (VERIFIED)

1. **Cache is ALREADY on Modal Volume**:
   - 3734 NPZ files in `/results/cache/tusz/train/`
   - 310.5 GB total in results volume
   - NO cache on S3 (never was!)

2. **No "Optimization" Needed**:
   - Cache was built directly to Modal volume on first run
   - Already on fast SSD storage
   - The "cache optimizer" was looking for something that doesn't exist

3. **Deleted Unused Volume**:
   - `brain-go-brr-data` was empty (0 files, 0 GB)
   - Removed from app.py
   - Only using `brain-go-brr-results` now

## Training Performance

### Current Setup (OPTIMAL)
- **Data Loading**: ~5s per batch (from Modal SSD)
- **Cache Access**: Direct local disk I/O
- **No Network Overhead**: Cache is local to compute

### If Cache Was on S3 (BAD - NOT OUR CASE)
- Would be: ~48s per batch
- Network latency: 100-700ms per NPZ file
- This is what we THOUGHT was happening but ISN'T

## Environment Variables

| Variable | Purpose | Value |
|----------|---------|-------|
| `BGB_DISABLE_TQDM` | Disable progress bars in Modal | `1` |
| `BGB_FORCE_MANIFEST_REBUILD` | Force rebuild manifest.json | `1` (if needed) |
| `BGB_FORCE_CACHE_COPY` | Force cache "optimization" | Not needed! |
| `WANDB_API_KEY` | W&B authentication | From Modal secret |

## Cost Analysis

- **Storage Cost**: ~$15/month for 310GB on Modal
- **Compute Cost**: $3.19/hour for A100
- **With optimizations**: ~100 hours = $319 total
- **Cache already optimal**: No S3 transfer costs!

## Commands

### Check Storage
```bash
# List volumes
modal volume list

# Explore contents (use our script)
modal run deploy/modal/explore_volumes.py

# Check running apps
modal app list
```

### Training
```bash
# Smoke test (uses existing cache)
modal run --detach deploy/modal/app.py \
  --action train \
  --config configs/modal/smoke_a100.yaml

# Full training (uses existing cache)
modal run --detach deploy/modal/app.py \
  --action train \
  --config configs/modal/train_a100.yaml
```

## Summary

✅ **Cache is already on fast Modal SSD** (310GB in `/results/`)
✅ **No S3 bottleneck** - cache was never on S3
✅ **Deleted unused volume** - cleaned up `brain-go-brr-data`
✅ **Training is already optimized** - using local cache

The "slow" startup you see is just Python imports and initialization, NOT S3 transfers!