# Modal Volume Architecture - FINAL CLARITY

## Overview
After thorough investigation and cleanup (Sep 25, 2025), here's the **CORRECT** Modal architecture:

## 1. Cache Architecture

### Local Caches
- **Location**: `cache/tusz/`
  - `train/`: 4,667 NPZ files (306GB)
  - `val/`: 1,832 NPZ files (143GB)
  - **Total**: 449GB
- **Smoke tests**: Use SAME cache with `BGB_LIMIT_FILES=3` env var
- **NO SEPARATE SMOKE CACHE EXISTS OR IS NEEDED**

### S3 Bucket Structure (raw data only)
- **Bucket**: `s3://brain-go-brr-eeg-data-20250919/`
- **Contents**:
  ```
  tusz/edf/           # Raw EDF data (266GB)
  ```
  NPZ caches should NOT be used directly from S3 for training.

### Modal Cache Location
- **Method**: Modal persistent volume (fast SSD)
- **Mount point**: `/results/cache/tusz/{train,dev}`
- Populate once (e.g., by copying from local or S3), then reuse across runs.

## 2. Modal Persistence Volume

### Purpose
- **Name**: `brain-go-brr-results`
- **Size**: 431 MiB (after cleanup)
- **Purpose**: Store training outputs ONLY (not caches!)

### Directory Structure (AFTER CLEANUP)
```
/results/
├── smoke/          # Smoke test results
│   ├── checkpoints/
│   ├── tensorboard/
│   └── wandb/
├── train/          # Full training results (created when needed)
├── checkpoints/    # Model checkpoints (created when needed)
├── tensorboard/    # TB logs (created when needed)
└── wandb/          # W&B logs (created when needed)
```

### DELETED Directories
- `/results/results/` - DELETED (confusing duplicate)

## 3. Modal Function Volumes

```python
@app.function(
    volumes={
        "/data": data_mount,         # S3: Raw EDF data (read‑only)
        "/results": results_volume,  # Persistent: Training outputs + NPZ caches
    }
)
```

## 4. Key Insights

### Why Modal Volumes Were Confusing
1. **Keep caches on the Modal SSD volume** — avoids S3 throttling and network variability
2. **One-time population** — copy NPZ caches into `/results/cache/tusz` once and reuse
3. **EDFs via S3 mount are fine** — raw inputs are streamed; caches are hot-path and must be local

### Smoke Tests Don't Need Separate Cache
- Smoke tests use `BGB_LIMIT_FILES=3` (local) or `=50` (Modal)
- This limits how many files are loaded from the SAME cache
- No need to maintain separate smoke cache!

## 5. Configuration Summary

### Local Training
```yaml
data:
  cache_dir: cache/tusz  # Local cache directory
```

### Modal Training
```yaml
data:
  cache_dir: /results/cache/tusz  # Modal persistent SSD volume
```

### Environment Variables
- `BGB_LIMIT_FILES=3` - Local smoke tests
- `BGB_LIMIT_FILES=50` - Modal smoke tests
- `BGB_SMOKE_TEST=1` - Skip seizure sampling for smoke tests

## 6. Cache Population Strategy

### Why NOT S3 Mount for Cache?
- **S3 is SLOW**: Network latency kills training performance
- **S3 throttling**: Can hit rate limits with parallel data loading
- **S3 costs**: Egress charges for repeatedly reading 450GB cache
- **Reliability**: Network hiccups can crash training

### Why Modal SSD Volume?
- **FAST**: Local NVMe SSD with microsecond latency
- **Reliable**: No network issues
- **Persistent**: Survives between runs
- **Cost-effective**: One-time population, then free reads

### One-Time Cache Population
```python
@app.function(
    volumes={
        "/results": results_volume,
        "/s3_cache": modal.CloudBucketMount(...),  # Temporary S3 mount
    },
    timeout=7200,  # 2 hours for 450GB copy
    cpu=24,
    memory=65536,
)
def populate_cache():
    """One-time copy of cache from S3 to Modal SSD volume."""
    import shutil
    from pathlib import Path

    src = Path("/s3_cache")  # S3 mount
    dst = Path("/results/cache/tusz")  # SSD volume

    # Copy train
    print(f"Copying {src}/train to {dst}/train...")
    shutil.copytree(src / "train", dst / "train", dirs_exist_ok=True)

    # Copy dev
    print(f"Copying {src}/dev to {dst}/dev...")
    shutil.copytree(src / "dev", dst / "dev", dirs_exist_ok=True)

    # Verify
    train_files = len(list((dst / "train").glob("*.npz")))
    dev_files = len(list((dst / "dev").glob("*.npz")))
    print(f"✅ Populated cache: {train_files} train, {dev_files} dev files")
```

### Commands Reference
```bash
# One-time cache population from S3 to Modal SSD
modal run deploy/modal/app.py --action populate-cache

# Verify cache on Modal volume
modal run deploy/modal/app.py --action verify-cache
```

### Modal Volume Management
```bash
# Inspect volume
modal run deploy/modal/inspect_volume.py

# Clean up volume
modal run deploy/modal/cleanup_volume.py
```

### Modal Training
```bash
# Smoke test
modal run --detach deploy/modal/app.py --action train --config configs/modal/smoke.yaml

# Full training
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml
```

## 7. Final Architecture

```
LOCAL                      MODAL (volume)
─────                      ──────────────
cache/tusz/
  ├── train/ ──sync──► /results/cache/tusz/train/
  └── dev/   ──sync──► /results/cache/tusz/dev/  # TUSZ naming: dev not val!

results/                  /results/
  └── local_runs/           ├── cache/tusz/{train,dev}
                             ├── smoke/
                             └── train/
```

## Summary
- **Caches**: Stored on Modal persistent volume at `/results/cache/tusz`
- **Results**: Stored on Modal persistent volume at `/results/`
- **Smoke tests**: Use same cache with file limits (no separate cache)
 - **Avoid S3 for caches**: eliminate throttling and timeouts on the hot path
