# Modal Volume Architecture - FINAL CLARITY

## Overview
After thorough investigation and cleanup (Sep 25, 2025), here's the **CORRECT** Modal architecture:

## 1. Cache Architecture

### Local Caches
- **Location**: `cache/tusz/`
  - `train/`: 4,667 NPZ files (306GB)
  - `dev/`: 1,832 NPZ files (143GB)
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

## 6. Commands Reference

### Populate Modal Cache (one-time)
```bash
# Example: copy from local to Modal volume (run in a context where the volume is mounted)
rsync -a cache/tusz/ /results/cache/tusz/
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
  └── dev/   ──sync──► /results/cache/tusz/dev/

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
