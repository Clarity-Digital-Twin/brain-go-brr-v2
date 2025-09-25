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

### S3 Bucket Structure
- **Bucket**: `s3://brain-go-brr-eeg-data-20250919/`
- **Contents**:
  ```
  tusz/edf/           # Raw EDF data (266GB)
  cache/tusz/train/   # Preprocessed training NPZ files (306GB)
  cache/tusz/dev/     # Preprocessed dev NPZ files (143GB)
  ```
- **Upload Status**: In progress via `aws s3 sync` (117.8/125.3 GB as of Sep 25)

### Modal Cache Access
- **Method**: CloudBucketMount (direct S3 mount)
- **Mount point**: `/cache/`
- **Config in app.py**:
  ```python
  cache_mount = modal.CloudBucketMount(
      "brain-go-brr-eeg-data-20250919",
      secret=s3_secret,
      key_prefix="cache/tusz/",  # Maps to /cache/{train,dev}/
      read_only=True,
  )
  ```

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
- `/results/cache/` - DELETED (we use S3 mount now)
- `/results/results/` - DELETED (confusing duplicate)

## 3. Modal Function Volumes

```python
@app.function(
    volumes={
        "/data": data_mount,      # S3: Raw EDF data
        "/cache": cache_mount,    # S3: Preprocessed NPZ cache
        "/results": results_volume,  # Persistent: Training outputs
    }
)
```

## 4. Key Insights

### Why Modal Volumes Were Confusing
1. **Modal volumes are S3-backed internally** - That's why we saw "SlowDown signal from S3"
2. **Double upload was redundant** - Uploading to Modal volume = uploading to S3
3. **Direct S3 mount is cleaner** - Skip the middleman, mount S3 directly

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
  cache_dir: /cache  # S3 mount point (set by app.py)
```

### Environment Variables
- `BGB_LIMIT_FILES=3` - Local smoke tests
- `BGB_LIMIT_FILES=50` - Modal smoke tests
- `BGB_SMOKE_TEST=1` - Skip seizure sampling for smoke tests

## 6. Commands Reference

### Check S3 Status
```bash
# List S3 contents
aws s3 ls s3://brain-go-brr-eeg-data-20250919/cache/tusz/ --recursive --summarize

# Upload cache (if needed)
aws s3 sync cache/tusz/ s3://brain-go-brr-eeg-data-20250919/cache/tusz/
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
LOCAL                          S3                           MODAL
─────                          ──                           ─────
cache/tusz/
  ├── train/ ──────sync──────> cache/tusz/train/ ─mount─> /cache/train/
  └── dev/   ──────sync──────> cache/tusz/dev/   ─mount─> /cache/dev/

results/                                                    /results/
  └── local_runs/                                            ├── smoke/
                                                              └── train/
```

## Summary
- **Caches**: Live in S3, mounted directly to Modal at `/cache/`
- **Results**: Stored in Modal persistent volume at `/results/`
- **Smoke tests**: Use same cache with file limits (no separate cache)
- **No redundancy**: Direct S3 mount eliminates double storage