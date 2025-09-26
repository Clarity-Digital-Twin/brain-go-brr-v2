# MODAL CACHE STRATEGY - SINGLE SOURCE OF TRUTH

## THE PROBLEM
We have cache confusion between S3 mounts and Modal SSD volumes. This document is the FINAL, DEFINITIVE strategy.

## THE SOLUTION: USE MODAL SSD VOLUME

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

## ARCHITECTURE

### Current State (WRONG in app.py)
```python
# app.py currently has this WRONG setup:
cache_mount = modal.CloudBucketMount(...)  # S3 mount at /cache
volumes={
    "/cache": cache_mount,  # WRONG! This is slow S3!
}
cache_dir = "/cache"  # WRONG! Points to S3!
```

### Target State (CORRECT)
```python
# NO cache_mount from S3!
volumes={
    "/results": results_volume,  # Persistent SSD volume only
}
cache_dir = "/results/cache/tusz"  # CORRECT! SSD volume!
```

## IMPLEMENTATION PLAN

### 1. Modal Volume Structure
```
/results/                    # Modal persistent volume (SSD)
├── cache/                   # Cache data (one-time population)
│   └── tusz/
│       ├── train/          # 4667 NPZ files (~306GB)
│       │   ├── *.npz
│       │   ├── manifest.json
│       │   └── _dataset_index.json
│       └── dev/            # 1832 NPZ files (~143GB)
│           ├── *.npz
│           ├── manifest.json
│           └── _dataset_index.json
├── checkpoints/            # Training outputs
├── tensorboard/
└── wandb/
```

### 2. One-Time Cache Population
We need to copy cache from S3 to Modal SSD volume ONCE:

```python
@app.function(
    volumes={
        "/results": results_volume,
        "/s3_cache": cache_mount,  # Temporary S3 mount
    },
    timeout=7200,  # 2 hours for 450GB copy
    cpu=16,
    memory=32768,
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

### 3. Fix app.py Training Function
```python
@app.function(
    gpu="A100-80GB",
    volumes={
        "/data": data_mount,      # Raw EDF data (rarely used)
        "/results": results_volume,  # SSD with cache AND outputs
    },
    # NO /cache mount!
)
def train(...):
    # Use SSD cache
    cache_dir = "/results/cache/tusz"  # NOT /cache!
    ...
```

### 4. Config Updates
```yaml
# configs/modal/smoke.yaml and train.yaml
data:
  cache_dir: /results/cache/tusz  # SSD volume, NOT /cache!
```

## COMMANDS

### First Time Setup
```bash
# 1. Populate cache from S3 to Modal SSD (ONE TIME ONLY)
modal run deploy/modal/app.py --action populate-cache

# 2. Verify cache
modal run deploy/modal/app.py --action verify-cache
```

### Training (After Cache Populated)
```bash
# Smoke test
modal run deploy/modal/app.py --action train --config configs/modal/smoke.yaml

# Full training
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml
```

## KEY POINTS

1. **NEVER** use S3 mount for cache during training
2. **ALWAYS** use Modal SSD volume at `/results/cache/tusz`
3. **ONE-TIME** population from S3 to SSD
4. **REUSE** the populated cache across all training runs
5. **SMOKE** tests use SAME cache with `BGB_LIMIT_FILES=50`

## VERIFICATION CHECKLIST

- [ ] app.py REMOVED cache_mount CloudBucketMount
- [ ] app.py train() uses `/results/cache/tusz` NOT `/cache`
- [ ] configs/modal/*.yaml point to `/results/cache/tusz`
- [ ] populate_cache() function added to app.py
- [ ] Cache populated to Modal SSD volume (4667 + 1832 files)
- [ ] Training logs show "Using valid Modal SSD cache"

## COST ANALYSIS

### S3 Mount (BAD)
- 450GB cache × N reads = massive egress costs
- Network latency = slower training
- Risk of throttling/failures

### SSD Volume (GOOD)
- One-time 450GB transfer from S3
- Then FREE, FAST local reads forever
- Modal volume costs ~$0.15/GB/month = $67/month for cache
- BUT training is 3-10x faster!

## THIS IS THE WAY
No more confusion. Modal cache lives on SSD at `/results/cache/tusz`. Period.