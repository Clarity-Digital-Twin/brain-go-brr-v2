# Modal Deployment - Complete Architecture Guide

## 🎯 TL;DR - How It Actually Works

```
LOCAL CACHE → S3 BUCKET → MODAL S3 MOUNT → TRAINING
    ↓            ↓              ↓               ↓
cache/tusz/  s3://...     /cache/      /results/{smoke,train}/
```

**NO SEPARATE SMOKE CACHE EXISTS!** Smoke tests use the same cache with `BGB_LIMIT_FILES=50`.

## 📁 Storage Architecture (CRITICAL TO UNDERSTAND)

### 1. S3 Bucket (`brain-go-brr-eeg-data-20250919`)
**Purpose**: Central storage for all EEG data and preprocessed caches

```
s3://brain-go-brr-eeg-data-20250919/
├── tusz/edf/           # Raw EDF files (266GB)
│   ├── train/          # Training EDF files
│   └── dev/            # Development EDF files
└── cache/tusz/         # Preprocessed NPZ files (449GB total)
    ├── train/          # 4,667 NPZ files (306GB)
    └── dev/            # 1,832 NPZ files (143GB)
```

### 2. Modal S3 Mounts (READ-ONLY)
**Purpose**: Direct access to S3 data without copying

- `/data/` → S3 `tusz/edf/` (raw EDF files)
- `/cache/` → S3 `cache/tusz/` (preprocessed NPZ files)

### 3. Modal Persistence Volume (`brain-go-brr-results`)
**Purpose**: Store training outputs ONLY (checkpoints, logs, metrics)

```
/results/                   # Modal persistence volume
├── smoke/                  # Smoke test outputs
│   ├── checkpoints/
│   ├── tensorboard/
│   └── wandb/
└── train/                  # Full training outputs (created when needed)
```

**NEVER STORE CACHES HERE!** Caches come from S3 mount.

## 🚀 Quick Commands

```bash
# Smoke test (50 files via BGB_LIMIT_FILES)
modal run --detach deploy/modal/app.py --action train --config configs/modal/smoke.yaml

# Full A100 training (all 4,667 train files)
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml

# Resume from checkpoint
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml --resume true

# Inspect persistence volume
modal run deploy/modal/inspect_volume.py

# Clean up old cache dirs (if needed)
modal run deploy/modal/cleanup_volume.py
```

## ⚠️ Common Confusions (RESOLVED)

### Q: Do smoke tests need a separate cache?
**A**: NO! They use the SAME cache with `BGB_LIMIT_FILES=50` (set automatically).

### Q: Why not store cache in persistence volume?
**A**: S3 is 6x cheaper and cache is read-only after creation.

### Q: What's the directory structure?
**A**:
- Cache: `/cache/{train,dev}/` (S3 mount)
- Outputs: `/results/{smoke,train}/` (persistence volume)

## 📝 Key Files
- `app.py` - Main Modal deployment script
- `inspect_volume.py` - Check persistence volume contents
- `cleanup_volume.py` - Remove old cache directories
- ~~`explore_volumes.py`~~ - DELETED (outdated)

