# Modal Storage Architecture

## Overview

Modal deployment uses a hybrid storage approach: S3 for raw data, Modal volumes for everything else.

## Storage Components

### 1. S3 Bucket (Raw Data)
- **Bucket**: `brain-go-brr-eeg-data-20250919`
- **Contents**: Raw EDF files only
- **Path**: `tusz/edf/train/*.edf`
- **Access**: Read-only CloudBucketMount
- **Mount point**: `/data/edf/train/`

### 2. Modal Volume (Everything Else)
- **Volume**: `brain-go-brr-results` (310.5 GB)
- **Mount point**: `/results/`
- **Contents**:
  ```
  /results/
  ├── cache/tusz/
  │   ├── train/           # 3734 NPZ files (preprocessed)
  │   │   └── manifest.json # 22MB index file
  │   └── val/             # Validation cache
  ├── checkpoints/         # Model weights
  ├── tensorboard/         # Training logs
  └── wandb/              # W&B artifacts
  ```

## Data Flow

1. **First Epoch**: Reads raw EDF from S3 → preprocesses → saves NPZ to Modal volume
2. **Subsequent Epochs**: Reads NPZ directly from Modal SSD (10x faster)
3. **No S3 bottleneck**: Cache never leaves Modal infrastructure

## Performance Characteristics

| Storage | Location | Latency | Use Case |
|---------|----------|---------|----------|
| S3 Mount | `/data/` | 100-700ms/file | Raw EDF (read once) |
| Modal SSD | `/results/cache/` | 1-5ms/file | NPZ cache (read many) |
| Modal SSD | `/results/checkpoints/` | 1-5ms/file | Model weights |

## Why This Architecture Works

1. **Raw data on S3**: Cheap long-term storage for large EDF files
2. **Cache on Modal SSD**: Fast repeated access during training
3. **Built in-place**: Cache created directly on SSD, no copying needed
4. **Persistent across runs**: Cache survives between training sessions

## Cost Breakdown

- **S3 Storage**: ~$5/month for 300GB raw data
- **Modal Volume**: ~$15/month for 310GB persistent SSD
- **Total Storage**: ~$20/month
- **Compute**: $3.19/hour × 100 hours = $319 per full training

## Common Misconceptions

❌ **Myth**: "Cache is on S3 causing slowdowns"
✅ **Reality**: Cache is on Modal SSD from the start

❌ **Myth**: "Need to copy cache from S3 to Modal"
✅ **Reality**: Cache is built directly on Modal volume

❌ **Myth**: "S3 random access is the bottleneck"
✅ **Reality**: Only raw EDF is on S3, accessed sequentially once

## Commands

### Check Storage Usage
```bash
# View Modal volumes
modal volume list

# Explore volume contents
modal run deploy/modal/explore_volumes.py

# Download checkpoint locally
modal volume get brain-go-brr-results/checkpoints/best.pt ./best.pt
```

### Monitor Training
```bash
# View app status
modal app list

# Stream logs
modal app logs <app-id>
```

## Environment Variables

| Variable | Purpose | When to Use |
|----------|---------|-------------|
| `BGB_DISABLE_TQDM` | Disable progress bars | Always on Modal |
| `BGB_LIMIT_FILES` | Limit files for testing | Smoke tests only |
| `BGB_FORCE_MANIFEST_REBUILD` | Rebuild cache manifest | If cache corrupted |