# TUSZ Training Cache Archive

## Overview
This document describes the preprocessed TUSZ training data cache for Brain-Go-Brr v2.

## Cache Details

**Archive Name**: `tusz_train_cache_256hz_10-20_v1_20250920.tar.gz`
**Location**: Project root directory
**Size**: ~10-12GB compressed (27GB uncompressed)
**Files**: 246+ preprocessed EDF files as numpy arrays
**Created**: 2025-09-20

## Preprocessing Parameters
- **Sampling Rate**: 256 Hz (resampled from various original rates)
- **Montage**: Standard 10-20 system (19 channels)
- **Bandpass Filter**: 0.5-120 Hz
- **Notch Filter**: 60 Hz (US power line)
- **Normalization**: Per-channel z-score
- **Window Size**: 60 seconds (15,360 samples)
- **Stride**: 10 seconds (2,560 samples)

## Contents
Each `.npz` file contains:
- `windows`: Shape (N, 19, 15360) - N windows of 19-channel EEG
- `labels`: Shape (N, 15360) - Binary seizure labels per sample

## Usage

### Extract Cache
```bash
# Extract to cache directory
tar -xzf tusz_train_cache_256hz_10-20_v1_20250920.tar.gz
```

### For Modal Deployment
```python
# In Modal function setup
import tarfile
import os

# Download from storage (S3/Modal volume)
if not os.path.exists("cache/train"):
    with tarfile.open("tusz_train_cache_256hz_10-20_v1_20250920.tar.gz", "r:gz") as tar:
        tar.extractall(".")
```

## Compatibility
This cache is **ONLY** valid for training configs with:
- `sampling_rate: 256`
- `window_size: 60`
- `stride: 10`
- `bandpass: [0.5, 120]`
- `notch_freq: 60`
- `montage: "10-20"`

⚠️ **WARNING**: Changing ANY preprocessing parameter requires rebuilding the cache!

## Reproducibility
Cache was built from:
- **Dataset**: TUSZ v2.0.0 train split
- **Data Path**: `data_ext4/tusz/edf/train/`
- **Config**: `configs/tusz_train_wsl2.yaml`
- **Code Version**: Git hash at cache creation

## Storage Recommendations
1. Upload to S3 with versioning enabled
2. Keep copy on Modal volume for fast access
3. Document in experiment logs which cache version was used

## Time Savings
- **Without cache**: ~2-3 hours on Modal (network I/O bound)
- **With cache**: ~10 minutes (extraction only)
- **Speedup**: 12-18x faster startup

## Notes
- Cache building was interrupted multiple times but is deterministic
- Files are named: `{patient_id}_{session}_{segment}_windows.npz`
- Cache automatically grows as new files are encountered
- Current cache covers ~6.5% of full TUSZ train set (246/3734 files)