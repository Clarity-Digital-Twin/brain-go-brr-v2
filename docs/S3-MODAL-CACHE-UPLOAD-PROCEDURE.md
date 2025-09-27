# S3/Modal Cache Upload Procedure - COMPLETE GUIDE

## Current State (September 26, 2025)

### ✅ S3 Bucket Status
- **Bucket**: `s3://brain-go-brr-eeg-data-20250919/`
- **Current Contents**:
  ```
  tusz/edf/           # Raw EDF data (266GB) - PRESENT
  cache/              # NPZ cache directory - NOT YET CREATED (CLEAN!)
  ```
- **Status**: NO old caches exist - ready for fresh upload!

### ✅ Modal Volume Status
- **Volume**: `brain-go-brr-results` (431 MiB)
- **Contents**: Training outputs only
- **Cache Location**: `/results/cache/tusz/` - will be populated from S3

## Upload Procedure

### Step 1: Complete Local Cache Rebuild
```bash
# Monitor progress
echo "Progress: $(ls cache/tusz/dev/*.npz 2>/dev/null | wc -l)/1832 files"

# Wait for completion
# Expected: 4667 train files + 1832 dev files
```

### Step 2: Verify Local Cache
```bash
# Check file counts
ls -1 cache/tusz/train/*.npz | wc -l  # Should be 4667
ls -1 cache/tusz/dev/*.npz | wc -l    # Should be 1832

# Check for manifest files
ls -la cache/tusz/train/manifest.json
ls -la cache/tusz/dev/manifest.json
```

### Step 3: Upload to S3
```bash
# Run the upload script
./scripts/upload_cache_to_s3.sh

# This will:
# 1. Upload cache/tusz/train/ → s3://brain-go-brr-eeg-data-20250919/cache/tusz/train/
# 2. Upload cache/tusz/dev/ → s3://brain-go-brr-eeg-data-20250919/cache/tusz/dev/
# 3. Exclude .json and .log files
```

### Step 4: Verify S3 Upload
```bash
# Check S3 structure
~/.local/bin/aws s3 ls s3://brain-go-brr-eeg-data-20250919/cache/tusz/
~/.local/bin/aws s3 ls s3://brain-go-brr-eeg-data-20250919/cache/tusz/train/ | wc -l
~/.local/bin/aws s3 ls s3://brain-go-brr-eeg-data-20250919/cache/tusz/dev/ | wc -l
```

### Step 5: Clean Modal Volume (if needed)
```bash
# Clean any old caches on Modal
modal run deploy/modal/app.py --action clean-cache
```

### Step 6: Populate Modal SSD from S3
```bash
# One-time copy from S3 to Modal SSD
modal run deploy/modal/app.py --action populate-cache

# This will:
# 1. Mount S3 cache at /s3_cache
# 2. Copy to /results/cache/tusz/ on SSD
# 3. Verify file counts
# 4. Takes ~1-2 hours for 450GB
```

### Step 7: Verify Modal Cache
```bash
# Check cache on Modal volume
modal run deploy/modal/app.py --action verify-cache
```

## Architecture Summary

```
LOCAL BUILD                S3 STORAGE               MODAL SSD
───────────               ──────────               ─────────
cache/tusz/               cache/tusz/              /results/cache/tusz/
├── train/     ─upload→   ├── train/    ─copy→     ├── train/
│   (4667)                 │   (4667)               │   (4667)
└── dev/       ─upload→   └── dev/      ─copy→     └── dev/
    (1832)                     (1832)                   (1832)
```

## Critical Notes

### ✅ Naming Convention
- **ALWAYS use 'dev'** for validation split (NOT 'val')
- This matches TUSZ official naming: train/dev/eval
- All paths must use 'dev' consistently

### ✅ Cache Contents (Fixed)
- **mysz seizures**: Now properly labeled (44 seizures that were missing)
- **Outlier clipping**: Applied (±10σ)
- **Patient-disjoint**: Official TUSZ splits maintained

### ✅ Performance Strategy
- **S3**: Durable storage, backup only
- **Modal SSD**: Fast training access (10x faster than S3)
- **One-time population**: Copy once, reuse forever

## Troubleshooting

### If S3 upload fails
```bash
# Resume upload (aws s3 sync is incremental)
./scripts/upload_cache_to_s3.sh
```

### If Modal population fails
```bash
# Clean and retry
modal run deploy/modal/app.py --action clean-cache
modal run deploy/modal/app.py --action populate-cache
```

### Check S3 costs
```bash
# Monitor S3 usage
~/.local/bin/aws s3 ls s3://brain-go-brr-eeg-data-20250919/ --recursive --summarize --human-readable
```

## Final Verification

Before training:
1. ✅ S3 has cache/tusz/{train,dev}/ with correct file counts
2. ✅ Modal SSD has /results/cache/tusz/{train,dev}/ populated
3. ✅ All paths use 'dev' naming (not 'val')
4. ✅ Manifest files are generated for both splits

## Commands Reference

```bash
# Upload to S3
./scripts/upload_cache_to_s3.sh

# Populate Modal from S3
modal run deploy/modal/app.py --action populate-cache

# Start training
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml
```

---

**Status**: Ready for cache upload once dev split rebuild completes (currently 43% done)