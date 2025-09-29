# Brain-Go-Brr V3.2.0 Cache & Training Workflow

**CRITICAL**: This document describes the ACTUAL workflow as it exists today, including gaps and manual steps. Read this COMPLETELY before attempting training.

## Overview

The Brain-Go-Brr seizure detection system requires a complex data pipeline from raw EDF files to trained models. This pipeline involves:
1. Local cache building from TUSZ EDF files
2. Manifest generation for balanced sampling
3. S3 storage as intermediate transfer
4. Modal cloud deployment for A100 training

**Total data size**: ~450GB preprocessed cache (306GB train + 143GB dev)

## ⚠️ Current Pipeline Issues

### Critical Gaps
1. **Manifest files NOT uploaded to S3** - The `upload_cache_to_s3.sh` script excludes `*.json` files
2. **No automated S3 upload** - Completely manual process
3. **Modal populate-cache requires --detach** - Will stop if terminal disconnects
4. **Dev manifest not auto-created** - Must run `scan-cache` manually
5. **No end-to-end automation** - Multiple manual steps required

### What Can Go Wrong
- Training with zero seizures in batches (missing manifests)
- 2+ hour startup delays (manifest rebuilding)
- Wasted A100 compute rebuilding cache ($3.19/hour)
- Modal functions stopping when terminal closes
- Cache corruption from interrupted builds

## Complete Workflow (Current State)

### Phase 1: Local Cache Building

#### Automatic (during training)
```bash
# Cache builds automatically on first training run
make train-local  # or: python -m src train configs/local/train.yaml
# Takes 2-3 hours on RTX 4090
# Creates: cache/tusz/train/*.npz (4667 files)
#          cache/tusz/dev/*.npz (1832 files)
```

#### Manual (CLI)
```bash
# Build train cache
python -m src build-cache \
  --data-dir data/edf \
  --cache-dir cache/tusz/train \
  --split train

# Build dev cache
python -m src build-cache \
  --data-dir data/edf \
  --cache-dir cache/tusz/dev \
  --split dev
```

**Output Structure**:
```
cache/tusz/
├── train/
│   ├── aaaaaaac_s001_t000_windows.npz
│   ├── aaaaaaac_s001_t001_windows.npz
│   └── ... (4667 total)
└── dev/
    ├── aaaaaajy_s001_t000_windows.npz
    └── ... (1832 total)
```

### Phase 2: Manifest Generation

**CRITICAL**: Train manifest enables balanced sampling and prevents zero-seizure batches!

#### Dataset Strategy (This is CORRECT, not a bug!)
- **Training**: Uses `BalancedSeizureDataset` with manifest to oversample seizures (learns better)
- **Validation**: Uses `EEGWindowDataset` with natural 8% seizure distribution (measures real performance)
- **Why different?**: Standard ML practice - train on balanced data, validate on real distribution

#### Check if manifests exist
```bash
ls -la cache/tusz/train/manifest.json  # REQUIRED! Should be ~27MB
ls -la cache/tusz/dev/manifest.json    # Optional (dev uses EEGWindowDataset) ~13MB
```

#### Create missing manifests
```bash
# Create train manifest (REQUIRED for training!)
python -m src scan-cache --cache-dir cache/tusz/train

# Create dev manifest (optional but recommended for future use)
python -m src scan-cache --cache-dir cache/tusz/dev
```

**Manifest Structure**:
```json
{
  "partial_seizure": [
    {"cache_file": "file.npz", "window_idx": 0},
    ...
  ],
  "full_seizure": [...],
  "no_seizure": [...]
}
```

### Phase 3: S3 Upload

**⚠️ BROKEN**: Current script excludes JSON files!

#### Current (Broken) Process
```bash
# This script EXCLUDES manifests (--exclude "*.json")
./scripts/upload_cache_to_s3.sh
```

#### Fixed Process (Manual)
```bash
# Upload NPZ files
aws s3 sync cache/tusz/train/ s3://brain-go-brr-eeg-data-20250919/cache/tusz/train/ \
  --exclude "*.log"
aws s3 sync cache/tusz/dev/ s3://brain-go-brr-eeg-data-20250919/cache/tusz/dev/ \
  --exclude "*.log"

# Upload manifests (CRITICAL!)
aws s3 cp cache/tusz/train/manifest.json \
  s3://brain-go-brr-eeg-data-20250919/cache/tusz/train/manifest.json
aws s3 cp cache/tusz/dev/manifest.json \
  s3://brain-go-brr-eeg-data-20250919/cache/tusz/dev/manifest.json

# Upload dataset indices
aws s3 cp cache/tusz/train/_dataset_index.json \
  s3://brain-go-brr-eeg-data-20250919/cache/tusz/train/_dataset_index.json
aws s3 cp cache/tusz/dev/_dataset_index.json \
  s3://brain-go-brr-eeg-data-20250919/cache/tusz/dev/_dataset_index.json
```

### Phase 4: Modal Cache Population

**⚠️ MUST use --detach or it stops when terminal closes!**

#### One-time setup (per Modal volume)
```bash
# WRONG (will stop if terminal disconnects after ~8 minutes)
modal run deploy/modal/app.py --action populate-cache

# CORRECT (survives disconnection)
modal run --detach deploy/modal/app.py --action populate-cache
```

**What happens**:
1. Mounts S3 bucket at `/s3_cache/`
2. Deletes any existing `/results/cache/tusz/` (clean slate)
3. Copies all NPZ files + manifests to Modal SSD
4. Takes 1-2 hours for 450GB
5. Creates validation metadata

**Monitor progress**:
```bash
# Check status
modal app list

# View logs (replace with actual app-id)
modal app logs ap-XXXXXXXXXXXXXXXXXXXXX
```

### Phase 5: Training

#### Smoke test first
```bash
modal run --detach deploy/modal/app.py \
  --action train \
  --config configs/modal/smoke.yaml
```

#### Full training
```bash
modal run --detach deploy/modal/app.py \
  --action train \
  --config configs/modal/train.yaml
```

**Expected behavior with proper cache**:
- Immediate start (no cache building)
- Uses `BalancedSeizureDataset` (not fallback)
- Shows "Seizure ratio: X% (from manifest)"
- Consistent seizures in every batch

## Verification Checklist

### Before S3 Upload
- [ ] `cache/tusz/train/` has 4667 NPZ files
- [ ] `cache/tusz/dev/` has 1832 NPZ files
- [ ] `cache/tusz/train/manifest.json` exists (~27MB) **REQUIRED!**
- [ ] `cache/tusz/dev/manifest.json` exists (~13MB) *Optional but upload if present*
- [ ] Both `_dataset_index.json` files exist

### After S3 Upload
```bash
# Verify all files uploaded
aws s3 ls s3://brain-go-brr-eeg-data-20250919/cache/tusz/train/ --recursive | grep manifest.json
aws s3 ls s3://brain-go-brr-eeg-data-20250919/cache/tusz/dev/ --recursive | grep manifest.json
```

### After Modal Population
Check logs for:
```
[COPY] ✅ Copied 4667 train files
[COPY] ✅ Copied 1832 dev files
[COPY] ✅ Copied metadata file
✅ Cache population complete! 4667 train, 1832 dev files
```

### During Training
Check logs for:
```
[CACHE] ✅ Cache built with official_tusz policy
[DATASET] BalancedSeizureDataset: XXXX windows from manifest
[DATASET] Seizure ratio: XX% (from manifest)
[DATASET] Using pos_weight: X.XX (sqrt scaling)
```

## Emergency Recovery

### If training shows zero seizures
```bash
# Force manifest rebuild locally
BGB_FORCE_MANIFEST_REBUILD=1 python -m src scan-cache --cache-dir cache/tusz/train
BGB_FORCE_MANIFEST_REBUILD=1 python -m src scan-cache --cache-dir cache/tusz/dev

# Re-upload to S3 (including JSONs!)
# Then re-run populate-cache
```

### If Modal populate-cache stops
```bash
# Always use --detach
modal run --detach deploy/modal/app.py --action populate-cache

# Or run in tmux for safety
tmux new -s populate
modal run deploy/modal/app.py --action populate-cache
# Ctrl+B then D to detach
```

### If cache is corrupted
```bash
# Clean Modal cache
modal run deploy/modal/app.py --action clean-cache

# Delete local cache
rm -rf cache/tusz/

# Rebuild from scratch
make train-local  # Will rebuild cache
```

## Proposed Fixes (TODO)

### Immediate Fixes Needed

1. **Fix upload script** - Remove `--exclude "*.json"` from `upload_cache_to_s3.sh`
2. **Add Makefile targets**:
   ```makefile
   upload-cache:
       ./scripts/upload_cache_to_s3.sh

   populate-modal:
       modal run --detach deploy/modal/app.py --action populate-cache
   ```

3. **Auto-create dev manifest** - Add to cache building process
4. **Document --detach requirement** - Update all Modal examples

### Long-term Improvements

1. **Single command deployment**:
   ```bash
   make deploy-modal  # Builds cache, uploads S3, populates Modal, starts training
   ```

2. **Validation commands**:
   ```bash
   python -m src validate-pipeline  # Checks all components ready
   ```

3. **Automated manifest generation** - Always create after cache building
4. **Direct Modal upload** - Skip S3 intermediate (if possible)

## Cost Implications

- **Cache building locally**: Free (uses your GPU)
- **S3 storage**: ~$10/month for 450GB
- **S3 egress**: ~$40 per full transfer to Modal
- **Modal populate-cache**: ~$0.50-1.00 (CPU compute)
- **Modal training without cache**: +$100-300 WASTED on A100 rebuilding

**ALWAYS populate cache before training to avoid waste!**

## Contact

For issues with this pipeline:
- GitHub Issues: https://github.com/anthropics/claude-code/issues
- TUSZ data questions: Refer to official TUSZ documentation
- Modal questions: https://modal.com/docs

---

**Last Updated**: September 29, 2025
**Version**: 3.2.0
**Status**: Pipeline functional but requires manual coordination