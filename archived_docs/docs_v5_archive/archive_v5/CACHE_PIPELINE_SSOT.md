═══════════════════════════════════════════════════════════════════════════════
🎯 CACHE PIPELINE SSOT - COMPLETE UNDERSTANDING
═══════════════════════════════════════════════════════════════════════════════

## Current Cache Architecture

### 1. Storage Locations

```
LOCAL                          S3                                   MODAL
─────────────────────          ──────────────────────────          ─────────────────────
cache/tusz/                    s3://.../cache/tusz/                /results/cache/tusz/
├── .cache_metadata.json  →    ├── .cache_metadata.json   →        ├── .cache_metadata.json
├── train/                     ├── train/                          ├── train/
│   ├── manifest.json  (27MB) →│   ├── manifest.json      →        │   ├── manifest.json
│   ├── _dataset_index.json → │   ├── _dataset_index.json →       │   ├── _dataset_index.json
│   └── *.npz (4,667 files) → │   └── *.npz               →        │   └── *.npz
└── dev/                       └── dev/                            └── dev/
    ├── manifest.json  (13MB) →    ├── manifest.json     →            ├── manifest.json
    ├── _dataset_index.json →      ├── _dataset_index.json →         ├── _dataset_index.json
    └── *.npz (1,832 files) →      └── *.npz              →            └── *.npz

SIZES:
- Local/S3/Modal: 449 GB total (306 GB train + 143 GB dev)
- Compressed NPZ: avg 75 MB on disk
- Decompressed in RAM: avg 85 MB (1.13x compression ratio)
```

### 2. File Types & Roles

| File | Required? | Used By | Purpose |
|------|-----------|---------|---------|
| `*.npz` | ✅ YES | All datasets | Compressed windows + labels |
| `manifest.json` (train) | ✅ YES | `BalancedSeizureDataset` | Balanced sampling |
| `manifest.json` (dev) | ⚠️ OPTIONAL | `ValidationDataset` | Sequential iteration metadata |
| `_dataset_index.json` | ⚠️ OPTIONAL | `EEGWindowDataset` | Fast init (avoids NPZ scans) |
| `.cache_metadata.json` | ⚠️ OPTIONAL | Validation | Audit trail |

### 3. Cache Flow

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: Local Build (RTX 4090)                             │
│ - Build cache: python -m src build-cache                    │
│ - Create manifests: python -m src scan-cache                │
│ - Result: cache/tusz/{train,dev}/ (~449 GB)                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: S3 Upload (Intermediate Storage)                   │
│ - aws s3 sync cache/tusz/train/ s3://.../cache/tusz/train/  │
│ - aws s3 sync cache/tusz/dev/ s3://.../cache/tusz/dev/      │
│ - Cost: ~$40 one-time egress                                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: Modal Populate Cache (Persistent SSD)              │
│ - modal run --detach app.py --action populate-cache         │
│ - Copies from S3 to /results/cache/tusz/ (Modal SSD)        │
│ - Cost: ~$0.50-1.00 CPU compute                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 4: Modal Training (A100)                              │
│ - modal run --detach app.py --action train                  │
│ - Reads from /results/cache/tusz/ (fast SSD)                │
│ - Cost: ~$300-400 for 100 epochs                            │
└─────────────────────────────────────────────────────────────┘
```

═══════════════════════════════════════════════════════════════════════════════
🚨 PROBLEM IDENTIFIED: Current NPZ Cache Causes OOM
═══════════════════════════════════════════════════════════════════════════════

## Why Current Cache WILL Fail on Modal

1. **Unlimited cache → OOM**:
   - 4,667 files × 85 MB avg = 387 GB needed
   - Modal A100 has 96 GB RAM
   - 387 GB > 96 GB = ❌ OOM GUARANTEED

2. **Limited cache (LRU) → Too slow**:
   - 6-file LRU = 99.87% miss rate
   - Constant re-decompression = 100x slower
   - Modal training becomes infeasible

3. **Root cause**:
   - NPZ compression forces full-file decompression into RAM
   - No way to partially load or memory-map compressed data

═══════════════════════════════════════════════════════════════════════════════
✅ SOLUTION: Memory-Mapped NPY Cache
═══════════════════════════════════════════════════════════════════════════════

## New Cache Structure

```
cache/tusz_mmap/                    # NEW mmap-friendly cache
├── .cache_metadata.json
├── train/
│   ├── manifest.json               # REGENERATED for .npy files
│   ├── _dataset_index.json         # REGENERATED for .npy files
│   ├── aaaaaajy_s001_t000_data.npy     # Windows array (uncompressed)
│   ├── aaaaaajy_s001_t000_labels.npy   # Labels array (uncompressed)
│   ├── aaaaaajy_s001_t001_data.npy
│   ├── aaaaaajy_s001_t001_labels.npy
│   └── ...
└── dev/
    ├── manifest.json
    ├── _dataset_index.json
    ├── aaaaaajy_s001_t000_data.npy
    ├── aaaaaajy_s001_t000_labels.npy
    └── ...

SIZES:
- Disk: ~400-450 GB (1.1-1.2x current size, uncompressed)
- RAM per worker: <1 GB (OS manages mmap automatically!)
```

## Why Mmap Solves Everything

| Metric | Current NPZ | Mmap NPY |
|--------|-------------|----------|
| Disk size | 449 GB | ~500 GB |
| RAM per worker | 85+ GB (OOM) | <1 GB |
| Load speed | 375ms decompress | 0.01ms mmap |
| OS memory management | ❌ Manual | ✅ Automatic |
| Workers share memory | ❌ No | ✅ Yes (page cache) |
| Scales to any size | ❌ No | ✅ Yes |

═══════════════════════════════════════════════════════════════════════════════
📋 COMPREHENSIVE IMPLEMENTATION PLAN
═══════════════════════════════════════════════════════════════════════════════

## Phase 1: Local Conversion (4-6 hours)

### Step 1.1: Create Conversion Script
```bash
# Create scripts/convert_cache_to_mmap.py
# - Read each NPZ file
# - Extract windows and labels arrays
# - Save as uncompressed NPY files
# - Verify mmap works
```

### Step 1.2: Run Local Conversion
```bash
# Convert train split (4,667 files, ~2-3 hours)
python scripts/convert_cache_to_mmap.py \
  --source cache/tusz/train \
  --dest cache/tusz_mmap/train

# Convert dev split (1,832 files, ~1 hour)
python scripts/convert_cache_to_mmap.py \
  --source cache/tusz/dev \
  --dest cache/tusz_mmap/dev

# Verify disk space
du -sh cache/tusz_mmap/  # Expect ~400-500 GB
```

### Step 1.3: Regenerate Manifests for NPY Files
```bash
# CRITICAL: Manifests reference file names, must regenerate!
python -m src scan-cache --cache-dir cache/tusz_mmap/train
python -m src scan-cache --cache-dir cache/tusz_mmap/dev
```

## Phase 2: Update Code (1 hour)

### Step 2.1: Update Datasets to Use Mmap
```python
# src/brain_brr/data/datasets.py

# OLD (NPZ):
self._cache_data: dict[Path, dict[str, Any]] = {}
with np.load(cache_path) as data:
    self._cache_data[cache_path] = {
        "windows": data["windows"][:],  # Decompresses entire file!
        "labels": data["labels"][:]
    }

# NEW (NPY mmap):
self._mmap_handles: dict[Path, tuple[np.ndarray, np.ndarray | None]] = {}
windows_file = cache_path.parent / f"{cache_path.stem}_data.npy"
labels_file = cache_path.parent / f"{cache_path.stem}_labels.npy"

windows_mmap = np.load(windows_file, mmap_mode='r')  # OS-managed memory!
labels_mmap = np.load(labels_file, mmap_mode='r') if labels_file.exists() else None

self._mmap_handles[cache_path] = (windows_mmap, labels_mmap)
```

### Step 2.2: Update Configs
```bash
# Point all configs at new mmap cache
sed -i 's|cache/tusz|cache/tusz_mmap|g' configs/local/*.yaml
sed -i 's|/results/cache/tusz|/results/cache/tusz_mmap|g' configs/modal/*.yaml
```

## Phase 3: S3 Upload (1-2 hours)

```bash
# Upload converted cache to NEW S3 location
aws s3 sync cache/tusz_mmap/train/ \
  s3://brain-go-brr-eeg-data-20250919/cache/tusz_mmap/train/

aws s3 sync cache/tusz_mmap/dev/ \
  s3://brain-go-brr-eeg-data-20250919/cache/tusz_mmap/dev/

# Verify manifests uploaded
aws s3 ls s3://.../cache/tusz_mmap/train/manifest.json
aws s3 ls s3://.../cache/tusz_mmap/dev/manifest.json
```

## Phase 4: Update Modal populate-cache (30 min)

```python
# deploy/modal/app.py

# Update S3 source path
src_path = "cache/tusz_mmap"  # NEW: Point to mmap cache

# Update Modal SSD destination
dst = Path("/results/cache/tusz_mmap")  # NEW: Separate from old cache

# Copy everything (NPY files + manifests)
shutil.copytree(src / "train", dst / "train")
shutil.copytree(src / "dev", dst / "dev")
```

## Phase 5: Modal Population (2-3 hours)

```bash
# Populate Modal SSD with mmap cache
modal run --detach deploy/modal/app.py --action populate-cache

# Monitor progress
modal app logs <app-id>
```

## Phase 6: Testing & Validation (1-2 hours)

```bash
# Local smoke test
make s  # Should use mmap cache now

# Local full test
make test

# Modal smoke test
modal run --detach deploy/modal/app.py --action train --config configs/modal/smoke.yaml

# Verify in logs:
# - RSS per worker < 2 GB
# - Window access < 1ms
# - No OOM errors
# - Fast validation (< 2 min per epoch)
```

═══════════════════════════════════════════════════════════════════════════════
⚠️ CRITICAL DEPENDENCIES & GOTCHAS
═══════════════════════════════════════════════════════════════════════════════

## Manifest Compatibility

**PROBLEM**: Current `manifest.json` references `.npz` files:
```json
{
  "partial_seizure": [
    {"cache_file": "aaaaaajy_s001_t000_windows.npz", "indices": [0, 5, 12]}
  ]
}
```

**SOLUTION**: Regenerate manifests after NPY conversion
- File names change: `_windows.npz` → `_data.npy` + `_labels.npy`
- Manifests must reference new file names
- Must run `scan-cache` on mmap directory

## Dataset Code Changes

**Files to modify**:
1. `src/brain_brr/data/datasets.py`:
   - `EEGWindowDataset._load_cache_for_worker()` → use mmap
   - `BalancedSeizureDataset._load_cache_for_worker()` → use mmap
   - `ValidationDataset.__getitem__()` → use mmap

2. `src/brain_brr/data/cache_utils.py`:
   - `scan_existing_cache()` → handle NPY file pairs instead of NPZ

## Config Updates

**Files to modify**:
- `configs/local/smoke.yaml` → `cache_dir: cache/tusz_mmap`
- `configs/local/train.yaml` → `cache_dir: cache/tusz_mmap`
- `configs/modal/smoke.yaml` → `cache_dir: /results/cache/tusz_mmap`
- `configs/modal/train.yaml` → `cache_dir: /results/cache/tusz_mmap`

## S3 Bucket Structure

```
s3://brain-go-brr-eeg-data-20250919/
├── cache/
│   ├── tusz/           # OLD compressed NPZ (keep for backup)
│   └── tusz_mmap/      # NEW uncompressed NPY
```

## Modal Volume Path

```
/results/
├── cache/
│   ├── tusz/           # OLD compressed NPZ (can delete after testing)
│   └── tusz_mmap/      # NEW uncompressed NPY
```

═══════════════════════════════════════════════════════════════════════════════
💰 COST & TIME ESTIMATES
═══════════════════════════════════════════════════════════════════════════════

## Storage Costs

| Item | Old (NPZ) | New (mmap NPY) | Delta |
|------|-----------|----------------|-------|
| Local disk | 449 GB | 500 GB | +51 GB |
| S3 storage | $10/mo | $12/mo | +$2/mo |
| S3 egress (one-time) | $40 | $45 | +$5 |

## Time Investment

| Phase | Time | Notes |
|-------|------|-------|
| Conversion script | 1 hr | One-time dev |
| Local conversion | 4 hrs | One-time |
| Code updates | 1 hr | Update 3 files |
| S3 upload | 2 hrs | One-time |
| Modal populate | 3 hrs | One-time |
| Testing | 2 hrs | Validation |
| **TOTAL** | **13 hrs** | **One-time investment** |

## ROI

- **Investment**: 13 hours + $50 one-time + $2/mo storage
- **Benefit**: Unlocks Modal training (saves $0 wasted A100 time)
- **Result**: Can actually train the model!

═══════════════════════════════════════════════════════════════════════════════
✅ SIGN-OFF CRITERIA
═══════════════════════════════════════════════════════════════════════════════

Before declaring DONE:

1. ✅ Local smoke test passes with mmap cache
2. ✅ Local full test passes
3. ✅ Worker RSS < 2 GB (measured via psutil)
4. ✅ Window access < 1ms (measured)
5. ✅ Modal smoke runs without OOM
6. ✅ Validation epoch < 2 min (49x speedup achieved)
7. ✅ make q && make test passes
8. ✅ All configs point to mmap cache
9. ✅ Manifests regenerated for mmap files
10. ✅ S3 and Modal have mmap cache populated

═══════════════════════════════════════════════════════════════════════════════
