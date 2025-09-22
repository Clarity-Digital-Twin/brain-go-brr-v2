# Cache & Manifest Architecture Explained

## Overview
The system uses a two-level cache structure with optional manifests for balanced sampling.

## Directory Structure

```
cache/tusz/
├── train/              # 80% of data (3734 files)
│   ├── *.npz          # Cached windows
│   └── manifest.json  # Seizure categorization
└── val/               # 20% of data (933 files)
    └── *.npz          # Cached windows (NO manifest)
```

## How It Works

### 1. Data Split (Line 1003-1006)
```python
val_split = int(len(edf_files) * 0.2)  # 20% validation
train_files = edf_files[val_split:]    # Last 80%
val_files = edf_files[:val_split]      # First 20%
```

### 2. Cache Building
Each dataset builds its own cache independently:
- **Train**: `cache/tusz/train/*.npz` (3734 files)
- **Val**: `cache/tusz/val/*.npz` (933 files)

### 3. Manifest Generation (ONLY for train!)

**Why only train has a manifest?**
- **Train uses BalancedSeizureDataset** (if `use_balanced_sampling=true`)
- **Val ALWAYS uses EEGWindowDataset** (random sampling)

The manifest (`manifest.json`) categorizes windows:
```json
{
  "partial_seizure": [...],  // Most valuable (seizure boundaries)
  "full_seizure": [...],     // 100% seizure
  "no_seizure": [...]        // Background
}
```

### 4. Sampling Strategies

#### TRAINING Dataset:
```
if use_balanced_sampling = true:
  if manifest exists:
    → BalancedSeizureDataset (SeizureTransformer approach)
      • ALL partial seizure windows
      • 0.3x full seizure windows
      • 2.5x background windows
  else:
    → EEGWindowDataset + WeightedRandomSampler (fallback)
      • Samples 20,000 windows to find seizures
      • Creates weights for balanced mini-batches
else:
  → EEGWindowDataset (pure random sampling)
```

#### VALIDATION Dataset:
```
ALWAYS → EEGWindowDataset (random sampling)
- No manifest needed
- No balanced sampling
- Pure random to test real distribution
```

## Local vs Cloud Differences

### LOCAL (your machine):
```
data_ext4/tusz/edf/train/  → split 80/20 → cache/tusz/train/ (3734 files + manifest)
                                        → cache/tusz/val/   (933 files, NO manifest)
```

### MODAL (cloud):
```
/data/edf/train/  → split 80/20 → /results/cache/tusz/train/ (3734 files + manifest?)
                                → /results/cache/tusz/val/   (933 files, NO manifest)
```

**Modal Status**: Building same structure but in `/results/` volume

## Why This Design?

1. **Train needs balanced sampling** because:
   - TUSZ is 99.9% non-seizure
   - Without balancing, model learns "always predict no seizure"
   - Manifest enables efficient categorization

2. **Val doesn't need balancing** because:
   - Validation should reflect true data distribution
   - Tests model on realistic imbalanced data
   - Random sampling is appropriate

3. **Manifest only for train** because:
   - Only train uses BalancedSeizureDataset
   - Val doesn't need categorization
   - Saves computation time

## Current Status

### Local ✅
- Train cache: 3734 files built
- Train manifest: Built with 13,095 partial, 6,759 full, 232,235 no-seizure
- Val cache: 933 files built
- Val manifest: Not needed/built

### Modal (In Progress)
- At file 3121/3734 (83%)
- Building to `/results/cache/tusz/`
- Will create same train/val structure
- Manifest will be built when training starts

## Key Takeaways

1. **Two separate caches**: train/ and val/ (80/20 split)
2. **One manifest**: Only train/ needs it for balanced sampling
3. **Val is always random**: Reflects true distribution
4. **Modal mirrors local**: Same structure, different path
5. **Manifest is optional**: Only needed if `use_balanced_sampling=true`

## Validation & Rebuild

- Training validates the manifest against the cache directory and deletes/rebuilds it if empty or stale.
- You can force a rebuild on startup by setting `BGB_FORCE_MANIFEST_REBUILD=1`.
- To rebuild explicitly at any time:

```
python -m src scan-cache --cache-dir <cache/tusz/train or /results/cache/tusz/train>
```
