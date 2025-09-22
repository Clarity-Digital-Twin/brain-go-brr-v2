# Data Pipeline Architecture

## Overview
Complete EDF → Training pipeline with manifest-driven balanced sampling that eliminates Modal bottlenecks.

## Pipeline Stages

### 1. Raw Input & Parsing
- **EDF Files**: 19-channel EEG recordings (50-500MB each)
- **CSV_BI Annotations**: TUSZ format with channel-specific seizure events
- **Parser**: `parse_tusz_csv()` extracts (start, stop, label, confidence)

### 2. Preprocessing & Caching
```python
For each EDF file:
1. Load EDF → Map channels to canonical 10-20 montage
2. Bandpass: 0.5-120 Hz (remove drift & noise)
3. Notch: 60 Hz (remove power line interference)
4. Resample: → 256 Hz (standardize)
5. Window: 60s with 10s stride (83% overlap)
6. Save NPZ: windows[N, 19, 15360], labels[N, 15360]
```

**Cache Structure**:
```
cache/tusz/
├── train/              # 80% of data (3734 files)
│   ├── *.npz          # 26-152MB each
│   └── manifest.json  # Window categorization
└── val/               # 20% of data (933 files)
    └── *.npz          # No manifest (random sampling)
```

### 3. Manifest Generation (One-Time Scan)
```python
def scan_existing_cache(cache_dir):
    # Scan ALL NPZ files once to categorize windows
    manifest = {
        "partial_seizure": [],  # 0% < seizure < 50%
        "full_seizure": [],     # seizure ≥ 50%
        "no_seizure": []        # seizure = 0%
    }
    # Each entry: {"cache_file": "X.npz", "window_idx": N}
```

### 4. BalancedSeizureDataset (SeizureTransformer Formula)
```python
# Dataset composition:
# D = Dps ∪ D*fs ∪ D*ns
entries = []
entries.extend(ALL partial windows)           # Most informative
entries.extend(0.3 × partial full windows)    # Less novel
entries.extend(2.5 × partial background)      # Necessary negatives

# Result: ~34.2% seizure ratio (deterministic from manifest)
```

## Critical Optimization: Eliminating Modal Bottleneck

### The Problem (2+ Hour Delay)
```python
# OLD: Sample 1000 windows to estimate seizure ratio
for idx in random.sample(range(len(dataset)), 1000):
    window, label = dataset[idx]  # LOADS NPZ FROM DISK!
    # Modal: 100-700ms per network file access
    # Total: 12 minutes to 2+ hours
```

### The Solution (Instant)
```python
# NEW: BalancedSeizureDataset knows exact ratio from manifest
if isinstance(dataset, BalancedSeizureDataset):
    pos_ratio = dataset.seizure_ratio  # Just returns 0.342
    # No file I/O, no sampling, instant!
```

### Performance Impact
| Environment | Old (Sampling) | New (Direct) | Improvement |
|------------|---------------|--------------|-------------|
| Modal | 2+ hours | <1 second | **7200× faster** |
| Local | 1-2 seconds | <1ms | 1000× faster |

## Key Implementation Files

- **CSV Parser**: `src/brain_brr/data/io.py:parse_tusz_csv()`
- **Preprocessing**: `src/brain_brr/data/preprocess.py`
- **Window Extraction**: `src/brain_brr/data/windows.py`
- **Manifest Scan**: `src/brain_brr/data/cache_utils.py:scan_existing_cache()`
- **Balanced Dataset**: `src/brain_brr/data/datasets.py:BalancedSeizureDataset`
- **Training Loop**: `src/brain_brr/train/loop.py:train_epoch()`

## Critical Issues Resolved

### Zero-Seizure Cache (254GB Wasted)
- **Cause**: Wrong CSV column parsing
- **Fix**: Correct CSV_BI parser, manifest validation
- **Prevention**: Always run `scan-cache` before training

### Channel Naming Mismatches
- **Cause**: TUSZ uses 'EEG FP1-LE' not 'Fp1'
- **Fix**: Robust `clean_tusz_name()` with synonyms
- **Prevention**: Channel canonicalization in pipeline

### Missing Myoclonic Seizures
- **Cause**: Label set omitted 'mysz'
- **Fix**: Complete seizure set `{gnsz,fnsz,cpsz,absz,spsz,tcsz,tnsz,mysz}`
- **Prevention**: Empirical validation against corpus

## Validation & Recovery

### Automatic Safeguards
1. Training validates manifest on startup
2. Deletes/rebuilds if empty or stale
3. Warns if seizure_ratio < 0.001

### Manual Controls
```bash
# Force manifest rebuild
export BGB_FORCE_MANIFEST_REBUILD=1

# Scan cache manually
python -m src scan-cache --cache-dir cache/tusz/train

# Verify before training
python -m src preflight --data-dir data --cache-dir cache
```

## Mathematical Correctness

Both methods produce identical pos_weight:
```
Sampling: ratio ≈ 0.342 ± error → pos_weight ≈ 1.387
Direct:   ratio = 0.342 (exact) → pos_weight = 1.387
```

The optimization only eliminates I/O, not the calculation.

## Summary
The pipeline transforms raw EDF through preprocessing, caching, and manifest-driven balanced sampling. The key optimization—using pre-computed ratios instead of sampling—eliminates a 2+ hour Modal bottleneck while maintaining exact mathematical equivalence.