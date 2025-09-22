# Complete Data Pipeline Architecture: EDF → Training

## Overview
This document provides the complete, unambiguous data flow from raw EDF files to model training, including the critical BalancedSeizureDataset optimization that eliminates 2+ hour bottlenecks on Modal.

## Pipeline Stages

### Stage 1: Raw Data Input
```
INPUT:
├── EDF Files: 19-channel EEG recordings at various sampling rates
│   └── Format: European Data Format, ~50-500MB each
└── CSV Files: TUSZ seizure annotations (CSV_BI format)
    └── Format: start_time, stop_time, label, probability
```

### Stage 2: Preprocessing & Caching
**Location**: `src/brain_brr/data/io.py`, `preprocess.py`, `windows.py`

```python
For each EDF file:
1. Load EDF → Parse channels → Map to 10-20 montage
2. Bandpass filter: 0.5-120 Hz (remove DC drift & high freq noise)
3. Notch filter: 60 Hz (remove power line interference)
4. Resample: Original Hz → 256 Hz (standardize)
5. Window: 60s windows with 10s stride
6. Save as NPZ: windows[N, 19, 15360], labels[N, 15360]
```

**Cache Structure**:
```
cache_dir/
├── file1_windows.npz  (26-152MB, contains 10-60 windows)
├── file2_windows.npz
└── ...
```

### Stage 3: Manifest Generation
**Location**: `src/brain_brr/data/cache_utils.py::scan_existing_cache()`

**Purpose**: One-time scan of ALL cache files to categorize windows by seizure content.

```python
def scan_existing_cache(cache_dir):
    manifest = {"partial_seizure": [], "full_seizure": [], "no_seizure": []}

    for npz_file in cache_dir.glob("*.npz"):
        data = np.load(npz_file)
        windows = data["windows"]  # Shape: (N, 19, 15360)
        labels = data["labels"]     # Shape: (N, 15360)

        for window_idx in range(len(windows)):
            seizure_fraction = (labels[window_idx] > 0).mean()

            entry = {"cache_file": npz_file.name, "window_idx": window_idx}

            if seizure_fraction > 0.5:
                manifest["full_seizure"].append(entry)
            elif seizure_fraction > 0:
                manifest["partial_seizure"].append(entry)
            else:
                manifest["no_seizure"].append(entry)

    # Save manifest.json with exact window locations
    return manifest
```

**Manifest Format**:
```json
{
  "partial_seizure": [
    {"cache_file": "file1_windows.npz", "window_idx": 3},
    {"cache_file": "file2_windows.npz", "window_idx": 7},
    ...
  ],
  "full_seizure": [...],
  "no_seizure": [...]
}
```

### Stage 4: BalancedSeizureDataset Creation
**Location**: `src/brain_brr/data/datasets.py::BalancedSeizureDataset`

**Algorithm** (from SeizureTransformer paper):
```python
class BalancedSeizureDataset:
    def __init__(self, cache_dir, full_ratio=0.3, background_ratio=2.5):
        # Load manifest (instant - just JSON)
        manifest = load_manifest(cache_dir)

        # Build balanced dataset:
        entries = []

        # 1. Take ALL partial seizure windows (most informative)
        n_partial = len(manifest["partial_seizure"])
        entries.extend(manifest["partial_seizure"])

        # 2. Add 30% as many full seizure windows
        n_full = int(0.3 * n_partial)
        entries.extend(random.sample(manifest["full_seizure"], n_full))

        # 3. Add 2.5x as many background windows
        n_bg = int(2.5 * n_partial)
        entries.extend(random.sample(manifest["no_seizure"], n_bg))

        # Calculate exact seizure ratio
        self.seizure_ratio = (n_partial + n_full) / len(entries)
        # Typically: (17000 + 5100) / 49760 ≈ 0.342 (34.2%)
```

**Why These Ratios?**
- **Partial (100%)**: Windows with seizure onset/offset are most informative for boundary detection
- **Full (30%)**: Already obvious seizures, less informative for learning transitions
- **Background (250%)**: Need negatives but not too many to avoid class imbalance

### Stage 5: Training Optimization
**Location**: `src/brain_brr/train/loop.py::train_epoch()`

#### OLD METHOD (2+ Hours on Modal):
```python
# Sample 1000 random windows to estimate seizure ratio
sample_indices = random.sample(range(len(dataset)), 1000)
seizure_count = 0

for idx in sample_indices:
    window, label = dataset[idx]  # LOADS NPZ FROM DISK!
    # On Modal: 100-700ms per file over network storage
    # 1000 files × 0.7s = 700s = 12 minutes (best case)
    # 1000 files × 7s = 7000s = 2 hours (worst case)
    if (label > 0).any():
        seizure_count += 1

estimated_ratio = seizure_count / 1000  # Statistical estimate
pos_weight = sqrt((1 - estimated_ratio) / estimated_ratio)
```

#### NEW METHOD (Instant):
```python
if isinstance(dataset, BalancedSeizureDataset):
    # Dataset already knows exact ratio from manifest!
    pos_ratio = dataset.seizure_ratio  # Just returns 0.342
    # No file I/O, no sampling, no waiting!
else:
    # Fallback for other datasets (reduced to 100 samples)
    sample_and_estimate()

pos_weight = sqrt((1 - pos_ratio) / pos_ratio)  # = 1.387
```

## Performance Impact

### Local (WSL2 with SSD):
- NPZ file access: <1ms (Linux page cache)
- 1000 samples: ~1-2 seconds
- **Not a bottleneck**

### Modal (Network Storage):
- NPZ file access: 100-700ms (no page cache, network latency)
- 1000 samples: 12 minutes to 2+ hours
- **CRITICAL BOTTLENECK** → Fixed by optimization

## Mathematical Correctness

**Sampling Method**:
- Provides statistical estimate: `ratio ≈ 0.342 ± sampling_error`
- Larger sample → smaller error
- Always has uncertainty

**Direct Method**:
- Provides exact value: `ratio = 0.342`
- No sampling error
- Deterministic from manifest

**Both methods produce same pos_weight**:
```
pos_weight = sqrt((1 - 0.342) / 0.342)
          = sqrt(0.658 / 0.342)
          = sqrt(1.924)
          = 1.387
```

## Key Invariants

1. **Data Integrity**: Same windows, same labels, same preprocessing
2. **Statistical Validity**: Exact ratio > estimated ratio (no sampling error)
3. **Training Equivalence**: Same pos_weight → same loss scaling → same gradients
4. **No Hardcoding**: Ratio comes from actual data distribution via manifest

## Common Misconceptions

**Q: Are we cheating by skipping the sampling?**
A: No. We're using exact knowledge instead of statistical estimation. The manifest was built by scanning ALL windows.

**Q: Does this change the training data?**
A: No. Exact same windows are used for training. We only skip redundant I/O to estimate what we already know.

**Q: Why not just hardcode pos_weight=1.387?**
A: Because it changes with data! Different datasets or filtering criteria produce different ratios. The manifest ensures we always use the correct value.

## Validation Mechanisms

1. **Manifest Validation**: Training auto-validates manifest against cache files
2. **Forced Rebuild**: `BGB_FORCE_MANIFEST_REBUILD=1` to regenerate
3. **Sanity Checks**: Training stops if seizure_ratio < 0.001 (likely corrupt)
4. **Logging**: All steps logged for audit trail

## File References

- Manifest generation: `src/brain_brr/data/cache_utils.py:45-120`
- BalancedSeizureDataset: `src/brain_brr/data/datasets.py:191-313`
- Training optimization: `src/brain_brr/train/loop.py:311-340`
- Modal bottleneck analysis: `docs/03-operations/troubleshooting.md`

## Summary

The complete pipeline transforms raw EDF files through preprocessing, caching, manifest generation, and balanced dataset creation to enable efficient training. The key optimization - using pre-computed seizure ratios from the manifest instead of sampling - eliminates a 2+ hour bottleneck on Modal while maintaining mathematical equivalence and data integrity.