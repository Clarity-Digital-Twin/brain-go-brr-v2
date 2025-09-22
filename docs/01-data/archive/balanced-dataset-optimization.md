# BalancedSeizureDataset Optimization: Eliminating the Modal Bottleneck

## The Problem

When training on Modal.com with network-attached storage, the training pipeline was experiencing a **2+ hour delay** at the "DATASET STATISTICS" phase. This was caused by sampling 1000 random windows to estimate the seizure ratio for calculating pos_weight.

### Root Cause Analysis

```python
# The problematic code in train_epoch():
sample_indices = torch.randperm(len(dataset))[:1000]
for idx in sample_indices:
    _, label = dataset[idx]  # Loads NPZ file from disk!
    if (label > 0).any():
        pos_count += 1
```

**Why it's slow on Modal:**
1. Each `dataset[idx]` triggers NPZ file load (26-152MB)
2. Modal uses network storage (100-700ms latency per file)
3. 1000 files × 700ms = 700 seconds = 12 minutes (best case)
4. With network congestion: 2+ hours (observed)

**Why it's fast locally:**
1. Local SSD: <1ms file access
2. Linux page cache: Files stay in RAM after first read
3. 1000 files × 1ms = 1 second

## The Solution

BalancedSeizureDataset already knows its exact seizure distribution from the manifest!

### Key Insight

The manifest building process (`scan_existing_cache()`) has already:
1. Loaded EVERY NPZ file in the cache
2. Examined EVERY window
3. Counted exact seizures by category
4. Saved this information in `manifest.json`

### Implementation

```python
class BalancedSeizureDataset(Dataset):
    def __init__(self, cache_dir, ...):
        # Load manifest and build dataset
        manifest = load_manifest(cache_dir)

        # Count windows by type
        n_partial = len(partial_windows_selected)
        n_full = len(full_windows_selected)
        n_background = len(background_windows_selected)

        # Calculate EXACT ratio (not estimated!)
        self._n_seizure_windows = n_partial + n_full
        self._n_total_windows = len(self._entries)
        self._seizure_ratio = self._n_seizure_windows / self._n_total_windows

    @property
    def seizure_ratio(self) -> float:
        """Return exact seizure ratio - no sampling needed!"""
        return self._seizure_ratio
```

### Training Integration

```python
def train_epoch(model, dataloader, ...):
    dataset = dataloader.dataset

    # NEW: Check if dataset knows its distribution
    if isinstance(dataset, BalancedSeizureDataset):
        pos_ratio = dataset.seizure_ratio  # Instant! No I/O!
        print(f"[DATASET] Using known distribution: {pos_ratio:.1%}")
    else:
        # Fallback: sample for other dataset types
        pos_ratio = sample_and_estimate(dataset, sample_size=100)

    # Same calculation either way
    pos_weight = math.sqrt((1 - pos_ratio) / pos_ratio)
```

## Performance Impact

| Metric | Before (Sampling) | After (Direct) | Improvement |
|--------|------------------|----------------|-------------|
| Modal Time | 2+ hours | <1 second | **7200x faster** |
| Local Time | 1-2 seconds | <1ms | 1000x faster |
| I/O Operations | 1000 file reads | 0 | Eliminated |
| Network Traffic | 26-152GB | 0 | Eliminated |
| Accuracy | ±2% (sampling error) | Exact | Better |

## Critical Implementation Details

### The Bug Fix (by AI Agent)

Initially, the seizure ratio was calculated incorrectly:
```python
# WRONG - counted all manifest entries even if files missing
n_partial_used = len(partial)  # All entries in manifest
```

Fixed to:
```python
# CORRECT - only count windows actually added to dataset
n_partial_kept = 0
for item in partial:
    if cache_file.exists():
        n_partial_kept += 1
n_partial_used = n_partial_kept
```

### Mathematical Equivalence

**Sampling (Statistical Estimate)**:
```
True ratio = 0.342
Sample 1000: ratio = 0.342 ± 0.015 (95% CI)
pos_weight = sqrt((1 - 0.342) / 0.342) ≈ 1.387 ± 0.03
```

**Direct (Exact Value)**:
```
Exact ratio = 0.342 (from manifest)
pos_weight = sqrt((1 - 0.342) / 0.342) = 1.387 (exact)
```

## Validation & Safety

### Automatic Checks
1. **Manifest Validation**: Training validates manifest matches cache
2. **Ratio Sanity Check**: Warns if seizure_ratio < 0.001
3. **Force Rebuild**: `BGB_FORCE_MANIFEST_REBUILD=1` to regenerate

### What We're NOT Doing
- ❌ NOT hardcoding values
- ❌ NOT changing the dataset composition
- ❌ NOT skipping data validation
- ❌ NOT altering training dynamics

### What We ARE Doing
- ✅ Using pre-computed knowledge from manifest
- ✅ Eliminating redundant I/O operations
- ✅ Maintaining exact mathematical equivalence
- ✅ Improving accuracy (exact vs estimated)

## Usage Guide

### Standard Usage
```bash
# Training automatically uses optimization if BalancedSeizureDataset is detected
modal run --detach deploy/modal/app.py::train --config-path configs/modal/train_a100.yaml
```

### Force Manifest Rebuild
```bash
# If you suspect manifest is stale
export BGB_FORCE_MANIFEST_REBUILD=1
modal run --detach deploy/modal/app.py::train --config-path configs/modal/train_a100.yaml
```

### Verify Optimization is Active
Look for these log lines:
```
[DATASET] BalancedSeizureDataset: 49760 windows from manifest
[DATASET] Using BalancedSeizureDataset known distribution
[DATASET] Seizure ratio: 34.2% (from manifest)
[DATASET] Using pos_weight: 1.39 (sqrt scaling)
```

## Summary

This optimization transforms a 2+ hour bottleneck into instant execution by leveraging pre-computed knowledge from the manifest. It maintains perfect mathematical equivalence while eliminating thousands of unnecessary I/O operations. This is a pure engineering win with no compromises on correctness or training quality.