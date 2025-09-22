# Balanced Sampling Optimization

## The 2-Hour Modal Bottleneck (SOLVED)

### The Problem
Training on Modal was experiencing **2+ hour delays** during dataset statistics calculation:

```python
# OLD CODE - Caused 2+ hour delay on Modal
sample_indices = torch.randperm(len(dataset))[:1000]
for idx in sample_indices:
    _, label = dataset[idx]  # Loads NPZ from network storage!
    if (label > 0).any():
        pos_count += 1
```

**Why it was slow:**
- Each `dataset[idx]` loads a 26-152MB NPZ file
- Modal network storage: 100-700ms per file
- 1000 files × 700ms = 12+ minutes (best case)
- With congestion: 2+ hours (observed)

### The Solution
BalancedSeizureDataset already knows its exact distribution from the manifest!

```python
class BalancedSeizureDataset(Dataset):
    def __init__(self, cache_dir, ...):
        # Load manifest - already has all stats!
        manifest = load_manifest(cache_dir)

        # Calculate exact ratio (no sampling!)
        self._seizure_ratio = n_seizure_windows / n_total_windows

    @property
    def seizure_ratio(self) -> float:
        """Return exact ratio - instant, no I/O!"""
        return self._seizure_ratio
```

## Performance Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Modal Time | 2+ hours | <1 second | **7200x faster** |
| I/O Operations | 1000 file reads | 0 | Eliminated |
| Network Traffic | 26-152GB | 0 | Eliminated |
| Accuracy | ±2% error | Exact | Better |

## How It Works

### 1. Manifest Generation (One-time)
During cache building, `scan_existing_cache()`:
- Loads EVERY NPZ file
- Counts windows by seizure type
- Saves counts to `manifest.json`

### 2. Dataset Creation (Fast)
BalancedSeizureDataset:
- Loads manifest (small JSON)
- Knows exact seizure ratio
- No sampling needed!

### 3. Training Integration
```python
def train_epoch(model, dataloader, ...):
    dataset = dataloader.dataset

    # Check if dataset has pre-computed stats
    if isinstance(dataset, BalancedSeizureDataset):
        pos_ratio = dataset.seizure_ratio  # Instant!
        print(f"[DATASET] Using known distribution: {pos_ratio:.1%}")
    else:
        # Fallback for other datasets
        pos_ratio = sample_and_estimate(dataset)

    pos_weight = math.sqrt((1 - pos_ratio) / pos_ratio)
```

## Verification

Look for these log lines to confirm optimization is active:
```
[DATASET] BalancedSeizureDataset: 49760 windows from manifest
[DATASET] Using BalancedSeizureDataset known distribution
[DATASET] Seizure ratio: 34.2% (from manifest)
[DATASET] Using pos_weight: 1.39 (sqrt scaling)
```

## Key Points

✅ **Mathematically equivalent** - Same pos_weight calculation
✅ **More accurate** - Exact ratio vs statistical estimate
✅ **7200x faster** - Eliminates all I/O operations
✅ **Automatic** - No config changes needed

## Troubleshooting

If manifest seems stale:
```bash
export BGB_FORCE_MANIFEST_REBUILD=1
python -m src train configs/modal/train_a100.yaml
```

This forces a rebuild of the manifest from cache files.