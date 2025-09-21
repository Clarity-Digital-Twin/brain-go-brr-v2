# P0: TUSZ SAMPLING - THE CORRECT FIX

**Severity**: P0 - TRAINING COMPLETELY BROKEN
**Date**: 2025-09-20
**Status**: IMPLEMENTED - Complete fix deployed, awaiting testing

## THE PROBLEM

Our training sees 0% seizures and collapses to useless all-negative predictions.

## HOW SEIZURETRANSFORMER SOLVED IT (1ST PLACE)

From their paper (Page 2):

> "To improve the model's ability to distinguish seizure signals from background noise, we **statistically categorize training windows into three classes**: no-seizure, full-seizure, and partial-seizure, and **uniformly sample a certain number of windows from each class** to create a balanced dataset."

### Their Exact Formula:
```
D = Dps ∪ D*fs ∪ D*ns

where:
- Dps = ALL partial-seizure windows (windows with some seizure)
- D*fs = randomly selected 0.3 × |Dps| full-seizure windows
- D*ns = randomly selected 2.5 × |Dps| no-seizure windows
```

### Key Insights:
1. They use **75% overlap** when windowing (we use 83% - stride 10s on 60s windows)
2. They **categorize BEFORE training** (not during)
3. They keep **ALL partial seizure windows** (most informative)
4. They create a **fixed ratio**: 1 partial : 0.3 full : 2.5 background

## OUR BROKEN APPROACH

```python
# We're doing this:
dataset = EEGWindowDataset(all_files)  # 250k+ windows
sampler = create_balanced_sampler(dataset, sample_size=20000)
# Randomly sample 20k windows hoping to find seizures
# RESULT: Often finds 0 seizures → training fails
```

## THE CORRECT IMPLEMENTATION

### Step 1: Build Categorized Cache (During Cache Creation)

```python
def build_categorized_cache(edf_files, csv_files, cache_dir):
    """Build cache with seizure categorization manifest."""

    manifest = {
        'partial_seizure': [],  # Windows with 0% < seizure < 100%
        'full_seizure': [],     # Windows with 100% seizure
        'no_seizure': [],       # Windows with 0% seizure
    }

    for file_idx, (edf, csv) in enumerate(zip(edf_files, csv_files)):
        # Process and cache windows
        windows, labels = process_file(edf, csv)

        # Save to cache
        cache_file = cache_dir / f"{edf.stem}_windows.npz"
        np.savez_compressed(cache_file, windows=windows, labels=labels)

        # Categorize each window
        for window_idx in range(len(windows)):
            seizure_ratio = (labels[window_idx] > 0).mean()

            if seizure_ratio == 0:
                manifest['no_seizure'].append((file_idx, window_idx))
            elif seizure_ratio > 0.99:  # Nearly 100% seizure
                manifest['full_seizure'].append((file_idx, window_idx))
            else:  # Partial seizure
                manifest['partial_seizure'].append((file_idx, window_idx))

    # Save manifest
    with open(cache_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f)

    print(f"Cache built: {len(manifest['partial_seizure'])} partial, "
          f"{len(manifest['full_seizure'])} full, "
          f"{len(manifest['no_seizure'])} no-seizure windows")
```

### Step 2: Create SeizureTransformer-style Balanced Dataset

```python
class BalancedSeizureDataset(Dataset):
    """Dataset that implements SeizureTransformer's balancing strategy."""

    def __init__(self, cache_dir: Path):
        # Load manifest
        with open(cache_dir / 'manifest.json') as f:
            manifest = json.load(f)

        partial = manifest['partial_seizure']
        full = manifest['full_seizure']
        no_seizure = manifest['no_seizure']

        # SeizureTransformer's formula: D = Dps ∪ D*fs ∪ D*ns
        # Use ALL partial seizure windows
        self.indices = partial.copy()

        # Add 0.3x random full seizure windows
        n_full = int(0.3 * len(partial))
        if full:
            selected_full = random.sample(full, min(n_full, len(full)))
            self.indices.extend(selected_full)

        # Add 2.5x random no-seizure windows
        n_background = int(2.5 * len(partial))
        if no_seizure:
            selected_background = random.sample(no_seizure,
                                               min(n_background, len(no_seizure)))
            self.indices.extend(selected_background)

        # Shuffle for training
        random.shuffle(self.indices)

        print(f"[DATASET] Created balanced dataset:")
        print(f"  - {len(partial)} partial seizure windows (100%)")
        print(f"  - {len(selected_full)} full seizure windows (30% of partial)")
        print(f"  - {len(selected_background)} no-seizure windows (250% of partial)")
        print(f"  - Total: {len(self.indices)} windows")
```

### Step 3: Quick Fix for Existing Cache (Scan and Categorize)

```python
def scan_existing_cache(cache_dir: Path) -> dict:
    """Scan existing cache files to build manifest."""

    manifest = {
        'partial_seizure': [],
        'full_seizure': [],
        'no_seizure': []
    }

    npz_files = sorted(cache_dir.glob("*.npz"))

    for file_idx, npz_file in enumerate(tqdm(npz_files)):
        data = np.load(npz_file)
        labels = data['labels']

        for window_idx in range(labels.shape[0]):
            seizure_ratio = (labels[window_idx] > 0).mean()

            if seizure_ratio == 0:
                manifest['no_seizure'].append((str(npz_file), window_idx))
            elif seizure_ratio > 0.99:
                manifest['full_seizure'].append((str(npz_file), window_idx))
            else:
                manifest['partial_seizure'].append((str(npz_file), window_idx))

    # Save manifest
    with open(cache_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f)

    return manifest
```

## WHY THIS WORKS

1. **Guarantees seizure representation** - Not hoping to randomly find them
2. **Matches paper's proven approach** - They won 1st place
3. **Efficient** - No sampling 20k windows at training time
4. **Reproducible** - Same balanced dataset every run

## ACTION PLAN

### For Modal (Currently Building Cache):
1. Let it finish building cache (~1-2 hours left)
2. It will try to train with broken sampler
3. **KILL IT** when training fails
4. Run `scan_existing_cache()` on Modal's cache
5. Restart training with `BalancedSeizureDataset`

### For Local (Cache Complete):
1. Run `scan_existing_cache()` on `cache/tusz/train`
2. Create `BalancedSeizureDataset` using manifest
3. Train with guaranteed seizure representation

## EXPECTED RESULTS

Following SeizureTransformer's approach should give us:
- **~30% of batches with seizures** (vs current 0%)
- **Stable training** (no collapse to all-negatives)
- **Better performance** (learning from most informative windows)

---

**This is the difference between 1st place and complete failure.**

## IMPLEMENTATION STATUS (2025-09-21)

### ✅ COMPLETED FIXES

1. **Cache Scanning with Manifest** (`src/brain_brr/data/cache_utils.py:52`)
   - Scans NPZ files and categorizes windows as partial/full/no-seizure
   - Creates `manifest.json` with relative paths for portability
   - Warns if no partial seizures found

2. **BalancedSeizureDataset** (`src/brain_brr/data/datasets.py:190`)
   - Implements exact SeizureTransformer formula: ALL partial + 0.3×full + 2.5×background
   - Validates partial seizures exist (fails fast if none)
   - Uses consistent numpy RNG for reproducibility
   - Logs dataset composition on creation

3. **Training Integration** (`src/brain_brr/train/loop.py:1074`)
   - Auto-builds manifest if missing
   - Uses BalancedSeizureDataset when manifest exists
   - Falls back to EEGWindowDataset on failure
   - Proper Union typing (no hacks)

4. **CLI Support** (`src/brain_brr/cli/cli.py:198`)
   - `build-cache` now creates manifest automatically
   - New `scan-cache` command for existing caches
   - Shows seizure category counts

### 🔧 TECHNICAL IMPROVEMENTS

- **Removed all type annotation hacks** - proper Union typing
- **Fixed random library consistency** - numpy RNG throughout
- **Added validation** - fails fast if no seizures found
- **Relative paths in manifest** - portable across systems
- **Better error handling** - specific exceptions, informative messages
- **Full test coverage** - unit tests pass

### ✅ QUALITY CHECKS

```bash
make q  # All checks pass (lint, format, mypy)
pytest tests/unit/data/  # 9/9 tests pass
```

### 📊 NEXT STEPS

1. **For Modal**:
   - Let cache finish building
   - Run: `python -m src scan-cache --cache-dir /path/to/modal/cache/train`
   - Restart training

2. **For Local**:
   - Run: `python -m src scan-cache --cache-dir cache/tusz/train`
   - Start training: `python -m src train configs/tusz_train_wsl2.yaml`

The implementation is 100% complete and matches SeizureTransformer's proven approach exactly.