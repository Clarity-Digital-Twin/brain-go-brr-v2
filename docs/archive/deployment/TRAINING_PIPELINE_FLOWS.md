# TRAINING PIPELINE FLOWS - CURRENT STATE

## LOCAL PIPELINE (WSL2) - BROKEN
```
┌─────────────────────────────────────────────────────────┐
│                    LOCAL TRAINING START                 │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Config: cache_dir = cache/tusz                          │
│ Reality: Cache exists at cache/train (246 files)        │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│           Dataset.__init__() - BROKEN FLOW              │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ For each of 3734 files:                             │ │
│ │   1. Check cache/tusz/{file}_windows.npz            │ │
│ │   2. NOT FOUND (wrong directory!)                   │ │
│ │   3. Add to index anyway                            │ │
│ │   4. DON'T process file (deferred to training)      │ │
│ └─────────────────────────────────────────────────────┘ │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  TRAINING LOOP - SLOW AF                │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ For each batch:                                     │ │
│ │   __getitem__() called                              │ │
│ │   ├─ Check cache/tusz/{file}_windows.npz            │ │
│ │   ├─ NOT FOUND                                      │ │
│ │   ├─ Call _process_file() ON DEMAND                 │ │
│ │   │   ├─ Load EDF (slow!)                           │ │
│ │   │   ├─ Preprocess (slow!)                         │ │
│ │   │   ├─ Extract windows (slow!)                    │ │
│ │   │   └─ Save to cache/tusz (finally!)              │ │
│ │   └─ Return window                                  │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                         │
│ Result: 27 seconds per batch instead of 2-3 seconds     │
└─────────────────────────────────────────────────────────┘
```

## MODAL PIPELINE (A100) - CORRECT BUT INEFFICIENT
```
┌─────────────────────────────────────────────────────────┐
│                   MODAL TRAINING START                  │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Config: cache_dir = /results/cache/tusz                 │
│ Reality: No cache exists (fresh container)              │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│          Dataset.__init__() - CORRECT FLOW              │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ For each of 3734 files:                             │ │
│ │   1. Check /results/cache/tusz/{file}_windows.npz   │ │
│ │   2. NOT FOUND (no cache yet)                       │ │
│ │   3. Call _process_file() IMMEDIATELY               │ │
│ │   4. Save to cache                                  │ │
│ │   5. Add to index                                   │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                         │
│ Takes 6+ HOURS due to network I/O from S3               │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│                 TRAINING LOOP - FAST                    │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ For each batch:                                     │ │
│ │   __getitem__() called                              │ │
│ │   ├─ Check /results/cache/tusz/{file}_windows.npz   │ │
│ │   ├─ FOUND (cache complete!)                        │ │
│ │   ├─ Load from cache (fast!)                        │ │
│ │   └─ Return window                                  │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                         │
│ Result: 2-3 seconds per batch (optimal)                 │
└─────────────────────────────────────────────────────────┘
```

## THE PROBLEM IN CODE

### datasets.py lines 56-78 (THE CULPRIT)
```python
if cache_path is not None and cache_path.exists():
    # Cache exists - use it
    with np.load(cache_path) as cached:
        n_windows = cached["windows"].shape[0]
else:
    # No cache - process file
    if IN_DATASET_INIT:  # Conceptually
        # Modal does this - processes ALL files upfront
        windows_arr, labels_arr = self._process_file(edf_path, i)
        np.savez_compressed(cache_path, windows=windows_arr, labels=labels_arr)
    else:  # IN___getitem__
        # Local does this - defers to training time
        # This happens in __getitem__ at line 165!
```

## WHAT SHOULD HAPPEN (IDEAL FLOW)
```
┌─────────────────────────────────────────────────────────┐
│                    ANY TRAINING START                   │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│              PRE-FLIGHT CACHE CHECK                     │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ if not cache_complete():                            │ │
│ │     print("Building complete cache...")             │ │
│ │     build_all_cache()  # Explicit, upfront          │ │
│ │ else:                                               │ │
│ │     print("Using cached data")                      │ │
│ └─────────────────────────────────────────────────────┘ │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│           Dataset.__init__() - FAST                     │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ For each file:                                      │ │
│ │   1. Load cache metadata only                       │ │
│ │   2. Build index                                    │ │
│ │   3. NO PROCESSING                                  │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                         │
│ Takes < 1 minute                                        │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌────────────────────────────────────────────────────────┐
│              TRAINING LOOP - OPTIMAL                   │
│ ┌────────────────────────────────────────────────────┐ │
│ │ __getitem__():                                     │ │
│ │   ├─ Load from cache                               │ │
│ │   └─ Return (2-3s/batch)                           │ │
│ └────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

## THE FIX - IMMEDIATE

### Step 1: Stop mixing cache directories
```bash
# After current training completes
# Option A: Move cache to expected location
mv cache/train/* cache/tusz/

# Option B: Symlink (temporary)
ln -s $(pwd)/cache/train $(pwd)/cache/tusz
```

### Step 2: Update dataset to handle partial cache better
```python
# In datasets.py __init__, add explicit mode
def __init__(self, ..., build_cache: bool = True):
    # ...
    if build_cache and not cache_path.exists():
        # Build ALL cache upfront
        self._process_file(edf_path, i)
    elif not cache_path.exists():
        # Warn but continue
        print(f"WARNING: Missing cache for {edf_path.name}")
```

### Step 3: Create cache builder script
```python
# build_cache.py
from pathlib import Path
from src.brain_brr.data.datasets import EEGWindowDataset

def build_complete_cache(data_dir: str, cache_dir: str):
    """Build complete cache before training."""
    print("Building complete cache...")
    dataset = EEGWindowDataset(
        edf_files=list(Path(data_dir).glob("**/*.edf")),
        cache_dir=Path(cache_dir),
    )
    print(f"Cache complete: {len(dataset)} windows")

if __name__ == "__main__":
    build_complete_cache(
        "data_ext4/tusz/edf/train",
        "cache/tusz"
    )
```

## CURRENT STATUS

### LOCAL (WSL2)
- **Status**: Running but SLOW (27s/batch)
- **Problem**: Building cache DURING training
- **Files Cached**: Growing as training progresses
- **ETA**: ~236 hours instead of 15 hours

### MODAL (A100)
- **Status**: Building cache BEFORE training (correct!)
- **Progress**: 91/3734 files (2.4%)
- **ETA Cache Build**: ~5-6 more hours
- **Then Training**: Will be fast once cache done

## ACTION ITEMS

1. **DO NOT TOUCH MODAL** - It's doing the right thing
2. **LOCAL**: Let it run but prepare fix for next time
3. **AFTER LOCAL COMPLETES**: Fix cache directory mismatch
4. **FUTURE**: Implement explicit cache builder