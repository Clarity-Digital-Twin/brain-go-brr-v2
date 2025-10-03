# Dataset Pipeline - Complete SSOT

**Date**: October 3, 2025
**Status**: ✅ COMPLETE & OPTIMIZED (v3.4.1)
**Purpose**: Single source of truth for all dataset types, indexes, manifests, and processing

---

## Executive Summary

**Three Dataset Types**:
1. **EEGWindowDataset**: Natural distribution (~8% seizures), uses `_dataset_index.json` for fast startup
2. **BalancedSeizureDataset**: Balanced sampling (~30% seizures), uses `manifest.json` for instant loading
3. **ValidationDataset**: Natural distribution (~8% seizures), uses `manifest.json` for instant loading (NEW Oct 3!)

**Two Index/Manifest Files**:
1. **`manifest.json`**: Categorizes windows by seizure content (partial/full/none), ~27MB train, ~13MB dev
2. **`_dataset_index.json`**: Simple window counts per file, ~150KB, gets invalidated when file list changes

**Current Optimization Status (Oct 3, 2025)**:
- ✅ **Full training**: Both datasets load instantly (<2s) from manifests
- ✅ **Full validation**: Loads instantly from manifest (ValidationDataset fix)
- ❌ **Smoke training**: Rebuilds index (30-40 min) - acceptable for testing
- ✅ **Smoke validation**: Loads instantly from manifest

---

## Dataset Types Explained

### 1. EEGWindowDataset (Original)

**Purpose**: Natural distribution dataset with optional caching
**File Used**: `_dataset_index.json` (window counts)
**Load Time**: 30-40 min first time (scans NPZ), <1s with valid index

**How It Works**:
```python
# src/brain_brr/data/datasets.py:20-140
class EEGWindowDataset(Dataset):
    def __init__(self, edf_files: list[Path], cache_dir: Path, ...):
        # Try to load _dataset_index.json
        index_path = cache_dir / "_dataset_index.json"
        if index_path.exists():
            # Validate: does index match current file list?
            if cached["files"] == [str(p) for p in edf_files]:
                # Use cached counts ✅
                self.window_counts = cached["window_counts"]
            else:
                # File list changed → rebuild index ❌
                self._build_index()
        else:
            # No index → scan all NPZ files (30-40 min)
            self._build_index()
```

**When Used**:
- Smoke training (`use_balanced_sampling: false`)
- LEGACY validation (before Oct 3 fix)

**Pros**:
- Natural distribution (realistic evaluation)
- Can work on-demand without preprocessing

**Cons**:
- Slow startup when index invalid (file list changes)
- Index invalidated when switching smoke ↔ full

---

### 2. BalancedSeizureDataset (Training Optimizer)

**Purpose**: Oversample seizures for effective training
**File Used**: `manifest.json` (window categorization)
**Load Time**: <2s (reads JSON)

**How It Works**:
```python
# src/brain_brr/data/datasets.py:200-340
class BalancedSeizureDataset(Dataset):
    def __init__(self, cache_dir: Path, ...):
        # Load manifest.json (instant)
        with (cache_dir / "manifest.json").open() as f:
            manifest = json.load(f)

        # Sample windows:
        # - 100% of partial seizure windows
        # - 30% of full seizure windows
        # - 250% of partial = background windows
        # Result: ~30% seizures in dataset (vs natural 8%)
```

**When Used**:
- Full training (`use_balanced_sampling: true`)

**Pros**:
- Instant loading from manifest
- Balanced sampling → better training
- Deterministic (seeded RNG)

**Cons**:
- Requires manifest.json to exist
- Only for training (validation needs natural distribution)

---

### 3. ValidationDataset (NEW - Oct 3, 2025)

**Purpose**: Natural distribution validation with instant loading
**File Used**: `manifest.json` (same as BalancedSeizureDataset)
**Load Time**: <2s (reads JSON)

**How It Works**:
```python
# src/brain_brr/data/datasets.py:402-494
class ValidationDataset(Dataset):
    def __init__(self, cache_dir: Path, ...):
        # Load manifest.json (instant)
        with (cache_dir / "manifest.json").open() as f:
            manifest = json.load(f)

        # Collect ALL windows (no sampling)
        all_entries = []
        all_entries.extend(manifest["partial_seizure"])
        all_entries.extend(manifest["full_seizure"])
        all_entries.extend(manifest["no_seizure"])

        # CRITICAL: Group by file, sort by window index
        # (Validation streaming requires file grouping)
        file_to_windows = defaultdict(list)
        for item in all_entries:
            file_to_windows[item["cache_file"]].append(
                (item["window_idx"], cache_path)
            )

        # Build ordered list: files sorted, windows sorted within file
        for cache_file_name in sorted(file_to_windows.keys()):
            windows_sorted = sorted(file_to_windows[cache_file_name])
            self._entries.extend(windows_sorted)
```

**When Used**:
- ALL validation (smoke + full)

**Pros**:
- Instant loading from manifest (<2s vs 30-40 min)
- Natural distribution (realistic metrics)
- Maintains file grouping for streaming validation

**Cons**:
- Requires manifest.json to exist
- Fallback to EEGWindowDataset if manifest missing

---

## Index & Manifest Files

### 1. `manifest.json` (Detailed Categorization)

**Location**: `cache/tusz/{train,dev}/manifest.json`
**Size**: 27 MB (train), 13 MB (dev)
**Created By**: `src/brain_brr/data/cache_utils.py::scan_existing_cache()`
**Generation Time**: 5-10 minutes (scans all NPZ files once)

**Structure**:
```json
{
  "partial_seizure": [
    {
      "cache_file": "aaaaaaac_s001_t000_windows.npz",
      "window_idx": 0,
      "seizure_ratio": 0.45
    },
    ...
  ],
  "full_seizure": [...],
  "no_seizure": [...]
}
```

**Categorization Logic**:
```python
# Per-window seizure ratio (fraction of samples labeled as seizure)
if ratio == 0.0:
    category = "no_seizure"
elif ratio >= 0.99:
    category = "full_seizure"
else:
    category = "partial_seizure"
```

**Used By**:
- ✅ BalancedSeizureDataset (training)
- ✅ ValidationDataset (validation)
- ❌ EEGWindowDataset (doesn't use it)

**When Regenerated**:
- First time: Created automatically by `scan_existing_cache()`
- Explicit: `export BGB_FORCE_MANIFEST_REBUILD=1`
- Manual: Delete and run training

---

### 2. `_dataset_index.json` (Simple Window Counts)

**Location**: `cache/tusz/{train,dev}/_dataset_index.json`
**Size**: 150 KB (1832 files), 282 B (3 files for smoke)
**Created By**: `EEGWindowDataset.__init__()` during first load
**Generation Time**: 30-40 minutes (opens all NPZ files)

**Structure**:
```json
{
  "files": [
    "/full/path/to/aaaaaaac_s001_t000.edf",
    "/full/path/to/aaaaaaac_s001_t001.edf",
    ...
  ],
  "window_counts": [25, 18, 21, ...]
}
```

**Invalidation Logic**:
```python
# Index is valid ONLY if:
# 1. File exists
# 2. File list EXACTLY matches current EDF list (including order and paths)
if cached["files"] != [str(p) for p in self.edf_files]:
    # Rebuild index
```

**Why It Gets Invalidated**:
1. File list order changes
2. File paths change (absolute vs relative)
3. Switching smoke ↔ full (different file counts)
4. File corruption/deletion

**Used By**:
- ✅ EEGWindowDataset (both train and val)
- ❌ BalancedSeizureDataset (doesn't use it)
- ❌ ValidationDataset (doesn't use it)

---

## Usage Matrix: When Each Dataset is Used

### Training Dataset Selection

**Config**: `data.use_balanced_sampling`

| Config Value | Dataset Used | File Used | Load Time |
|--------------|-------------|-----------|-----------|
| `true` (full training) | BalancedSeizureDataset | `train/manifest.json` | <2s ✅ |
| `false` (smoke test) | EEGWindowDataset | `train/_dataset_index.json` | 30-40 min ❌ |

**Why Smoke Uses EEGWindowDataset**:
- Smoke limits files with `BGB_LIMIT_FILES` or `BGB_SMOKE_TEST=1`
- Balanced sampling doesn't work with file limiting
- So smoke sets `use_balanced_sampling: false`
- Result: EEGWindowDataset (natural distribution, rebuilds index)

**Code Reference**:
```python
# src/brain_brr/train/loop.py:490-540
if cfg.data.use_balanced_sampling and manifest_path.exists():
    train_dataset = BalancedSeizureDataset(train_cache_dir)
else:
    train_dataset = EEGWindowDataset(
        train_files,
        cache_dir=train_cache_dir,
        ...
    )
```

---

### Validation Dataset Selection (NEW - Oct 3, 2025)

**Always tries ValidationDataset first**, falls back to EEGWindowDataset if manifest missing:

| Manifest Exists? | Dataset Used | File Used | Load Time |
|-----------------|-------------|-----------|-----------|
| ✅ Yes | ValidationDataset | `dev/manifest.json` | <2s ✅ |
| ❌ No | EEGWindowDataset | `dev/_dataset_index.json` | 30-40 min ❌ |

**Code Reference**:
```python
# src/brain_brr/train/loop.py:539-567
val_manifest_path = val_cache_dir / MANIFEST_FILENAME

if val_manifest_path.exists():
    try:
        val_dataset = ValidationDataset(val_cache_dir)
    except Exception as e:
        logger.warning(f"ValidationDataset failed: {e}")
        val_dataset = EEGWindowDataset(...)
else:
    logger.info("No validation manifest, using EEGWindowDataset")
    val_dataset = EEGWindowDataset(...)
```

---

## Processing Timeline: What Happens When

### First Time Ever (No Cache)

**Step 1**: NPZ Creation (Hours)
```bash
# Preprocess raw EDF → NPZ windows
# - Bandpass, resample, z-score, clip ±10σ
# - Extract 60s windows with 10s stride
# - Save as *.npz (windows + labels)
# Time: ~5-10 hours for 4667 train files
```

**Step 2**: Manifest Creation (5-10 min)
```bash
# Scan NPZ files, categorize windows by seizure ratio
# - Creates train/manifest.json (27 MB)
# - Creates dev/manifest.json (13 MB)
# Time: ~5-10 minutes total
```

**Step 3**: Training Starts
```bash
# Full training: BalancedSeizureDataset loads manifest (<2s) ✅
# Validation: ValidationDataset loads manifest (<2s) ✅
```

**Total First-Time Cost**: ~5-10 hours (one-time NPZ creation) + ~5-10 min (manifests)

---

### Subsequent Runs (Cache Exists)

**Scenario 1: Full Training** (`use_balanced_sampling: true`)
```bash
[14:25:28] INFO [BalancedSeizureDataset] Created with 61616 windows  ← <2s ✅
[14:25:29] INFO [ValidationDataset] Created with 142000 windows      ← <2s ✅
[14:25:30] INFO Starting epoch 1...
```

**Scenario 2: Smoke Test** (`use_balanced_sampling: false`)
```bash
[14:25:28] INFO [DATA] Building dataset index for 4667 files...     ← 30-40 min ❌
[14:25:31] INFO [DATA] Processing file 11/4667...
... (30-40 minutes)
[14:55:00] INFO [DATA] Saved index cache                            ← Done
[14:55:01] INFO [ValidationDataset] Created with 142000 windows     ← <2s ✅
[14:55:02] INFO Starting epoch 1...
```

**Why Smoke is Slow**:
- EEGWindowDataset rebuilds index every time (index invalidated by file list changes)
- This is ACCEPTABLE because smoke is for quick testing, not production training
- The 30-40 min cost is one-time per smoke run

---

### Switching Between Modes

**Smoke → Full Training**:
```bash
# Smoke creates: train/_dataset_index.json (3 files)
# Full training: IGNORES index, uses train/manifest.json instead ✅
# No conflict!
```

**Full → Smoke Training**:
```bash
# Full uses: train/manifest.json
# Smoke: IGNORES manifest, rebuilds train/_dataset_index.json ❌
# 30-40 min rebuild (acceptable for testing)
```

**Key Insight**: The two dataset types use DIFFERENT files, so no conflicts!

---

## Optimization Status (Oct 3, 2025)

### ✅ What's Optimized

1. **Full Training (Production)**:
   - Train: BalancedSeizureDataset → manifest.json → <2s ✅
   - Val: ValidationDataset → manifest.json → <2s ✅
   - **Total startup: <5s** (down from 40-50 minutes!)

2. **Validation (All Modes)**:
   - ValidationDataset → manifest.json → <2s ✅
   - **99.6% improvement** (was 5-10 minutes with EEGWindowDataset)

### ❌ What's NOT Optimized (By Design)

1. **Smoke Training**:
   - EEGWindowDataset → rebuilds index → 30-40 min ❌
   - **Why acceptable**: Smoke is for testing, runs rarely
   - **Why not fixed**: Balanced sampling doesn't work with file limiting

### 📋 Files Created During Training

| File | Created When | Size | Purpose |
|------|-------------|------|---------|
| `*.npz` | First time (hours) | 306 GB train, 143 GB dev | Window data |
| `train/manifest.json` | First time or rebuild | 27 MB | Balanced sampling |
| `dev/manifest.json` | First time or rebuild | 13 MB | Fast validation |
| `train/_dataset_index.json` | Smoke mode only | 282 B - 1 MB | EEGWindowDataset cache |
| `dev/_dataset_index.json` | LEGACY (unused now) | 148 KB | EEGWindowDataset cache |
| `.cache_metadata.json` | First time | 282 B | Split validation |

---

## Configuration Reference

### configs/local/smoke.yaml
```yaml
data:
  cache_dir: cache/tusz
  use_balanced_sampling: false  # Forces EEGWindowDataset (for file limiting)
```

**Result**:
- Train: EEGWindowDataset → rebuilds index (30-40 min) ❌
- Val: ValidationDataset → reads manifest (<2s) ✅

---

### configs/local/train.yaml
```yaml
data:
  cache_dir: cache/tusz
  use_balanced_sampling: true  # Uses BalancedSeizureDataset
```

**Result**:
- Train: BalancedSeizureDataset → reads manifest (<2s) ✅
- Val: ValidationDataset → reads manifest (<2s) ✅

---

## Troubleshooting

### Problem: Validation Takes 5-10 Minutes to Start

**Diagnosis**:
```bash
# Check if manifest exists
ls -lh cache/tusz/dev/manifest.json

# Check logs for dataset type
grep -E "(ValidationDataset|EEGWindowDataset)" training.log
```

**Solution**:
- If manifest missing: Let it generate once (~5 min), then cached forever
- If ValidationDataset not used: Check code version (needs Oct 3 fix)

---

### Problem: Smoke Test Takes 40 Minutes to Start

**Diagnosis**:
```bash
# This is EXPECTED behavior
# Smoke uses EEGWindowDataset for training (rebuilds index)
```

**Why This Happens**:
1. Smoke config: `use_balanced_sampling: false`
2. Result: EEGWindowDataset (not BalancedSeizureDataset)
3. EEGWindowDataset rebuilds `_dataset_index.json` (30-40 min)

**Is This a Bug?**: NO - this is by design
- Smoke is for testing, not production
- Full training is optimized (<5s startup)
- Smoke 30-40 min cost is acceptable

**Workaround**: Use full training config for faster startup

---

### Problem: Index Cache Invalid After Switching Modes

**Diagnosis**:
```bash
# Check index contents
cat cache/tusz/train/_dataset_index.json | python3 -m json.tool | head -20

# Look for file count mismatch
# Smoke: 3 files
# Full: 4667 files
```

**Why This Happens**:
- `_dataset_index.json` stores exact file list
- Switching modes changes file list
- EEGWindowDataset detects mismatch → rebuilds

**Solution**: This is normal behavior, no fix needed

---

### Problem: Want to Force Manifest Rebuild

**Solution**:
```bash
# Delete manifest
rm cache/tusz/train/manifest.json
rm cache/tusz/dev/manifest.json

# Or force rebuild
export BGB_FORCE_MANIFEST_REBUILD=1
.venv/bin/python -m src train configs/local/train.yaml
```

---

## Future Optimizations (Not Needed Now)

### Could Smoke Use Manifests?

**Idea**: Make smoke also use BalancedSeizureDataset

**Problem**: Balanced sampling doesn't work with file limiting
- `BGB_LIMIT_FILES=3` → Only 3 NPZ files
- Balanced sampling needs full manifest (all files)
- Result: Can't sample properly

**Verdict**: Not worth optimizing - smoke is for testing only

---

### Could We Cache Index with Relative Paths?

**Idea**: Store relative paths in `_dataset_index.json` so it survives mode switches

**Problem**: File LIST still changes between smoke/full
- Smoke: 3 files
- Full: 4667 files
- Index would still be invalid

**Verdict**: Doesn't solve the core issue

---

## Summary

**What We Fixed (Oct 3, 2025)**:
- ✅ Validation loads in <2s (was 5-10 minutes)
- ✅ ValidationDataset uses manifest.json for instant loading
- ✅ Full training startup: <5s total (was 40-50 minutes)

**What's Still "Slow" (By Design)**:
- ❌ Smoke training: 30-40 min index rebuild
- **This is acceptable** - smoke is for testing, full training is optimized

**Key Takeaway**:
- **Production (full training)**: 100% optimized ✅
- **Testing (smoke)**: Acceptable tradeoff for flexibility ✅

---

**Last Updated**: October 3, 2025
**Next Review**: When dataset architecture changes
