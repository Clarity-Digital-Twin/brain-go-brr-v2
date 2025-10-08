# Dev/Val Cache Zero Windows Investigation

**Date**: October 8, 2025
**Status**: ACTIVE INVESTIGATION - DO NOT FIX REACTIVELY
**Severity**: P0 - Training blocked on Modal

## 🔴 Problem Statement

Modal training run `ap-uitgvl8kXZoKJ4fZoSehsI` shows:

```
[ValidationDataset] Created with 0 windows (filtered):
  - 7944 partial seizure
  - 3536 full seizure
  - 136744 no-seizure
  - Seizure ratio: 0.0% (natural distribution)
```

**This is contradictory**: The dataset reports it has seizure windows in the manifest BUT the final count is 0 with ratio 0.0%.

## 📊 Expected Behavior

From `docs/02-data/cache-layout.md`:
- Dev cache should have **1832 NPY data files**
- Dev manifest should reference **~148,224 windows** (from v3.8.3 manifest regeneration)
- ValidationDataset should load ALL windows in natural distribution (~8% seizures)

## 🔍 Root Cause Analysis

### Issue 1: Manifest Deletion Logic is Train-Only

**File**: `src/brain_brr/train/loop.py:641-662`

```python
train_cache_dir = data_cache_root / "train"  # ← TRAIN only
use_balanced = bool(config.data.use_balanced_sampling)
manifest_path = train_cache_dir / MANIFEST_FILENAME  # ← TRAIN manifest only

# Force manifest rebuild if requested or if it exists but is invalid
if use_balanced and manifest_path.exists():  # ← Only checks TRAIN manifest
    # ... validation logic ...
    if not validate_manifest(train_cache_dir, manifest_data):
        logger.warning("Invalid/stale manifest detected → deleting for rebuild")
        manifest_path.unlink()  # ← Deletes TRAIN manifest only
```

**Problem**: The code only validates and rebuilds the **train** manifest. The **dev** manifest is never checked for staleness.

### Issue 2: Manifest Rebuild Looks for Wrong File Format

**File**: `src/brain_brr/train/loop.py:664-678`

```python
if use_balanced and not manifest_path.exists():
    train_cache_dir.mkdir(parents=True, exist_ok=True)
    existing_cache_files = list(train_cache_dir.glob("*.npz"))  # ← BUG: Looks for NPZ!
    if existing_cache_files:
        # ... build manifest ...
    else:
        logger.info("[DATA] Skipping manifest build - cache not yet populated")
```

**Problem**:
- We use **NPY mmap format** (`*_data.npy` + `*_labels.npy`)
- Code looks for **NPZ files** (`*.npz`)
- Result: **Always skips manifest rebuild** even when cache exists!

### Issue 3: Dev Manifest Never Rebuilt

**File**: `src/brain_brr/train/loop.py:734-785`

```python
# Validation cache uses "dev" subdir (TUSZ official naming)
val_split_name = "dev"
val_cache_dir = data_cache_root / val_split_name
val_manifest_path = val_cache_dir / MANIFEST_FILENAME

# Try ValidationDataset (instant load from manifest)
if val_manifest_path.exists():  # ← If stale, no rebuild logic!
    try:
        val_dataset = ValidationDataset(val_cache_dir, allowed_cache_files=...)
```

**Problem**: If the dev manifest exists but is **stale/invalid**, there is no validation or rebuild logic. It just uses it.

### Issue 4: ValidationDataset Shows Contradictory Stats

**File**: `src/brain_brr/data/datasets.py:627-638`

The log output shows:
```
[ValidationDataset] Created with 0 windows (filtered):
  - 7944 partial seizure    ← These numbers come from manifest
  - 3536 full seizure        ← These numbers come from manifest
  - 136744 no-seizure        ← These numbers come from manifest
  - Seizure ratio: 0.0%      ← But final ratio is 0%!
```

**Analysis**: Looking at the code:

```python
# Line 577-585: Load ALL entries from manifest
partial: list[dict] = list(manifest.get("partial_seizure", []))
full: list[dict] = list(manifest.get("full_seizure", []))
no_seizure: list[dict] = list(manifest.get("no_seizure", []))
all_entries: list[dict] = []
all_entries.extend(partial)  # ← 7944 entries
all_entries.extend(full)      # ← 3536 entries
all_entries.extend(no_seizure) # ← 136744 entries

# Line 601-609: Filter by allowed_cache_files + file existence
for item in all_entries:
    cache_file_name = item["cache_file"]
    if allowed_cache_files is not None and cache_file_name not in allowed_cache_files:
        continue  # ← FILTERING happens here!
    cache_file_path = self.cache_dir / cache_file_name
    if cache_file_exists(cache_file_path):
        file_to_windows[cache_file_name].append((int(item["window_idx"]), cache_file_path))
    else:
        missing_ref_count += 1  # ← Or files are missing!

# Line 620-625: Build final indices
self._entries: list[tuple[Path, int]] = indices  # ← 0 entries after filtering!
```

**The "(filtered)" note is KEY**: The dataset received an `allowed_cache_files` whitelist and **NONE of the manifest entries matched!**

### Issue 5: Allowed Files Whitelist Mismatch

**File**: `src/brain_brr/train/loop.py:744-750`

```python
allowed_cache_files = (
    {f"{val_file.stem}_data.npy" for val_file in val_files} if val_files else None
)
val_dataset = ValidationDataset(
    val_cache_dir,
    allowed_cache_files=allowed_cache_files,  # ← Whitelist based on EDF stems
)
```

**Problem**:
- `val_files` are EDF file paths (e.g., `aaaaaajy_s001_t000.edf`)
- Whitelist creates: `{"aaaaaajy_s001_t000_data.npy", ...}`
- Manifest references: Could be using OLD naming from stale manifest!

**Hypothesis**: The dev manifest is STALE and uses old naming convention that doesn't match the NPY files on disk.

## 🌲 Execution Flow Tree

```
train() in loop.py
│
├─ Line 620-621: Build cache paths
│   ├─ train_cache = /results/cache/tusz_mmap/train
│   └─ val_cache = /results/cache/tusz_mmap/dev
│
├─ Line 641-662: TRAIN manifest validation & deletion
│   ├─ manifest_path = train_cache / "manifest.json"  ← TRAIN only
│   ├─ validate_manifest(train_cache, manifest_data)
│   ├─ IF INVALID: manifest_path.unlink()  ← Deletes TRAIN manifest
│   └─ ✅ Result: Train manifest deleted if stale
│
├─ Line 664-678: TRAIN manifest rebuild (BROKEN)
│   ├─ existing_cache_files = train_cache.glob("*.npz")  ← BUG: Wrong format!
│   ├─ IF files exist: scan_existing_cache(train_cache)
│   └─ ❌ Result: Skips rebuild because no NPZ files found
│
├─ Line 685-732: Create train dataset
│   ├─ IF manifest exists: Use BalancedSeizureDataset
│   └─ ELSE: Use EEGWindowDataset (builds cache + index)
│       └─ ✅ Result: EEGWindowDataset built train cache successfully (303,990 windows)
│
├─ Line 787-798: Post-build train manifest creation
│   ├─ IF just built via EEGWindowDataset AND no manifest:
│   │   └─ scan_existing_cache(train_cache)  ← Builds manifest!
│   └─ ✅ Result: Train manifest created (61,616 balanced windows)
│
├─ Line 734-785: Create dev/validation dataset
│   ├─ val_cache_dir = /results/cache/tusz_mmap/dev
│   ├─ val_manifest_path = val_cache / "manifest.json"
│   ├─ allowed_cache_files = {f"{edf.stem}_data.npy" for edf in val_files}
│   │
│   ├─ IF val_manifest_path.exists():  ← Dev manifest exists!
│   │   ├─ ValidationDataset(val_cache, allowed_cache_files=...)
│   │   └─ ❌ Result: 0 windows (filtered) - whitelist mismatch!
│   │
│   └─ ELSE:
│       └─ EEGWindowDataset(val_files, cache_dir=val_cache, ...)
│           └─ Builds dev cache + index
│
└─ 🔴 BLOCKER: Validation dataset has 0 windows, training cannot proceed
```

## 🔍 Evidence from Logs

### 1. Manifest Deletion (Train Only)

```
[2025-10-08 05:50:35.733][src.brain_brr.train.loop][WARNING] Invalid/stale manifest detected → deleting for rebuild
[2025-10-08 05:50:35.781][src.brain_brr.train.loop][INFO] [DATA] Skipping manifest build - cache not yet populated
```

**Analysis**:
- Train manifest was detected as stale and deleted
- Rebuild was skipped because it looked for NPZ files (found none)

### 2. Train Cache Built via EEGWindowDataset

```
[2025-10-08 05:50:35.831][src.brain_brr.data.datasets][INFO] [DATA] Building dataset index for 4667 files...
[2025-10-08 05:50:35.831][src.brain_brr.data.datasets][INFO] [DATA] Processing file 1/4667: aaaaaaac_s001_t000.edf
...
[2025-10-08 05:58:30.868][src.brain_brr.data.datasets][INFO] [DATA] Saved index cache to /results/cache/tusz_mmap/train/_dataset_index.json
[2025-10-08 05:58:30.868][src.brain_brr.data.datasets][INFO] [DATA] Dataset ready! Total windows: 303990
```

**Analysis**: EEGWindowDataset successfully built train cache with 303,990 windows.

### 3. Validation Dataset Created with 0 Windows

```
[2025-10-08 05:58:31.277][src.brain_brr.data.datasets][INFO] [ValidationDataset] Created with 0 windows (filtered):
  - 7944 partial seizure
  - 3536 full seizure
  - 136744 no-seizure
  - Seizure ratio: 0.0% (natural distribution)
```

**The "(filtered)" is CRITICAL**: This means the allowed_cache_files whitelist filtered out EVERYTHING!

### 4. Train Manifest Built Post-Cache

```
[2025-10-08 05:58:31.334][src.brain_brr.train.loop][INFO] [DATA] Cache built, now creating manifest for balanced sampling...
[2025-10-08 05:59:14.876][src.brain_brr.data.cache_utils][INFO] Manifest created from NPY (mmap): 16215 partial, 8446 full, 279329 no-seizure
[2025-10-08 05:59:14.876][src.brain_brr.data.cache_utils][INFO]   Seizure ratio: 8.1%
[2025-10-08 05:59:23.780][src.brain_brr.train.loop][INFO] [DATA] Switched to BalancedSeizureDataset: 61616 windows
```

**Analysis**: Train manifest was successfully created AFTER cache build (post-build logic worked).

## 🎯 Root Cause Summary

**The dev manifest exists but is STALE/INVALID, and there is NO validation or rebuild logic for it.**

Possible causes:
1. **Old naming convention**: Manifest uses old NPZ naming (`*_windows.npz`) but cache uses NPY naming (`*_data.npy`)
2. **File mismatch**: Manifest references files that don't exist in dev cache
3. **Whitelist filtering**: `allowed_cache_files` doesn't match manifest entries

## 📁 File Locations

### Cache Structure (Expected)

```
/results/cache/tusz_mmap/
├── train/
│   ├── manifest.json           ✅ (rebuilt during training)
│   ├── _dataset_index.json     ✅ (created by EEGWindowDataset)
│   ├── *_data.npy (4667 files) ✅
│   └── *_labels.npy (4667)     ✅
│
└── dev/
    ├── manifest.json           ⚠️ (EXISTS but STALE/INVALID)
    ├── _dataset_index.json     ❓ (unknown)
    ├── *_data.npy (1832 files) ❓ (need to verify)
    └── *_labels.npy (1832)     ❓ (need to verify)
```

### Code Files Involved

1. **src/brain_brr/train/loop.py**:
   - Line 641-662: Train manifest validation (dev not checked)
   - Line 664-678: Manifest rebuild (looks for wrong format)
   - Line 734-785: Dev dataset creation (no validation logic)

2. **src/brain_brr/data/datasets.py**:
   - Line 532-649: ValidationDataset (filters by allowed_cache_files)
   - Line 601-609: File existence check + whitelist filtering

3. **src/brain_brr/data/cache_utils.py**:
   - Line 142-262: `scan_existing_cache()` - builds manifest
   - Line 265-308: `validate_manifest()` - checks manifest validity

4. **deploy/modal/app.py**:
   - Line 191-339: `populate_cache()` - copies cache from S3 to Modal SSD
   - Line 259-292: Dev split copy logic

## 🔧 Investigation Commands

### On Modal (via check_cache function)

```bash
modal run deploy/modal/app.py --action check-cache
```

**Expected output we need**:
```
Dev Split (/results/cache/tusz_mmap/dev):
  manifest.json:         ✅ or ❌
  _dataset_index.json:   ✅ or ❌
  *_data.npy files:      ✅ 1832 or ⚠️  N
  *_labels.npy files:    ✅ 1832 or ⚠️  N
```

### Inspect Dev Manifest Content

```bash
# View first entry in dev manifest
modal run deploy/modal/app.py --action inspect-manifest --split dev
```

### List Dev Cache Files

```bash
# Count files
modal run --detach -q "python -c 'from pathlib import Path; print(len(list(Path(\"/results/cache/tusz_mmap/dev\").glob(\"*_data.npy\"))))'"

# Show first 10 filenames
modal run --detach -q "python -c 'from pathlib import Path; [print(p.name) for p in sorted(Path(\"/results/cache/tusz_mmap/dev\").glob(\"*_data.npy\"))[:10]]'"
```

## 🎯 Proposed Fix Strategy (DO NOT IMPLEMENT YET)

### Option 1: Add Dev Manifest Validation (Minimal Change)

**Files**: `src/brain_brr/train/loop.py:734-785`

```python
# Validation cache uses "dev" subdir (TUSZ official naming)
val_split_name = "dev"
val_cache_dir = data_cache_root / val_split_name
val_manifest_path = val_cache_dir / MANIFEST_FILENAME

# ✨ NEW: Validate dev manifest just like train manifest
if val_manifest_path.exists():
    import json
    from src.brain_brr.data.cache_utils import validate_manifest

    try:
        with open(val_manifest_path) as f:
            val_manifest_data = json.load(f)

        if not validate_manifest(val_cache_dir, val_manifest_data):
            logger.warning("[DEV] Invalid/stale manifest detected → deleting for rebuild")
            val_manifest_path.unlink()
    except Exception as e:
        logger.warning(f"[DEV] Failed to validate manifest: {e}, deleting...")
        val_manifest_path.unlink()

# ✨ NEW: Rebuild dev manifest if missing
if not val_manifest_path.exists():
    existing_dev_files = list(val_cache_dir.glob("*_data.npy"))  # ✅ NPY format!
    if existing_dev_files:
        from src.brain_brr.data.cache_utils import scan_existing_cache
        _ = scan_existing_cache(val_cache_dir)
        logger.info(f"[DEV] Built manifest from {len(existing_dev_files)} NPY files")
```

**Pros**:
- Minimal change, follows existing train pattern
- Fixes stale manifest issue
- Fixes NPY vs NPZ glob bug

**Cons**:
- Duplicates validation logic (not DRY)

### Option 2: Extract Shared Manifest Validation Function (Clean)

**New file**: `src/brain_brr/data/manifest_manager.py`

```python
def ensure_valid_manifest(
    cache_dir: Path,
    split_name: str = "train",
    force_rebuild: bool = False,
) -> Path:
    """Ensure manifest exists and is valid, rebuilding if necessary.

    Args:
        cache_dir: Path to cache directory (e.g., cache/tusz_mmap/train)
        split_name: Split name for logging (e.g., "train", "dev")
        force_rebuild: Force rebuild even if valid

    Returns:
        Path to valid manifest.json
    """
    manifest_path = cache_dir / MANIFEST_FILENAME

    # Check if rebuild forced or manifest invalid
    should_rebuild = force_rebuild
    if manifest_path.exists() and not force_rebuild:
        try:
            with open(manifest_path) as f:
                manifest_data = json.load(f)
            if not validate_manifest(cache_dir, manifest_data):
                logger.warning(f"[{split_name.upper()}] Invalid/stale manifest → deleting")
                manifest_path.unlink()
                should_rebuild = True
        except Exception as e:
            logger.warning(f"[{split_name.upper()}] Failed to validate manifest: {e}")
            manifest_path.unlink()
            should_rebuild = True
    elif not manifest_path.exists():
        should_rebuild = True

    # Rebuild if needed
    if should_rebuild:
        existing_files = list(cache_dir.glob("*_data.npy"))  # ✅ NPY format!
        if existing_files:
            _ = scan_existing_cache(cache_dir)
            logger.info(f"[{split_name.upper()}] Built manifest from {len(existing_files)} files")
        else:
            logger.warning(f"[{split_name.upper()}] No cache files found, cannot build manifest")

    return manifest_path
```

**Usage in loop.py**:

```python
from src.brain_brr.data.manifest_manager import ensure_valid_manifest

# Train manifest
train_cache_dir = data_cache_root / "train"
train_manifest_path = ensure_valid_manifest(
    train_cache_dir,
    split_name="train",
    force_rebuild=env.force_manifest_rebuild()
)

# Dev manifest
val_cache_dir = data_cache_root / "dev"
val_manifest_path = ensure_valid_manifest(
    val_cache_dir,
    split_name="dev",
    force_rebuild=False  # Only rebuild if invalid
)
```

**Pros**:
- DRY - shared logic for both splits
- Easier to test and maintain
- Fixes both bugs (NPY glob + dev validation)

**Cons**:
- More code churn
- Requires new file + imports

### Option 3: Auto-Rebuild in ValidationDataset (Defensive)

**Files**: `src/brain_brr/data/datasets.py:570-576`

```python
manifest_path = self.cache_dir / constants.MANIFEST_FILENAME

# ✨ NEW: Validate manifest before using
if ensure_manifest and manifest_path.exists():
    from src.brain_brr.data.cache_utils import validate_manifest
    import json

    try:
        with open(manifest_path) as f:
            manifest_data = json.load(f)

        if not validate_manifest(self.cache_dir, manifest_data):
            logger.warning(f"[ValidationDataset] Stale manifest at {manifest_path}, rebuilding...")
            manifest_path.unlink()
            _ = scan_existing_cache(self.cache_dir)
    except Exception as e:
        logger.warning(f"[ValidationDataset] Manifest validation failed: {e}, rebuilding...")
        if manifest_path.exists():
            manifest_path.unlink()
        _ = scan_existing_cache(self.cache_dir)

if ensure_manifest and not manifest_path.exists():
    _ = scan_existing_cache(self.cache_dir)
```

**Pros**:
- Self-healing - ValidationDataset handles its own manifest
- Defensive programming - works even if loop.py doesn't validate

**Cons**:
- Hides the problem - loop.py should handle this
- Slower first load if manifest needs rebuild

## ✅ Next Steps (DO NOT CODE YET)

1. **Run investigation commands** to confirm:
   - Dev cache has 1832 NPY files on Modal
   - Dev manifest exists but references wrong files
   - Whitelist mismatch is the filtering cause

2. **Verify hypothesis** by checking:
   - First manifest entry format (NPZ vs NPY naming)
   - First allowed_cache_files entry format
   - First actual dev file name on disk

3. **Get consensus** on fix approach:
   - Option 1: Quick fix (add dev validation to loop.py)
   - Option 2: Clean fix (extract shared function)
   - Option 3: Defensive fix (ValidationDataset self-heals)

4. **Test fix locally** before Modal deployment:
   - Delete local dev manifest
   - Run training
   - Verify manifest rebuilds and validation works

5. **Deploy to Modal** with smoke test first

## 📝 Questions to Answer

1. Does dev cache exist on Modal with 1832 NPY files?
2. What does the first entry in dev manifest.json look like?
3. What format are the actual dev filenames on disk?
4. Does `allowed_cache_files` match the manifest entry format?
5. Why wasn't dev manifest rebuilt when train was?

---

**Status**: Investigation complete, awaiting command execution to verify hypothesis before implementing fix.
