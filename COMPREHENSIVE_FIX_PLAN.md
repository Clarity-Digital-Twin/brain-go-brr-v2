# Comprehensive Fix Plan - All Remaining Issues

**Date**: October 6, 2025
**Status**: 🔴 BLOCKED - Must fix before training
**Scope**: 1 P0 blocker + 6 P2 code quality issues
**Total Time**: ~4-5 hours (P0: 1-2 hours, P2: 2-3 hours)

---

## Executive Summary

**What's Blocking Training** (P0):
- ❌ **NPZ files contaminating Modal cache** (3 stray files from first failed smoke test)
- ❌ **datasets.py creates NPZ files on cache miss** (will re-contaminate cache)
- ❌ **Cache validation logic expects NPZ** (finds contamination, reports wrong count)

**What's Not Blocking Training** (P2 - Code Quality):
- ⚠️ clean_cache() references old path
- ⚠️ Type annotations use `Any | None`
- ⚠️ NPZ references in comments
- ⚠️ Duplicate _load_cache_for_worker (3x identical)

**Training Impact**:
- **P0**: 🔴 BLOCKS training - NPZ files will cause confusion, cache miss risk
- **P2**: 🟡 Does NOT block - purely code quality/maintainability

---

## P0 BLOCKER: NPZ Cache Contamination

### Problem Statement

**Discovery**: Modal cache contains BOTH NPY (correct) and NPZ (wrong) files from first failed smoke test.

**Evidence from Modal Browser**:
```
/results/cache/tusz_mmap/train/
├── aaaaaaac_s001_t000_data.npy (27.8 MiB)    ✅ From populate_cache
├── aaaaaaac_s001_t000_labels.npy (1.5 MiB)   ✅ From populate_cache
└── aaaaaaac_s001_t000_windows.npz (25.8 MiB) ❌ FROM WHERE?!
```

**Root Cause Timeline**:
1. Oct 6, 12:16pm: Launched first smoke test with WRONG cache path (`/results/cache/tusz`)
2. Training didn't find cache at that path, started building cache on-the-fly
3. **datasets.py:119-130 created NPZ files** (not NPY!) when building cache
4. Created 3 NPZ files before we killed the test at 12:21pm
5. Oct 6, 12:38pm: Fixed cache path, launched second smoke test
6. Second test reported: "[CACHE] ✅ Using valid Modal SSD cache: 3 NPZ files"
7. **BUT**: We populated 4667 train + 1832 dev NPY files! Why only 3 NPZ?

**Answer**: The 3 NPZ files are CONTAMINATION from the first failed test.

---

### Root Cause: datasets.py Creates NPZ Files

**Location**: `src/brain_brr/data/datasets.py:117-130`

**Code Analysis**:
```python
# Line 108: Creates cache path with .npz extension
cache_path = self.cache_dir / f"{edf_path.stem}_windows.npz"  # ❌ NPZ!

# Lines 117-121: Exception handler creates NPZ files
if cache_path is not None:
    if labels_arr is not None:
        np.savez_compressed(cache_path, windows=windows_arr, labels=labels_arr)  # ❌ NPZ!
    else:
        np.savez_compressed(cache_path, windows=windows_arr)  # ❌ NPZ!

# Lines 126-130: Cache miss handler creates NPZ files
if cache_path is not None:
    if labels_arr is not None:
        np.savez_compressed(cache_path, windows=windows_arr, labels=labels_arr)  # ❌ NPZ!
    else:
        np.savez_compressed(cache_path, windows=windows_arr)  # ❌ NPZ!
```

**Why This Is Wrong**:
- populate_cache creates NPY files (`_data.npy`, `_labels.npy`) ✅
- _load_cache_for_worker loads NPY files ✅
- BUT on cache miss, datasets.py creates NPZ files ❌

**This is a CODE INCONSISTENCY BUG!**

---

### Impact Analysis

**Current Impact** (3 NPZ files):
- ✅ Training works (manifest system uses NPY files)
- ⚠️ Confusing logs ("[CACHE] 3 NPZ files" - what does this mean?)
- ⚠️ Wasted disk space (NPZ files duplicate NPY data)
- ⚠️ Cache validation logic misleading

**Future Impact** (if unfixed):
- 🔴 ANY cache miss will create NPZ files (not NPY!)
- 🔴 Cache directory becomes mixed format (NPZ + NPY)
- 🔴 Manifest system may get confused
- 🔴 Debugging becomes nightmare ("which files are real?")
- 🔴 Mmap benefits lost for new files (NPZ requires decompression)

**Why This Matters**:
- Cache misses can happen during training if:
  - Network issues cause file corruption
  - OOM kills worker mid-load
  - File system errors
  - We add new EDF files to dataset
- Each cache miss = new NPZ file contamination

---

### Solution: Two-Part Fix

#### Part 1: Clean Modal Cache (30 min)

**Goal**: Remove 3 stray NPZ files from Modal SSD volume

**Implementation**:
1. Create cleanup script: `deploy/modal/clean_stray_npz.py`
2. Script logic:
   ```python
   # Pseudocode:
   - Mount /results volume
   - Find ALL *.npz files in /results/cache/tusz_mmap/
   - For each NPZ file:
     - Check if corresponding NPY files exist (_data.npy, _labels.npy)
     - If NPY files exist: DELETE NPZ (it's a duplicate)
     - If NPY files DON'T exist: WARN (unexpected case!)
   - Report files deleted
   ```
3. Safety checks:
   - Dry-run mode (list files, don't delete)
   - Require confirmation flag
   - Only delete from /results/cache/tusz_mmap/ (not other paths)
   - Verify NPY files exist before deleting NPZ

**Testing**:
```bash
# Dry run (list files to delete)
modal run deploy/modal/clean_stray_npz.py --dry-run

# Actual deletion (requires confirmation)
modal run deploy/modal/clean_stray_npz.py --confirm
```

**Expected Result**:
```
[CLEANUP] Found 3 NPZ files to clean:
  - aaaaaaac_s001_t000_windows.npz (25.8 MiB) → NPY files exist ✓
  - aaaaaaac_s001_t001_windows.npz (18.6 MiB) → NPY files exist ✓
  - aaaaaaac_s002_t000_windows.npz (21.7 MiB) → NPY files exist ✓
[CLEANUP] Deleted 3 NPZ files (66.1 MiB freed)
[CLEANUP] ✅ Cache clean - only NPY files remain
```

---

#### Part 2: Fix datasets.py NPZ Creation (1 hour)

**Goal**: Stop datasets.py from creating NPZ files on cache miss

**Two Options**:

##### Option A (RECOMMENDED): Remove On-The-Fly Cache Creation
**Philosophy**: Datasets should ONLY read cache, never write it.

**Changes**:
```python
# datasets.py:108-130
# BEFORE (current code):
cache_path = self.cache_dir / f"{edf_path.stem}_windows.npz"
if cache_path.exists():
    # Load from cache...
else:
    # Process file and CREATE NPZ CACHE ❌
    windows_arr, labels_arr = self._process_file(edf_path, i)
    np.savez_compressed(cache_path, windows=windows_arr, labels=labels_arr)

# AFTER (Option A):
cache_stem = self.cache_dir / f"{edf_path.stem}_windows"
# Try to load from cache (NPY format)
windows_mmap, labels_mmap = self._load_cache_for_worker(cache_stem)
if windows_mmap is None:
    # Cache miss - FAIL FAST with helpful error
    raise FileNotFoundError(
        f"Cache not found for {edf_path.name}. "
        f"Run populate_cache first: "
        f"modal run deploy/modal/app.py --action populate-cache"
    )
```

**Pros**:
- ✅ Clear separation: populate_cache writes, datasets read
- ✅ No accidental cache creation with wrong format
- ✅ Forces proper cache pre-population workflow
- ✅ Simpler code (no write paths in datasets)

**Cons**:
- ❌ Requires populate_cache to run first (but we already do this!)
- ❌ Loses "just-in-time" cache building (was this ever used?)

##### Option B: Convert NPZ → NPY Creation
**Philosophy**: Keep on-the-fly caching, but fix format

**Changes**:
```python
# datasets.py:108-130
# BEFORE:
cache_path = self.cache_dir / f"{edf_path.stem}_windows.npz"
np.savez_compressed(cache_path, windows=windows_arr, labels=labels_arr)

# AFTER:
cache_stem = self.cache_dir / f"{edf_path.stem}_windows"
data_file = Path(str(cache_stem) + "_data.npy")
labels_file = Path(str(cache_stem) + "_labels.npy")

# Save as NPY (not NPZ!)
np.save(data_file, windows_arr)
if labels_arr is not None:
    np.save(labels_file, labels_arr)
```

**Pros**:
- ✅ Keeps on-the-fly caching (backward compatible)
- ✅ Fixes format inconsistency

**Cons**:
- ❌ Still mixing cache writes across two locations (populate_cache + datasets)
- ❌ More complex code (read + write paths)
- ❌ Risk of format drift (what if populate_cache changes?)

**RECOMMENDATION**: **Option A** (remove on-the-fly caching)
- Cleaner architecture
- We already require populate_cache before training
- No risk of future format inconsistencies

---

#### Part 3: Update Cache Validation (15 min)

**Location**: `deploy/modal/app.py:672`

**Current Code**:
```python
if cache_path.exists():
    npz_files = list(cache_path.glob("*.npz"))  # ❌ Looking for NPZ!
    # ... validation logic ...
```

**Problem**: After NPY conversion, should check for NPY files, not NPZ

**Fix**:
```python
if cache_path.exists():
    # Check for NPY files (mmap format)
    data_files = list(cache_path.glob("*_data.npy"))  # ✅ NPY format!
    labels_files = list(cache_path.glob("*_labels.npy"))

    # Validation: data and labels files should match
    if len(data_files) != len(labels_files):
        logger.warning(
            f"[CACHE] Mismatch: {len(data_files)} data files, "
            f"{len(labels_files)} labels files"
        )

    # Check for stray NPZ files (contamination)
    npz_files = list(cache_path.glob("*.npz"))
    if npz_files:
        logger.warning(
            f"[CACHE] Found {len(npz_files)} stray NPZ files - "
            f"run clean_stray_npz.py to remove"
        )
```

---

### Testing Plan (P0)

#### Stage 1: Clean Cache (30 min)
```bash
# 1. Verify 3 NPZ files exist
modal run deploy/modal/app.py --action check-cache | grep "NPZ"
# Expected: "3 NPZ files"

# 2. Run cleanup (dry-run first)
modal run deploy/modal/clean_stray_npz.py --dry-run
# Expected: Lists 3 NPZ files

# 3. Run cleanup (actual deletion)
modal run deploy/modal/clean_stray_npz.py --confirm
# Expected: "Deleted 3 NPZ files"

# 4. Verify NPZ files gone
modal run deploy/modal/app.py --action check-cache | grep "NPZ"
# Expected: "0 NPZ files" or no NPZ mention
```

#### Stage 2: Fix datasets.py (1 hour)
```bash
# 1. Update datasets.py (implement Option A)
# ... code changes ...

# 2. Unit tests
pytest tests/unit/data/test_datasets.py -xvs -k "test_cache"
# Expected: All pass

# 3. Test cache miss behavior
BGB_LIMIT_FILES=1 python -m src train configs/local/smoke.yaml
# Expected: FileNotFoundError with helpful message

# 4. Verify no NPZ creation
ls cache/tusz_mmap/train/*.npz 2>/dev/null
# Expected: No such file (or empty)
```

#### Stage 3: Update Cache Validation (15 min)
```bash
# 1. Update app.py:672
# ... code changes ...

# 2. Test validation logic
modal run deploy/modal/app.py --action check-cache
# Expected: Reports NPY file counts, no NPZ warnings

# 3. Verify smoke test logs
modal run deploy/modal/app.py --action train --config configs/modal/smoke.yaml
# Expected: "[CACHE] Using 4667 NPY data files" (or similar)
```

#### Stage 4: Full Smoke Test (10 min)
```bash
# Launch smoke test to verify everything works
modal run --detach deploy/modal/app.py --action train --config configs/modal/smoke.yaml

# Monitor logs
modal app logs <app-id> | grep -E "CACHE|mmap|NPZ"

# Expected:
# - "[CACHE] Using cache directory: /results/cache/tusz_mmap" ✅
# - No NPZ file mentions ✅
# - Fast startup (<1 min dataset initialization) ✅
# - Memory usage <2 GB per worker ✅
```

---

### Success Criteria (P0)

Before declaring P0 fixed:
- [ ] Zero NPZ files in `/results/cache/tusz_mmap/train/`
- [ ] Zero NPZ files in `/results/cache/tusz_mmap/dev/`
- [ ] datasets.py does NOT create NPZ files on cache miss
- [ ] Cache validation reports NPY files (not NPZ)
- [ ] Smoke test completes without errors
- [ ] Smoke test logs show correct cache usage
- [ ] Memory usage stays <2 GB per worker (mmap working)

---

## P2 Code Quality Issues (Non-Blocking)

### Issue 1: clean_cache() References Old Path

**Location**: `deploy/modal/app.py:459`

**Problem**:
```python
cache_paths = [
    Path("/results/cache/tusz"),      # ❌ Old path!
    Path("/results/cache/smoke"),     # ❌ Never used!
]
```

**Should Be**:
```python
cache_paths = [
    Path("/results/cache/tusz_mmap"),  # ✅ Current mmap cache
    Path("/results/cache/tusz"),       # 🔵 Legacy (optional cleanup)
]
```

**Fix** (5 min):
- Update to reference `/results/cache/tusz_mmap`
- Optionally keep old path for legacy cleanup
- Remove `/results/cache/smoke` (never used)

---

### Issue 2: Type Annotations Use `Any | None`

**Locations**:
- `src/brain_brr/utils/logging_config.py:147`
- `src/brain_brr/utils/training_logger.py:111`

**Problem**:
```python
# logging_config.py:147
self.console: Any | None = None  # ❌ Weak typing

# training_logger.py:111
self.console: Any | None = None  # ❌ Weak typing
```

**Fix** (15 min):
```python
# Both files:
from rich.console import Console

self.console: Console | None = None  # ✅ Strong typing
```

**Also check**: `train_step.py:181` (wandb_logger)
```python
# BEFORE:
wandb_logger: Any | None = None

# AFTER:
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run as WandBRun

wandb_logger: "WandBRun | None" = None
```

---

### Issue 3: NPZ References in Comments

**Locations**:
- `src/brain_brr/data/datasets.py` (lines 108, 262, 299, 405, 516, 692)
- `src/brain_brr/data/cache_utils.py` (lines 23, 41, 73, 88, 95, 134, 208, 210, 212, 215)

**Problem**: Comments reference `.npz` format despite NPY conversion

**Examples**:
```python
# datasets.py:108
cache_path = self.cache_dir / f"{edf_path.stem}_windows.npz"  # ❌ Variable name outdated

# datasets.py:262
# Old: aaaaaajy_s001_t000_windows.npz  # ❌ Comment outdated
```

**Fix** (30 min):
1. Rename variables: `cache_path` → `cache_stem` (since it's now base for NPY files)
2. Update comments: Replace NPZ references with NPY format
3. Remove "Old/New" comparison comments (no longer relevant)

**Example**:
```python
# BEFORE:
cache_path = self.cache_dir / f"{edf_path.stem}_windows.npz"

# AFTER:
cache_stem = self.cache_dir / f"{edf_path.stem}_windows"
# Format: file_windows_data.npy + file_windows_labels.npy (mmap)
```

---

### Issue 4: Duplicate `_load_cache_for_worker` Implementations

**Locations**:
- `EEGWindowDataset._load_cache_for_worker()` (lines 240-275)
- `BalancedSeizureDataset._load_cache_for_worker()` (lines 494-529)
- `ValidationDataset._load_cache_for_worker()` (lines 670-705)

**Problem**: Identical 40-line method duplicated 3 times (120 lines total, DRY violation)

**Fix** (1 hour):

1. **Extract to cache_utils.py**:
```python
# src/brain_brr/data/cache_utils.py

def load_cache_mmap(
    cache_stem: Path,
    mmap_handles: dict[Path, tuple[np.ndarray, np.ndarray | None]],
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load memory-mapped cache arrays (shared across all datasets).

    Args:
        cache_stem: Base path without extension (e.g., cache/train/file_windows)
        mmap_handles: Shared cache dict to store mmap handles

    Returns:
        Tuple of (windows_mmap, labels_mmap) where labels may be None

    Raises:
        FileNotFoundError: If cache files don't exist
    """
    if cache_stem not in mmap_handles:
        # Construct NPY file paths
        windows_file = Path(str(cache_stem) + "_data.npy")
        labels_file = Path(str(cache_stem) + "_labels.npy")

        if not windows_file.exists():
            raise FileNotFoundError(f"Cache not found: {windows_file}")

        # Open as memory-mapped (ZERO copies to RAM!)
        windows_mmap = np.load(windows_file, mmap_mode="r")
        labels_mmap = np.load(labels_file, mmap_mode="r") if labels_file.exists() else None

        mmap_handles[cache_stem] = (windows_mmap, labels_mmap)

    return mmap_handles[cache_stem]
```

2. **Update each dataset class**:
```python
# BEFORE (40 lines):
def _load_cache_for_worker(self, cache_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    # ... 40 lines of duplicate logic ...

# AFTER (3 lines):
def _load_cache_for_worker(self, cache_stem: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load memory-mapped cache (delegates to shared function)."""
    from src.brain_brr.data.cache_utils import load_cache_mmap
    return load_cache_mmap(cache_stem, self._mmap_handles)
```

**Testing**:
```bash
# Test new shared function
pytest tests/unit/data/test_cache_utils.py -xvs -k "test_load_cache_mmap"

# Test each dataset class
pytest tests/unit/data/test_datasets.py -xvs

# Full test suite
make test
```

---

## P2 Implementation Sequence

**NOT BLOCKING TRAINING** - Can be done in parallel or after training starts

### Phase 1: Quick Fixes (30 min)
1. Fix clean_cache() path (5 min)
2. Fix type annotations (15 min)
3. Test: `make q && make test`

### Phase 2: Code Cleanup (1.5 hours)
1. Update NPZ comments/variables (30 min)
2. Extract shared _load_cache_for_worker (1 hour)
3. Test after EACH change

### Phase 3: Validation (15 min)
```bash
make q          # Lint + format + mypy
make test       # Full test suite
make s          # Smoke test (3 files)
```

---

## Overall Priority & Timeline

### Critical Path (P0 - MUST DO BEFORE TRAINING)

**Estimated Time**: 1-2 hours

1. ✅ **Fix cache path hardcoding** (15 min) - DONE!
2. ⏱️ **Clean stray NPZ files** (30 min) - Create script + run
3. ⏱️ **Fix datasets.py NPZ creation** (30-60 min) - Implement Option A
4. ⏱️ **Update cache validation** (15 min) - Check for NPY not NPZ
5. ⏱️ **Test smoke test** (10 min) - Verify everything works

**Goal**: Launch smoke test with 100% clean cache (NPY only)

### Optional Path (P2 - CAN WAIT)

**Estimated Time**: 2-3 hours

- Can be done AFTER smoke test launches
- Can be done DURING full training (no impact)
- Can be split across multiple PRs

---

## Rollback Plan

### If P0 Fixes Fail:

```bash
# 1. Rollback datasets.py changes
git checkout src/brain_brr/data/datasets.py

# 2. Keep NPZ files (training will work with manifest)
# (No harm in keeping stray files, just confusing)

# 3. Test smoke test
modal run --detach deploy/modal/app.py --action train --config configs/modal/smoke.yaml
```

**Note**: Current state (3 NPZ files + 4667+1832 NPY files) is FUNCTIONAL, just messy.

### If P2 Fixes Break Tests:

```bash
# Rollback individual files
git checkout src/brain_brr/data/datasets.py
git checkout src/brain_brr/data/cache_utils.py
# etc.

# Verify tests pass
make test
```

---

## Documentation Updates Needed

### After P0 Fixes:
- [ ] Update P0_CACHE_PATH_BUG_INVESTIGATION.md (add NPZ contamination section)
- [ ] Update TECHNICAL_DEBT.md (mark P0 issues resolved)
- [ ] Update CLAUDE.md (clarify cache should be 100% NPY)

### After P2 Fixes:
- [ ] Update TECHNICAL_DEBT.md (mark all P2 issues resolved)
- [ ] Update REMAINING_DEBT_IMPLEMENTATION.md (mark completed)

---

## AI Agent Review Checklist

**Before implementing, confirm**:

### P0 (Critical):
- [ ] Option A (remove on-the-fly caching) is correct approach?
- [ ] clean_stray_npz.py script logic is safe?
- [ ] Testing sequence is sufficient to catch regressions?
- [ ] Any edge cases we're missing?

### P2 (Code Quality):
- [ ] Shared load_cache_mmap() signature is correct?
- [ ] All 3 dataset classes have identical logic?
- [ ] Type hints (WandB/Rich) are available for import?
- [ ] Variable rename (cache_path → cache_stem) won't break tests?

### Questions for Consensus:
1. Should we keep legacy `/results/cache/tusz` in clean_cache() for backward compat?
2. Better name than `cache_stem` for NPY base path?
3. Should clean_stray_npz.py be permanent or one-time script?
4. Any concerns about removing on-the-fly cache creation?

---

**Status**: 📋 Ready for AI agent consensus review
**Next Step**: Get consensus, then implement P0 fixes immediately
**Timeline**: P0 fixes (1-2 hrs) → Smoke test → Full training
**P2 Timeline**: Can be done anytime (not blocking)
