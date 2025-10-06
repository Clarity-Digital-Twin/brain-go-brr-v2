# Remaining Technical Debt - Implementation Plan

**⚠️ DEPRECATED**: This document has been superseded by `COMPREHENSIVE_FIX_PLAN.md`

**Date**: October 6, 2025
**Status**: See `COMPREHENSIVE_FIX_PLAN.md` for current fix plan
**Scope**: Originally 3 P2 issues, NOW 1 P0 + 3 P2 issues

**New P0 Blocker Discovered**: NPZ cache contamination (3 stray files + datasets.py creates NPZ on cache miss)

**👉 GO TO**: `COMPREHENSIVE_FIX_PLAN.md` for complete analysis and implementation plan

---

## Historical Content (For Reference Only)

---

## Executive Summary

**What's Fixed** (October 6, 2025):
- ✅ P1: Documentation cache paths (`cache/tusz` → `cache/tusz_mmap`)
- ✅ P2: Magic numbers → constants (`HEARTBEAT_INTERVAL_SEC`)
- ✅ P2: Environment variable naming (`BGB_SKIP_PERF_TESTS`)
- ✅ P3: YAML double comments cleanup

**What Remains** (3 issues, all P2):
1. Lingering NPZ references in comments/variable names (30 min)
2. Duplicate `_load_cache_for_worker` implementations (1 hour)
3. Type annotation improvements (30 min)

**Training Impact**: ✅ ZERO - All issues are code quality/maintainability only

---

## P2 Issue #1: NPZ References in Comments

### Problem Statement
Code comments and variable names reference `.npz` format despite complete NPY mmap conversion. Code works correctly (auto-converts paths), but this creates confusion for developers.

### Locations
- `src/brain_brr/data/datasets.py:108,262,299,405,516,692`
- `src/brain_brr/data/cache_utils.py:41,73,88,95,134,208,210,212,215`

### Current State (Example)
```python
# datasets.py:108
cache_path = self.cache_dir / f"{edf_path.stem}_windows.npz"  # ❌ Variable name outdated

# datasets.py:262
# Old: aaaaaajy_s001_t000_windows.npz  # ❌ Comment outdated
# New: aaaaaajy_s001_t000_data.npy + aaaaaajy_s001_t000_labels.npy
```

### Implementation Plan

#### Step 1: Update Variable Names (15 min)
**Goal**: Rename `cache_path` variables to reflect they're now NPY stems, not NPZ paths

**Changes**:
1. `datasets.py:108` - Rename `cache_path` → `cache_stem` (since it becomes base for `_data.npy` / `_labels.npy`)
2. Update all references to this variable within methods
3. Update docstrings to reflect "NPY stem" instead of "NPZ file"

**Example**:
```python
# BEFORE:
cache_path = self.cache_dir / f"{edf_path.stem}_windows.npz"  # Returns NPZ path
windows_mmap, labels_mmap = self._get_windows_mmap(cache_path)

# AFTER:
cache_stem = self.cache_dir / f"{edf_path.stem}_windows"  # Returns NPY base stem
windows_mmap, labels_mmap = self._get_windows_mmap(cache_stem)
```

#### Step 2: Update Comments (15 min)
**Goal**: Replace NPZ references with NPY format descriptions

**Changes**:
1. `datasets.py:262` - Update comment to show NPY format:
   ```python
   # Format: aaaaaajy_s001_t000_data.npy + aaaaaajy_s001_t000_labels.npy (mmap)
   ```
2. `cache_utils.py:41,73,88,95` - Update comments referencing NPZ to NPY
3. Remove outdated "Old/New" comparison comments (no longer relevant post-conversion)

#### Testing
```bash
# Verify tests still pass
make test

# Smoke test
make s

# Type check
mypy src/
```

---

## P2 Issue #2: Duplicate `_load_cache_for_worker` Methods

### Problem Statement
Identical 40-line `_load_cache_for_worker()` method duplicated across 3 dataset classes (120 lines total). Violates DRY principle - bugs must be fixed 3 times.

### Locations
- `EEGWindowDataset._load_cache_for_worker()` (lines 240-275)
- `BalancedSeizureDataset._load_cache_for_worker()` (lines 494-529)
- `ValidationDataset._load_cache_for_worker()` (lines 670-705)

### Current State
All three methods have IDENTICAL logic:
1. Check if cache_stem in `self._mmap_handles`
2. If not, construct `data.npy` and `labels.npy` paths
3. Open with `np.load(..., mmap_mode='r')`
4. Store in `self._mmap_handles`
5. Return tuple `(windows_mmap, labels_mmap)`

### Implementation Plan

#### Step 1: Extract Shared Function (30 min)
**Goal**: Move logic to `cache_utils.py` as shared function

**New function signature**:
```python
# src/brain_brr/data/cache_utils.py

def load_cache_mmap(
    cache_stem: Path,
    mmap_handles: dict[Path, tuple[np.ndarray, np.ndarray | None]],
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load memory-mapped cache arrays (shared across all datasets).

    Args:
        cache_stem: Base path without extension (e.g., cache/train/file_windows)
        mmap_handles: Shared cache dict to store mmap handles
        logger: Logger for diagnostics

    Returns:
        Tuple of (windows_mmap, labels_mmap) where labels may be None

    Raises:
        FileNotFoundError: If cache files don't exist

    Notes:
        - Mmap handles are stored in mmap_handles dict to prevent re-opening
        - OS manages memory automatically via page cache
        - Multiple workers share same physical pages (zero-copy)
    """
    if cache_stem not in mmap_handles:
        # Construct file paths
        windows_file = Path(str(cache_stem) + "_data.npy")
        labels_file = Path(str(cache_stem) + "_labels.npy")

        if not windows_file.exists():
            raise FileNotFoundError(f"Cache data file not found: {windows_file}")

        # Open as memory-mapped (ZERO copies to RAM!)
        windows_mmap = np.load(windows_file, mmap_mode="r")
        labels_mmap = np.load(labels_file, mmap_mode="r") if labels_file.exists() else None

        mmap_handles[cache_stem] = (windows_mmap, labels_mmap)

        logger.debug(
            f"[MMAP] Loaded {windows_file.name}: "
            f"{windows_mmap.shape} windows, "
            f"{labels_mmap.shape if labels_mmap is not None else 'no'} labels"
        )

    return mmap_handles[cache_stem]
```

#### Step 2: Update Dataset Classes (30 min)
**Goal**: Replace each class's method with call to shared function

**Changes to each dataset**:
```python
# BEFORE (40 lines per class):
def _load_cache_for_worker(self, cache_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    # 40 lines of duplicate logic...
    pass

# AFTER (3 lines per class):
def _load_cache_for_worker(self, cache_stem: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load memory-mapped cache (delegates to shared cache_utils function)."""
    from src.brain_brr.data.cache_utils import load_cache_mmap
    return load_cache_mmap(cache_stem, self._mmap_handles, logger)
```

**Classes to update**:
1. `EEGWindowDataset` (line 240)
2. `BalancedSeizureDataset` (line 494)
3. `ValidationDataset` (line 670)

#### Testing
```bash
# Unit tests for cache_utils.load_cache_mmap
pytest tests/unit/data/test_cache_utils.py -xvs -k "test_load_cache_mmap"

# Dataset tests
pytest tests/unit/data/test_datasets.py -xvs

# Integration smoke test
make s

# Full test suite
make test
```

---

## P2 Issue #3: Type Annotation Improvements

### Problem Statement
Using `Any | None` instead of proper types weakens type safety and makes mypy less effective.

### Locations
- `src/brain_brr/train/train_step.py:181` - `wandb_logger: Any | None`
- `src/brain_brr/train/training_logger.py:111` - `self.console: Any | None`

### Current State
```python
# train_step.py:181
wandb_logger: Any | None = None  # ❌ Weak typing

# training_logger.py:111
self.console: Any | None = None  # ❌ Weak typing
```

### Implementation Plan

#### Step 1: Fix train_step.py (15 min)
**Goal**: Replace `Any` with proper WandB type

**Changes**:
```python
# BEFORE:
from typing import Any
wandb_logger: Any | None = None

# AFTER:
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run as WandBRun

wandb_logger: "WandBRun | None" = None  # String annotation for forward ref
```

**Alternative** (if wandb types are available at runtime):
```python
from wandb.sdk.wandb_run import Run as WandBRun
wandb_logger: WandBRun | None = None
```

#### Step 2: Fix training_logger.py (15 min)
**Goal**: Replace `Any` with proper Console type

**Changes**:
```python
# BEFORE:
from typing import Any
self.console: Any | None = None

# AFTER:
from rich.console import Console
self.console: Console | None = None
```

**Note**: `rich.console.Console` is a concrete class, so no forward reference needed.

#### Testing
```bash
# Type check
mypy src/brain_brr/train/train_step.py
mypy src/brain_brr/train/training_logger.py

# Verify no runtime errors
make test
```

---

## Implementation Sequence

### Phase 1: Preparation (5 min)
1. Create feature branch: `git checkout -b fix/remaining-debt-p2`
2. Verify tests pass on main: `make test`
3. Review this document with AI agent for consensus

### Phase 2: Execute Fixes (2 hours)
**Order of operations** (minimize test breakage):

1. **Issue #1 (30 min)**: NPZ reference cleanup
   - Low risk, mostly comments
   - Run tests after: `make test`

2. **Issue #2 (1 hour)**: Extract shared function
   - Higher risk (code refactor)
   - Create new function in `cache_utils.py`
   - Update each dataset class one at a time
   - Run tests after EACH class update: `pytest tests/unit/data/test_datasets.py -xvs -k TestClassName`
   - Full test suite after all classes updated: `make test`

3. **Issue #3 (30 min)**: Type annotations
   - Low risk (just type hints)
   - Run mypy after: `mypy src/`
   - Run tests after: `make test`

### Phase 3: Validation (15 min)
```bash
# Quality check (lint + format + mypy)
make q

# Full test suite
make test

# Smoke test (3 files, ~5 min)
make s

# Git status
git status
git diff
```

### Phase 4: Commit & Document (10 min)
```bash
# Stage changes
git add src/brain_brr/data/datasets.py
git add src/brain_brr/data/cache_utils.py
git add src/brain_brr/train/train_step.py
git add src/brain_brr/train/training_logger.py

# Commit
git commit -m "refactor: eliminate remaining P2 technical debt

- Update NPZ references → NPY in comments/variable names
- Extract shared load_cache_mmap() to cache_utils.py (DRY)
- Improve type annotations (Any → proper types)

Fixes 3 P2 code quality issues from TECHNICAL_DEBT.md"

# Update TECHNICAL_DEBT.md to mark all issues complete
# Create PR if needed
```

---

## Risk Assessment

### Low Risk Changes
- ✅ Issue #1 (NPZ comments): Comments-only, no logic changes
- ✅ Issue #3 (Type annotations): Type hints don't affect runtime

### Medium Risk Changes
- ⚠️ Issue #2 (Extract function): Code refactor, but well-tested

**Mitigation**:
- Test after EACH class update in Issue #2
- Full test suite before committing
- Smoke test validates end-to-end pipeline
- Can rollback easily if issues found

---

## Success Criteria

### Before Declaring DONE:
- [ ] All tests pass: `make test` (100% pass rate)
- [ ] Type checks pass: `mypy src/` (zero errors)
- [ ] Quality check passes: `make q` (lint + format + types)
- [ ] Smoke test passes: `make s` (3 files, <10 min)
- [ ] No regressions in existing functionality
- [ ] TECHNICAL_DEBT.md updated (all P2 issues marked complete)

### Code Quality Metrics:
- [ ] Duplicate code reduced: 120 lines → 3 lines (Issue #2)
- [ ] NPZ references eliminated: ~15 occurrences fixed (Issue #1)
- [ ] Type safety improved: 2 `Any` types → proper types (Issue #3)

---

## Rollback Plan

If any issues encountered:

```bash
# Discard changes
git checkout -- .

# Or reset to HEAD
git reset --hard HEAD

# Verify tests pass again
make test
```

**Partial completion is OK**: Each issue is independent and can be completed separately.

---

## AI Agent Review Checklist

Before implementing, confirm:

- [ ] **Issue #1**: Variable renames won't break existing code?
- [ ] **Issue #2**: Shared function signature is correct?
- [ ] **Issue #2**: All 3 dataset classes use identical logic?
- [ ] **Issue #3**: WandB/Rich types are available for import?
- [ ] **Testing**: Test sequence is sufficient?
- [ ] **Risk**: Any concerns about the refactor?

**Questions for AI Agent**:
1. Should `load_cache_mmap()` be a standalone function or a mixin class method?
2. Better name for `cache_stem` variable? (e.g., `cache_base_path`?)
3. Any edge cases we're missing in the extraction?

---

**Status**: Ready for AI agent consensus review and implementation
**Est. Total Time**: 2 hours 15 minutes (including testing)
**Training Impact**: ZERO (all changes are code quality only)
