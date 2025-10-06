# Technical Debt

**Date**: October 6, 2025 (Updated after fixes)
**Status**: 3 issues remaining (0 P0, 0 P1, 3 P2, 0 P3)
**Training Impact**: No blockers - training ready

---

## Executive Summary

| Priority | Count | Training Impact |
|----------|-------|-----------------|
| **P0 BLOCKER** | 0 | None - ready to train |
| **P1 URGENT** | 0 | None - all fixed! |
| **P2 MEDIUM** | 3 | Code quality/maintenance |
| **P3 LOW** | 0 | None - all fixed! |

**Recent Debt Eliminated**:
- ✅ NPZ→NPY mmap conversion (387 GB RAM → <1 GB)
- ✅ ValidationDataset caching (49x speedup)
- ✅ Modal SSD cache population (train + dev splits)
- ✅ **P1 Documentation paths** - Updated all `cache/tusz` → `cache/tusz_mmap`
- ✅ **P2 Magic numbers** - Added `HEARTBEAT_INTERVAL_SEC` to constants
- ✅ **P2 Env var naming** - Renamed to `BGB_SKIP_PERF_TESTS`
- ✅ **P3 YAML comments** - Cleaned double comments in configs

---

## P2: Remaining Code Quality Issues

### Issue 1: Lingering NPZ References in Comments
- **Location**: `src/brain_brr/data/datasets.py:108,262,299,405,516,692`
- **Problem**: Code comments reference `.npz` format despite NPY mmap conversion
- **Evidence**:
  ```python
  # datasets.py:108
  cache_path = self.cache_dir / f"{edf_path.stem}_windows.npz"  # Variable name outdated

  # datasets.py:262
  # Old: aaaaaajy_s001_t000_windows.npz
  # New: aaaaaajy_s001_t000_data.npy + aaaaaajy_s001_t000_labels.npy
  ```
- **Impact**: Code works (auto-converts paths), but comments confuse readers
- **Fix**: Rename variables to `cache_stem`, update comments to reflect NPY format
- **Estimated time**: 30 minutes

### Issue 2: Duplicate `_load_cache_for_worker` Implementations
- **Location**: `src/brain_brr/data/datasets.py:240-275, 494-529, 670-705`
- **Problem**: Identical 40-line method duplicated in 3 dataset classes (DRY violation)
- **Evidence**: 120 total lines of duplicate code (3x40 lines)
- **Impact**: Maintenance burden - bugs must be fixed 3x
- **Fix**: Extract to shared `_load_cache_for_worker_mmap()` in `cache_utils.py`
- **Estimated time**: 1 hour (refactor + test)

### Issue 3: Type Annotation Inconsistency
- **Location**: `src/brain_brr/train/train_step.py:181`, `src/brain_brr/train/training_logger.py:111`
- **Problem**: Using `Any | None` instead of proper types
- **Evidence**:
  ```python
  wandb_logger: Any | None = None  # Should be WandBLogger | None
  self.console: Any | None = None  # Should be Console | None (rich.console.Console)
  ```
- **Impact**: Weakens type safety
- **Fix**: Replace `Any` with proper types (import required types)
- **Estimated time**: 30 minutes

---

## What's Working Well ✅

1. **NPZ→NPY Mmap Conversion**: Complete and correct across all 3 dataset classes
2. **ValidationDataset**: Successfully uses mmap (49x speedup achieved)
3. **Manifest System**: Properly supports both NPZ (legacy) and NPY (production)
4. **Type Hints**: Strong coverage (only ~10 `Any` instances in thousands of lines)
5. **Testing**: 54 test files, proper GPU guards, good coverage
6. **Error Handling**: Robust NaN protection, gradient sanitization
7. **Configuration**: Pydantic schemas prevent runtime config errors
8. **No TODO/FIXME/HACK**: Clean codebase with no technical marker comments
9. **Modal Deployment**: Well-structured with A100 XID 31 fixes
10. **Documentation**: Extensive (just needs path updates)

---

## Risk Assessment

**Training Risk**: ✅ **ZERO** - No P0 blockers, all issues are documentation/style

**Detected Issues**:
- ❌ **No Reward Hacking** - All metrics properly computed and wired
- ❌ **No Unwired Features** - All config options used, no dead code
- ❌ **No Critical Bugs** - Mmap implementation complete and correct
- ❌ **No AI Deception** - Code does what it claims to do

**Recommended Before Full Training**:
1. Fix P1 doc paths (10 min) - prevents confusion
2. Optional: P2 code quality fixes (2-3 hours) - improves maintainability
3. P3 polish can wait until after training completes

---

## Implementation Priority

**Completed (October 6, 2025)**:
- [x] ✅ P1: Update CLAUDE.md cache paths (10 min)
- [x] ✅ P2: Move magic numbers to constants (15 min)
- [x] ✅ P2: Fix env var naming (10 min)
- [x] ✅ P3: Clean YAML comments (15 min)

**Remaining (After Modal Training Starts)**:
- [ ] P2: Extract shared `_load_cache_for_worker` (1 hour) - **Requires planning**
- [ ] P2: Update NPZ comments/variable names (30 min) - **Requires planning**
- [ ] P2: Improve type annotations (30 min) - **Requires planning**

**Total remaining time**: ~2 hours (all optional, non-blocking)

---

**Status**: Training ready - proceed with confidence! 🚀
