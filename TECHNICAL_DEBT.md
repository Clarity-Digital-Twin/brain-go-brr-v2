# Technical Debt

**Date**: October 6, 2025 (Updated after NPZ discovery)
**Status**: 4 issues remaining (1 P0, 0 P1, 3 P2, 0 P3)
**Training Impact**: 1 P0 blocker - NPZ contamination must be fixed

---

## Executive Summary

| Priority | Count | Training Impact |
|----------|-------|-----------------|
| **P0 BLOCKER** | 1 | NPZ cache contamination |
| **P1 URGENT** | 0 | None - all fixed! |
| **P2 MEDIUM** | 3 | Code quality/maintenance |
| **P3 LOW** | 0 | None - all fixed! |

**Recent Debt Eliminated**:
- ✅ NPZ→NPY mmap conversion (387 GB RAM → <1 GB)
- ✅ ValidationDataset caching (49x speedup)
- ✅ Modal SSD cache population (train + dev splits)
- ✅ **P0 Cache path hardcoding** - Fixed app.py lines 658, 811, 821
- ✅ **P1 Documentation paths** - Updated all `cache/tusz` → `cache/tusz_mmap`
- ✅ **P2 Magic numbers** - Added `HEARTBEAT_INTERVAL_SEC` to constants
- ✅ **P2 Env var naming** - Renamed to `BGB_SKIP_PERF_TESTS`
- ✅ **P3 YAML comments** - Cleaned double comments in configs

**New Issue Discovered (October 6, 2025)**:
- 🔴 **P0 NPZ contamination** - 3 stray NPZ files in Modal cache from failed smoke test
- 🔴 **P0 datasets.py bug** - Creates NPZ files on cache miss (should create NPY!)
- See `COMPREHENSIVE_FIX_PLAN.md` for complete analysis and fix plan

---

## P0: NPZ Cache Contamination (BLOCKS TRAINING)

### Critical Issue: Mixed NPZ/NPY Cache Files

**Problem**: Modal cache at `/results/cache/tusz_mmap/` contains BOTH:
- ✅ 4667+1832 NPY files (correct, from populate_cache)
- ❌ 3 NPZ files (wrong, from first failed smoke test)

**Root Cause**: `datasets.py:117-130` creates NPZ files on cache miss, not NPY files

**Impact**:
- Current: Confusing logs, wasted disk space
- Future: ANY cache miss will create more NPZ files (format drift!)

**Fix Required**:
1. Clean 3 stray NPZ files from Modal cache (30 min)
2. Fix datasets.py to either:
   - Option A (recommended): Remove on-the-fly cache creation entirely
   - Option B: Convert NPZ creation to NPY format
3. Update cache validation to check for NPY files (15 min)

**See**: `COMPREHENSIVE_FIX_PLAN.md` for complete analysis, implementation plan, and testing strategy

**Timeline**: 1-2 hours to fix, smoke test to validate

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
- **Location**:
  - `src/brain_brr/train/train_step.py:181` (wandb_logger)
  - `src/brain_brr/utils/training_logger.py:111` (console)
  - `src/brain_brr/utils/logging_config.py:147` (console)
- **Problem**: Using `Any | None` instead of proper types
- **Evidence**:
  ```python
  wandb_logger: Any | None = None  # Should be WandBRun | None
  self.console: Any | None = None  # Should be Console | None (rich.console.Console)
  ```
- **Impact**: Weakens type safety
- **Fix**: Replace `Any` with proper types (import required types)
- **Estimated time**: 15 minutes

---

## What's Working Well ✅

1. **NPZ→NPY Mmap Conversion**: Mostly complete (populate_cache ✅, _load_cache_for_worker ✅, write paths ❌ still create NPZ)
2. **ValidationDataset**: Successfully uses mmap (49x speedup achieved)
3. **Manifest System**: Properly supports both NPZ (legacy) and NPY (production)
4. **Type Hints**: Strong coverage (~10 `Any` instances need fixing)
5. **Testing**: 54 test files, proper GPU guards, good coverage
6. **Error Handling**: Robust NaN protection, gradient sanitization
7. **Configuration**: Pydantic schemas prevent runtime config errors
8. **No TODO/FIXME/HACK**: Clean codebase with no technical marker comments
9. **Modal Deployment**: Well-structured with A100 XID 31 fixes
10. **Documentation**: Extensive and accurate

---

## Risk Assessment

**Training Risk**: 🔴 **P0 BLOCKER** - NPZ contamination and code bugs must be fixed

**Detected Issues**:
- ✅ **Reward Hacking** - None detected, all metrics properly computed
- ✅ **Unwired Features** - None detected, all config options used
- 🔴 **Critical Bug** - datasets.py creates NPZ files on cache miss (wrong format!)
- 🔴 **Format Drift** - Mixed NPZ/NPY files in cache (3 stray NPZ files)

**Required Before Training**:
1. ✅ Fix P0 cache path hardcoding (COMPLETED)
2. 🔴 Clean 3 stray NPZ files from Modal cache (REQUIRED)
3. 🔴 Fix datasets.py NPZ creation bug (REQUIRED)
4. 🔴 Update cache validation logic (REQUIRED)
5. 🟡 Fix P2 code quality issues (2-3 hours, improves maintainability)

---

## Implementation Priority

**Completed (October 6, 2025)**:
- [x] ✅ P0: Fix cache path hardcoding (app.py lines 658, 811, 821)
- [x] ✅ P1: Update CLAUDE.md cache paths
- [x] ✅ P2: Move magic numbers to constants
- [x] ✅ P2: Fix env var naming
- [x] ✅ P3: Clean YAML comments

**P0 Remaining (BLOCKS TRAINING)**:
- [ ] 🔴 Clean 3 stray NPZ files from Modal cache (10 min)
- [ ] 🔴 Fix datasets.py NPZ creation bug (30 min)
- [ ] 🔴 Update cache validation logic (15 min)

**P2 Remaining (After P0)**:
- [ ] 🟡 Fix clean_cache() old path (5 min)
- [ ] 🟡 Fix type annotations - 3 files (15 min)
- [ ] 🟡 Update NPZ comments/variable names (30 min)
- [ ] 🟡 Extract shared `_load_cache_for_worker` (1 hour)

**Total remaining time**: P0 ~1 hour (BLOCKING), P2 ~2 hours (non-blocking)

---

## Next Steps

**Before Training** (P0 - Required):
1. Review `COMPREHENSIVE_FIX_PLAN.md` with AI agent for consensus
2. Implement NPZ cleanup script (30 min)
3. Fix datasets.py NPZ creation bug (30-60 min)
4. Update cache validation logic (15 min)
5. Run smoke test to validate fixes (10 min)

**After Training Starts** (P2 - Optional):
- P2 fixes can be done in parallel with training
- Total time: ~2-3 hours
- No impact on training performance

---

**Status**: 🔴 BLOCKED - Must fix P0 NPZ contamination before training
**See**: `COMPREHENSIVE_FIX_PLAN.md` for detailed implementation plan
