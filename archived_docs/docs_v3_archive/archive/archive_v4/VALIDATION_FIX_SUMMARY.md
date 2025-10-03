# Validation Dataset Startup Fix - Summary

**Date**: October 3, 2025
**Status**: ✅ FIXED AND VERIFIED
**Impact**: **5-10 minutes → 1.3 seconds** (99.6% faster!)

---

## Problem

Validation dataset was rebuilding its index on every training run, taking 5-10 minutes to scan 1832 NPZ files.

---

## Root Cause

Training loop used `EEGWindowDataset` for validation, which:
1. Tried to load `_dataset_index.json`
2. Compared file paths to see if index was valid
3. If paths didn't match → rebuilt index (5-10 minutes)
4. Index was frequently invalidated due to:
   - Path format changes (data_ext4 vs data)
   - Switching between smoke ↔ full mode
   - File list modifications

**Meanwhile**, `manifest.json` existed with all window data but was never used!

---

## Solution

Created `ValidationDataset` class that:
- Reads `manifest.json` directly (like `BalancedSeizureDataset`)
- Includes ALL windows from manifest (natural distribution ~8% seizures)
- No random sampling needed (validation uses sequential data)
- Falls back to `EEGWindowDataset` if manifest missing

---

## Changes Made

### 1. New Class: `ValidationDataset` (src/brain_brr/data/datasets.py:402-494)

```python
class ValidationDataset(Dataset):
    """Validation dataset using manifest without balanced sampling.

    Uses ALL windows from the manifest in natural distribution (~8% seizures).
    This is much faster than EEGWindowDataset (instant load vs 5-10 min scan).
    """
```

- Optional `allowed_cache_files` keeps smoke-mode limits (e.g., `BGB_LIMIT_FILES`) effective.

### 2. Updated Training Loop (src/brain_brr/train/loop.py:539-567)

```python
# Try ValidationDataset (instant load from manifest)
allowed_cache_files = {
    f"{val_file.stem}_windows.npz" for val_file in val_files
} if val_files else None

val_dataset: ValidationDataset | EEGWindowDataset
if val_manifest_path.exists():
    try:
        val_dataset = ValidationDataset(
            val_cache_dir,
            allowed_cache_files=allowed_cache_files,
        )
        logger.info(
            f"[DATASET] ValidationDataset: {len(val_dataset)} windows from manifest (instant load)"
        )
    except Exception as e:
        # Fallback to EEGWindowDataset
        val_dataset = EEGWindowDataset(...)
else:
    val_dataset = EEGWindowDataset(...)
```

### 3. Updated Exports (src/brain_brr/data/__init__.py)

Added `ValidationDataset` to `__all__` exports.

---

## Performance Results

### Before Fix

```
[13:21:48] INFO [BalancedSeizureDataset] Created with 61616 windows
                                          ↑ INSTANT (reads manifest)

[13:21:48] INFO [DATA] Building dataset index for 1832 files...
[13:21:51] INFO [DATA] Processing file 11/1832: aaaaaajy_s004_t000.edf
[13:24:55] INFO [DATA] Processing file 351/1832: aaaaahie_s023_t002.edf
                                          ↑ 5-10 MINUTES (scanning NPZ)
```

### After Fix

```
[14:09:51.719] INFO [BalancedSeizureDataset] Created with 61616 windows
[14:09:51.727] INFO [DATASET] BalancedSeizureDataset: 61616 windows from manifest

[14:09:53.002] INFO [ValidationDataset] Created with 148224 windows
[14:09:53.011] INFO [DATASET] ValidationDataset: 148224 windows from manifest (instant load)
                                          ↑ 1.3 SECONDS TOTAL!
```

**Improvement**: 5-10 minutes → 1.3 seconds = **99.6% faster!**

---

## Validation

### Dataset Statistics (Verified Correct)

**Training Dataset (BalancedSeizureDataset)**:
- 61,616 windows
- ~34% seizures (balanced sampling)
- Loads in <0.01 seconds from manifest

**Validation Dataset (ValidationDataset)**:
- 148,224 windows (full manifest) or subset when filters applied
- Breakdown:
  - 3,536 full seizure windows
  - 7,944 partial seizure windows
  - 136,744 no-seizure windows
- Seizure ratio: ~7.7% (natural distribution ✅)
- Loads in ~1.3 seconds from manifest

### Files Excluded (Zero-Window Files)

- Train manifest: 4438/4667 files (229 excluded = zero windows)
- Dev manifest: 1608/1832 files (224 excluded = zero windows)
- **This is correct** - empty files can't contribute windows!

---

## Impact

### Development Velocity

**Before**: 5-10 min wait per training run  
**After**: <2 sec wait per training run  
**Savings**: ~5-10 min per run × 3 runs/dev cycle = **15-30 min saved per cycle**

### Modal GPU Costs

**Before**: $0.60/hour × (10 min / 60) = **~$0.10-$0.20 wasted per run**  
**After**: Negligible (<2 sec)  
**Savings**: ~$0.10-$0.20 per training run

### Developer Experience

- No more waiting for index rebuild
- Consistent startup time (no surprise delays)
- Faster iteration cycles
- Less frustration 🎉

---

## Testing

✅ Quality checks passed (lint, format, mypy)  
✅ Training dataset uses BalancedSeizureDataset (manifest)  
✅ Validation dataset uses ValidationDataset (manifest, limit-aware)  
✅ Both load in <2 seconds total  
✅ Natural distribution preserved (7.7% seizures in validation)  
✅ Graceful fallback to EEGWindowDataset if manifest missing

---

## Next Steps

1. ✅ **DONE**: Fix implemented and verified
2. Run full training to ensure no regressions
3. Update Modal deployment to use same fix
4. Update CLAUDE.md with new ValidationDataset info

---

## Files Modified

1. `src/brain_brr/data/datasets.py` - Added `ValidationDataset` class
2. `src/brain_brr/data/__init__.py` - Exported `ValidationDataset`
3. `src/brain_brr/train/loop.py` - Updated validation dataset instantiation

---

**Status**: ✅ COMPLETE  
**Quality**: All checks passed  
**Performance**: 99.6% improvement  
**Risk**: Low (graceful fallback to old behavior if manifest missing)
