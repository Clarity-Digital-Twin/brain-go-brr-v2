# BUGS FIXED - 2025-09-24

## Summary of Critical Fixes Applied

This document records all bugs that were identified and fixed in the critical code review session on 2025-09-24.

## P0 Blockers (FIXED)

### ✅ Train/val split patient leakage - FIXED
- **Problem**: Used alphabetical file-based splitting causing patient `aaaaagxr` to appear in BOTH train and validation sets
- **Impact**: ALL previous training runs were scientifically invalid with artificially inflated validation metrics
- **Fix Applied**:
  - Created `src/brain_brr/data/tusz_splits.py` module for proper TUSZ split handling
  - Now using official TUSZ train/dev/eval splits with patient-level disjointness
  - Added `split_policy: official_tusz` to all configs
  - Added runtime validation that fails fast if any patient appears in multiple splits
  - Deleted all contaminated cache directories

### ✅ FA curve uses wrong thresholds - FIXED
- **Problem**: `sensitivity_at_fa_rates()` passed threshold parameter that was ignored by `batch_probs_to_events()`
- **Impact**: FA curve values were inconsistent; downstream analyses relying on fa_curve were wrong
- **Fix Applied**:
  - Updated `sensitivity_at_fa_rates()` to clone `post_cfg` and set `hysteresis.tau_on/off`
  - Now passes the cloned config instead of the ignored threshold parameter

## P2 Issues (FIXED)

### ✅ TCN channels config ignored - FIXED
- **Problem**: `ModelConfig.tcn.channels` was accepted but never used; channels hardcoded to `[64, 128, 256, 512]`
- **Impact**: Config ≠ behavior; wasted tuning time and reviewer confusion
- **Fix Applied**:
  - Removed `channels` field from `TCNConfig` schema
  - Removed `channels` from all YAML config files
  - Added comment noting channels are hardcoded

### ✅ TensorBoard not in base deps - FIXED
- **Problem**: TensorBoard imported unconditionally but not in base dependencies
- **Impact**: Runtime ImportError on fresh environments
- **Fix Applied**:
  - Made TensorBoard import optional with try/except
  - Added `HAS_TENSORBOARD` flag
  - Shows helpful message if TensorBoard not installed

### ✅ Manifest assumes unlabeled NPZs are no_seizure - FIXED
- **Problem**: `scan_existing_cache()` treated NPZ files without labels as all background
- **Impact**: Could flood manifest with false negatives if cache corrupted
- **Fix Applied**:
  - Now warns loudly about NPZ files without labels
  - Excludes them entirely from balanced sampling
  - Prevents flooding manifest with potentially incorrect data

## Verification

All fixes have been verified with a smoke test run showing:
```
[SPLIT STATS] OFFICIAL TUSZ SPLITS:
  Train: 579 patients, 4667 files
  Val:   53 patients, 1832 files
  ✅ PATIENT DISJOINTNESS VERIFIED - No leakage!
```

## Next Steps

1. Rebuild cache with correct splits:
   ```bash
   python -m src build-cache --data-dir data_ext4/tusz/edf --cache-dir cache/tusz
   ```

2. Restart training with proper patient-disjoint splits:
   ```bash
   make train-local  # or modal run for cloud
   ```

3. After training completes, evaluate ONCE on eval/ for final metrics

## Key Learnings

- **ALWAYS use patient-level splits** for medical data
- **NEVER split by files** when files belong to patients
- **ALWAYS validate disjointness** before training
- **Official splits exist for a reason** - use them!

---

**CRITICAL NOTE**: All previous training results are invalid due to patient leakage. Start fresh with the proper splits implemented here.