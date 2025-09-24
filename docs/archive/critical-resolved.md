# Critical Issues - Resolved

This document captures critical bugs and pain points from project history that have been resolved but should be remembered to prevent regression.

## P0 Critical Issues (Resolved)

### 1. Zero-Seizure Cache Catastrophe (254GB Wasted)
**Date:** 2025-09-21
**Impact:** 254GB useless cache, $60+ Modal credits burned, model learned nothing
**Root Cause:** CSV parser was reading wrong columns - expected simple format but TUSZ uses CSV_BI format with channel-specific annotations
**Resolution:**
- Fixed CSV_BI parser (`parse_tusz_csv`) and mask builder (`events_to_binary_mask`) in `src/brain_brr/data/io.py`
- Added manifest scan guards to CLI (`build-cache`, `scan-cache`) to stop when no seizures
- Implemented `BalancedSeizureDataset` with manifest validation
**Lesson:** Always validate cache has seizures before training!

### 1b. Balanced Manifest Order Bug (empty manifest built first)
**Impact:** BalancedSeizureDataset created with 0 windows; fell back to random sampling
**Root Cause:** Manifest was generated before any NPZ cache files were present
**Resolution:**
- Training loop now validates existing manifest and deletes/rebuilds if empty/stale
- Only builds manifest from a populated cache directory
- Added env toggle `BGB_FORCE_MANIFEST_REBUILD=1` for manual rebuilds on startup
**Lesson:** Generate manifests after cache population; validate at startup

### 2. TUSZ Channel Naming Mismatch
**Impact:** Training crashes with "Missing required channels"
**Root Cause:** TUSZ files have `'EEG FP1-LE'` but code expects `'Fp1'`
**Resolution:**
- Robust channel cleaning (`clean_tusz_name`) in `src/brain_brr/data/io.py`
- Channel synonym mapping in `src/brain_brr/constants.py`
- Ordered channel picking via `pick_and_order` in `src/brain_brr/utils/pick_utils.py`

### 3. Myoclonic (mysz) Seizure Type Missing
**Impact:** 44 mysz events mislabeled as background; missed during training
**Root Cause:** Seizure label set omitted `mysz` and mistakenly included non-existent `spkz`
**Resolution:**
- Updated seizure label set to `{seiz, gnsz, fnsz, cpsz, absz, spsz, tcsz, tnsz, mysz}`
- Rebuilt all caches (local and Modal) and restarted training
- Added tests and documentation of empirical type counts for v2.0.3
**Lesson:** Verify label sets empirically against the corpus version

### 4. Dataset Label Duration Bug
**Impact:** Labels wrong length (3.9M samples instead of 15360)
**Root Cause:** Passing `n_samples` instead of `duration_sec` to events_to_binary_mask
**Resolution:** Ensure `events_to_binary_mask` is called with true `duration_sec`; mask then trimmed/padded to `n_samples` during window extraction

### 5. Focal Loss Double-Counting
**Impact:** Training instability with extreme class imbalance
**Root Cause:** Using both focal_alpha and pos_weight caused double-counting
**Resolution:**
- Set safe defaults in configs; avoid combining `pos_weight` with focal `alpha` that double-counts imbalance

### 6. PyTorch Multiprocessing Hangs on WSL2
**Impact:** Training hangs after validation, DataLoader stuck
**Resolution:**
- WSL-safe defaults: `num_workers=0`, `pin_memory=false`
- Force spawn start method
- Document in configs with clear warnings

## P0 Critical Issues (Partially Resolved)

### 1. Mamba-SSM CUDA Fallback
**Impact:** Loses O(N) advantage, falls back to Conv1d
**Status:** Mitigated but not fully resolved
**Current State:**
- PyTorch 2.2.2 required for mamba-ssm compatibility
- d_conv=4 not supported by CUDA kernels (uses d_conv=4)
- `SEIZURE_MAMBA_FORCE_FALLBACK=1` forces Conv1d path
**Risk:** Performance degradation if CUDA kernels fail

### 2. EDF Header Malformation
**Impact:** Some TUSZ files have corrupt headers
**Resolution:**
- Header repair helper `_repair_edf_header_inplace` in `src/brain_brr/data/io.py` and retry logic on read failure
**Risk:** May miss edge cases in header corruption

## Lessons Learned

1. **Always validate data pipeline outputs** - Zero seizures = wasted compute
2. **Test with real data early** - Synthetic tests missed CSV_BI format
3. **Document critical env vars** - SEIZURE_MAMBA_FORCE_FALLBACK is essential
4. **WSL2 is fragile** - Always use num_workers=0 on WSL2
5. **Check both ends of pipeline** - Cache can be perfect but labels still wrong

## Prevention Checklist

Before any training run:
- [ ] Run `python -m src scan-cache` - verify seizures > 0
- [ ] Check manifest reports balanced sampling
- [ ] Verify channel canonicalization working
- [ ] Test one batch manually before full training
- [ ] Monitor first epoch closely for label collapse

## References
- Original postmortems in `/docs_archive/bugs/`
- Resolution tracking in `RESOLUTION_STATUS.md`
- Current guards and checks throughout codebase
