# Critical Issues - Resolved

This document captures critical bugs and pain points from project history that have been resolved but should be remembered to prevent regression.

## P0 Critical Issues (Resolved)

### 1. Zero-Seizure Cache Catastrophe (254GB Wasted)
**Date:** 2025-09-21
**Impact:** 254GB useless cache, $60+ Modal credits burned, model learned nothing
**Root Cause:** CSV parser was reading wrong columns - expected simple format but TUSZ uses CSV_BI format with channel-specific annotations
**Resolution:**
- Fixed CSV_BI parser in `src/brain_brr/data/io.py:220,294`
- Added guards to fail builds with zero seizures `cli.py:212`
- Implemented BalancedSeizureDataset with manifest validation
**Lesson:** Always validate cache has seizures before training!

### 2. TUSZ Channel Naming Mismatch
**Impact:** Training crashes with "Missing required channels"
**Root Cause:** TUSZ files have `'EEG FP1-LE'` but code expects `'Fp1'`
**Resolution:**
- Robust channel cleaning in `io.py:126` (clean_tusz_name)
- Channel synonym mapping in `constants.py:33`
- Ordered channel picking in `pick_utils.py:30`

### 3. Dataset Label Duration Bug
**Impact:** Labels wrong length (3.9M samples instead of 15360)
**Root Cause:** Passing `n_samples` instead of `duration_sec` to events_to_binary_mask
**Resolution:** Fixed calculation to use `n_samples / 256` for duration in seconds

### 4. Focal Loss Double-Counting
**Impact:** Training instability with extreme class imbalance
**Root Cause:** Using both focal_alpha and pos_weight caused double-counting
**Resolution:**
- Set focal_alpha=0.5 (neutral)
- Use pos_weight OR focal loss, never both
- Defaulted in all configs

### 5. PyTorch Multiprocessing Hangs on WSL2
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
- d_conv=5 not supported by CUDA kernels (coerces to 4)
- `SEIZURE_MAMBA_FORCE_FALLBACK=1` forces Conv1d path
**Risk:** Performance degradation if CUDA kernels fail

### 2. EDF Header Malformation
**Impact:** Some TUSZ files have corrupt headers
**Resolution:**
- Header repair function in `io.py:34` (_repair_edf_header_inplace)
- Automatic retry on failure
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