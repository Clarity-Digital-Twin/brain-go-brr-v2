# P0: Critical Cache Directory Mismatch Issue

**Severity**: P0 - CRITICAL
**Date**: 2025-09-20
**Status**: Active Training Affected

## Executive Summary
Current training is NOT using cached data as intended. Training is processing all 3734 files live, causing massive slowdown. Archive created contains wrong cache directory.

## Issue Description

### What We Thought
- Training would use cached data from `cache/train/` (246 files)
- Cache would speed up training by 10-100x
- Archive `tusz_train_cache_256hz_10-20_v1_20250920.tar.gz` contained the right data

### What's Actually Happening
- Config specifies `cache_dir: cache/tusz`
- But smoke test created cache in `cache/train/`
- Training is processing ALL files from scratch (no cache)
- 252,089 training windows + 51,901 validation windows being created live

## Root Cause Analysis

### Finding #1: Cache Directory Mismatch
```yaml
# configs/tusz_train_wsl2.yaml
experiment:
  cache_dir: cache/tusz  # <-- ACTUAL cache location

# But smoke test used:
experiment:
  cache_dir: cache/train  # <-- WHERE we archived from
```

### Finding #2: Dataset Size Discrepancy
- **Expected**: 3734 train files (from logs)
- **Actual**: 4667 EDF files in `data_ext4/tusz/edf/train/`
- **Cached**: Only 246 files in wrong directory
- **Processing**: All 3734 files live during training

### Finding #3: Performance Impact
- **With cache**: ~1-2 min to load
- **Without cache (current)**: 30-60+ min to process
- **Per-batch slowdown**: ~48s/batch instead of ~5-10s/batch

## Evidence

1. **Training logs show full processing**:
```
[DATA] Building dataset index for 3734 files...
[DATA] Processing file 3501/3734: aaaaates_s002_t005.edf
[DATA] Dataset ready! Total windows: 252089
```

2. **Cache directories**:
```bash
cache/train/     # 246 .npz files (27GB) - WRONG location
cache/tusz/      # DOESN'T EXIST - should be here
```

3. **Archive contains wrong cache**:
```bash
tusz_train_cache_256hz_10-20_v1_20250920.tar.gz
# Contains cache/train/* instead of cache/tusz/*
```

## Impact

### Current Training Run
- ⚠️ **30-60x slower** than expected
- Processing 3734 files live = ~45-60 minutes overhead
- Each epoch taking hours instead of minutes
- RTX 4090 underutilized during data processing

### Future Runs
- Archive is **useless** for this config
- Modal runs will be extremely slow
- Need to rebuild cache in correct location

## Resolution Steps

### Immediate (Don't interrupt current training)
1. Let current training continue (already 30+ batches in)
2. Monitor for completion
3. Document actual training time for comparison

### Next Steps
1. **After training completes**:
   ```bash
   # Copy cache to correct location
   cp -r cache/train cache/tusz

   # Or symlink for now
   ln -s $(pwd)/cache/train $(pwd)/cache/tusz
   ```

2. **For next run**:
   - Either update config to use `cache/train`
   - OR move cache to `cache/tusz`
   - OR standardize on one cache directory

3. **Create new archive**:
   ```bash
   tar -czf tusz_cache_CORRECT_v2.tar.gz cache/tusz/
   ```

## Lessons Learned

1. **Config validation needed**: Should verify cache_dir matches
2. **Cache path should be absolute or standardized**
3. **Training should log which cache is being used**
4. **Smoke test config should match production config paths**

## Questions Requiring Investigation

1. Why only 246 files cached from smoke test?
2. Why 3734 files reported but 4667 exist?
3. Are some files being filtered by the dataset loader?
4. Should we implement cache validation on startup?

## Recommendations

### Short Term
- Symlink `cache/train` → `cache/tusz` after current training
- Update configs to use consistent cache directory
- Add logging for cache hits/misses

### Long Term
- Implement cache validation
- Add progress bar for cache building
- Make cache directory configurable via CLI
- Add cache statistics to training logs

## Status Updates

### 2025-09-20 19:25
- Training continuing at batch 31/31512
- Decision: Let it run, fix cache after completion
- Archive created but contains wrong directory