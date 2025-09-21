# P0 BLOCKERS - CRITICAL PIPELINE FAILURES

**Date**: 2025-09-21
**Status**: 🔴 BLOCKING TRAINING

## 1. MANIFEST GENERATION FAILURE

### Problem
The manifest.json is EMPTY despite seizures existing in cache files:
```json
{"partial_seizure": [], "full_seizure": [], "no_seizure": []}
```

### Evidence
- Cache files DO contain seizures (verified):
  - `aaaaahge_s001_t001_windows.npz`: 2,206,438 seizure timesteps
  - `aaaaahlq_s001_t001_windows.npz`: 2,040,594 seizure timesteps
  - `aaaaahma_s001_t000_windows.npz`: 154,080 seizure timesteps

### Impact
- Training falls back to WeightedRandomSampler (inferior approach)
- Not using SeizureTransformer's proven balanced sampling
- Wastes time sampling 20,000 windows to estimate class distribution

### ROOT CAUSE IDENTIFIED ✅
**The manifest is generated BEFORE the cache is built!**

Pipeline Order Bug (line numbers from loop.py):
1. Line 1062: Check if manifest.json exists
2. Line 1066: Create empty train_cache_dir
3. Line 1067: scan_existing_cache() on EMPTY directory → creates empty manifest
4. Line 1088-1099: EEGWindowDataset builds the actual cache files
5. Line 1074: Manifest exists (but empty) so tries to use it
6. Line 1081: BalancedSeizureDataset gets 0 windows, falls back

**Current Flow (BROKEN)**:
```
Empty cache dir → Build manifest (empty) → Build cache → Use empty manifest → Fallback
```

**Correct Flow**:
```
Build cache → Build manifest from cache → Use manifest → Success
```

## 2. MODAL CACHE PERSISTENCE UNCERTAINTY

### Problem
Modal training at 83% complete (3121/3734 files), $80 spent, but we don't know:
- WHERE does Modal save the cache?
- WILL it persist after run completes?
- HOW do we retrieve it?

### Current Modal Status
```
App ID: ap-WLEzvqCQvRuN24fzKIk7FC
State: ephemeral (detached)
Progress: Building file 3121/3734
```

### Decision Needed
- [ ] Let it complete (risk: cache not saved)
- [ ] Kill it now (loss: $80 and 83% progress)
- [ ] Check Modal volume mounts first

## 3. SAMPLING STRATEGY CONFUSION

### Expected Flow (SeizureTransformer approach)
1. Build cache → 2. Generate manifest → 3. Use BalancedSeizureDataset
   - ALL partial seizure windows
   - 0.3x full seizure windows
   - 2.5x background windows

### Actual Flow (BROKEN)
1. Build cache → 2. Manifest EMPTY → 3. Fallback to WeightedRandomSampler
   - Samples 20,000 windows
   - Creates class weights
   - Random sampling (may miss rare seizures)

## IMMEDIATE ACTIONS REQUIRED

### 1. Fix Pipeline Order (PERMANENT FIX)
The manifest should ONLY be built AFTER cache is complete.
Options:
- Move manifest generation to AFTER EEGWindowDataset creation
- Or check if cache is complete before building manifest
- Or rebuild manifest if cache was built after manifest

### 2. Quick Workaround (TEMPORARY)
```bash
# Delete the empty manifest and rebuild from populated cache
rm cache/tusz/train/manifest.json
source .venv/bin/activate
python -c "from src.brain_brr.data.cache_utils import scan_existing_cache; from pathlib import Path; scan_existing_cache(Path('cache/tusz/train'))"
```

### 2. Check Modal Cache Location (URGENT)
```bash
# Check Modal deployment script for volume mounts
grep -n "volume\|cache\|persist" deploy/modal/app.py
```

### 3. Pipeline Fix Implementation
Need to modify `src/brain_brr/train/loop.py` to:
1. Build cache FIRST (EEGWindowDataset)
2. THEN build manifest from completed cache
3. THEN create BalancedSeizureDataset

Alternatively:
- Check cache completeness before building manifest
- If cache incomplete, skip manifest generation
- Let EEGWindowDataset build cache
- Rebuild manifest after cache complete

## SUCCESS CRITERIA
1. Manifest contains seizures: partial > 0, full > 0, no_seizure > 0
2. Training uses BalancedSeizureDataset, not WeightedRandomSampler
3. Modal cache persists and is retrievable
4. Pipeline works end-to-end without manual intervention

## NOTES
- TUSZ has <0.1% seizure data (extreme imbalance)
- SeizureTransformer won 2025 competition with balanced sampling
- WeightedRandomSampler is fallback, NOT the intended approach