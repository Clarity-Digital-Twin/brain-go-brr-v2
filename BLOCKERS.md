# P0/P1 BLOCKERS - CRITICAL ISSUES

## P0 BLOCKERS (BREAKING FUNCTIONALITY)

### 1. ✅ FIXED: TEST FAILURES - pick() API regression
**Status:** RESOLVED - All tests passing
**Previous Issue:** 4 tests failing with pick() API
**Root Cause:** Changed from `raw.pick_channels()` to `raw.pick()` but:
  - Mock objects in tests don't have `pick()` method
  - Real MNE `pick()` doesn't accept `ordered=True` parameter
**Fix Applied:** Reverted to `pick_channels()` with detailed comment explaining why:
  - MNE's `pick()` method does NOT support `ordered` parameter (verified)
  - `pick_channels()` has `ordered=True` which we need for channel ordering
  - Must keep using `pick_channels()` despite deprecation until MNE adds ordered to `pick()`

### 2. ⚠️ RUFF LINT ERROR - SIM113
**Status:** Blocking `make lint-fix`
**Error:** `SIM113 Use enumerate() for index variable global_step in for loop`
**Location:** `src/brain_brr/train/loop.py:240`
**Issue:** Ruff wants us to use enumerate instead of manual counter
**Fix Required:** Refactor to use enumerate or suppress warning

## P1 ISSUES (NOT BLOCKING BUT ANNOYING)

### 3. ⚠️ LR Scheduler Warning
**Status:** Warning appears but training works
**Warning:** "Detected call of `lr_scheduler.step()` before `optimizer.step()`"
**Location:** First training batch
**Root Cause:** Despite fix attempt, `global_step` tracking not working correctly
**Impact:** PyTorch skips first LR value but training still works
**Fix Attempted:** Added global_step tracking but needs more work

### 4. ✅ FIXED: Missing Training Metrics
**Status:** RESOLVED
**Previous Issue:** Training completed without showing metrics
**Fix Applied:**
  - Changed argparse to positional argument
  - Fixed CLI wrapper to pass positional argument
  - Added verbose metric printing
**Result:** Metrics now properly displayed

### 5. ✅ FIXED: pick_channels Deprecation Warning
**Status:** RESOLVED (but caused test failures)
**Previous Warning:** "pick_channels() is a legacy function"
**Fix Applied:** Changed to `raw.pick()`
**Side Effect:** Tests now failing (see P0 #1)

## TESTING SUMMARY

### Smoke Test Results (BGB_LIMIT_FILES=3)
- **Train Loss:** 1.6835
- **Val Loss:** 4.4516
- **TAES:** 0.0000 (expected - too little data)
- **AUROC:** 0.5164 (essentially random, expected with tiny dataset)
- **Sensitivity@10FA:** 0.0000 (no seizures detected)

### What AUROC=0.5164 means:
- AUROC (Area Under ROC Curve) measures discrimination ability
- 0.5 = random guessing (coin flip)
- 1.0 = perfect classification
- **0.5164 is essentially random** - expected because:
  - Only 3 training files
  - Model barely trained (1 epoch)
  - Insufficient data for learning patterns
  - This is just a smoke test to verify code runs

## IMMEDIATE ACTION ITEMS

1. **FIX TESTS (P0):** Revert pick() change or fix test mocks
2. **FIX LINT (P0):** Address SIM113 error
3. **FIX LR SCHEDULER (P1):** Properly handle first step
4. **RUN FULL TRAINING:** Once fixes applied, run with full dataset

## REGRESSION ROOT CAUSE

The regression was introduced when fixing deprecation warnings:
- Changed `pick_channels()` → `pick()` to fix MNE deprecation
- But didn't test the change properly
- Test mocks don't match new API
- Real MNE API doesn't accept `ordered` parameter

## Commands to Verify Fixes

```bash
# After fixes:
make lint-fix  # Should pass
make test       # Should have 0 failures
make q          # Full quality check
python -m src train configs/smoke_test.yaml  # Should run without warnings
```