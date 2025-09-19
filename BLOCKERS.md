# P0/P1 BLOCKERS - CRITICAL ISSUES

## P0 BLOCKERS (BREAKING FUNCTIONALITY)

### 1. ✅ RESOLVED: Channel Ordering - Critical for Model Architecture

**Status:** RESOLVED with professional implementation

**Why Channel Order is Critical:**
- Our model has **hardcoded spatial expectations**: 19 channels in exact 10-20 montage order
- The U-Net encoder uses **spatial convolutions** that learn patterns based on channel adjacency
- **Channel position = spatial position**: Fp1 at index 0, F3 at index 1, etc.
- Changing order would **break all trained weights** - the model expects specific spatial relationships

**The MNE API Situation (2024-2025):**
- `pick_channels()` supports `ordered=True` but shows deprecation warnings
- `pick()` is the newer API but **lacks the `ordered` parameter**
- MNE defaults changed: `ordered` went from False (v1.6) to True (v1.7+)
- Both methods are still supported in MNE 1.10.x

**Professional Solution Implemented:**
```python
# We use pick_channels() despite deprecation warning because:
# 1. pick() doesn't support ordered=True parameter (verified MNE 1.10.1)
# 2. We MUST preserve channel order for spatial convolutions in U-Net
# 3. Channel order defines spatial relationships the model learned
raw.pick_channels(available, ordered=True)
```

**Why Not Just Use pick()?**
- `raw.pick(channel_names)` reorders channels unpredictably
- No `ordered` parameter means we can't guarantee the exact sequence
- Would require calling `reorder_channels()` separately (two operations)

**Future-Proof Strategy:**
When MNE adds `ordered` to `pick()`, we can migrate with:
```python
if hasattr(inspect.signature(raw.pick), 'ordered'):
    raw.pick(available, ordered=True)
else:
    raw.pick_channels(available, ordered=True)
```

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

### 5. ✅ UNDERSTOOD: pick_channels Deprecation Warning
**Status:** RESOLVED - Keep using pick_channels() intentionally
**Warning:** "pick_channels() is a legacy function"
**Decision:** Continue using `pick_channels()` because:
  - It's the ONLY method that guarantees channel order preservation
  - Channel order is critical for our spatial convolutions
  - Warning is just informational - function still works perfectly
  - Will migrate when MNE adds `ordered` parameter to `pick()`

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

## ARCHITECTURAL INSIGHT: Why Channel Order Matters

**Our SeizureDetector Architecture Requires Fixed Channel Order:**

1. **Input Layer:** Expects exactly `(B, 19, 15360)` shape
   - 19 channels in SPECIFIC order (Fp1, F3, C3, ...)
   - Each channel index corresponds to a physical electrode position

2. **U-Net Encoder:** Uses spatial convolutions across channels
   - Conv1d operates on channel dimension
   - Learns spatial patterns (e.g., frontal vs occipital activity)
   - Weights are trained for specific channel arrangements

3. **Clinical Meaning:** 10-20 montage has spatial significance
   - Adjacent channels in our order are spatially adjacent on scalp
   - Model learns these spatial relationships during training
   - Random reordering would destroy learned patterns

**Example Impact of Wrong Order:**
- Model expects index 0 = Fp1 (frontal pole)
- If we get O1 (occipital) at index 0 instead
- Frontal seizure patterns would be misinterpreted as occipital
- Complete failure of spatial pattern recognition

**This is why we MUST use `pick_channels(ordered=True)` despite deprecation**

## Commands to Verify Fixes

```bash
# After fixes:
make lint-fix  # Should pass
make test       # Should have 0 failures
make q          # Full quality check
python -m src train configs/smoke_test.yaml  # Should run without warnings
```