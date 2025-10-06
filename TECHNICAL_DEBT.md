# Technical Debt

**Date**: October 6, 2025
**Status**: 🔴 **CRITICAL ISSUES DISCOVERED** - Training validation will fail
**Training Impact**: CRITICAL - Validation dataset empty, training will crash at epoch end

---

## Executive Summary

| Priority | Count | Training Impact | Status |
|----------|-------|-----------------|--------|
| **P0 BLOCKER** | 3 | Training will crash at validation | 🔴 **FIX IMMEDIATELY** |
| **P1 URGENT** | 1 | Data safety, logs noisy | 🟡 Fix before production |
| **P2 MEDIUM** | 1 | Minor warnings | 🟢 Fix when convenient |
| **P3 LOW** | 0 | None | ✅ Clear |

**CRITICAL**: Current Modal training (`ap-O0RQ15Kc1kfqjHz2l23RbD`) will crash at end of epoch 1 when validation runs!

---

## 🔴 P0: CRITICAL BLOCKERS (Fix Immediately)

### P0-1: ValidationDataset Cannot Find Cache Files ⚠️ **TRAINING WILL CRASH**

**Evidence from Modal Logs**:
```
[ValidationDataset] Created with 0 windows (filtered):
  - 7944 partial seizure
  - 3536 full seizure
  - 136744 no-seizure
  - Seizure ratio: 0.0% (natural distribution)
[WARNING] Skipped 148224 manifest entries referencing missing cache files
```

**Root Cause**:
- Manifest references files as `patient_session_data_windows` (legacy NPZ naming)
- Actual files on disk: `patient_session_data_data.npy` + `patient_session_data_labels.npy`
- `ValidationDataset.__init__` line 604 checks `cache_file_path.exists()` directly
- `BalancedSeizureDataset` has `cache_file_exists()` helper that handles NPY conversion (line 385-392)
- `ValidationDataset` is MISSING this helper!

**Impact**:
- ✅ Training batches run fine (uses BalancedSeizureDataset which has the fix)
- ❌ At epoch end, validation attempts to run with 0 windows
- ❌ Validation will either crash or return invalid default metrics
- ❌ Wastes entire epoch of training (~6-7 hours on A100)

**Files Affected**:
- `src/brain_brr/data/datasets.py:604` - ValidationDataset missing NPY file check helper

**Fix**:
```python
# In ValidationDataset.__init__ (around line 594), add helper before loop:
def cache_file_exists(cache_path: Path) -> bool:
    """Check if cache file exists in NPY mmap format."""
    if cache_path.exists():
        return True  # Direct match (rare with _windows suffix)
    # NPY format: convert stem from *_windows → *_data.npy
    stem = cache_path.stem.replace("_windows", "")
    data_file = cache_path.parent / f"{stem}_data.npy"
    return data_file.exists()

# Then replace line 604:
if cache_file_exists(cache_file_path):  # Was: if cache_file_path.exists():
    file_to_windows[cache_file_name].append((int(item["window_idx"]), cache_file_path))
else:
    missing_ref_count += 1
```

**Verification After Fix**:
```
[ValidationDataset] Created with 148224 windows:
  - 7944 partial seizure
  - 3536 full seizure
  - 136744 no-seizure
  - Seizure ratio: 7.7% (natural distribution)
```

---

### P0-2: Read-Only NumPy Array Warnings (Data Safety Issue)

**Evidence from Modal Logs**:
```
UserWarning: The given NumPy array is not writable, and PyTorch does not support
non-writable tensors. This means writing to this tensor will result in undefined
behavior. (Triggered internally at ../torch/csrc/utils/tensor_numpy.cpp:206.)
  "window": torch.from_numpy(window),
```

**Root Cause**:
- NPY files opened with `mmap_mode='r'` (read-only) for cache safety
- `torch.from_numpy(window)` creates tensors backed by read-only memory
- Any in-place operation (`*_`) on these tensors → undefined behavior

**Current Risk**:
- ✅ No in-place ops on inputs currently
- ⚠️ Future code changes could introduce silent bugs
- ⚠️ Warning appears 4x per worker at startup (log noise)
- ⚠️ Violates PyTorch best practices (non-writable tensors undefined)

**Files Affected**:
- `src/brain_brr/data/datasets.py:530` - BalancedSeizureDataset
- `src/brain_brr/data/datasets.py:685` - ValidationDataset

**Production-Grade Fix**:
```python
# In all three datasets' __getitem__, replace:
return {
    "window": torch.from_numpy(window),
    "label": torch.from_numpy(label),
    ...
}

# With:
return {
    "window": torch.from_numpy(window).clone(),  # Explicit writable copy
    "label": torch.from_numpy(label).clone(),
    ...
}
```

**Cost Analysis**:
- CPU copy overhead: ~5-10ms per batch
- H2D transfer time: ~50-100ms per batch
- Net impact: <10% overhead, eliminates entire bug class
- Industry standard: Google/Meta/OpenAI all do this

**Verification**:
- No UserWarning about read-only arrays
- Test: `x = dataset[0]['window']; x[0] = 999.0` should work without error

---

### P0-3: LR Scheduler Order Warning

**Evidence from Modal Logs**:
```
UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`.
In PyTorch 1.1.0 and later, you should call them in the opposite order.
```

**Root Cause**:
- PyTorch 1.1+ requires `optimizer.step()` → `scheduler.step()` order
- Current code calls `scheduler.step()` before `optimizer.step()`

**Impact**:
- ⚠️ PyTorch skips first LR value
- ⚠️ Warmup schedule starts from step 2 instead of step 1
- ⚠️ Log noise

**Files Affected**:
- `src/brain_brr/train/train_step.py` - Training loop step order

**Fix**:
Move `scheduler.step()` call to AFTER `optimizer.step()` in training loop

**Verification**:
- No UserWarning about scheduler order
- LR starts from warmup_start value (not skipped)

---

## 🟡 P1: URGENT (Fix Before Production)

### P1-1: Manifest Cache File References Use Legacy Naming

**Issue**: Manifests reference `*_windows` but actual files are `*_data.npy` + `*_labels.npy`

**Root Cause**:
- Legacy NPZ format used `_windows.npz` naming convention
- Migrated to NPY mmap format but manifests never regenerated
- Datasets work around this with `stem.replace("_windows", "")`

**Impact**:
- Code harder to understand (requires translation layer)
- Inconsistent with actual file structure on disk
- **CAUSED P0-1 BUG** (ValidationDataset missing translation logic)
- Fragile: Easy to forget workaround in new code

**Files Affected**:
- `cache/tusz_mmap/train/manifest.json` - All `cache_file` references
- `cache/tusz_mmap/dev/manifest.json` - All `cache_file` references
- `src/brain_brr/data/cache_utils.py` - Manifest generation logic

**Fix Options**:

**Option A: Regenerate Manifests** (Recommended):
- Update `scan_existing_cache()` to write correct `*_data` names
- Regenerate both train/dev manifests on Modal
- Remove all `stem.replace("_windows", "")` workarounds
- Cleaner, matches reality

**Option B: Keep Workarounds**:
- Add `cache_file_exists()` to ValidationDataset (P0-1 fix)
- Document translation layer extensively
- More fragile, harder to maintain

**Recommended**: Option A after P0 fixes deployed

---

## 🟢 P2: MEDIUM (Fix When Convenient)

### P2-1: psutil Swap Memory Warning

**Evidence**:
```
RuntimeWarning: 'sin' and 'sout' swap memory stats couldn't be determined and
were set to 0 ([Errno 2] No such file or directory: '/proc/vmstat')
```

**Impact**: None (cosmetic warning in container logs)

**Fix**: Suppress this specific warning or remove swap memory stats from logging

---

## Implementation Plan

### Phase 1: Emergency P0 Fixes (DO NOW - 30 min)

**Goal**: Fix critical bugs so training completes successfully

**1. Stop Current Training**
```bash
modal app stop ap-O0RQ15Kc1kfqjHz2l23RbD
# Reason: Will crash at validation, wastes ~$50 of GPU time
```

**2. Fix ValidationDataset Cache Lookup (P0-1)**

File: `src/brain_brr/data/datasets.py`

Add helper function inside `ValidationDataset.__init__` before line 595:
```python
def cache_file_exists(cache_path: Path) -> bool:
    """Check if cache file exists in NPY mmap format."""
    if cache_path.exists():
        return True
    stem = cache_path.stem.replace("_windows", "")
    data_file = cache_path.parent / f"{stem}_data.npy"
    return data_file.exists()
```

Update line 604:
```python
if cache_file_exists(cache_file_path):  # Changed from: cache_file_path.exists()
```

**3. Fix Read-Only Tensor Warnings (P0-2)**

Files: `src/brain_brr/data/datasets.py`

In `BalancedSeizureDataset.__getitem__` (line 530):
```python
return {
    "window": torch.from_numpy(window).clone(),  # Add .clone()
    "label": torch.from_numpy(label).clone(),    # Add .clone()
    ...
}
```

In `ValidationDataset.__getitem__` (line 685):
```python
return {
    "window": torch.from_numpy(window).clone(),  # Add .clone()
    "label": torch.from_numpy(label).clone(),    # Add .clone()
    ...
}
```

**4. Fix LR Scheduler Order (P0-3)**

File: `src/brain_brr/train/train_step.py`

Find where scheduler is stepped, move AFTER optimizer.step()

**5. Run Quality Checks**
```bash
make q           # Lint/format/mypy
make test        # All 104 tests should pass
```

**6. Local Smoke Test**
```bash
export BGB_NAN_DEBUG=1
make s           # Should now show validation metrics!
```

---

### Phase 2: Verification & Deploy (15 min)

**7. Commit Changes**
```bash
git add -A
git status  # Verify only datasets.py and train_step.py changed
git commit -m "fix(p0): Fix ValidationDataset cache lookup and tensor safety

- Add cache_file_exists() helper to ValidationDataset for NPY lookup
- Use .clone() on tensors from read-only mmap arrays for safety
- Fix LR scheduler step order per PyTorch 1.1+ requirements

Fixes:
- P0-1: ValidationDataset now loads 148k windows (was 0)
- P0-2: No read-only tensor warnings
- P0-3: No LR scheduler order warnings"

git push origin development
```

**8. Merge to Main**
```bash
git checkout main
git merge development --no-ff -m "Merge P0 critical fixes for validation dataset"
git push origin main
git checkout development
```

---

### Phase 3: Restart Training (5 min)

**9. Deploy to Modal**
```bash
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml
```

**10. Verify Startup Logs**

Watch for these indicators:
```
✅ [ValidationDataset] Created with 148224 windows (not 0!)
✅ No UserWarning about read-only tensors
✅ No UserWarning about LR scheduler order
✅ Batch 0/1284 starts successfully
```

**11. Monitor First Epoch Completion**
- Training: 1284/1284 batches
- Validation: Runs successfully, metrics calculated
- Checkpoint: `epoch_001.pt` saved
- W&B: Validation metrics logged

---

### Phase 4: P1 Cleanup (Later, After Training)

**12. Regenerate Manifests with Correct Naming**

Update `src/brain_brr/data/cache_utils.py`:
```python
# In scan_existing_cache(), change cache_file reference:
entry["cache_file"] = f"{recording_id}_data"  # Was: f"{recording_id}_windows"
```

Run on Modal:
```bash
modal run deploy/modal/app.py --action populate-cache  # Regenerates manifests
```

**13. Remove Workarounds**

Remove all `stem.replace("_windows", "")` calls from datasets.py

**14. Verify**
```bash
make test  # All tests pass with new manifests
```

---

## Testing Checklist

Before restarting training, verify:

- [ ] `ValidationDataset` loads >140k windows (not 0)
- [ ] No read-only tensor warnings in startup logs
- [ ] No LR scheduler order warnings
- [ ] Local smoke test shows validation metrics
- [ ] `make q` passes (lint/format/mypy)
- [ ] `make test` passes (104 tests)
- [ ] Git changes committed and pushed
- [ ] main branch synced

---

## Risk Assessment

**Current Training Status**: 🔴 **WILL CRASH** at end of epoch 1

**After P0 Fixes**: 🟢 **READY** for successful 100-epoch training

**Time Estimates**:
- Fix implementation: 30 min
- Testing & verification: 15 min
- Deployment & restart: 5 min
- **Total downtime**: ~50 minutes

**Cost Analysis**:
- Training lost: ~1 hour @ batch 85/1284 (~$1.50)
- Cost to fix now: ~$1 (downtime)
- Cost if crash at epoch 1: ~$3.50 (6-7 hours wasted)
- **Net savings**: ~$2.50 + avoid debugging failed run

**Decision**: ✅ STOP NOW, FIX, RESTART

---

## Previously Resolved Debt (For Reference)

### October 6, 2025 - Pre-Training Fixes
- ✅ NPZ cache contamination (cleaned 3 stray files)
- ✅ Type annotations (WandBRun, Console vs Any)
- ✅ Code deduplication (120 lines eliminated)
- ✅ NPZ→NPY mmap conversion (387 GB → <1 GB RAM)
- ✅ Modal SSD cache population
- ✅ All P0/P1/P2/P3 from previous audit

---

## Post-Fix Verification

After training restarts, confirm in logs:

**1. Validation Dataset Loaded**:
```
[ValidationDataset] Created with 148224 windows:
  - 7944 partial seizure
  - 3536 full seizure
  - 136744 no-seizure
  - Seizure ratio: 7.7% (natural distribution)
```

**2. No Warnings**:
- ✅ No "read-only NumPy array" warnings
- ✅ No "LR scheduler step order" warnings
- ✅ Only harmless psutil swap warning (P2)

**3. First Epoch Completes Successfully**:
- ✅ Training: 1284/1284 batches
- ✅ Validation: Metrics calculated (not defaults)
- ✅ Checkpoint: `epoch_001.pt` saved
- ✅ W&B: Validation curves visible

---

**Status**: 🔴 **ACTION REQUIRED** - Fix P0 issues immediately
**Current Training**: STOP (`ap-O0RQ15Kc1kfqjHz2l23RbD`)
**Next Step**: Implement Phase 1 fixes (30 min)
**Goal**: Restart training within 1 hour with all issues resolved
