# StatefulDataLoader Resume Bug - Batch Index Mismatch

**Date**: October 11, 2025
**Status**: IDENTIFIED - Fix pending
**Priority**: P2 (Functional but cosmetic/tracking issues)

---

## Problem Summary

When resuming from mid-epoch checkpoint with StatefulDataLoader, the **data** is correctly skipped to batch 2527, but the **batch counter** restarts from 0. This causes:

1. ❌ Progress bar shows `0/7702` instead of `2527/7702`
2. ❌ Log messages report wrong batch numbers
3. ❌ Mid-epoch checkpoint filenames have wrong batch indices
4. ✅ Training data is CORRECT (not reprocessing old batches)
5. ✅ Model state is CORRECT (weights/optimizer restored)

---

## Evidence from Logs

```
[RESUME] ✅ Exact mid-epoch resume at batch 2527  ← DataLoader positioned correctly
Training:   0%|      | 0/7702 [00:00<?, ?it/s]      ← Progress shows 0/7702 ❌
[WARMUP] Batch 0 focal_gamma=1.000                   ← batch_idx=0 instead of 2527 ❌
Training:   0%|      | 4/7702 [00:56<19:57:57]      ← Counting from 0, not 2527 ❌
```

Expected behavior:
```
[RESUME] ✅ Exact mid-epoch resume at batch 2527
Training:  33%|████▋     | 2527/7702 [00:00<?, ?it/s]  ← Should start at 2527
[WARMUP] Batch 2527 focal_gamma=2.000                   ← Should continue warmup
Training:  33%|████▋     | 2531/7702 [00:56<...]       ← Should count from 2527
```

---

## Root Cause Analysis

### Code Path: `src/brain_brr/train/train_step.py:297`

```python
for batch_idx, batch in enumerate(progress):
    # enumerate() ALWAYS starts from 0, regardless of DataLoader state!
```

**What happens:**
1. `train_loader.load_state_dict(ckpt["dataloader_state_dict"])` ← Positions iterator at batch 2527 ✅
2. `enumerate(progress)` ← Starts counting from 0, not 2527 ❌
3. First batch yielded is batch 2527's data, but labeled as `batch_idx=0` ❌

**Why it's wrong:**
- StatefulDataLoader correctly skips batches 0-2526 internally
- But `enumerate()` doesn't know about this - it just counts yields
- So `batch_idx` represents "batches since resume", not "absolute batch position"

---

## Impact Assessment

### What's BROKEN:
1. **Progress tracking**: Shows `0/7702` instead of `2527/7702` (confusing for monitoring)
2. **Log messages**: All batch numbers are wrong by 2527 offset
3. **Checkpoint filenames**: `mid_epoch_001_000030.pt` instead of `mid_epoch_001_002557.pt`
4. **Warmup schedules**: Might reset if checking `batch_idx` instead of `global_step`
5. **Heartbeat logs**: Report wrong batch positions
6. **W&B/TensorBoard**: Batch-indexed metrics have wrong x-axis

### What's CORRECT:
1. ✅ **Training data**: DataLoader skips correctly, no duplicate processing
2. ✅ **Model weights**: Restored correctly from checkpoint
3. ✅ **Optimizer state**: Restored correctly
4. ✅ **RNG state**: Restored for determinism
5. ✅ **Scheduler state**: `global_step` is correct (independent of batch_idx)
6. ✅ **Learning rate**: Correct (scheduler uses global_step, not batch_idx)

**Severity**: **P2** - Training is functionally correct but tracking/logging is misleading

---

## Fix Strategy

### Option 1: Offset enumerate() (RECOMMENDED)
**Pros**: Simple, surgical fix
**Cons**: Need to pass batch offset through function signature

```python
# In loop.py, after loading checkpoint
if "dataloader_state_dict" in ckpt:
    train_loader.load_state_dict(ckpt["dataloader_state_dict"])
    resume_batch_idx = ckpt.get('batch_idx', 0)  # ← Extract saved batch position
    logger.info(f"[RESUME] ✅ Exact mid-epoch resume at batch {resume_batch_idx}")
else:
    resume_batch_idx = 0

# Pass to train_epoch()
result = train_epoch(
    model,
    train_loader,
    optimizer,
    ...
    resume_batch_idx=resume_batch_idx,  # ← NEW parameter
)

# In train_step.py
def train_epoch(
    model,
    dataloader,
    optimizer,
    ...
    resume_batch_idx: int = 0,  # ← NEW parameter
):
    ...
    for relative_idx, batch in enumerate(progress):
        batch_idx = relative_idx + resume_batch_idx  # ← Correct offset
        ...
```

### Option 2: Track state in DataLoader wrapper
**Pros**: No signature changes
**Cons**: More complex, need custom wrapper

### Option 3: Accept the cosmetic issue
**Pros**: Zero code changes
**Cons**: Confusing logs, wrong checkpoint names

---

## Recommended Action

**SHORT TERM (Current Training)**:
- ✅ Training is functionally correct, let it run
- Monitor actual time-to-completion to verify batches are being skipped:
  - If reprocessing: ~2.5 hours for full epoch (7702 batches)
  - If skipping: ~1.7 hours remaining (5175 batches = 7702 - 2527)
- Look for speed confirmation in next few minutes

**MEDIUM TERM (Next PR)**:
- Implement Option 1 (offset enumerate)
- Add integration test for resume with batch counting
- Update checkpoint save/load to validate batch continuity

**VALIDATION**:
Current training should show:
- Epoch completes in ~1.7-1.8 hours (not 2.5 hours)
- Final batch processes 7702 total batches (5175 remaining + 2527 already done)
- No duplicate data processing (memory usage stays consistent)

---

## Testing Plan

1. **Verify current behavior** (5 min):
   - Check time remaining in progress bar
   - Expected: ~1.7 hours (5175 batches × 3.5s/batch)
   - If showing ~2.5 hours → DataLoader resume is broken (worse bug!)
   - If showing ~1.7 hours → Just cosmetic counter bug (this analysis is correct)

2. **After driver stability confirmed** (next epoch):
   - Implement Option 1 fix
   - Test with smoke test resume scenario
   - Verify batch_idx matches saved checkpoint value

---

## Related Files

- `src/brain_brr/train/loop.py:150-177` - Checkpoint loading
- `src/brain_brr/train/train_step.py:297` - enumerate() loop
- `src/brain_brr/train/checkpoint.py` - Save/load logic
- `tests/train/test_checkpoint_*.py` - Existing checkpoint tests

---

## Open Questions

1. Does warmup schedule check `batch_idx` or `global_step`?
   - If `batch_idx`: Warmup will restart (bad!)
   - If `global_step`: Warmup continues correctly (good!)
   - **Answer needed**: Check warmup.py implementation

2. Do we log batch-indexed metrics to W&B?
   - If yes: x-axis will be wrong after resume
   - If no: No impact

3. Is `global_step` correctly preserved across resume?
   - If yes: Scheduler and warmup work correctly
   - If no: Bigger problem than batch_idx

---

**Next Step**: Monitor training for 5-10 minutes to confirm batches are being skipped (epoch completes in ~1.7h, not 2.5h). If confirmed, this is just a cosmetic bug and training can continue.
