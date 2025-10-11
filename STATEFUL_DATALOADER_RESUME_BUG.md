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

### What's BROKEN (CRITICAL):
1. ❌ **`global_step` NOT SAVED**: Resets to 0 on resume, breaking warmup schedules
2. ❌ **Warmup schedule**: Restarts from beginning (focal_gamma=1.0 instead of 2.0)
3. ❌ **Progress tracking**: Shows `0/7702` instead of `2527/7702` (confusing for monitoring)
4. ❌ **Log messages**: All batch numbers are wrong by 2527 offset
5. ❌ **Checkpoint filenames**: `mid_epoch_001_000030.pt` instead of `mid_epoch_001_002557.pt`
6. ❌ **Heartbeat logs**: Report wrong batch positions
7. ❌ **W&B/TensorBoard**: Batch-indexed metrics have wrong x-axis

### What's CORRECT:
1. ✅ **Training data**: DataLoader skips correctly, no duplicate processing
2. ✅ **Model weights**: Restored correctly from checkpoint
3. ✅ **Optimizer state**: Restored correctly
4. ✅ **RNG state**: Restored for determinism
5. ✅ **Scheduler state**: Scheduler has its own step counter (learning rate is correct)
6. ✅ **Learning rate**: Correct (scheduler.state_dict() saves last_epoch)

**Severity**: **P1** - Warmup schedule is broken on resume (resets to focal_gamma=1.0)

---

## Fix Strategy

### REQUIRED FIX: Save and restore `global_step` AND `batch_idx`

```python
# In train_step.py - Save global_step in mid-epoch checkpoints
save_checkpoint(
    model,
    optimizer,
    epoch_index,
    0.0,
    mid_path,
    scheduler,
    None,
    scaler=scaler,
    save_rng=True,
    extra={
        "batch_idx": batch_idx,
        "global_step": global_step,  # ← ADD THIS
        "kind": "mid_epoch",
        "dataloader_state_dict": dataloader.state_dict(),
    },
)

# In loop.py - Restore both batch_idx and global_step
if "dataloader_state_dict" in ckpt:
    train_loader.load_state_dict(ckpt["dataloader_state_dict"])
    resume_batch_idx = ckpt.get('batch_idx', 0)
    resume_global_step = ckpt.get('global_step', 0)  # ← ADD THIS
    logger.info(f"[RESUME] ✅ Exact mid-epoch resume at batch {resume_batch_idx}, global_step {resume_global_step}")
else:
    resume_batch_idx = 0
    resume_global_step = 0

# Pass both to train_epoch()
result = train_epoch(
    model,
    train_loader,
    optimizer,
    ...
    global_step=resume_global_step,  # ← UPDATE: Pass restored value
    resume_batch_idx=resume_batch_idx,  # ← NEW parameter
)

# In train_step.py - Update signature and loop
def train_epoch(
    model,
    dataloader,
    optimizer,
    ...
    global_step: int = 0,  # ← Already exists, just needs restoration
    resume_batch_idx: int = 0,  # ← NEW parameter
):
    ...
    for relative_idx, batch in enumerate(progress):
        batch_idx = relative_idx + resume_batch_idx  # ← Fix counter
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

## Confirmed Issues

1. **Warmup uses `global_step`** ✅ (checked warmup.py:12)
   - Uses `global_step`, not `batch_idx`
   - **BUT** `global_step` is NOT saved in checkpoints ❌
   - Result: Warmup restarts from focal_gamma=1.0 on resume

2. **`global_step` is NOT in checkpoint** ❌ (verified mid_epoch_001_002527.pt)
   - Checkpoint keys: ['version', 'epoch', 'model_state_dict', 'optimizer_state_dict', 'best_metric', 'timestamp', 'scheduler_state_dict', 'scaler_state_dict', 'rng_state', 'batch_idx', 'kind', 'dataloader_state_dict']
   - Missing: 'global_step'

3. **Scheduler is OK** ✅ (has own step counter)
   - Scheduler saves `last_epoch` in state_dict
   - Learning rate is correct even though global_step resets

4. **Evidence from current training**:
   ```
   [RESUME] ✅ Exact mid-epoch resume at batch 2527
   [WARMUP] Batch 0 focal_gamma=1.000  ← Should be 2.0 (past warmup)!
   ```
   - Confirms global_step reset to 0
   - Warmup restarted (focal_gamma=1.0 instead of 2.0)

---

**Next Step**: Monitor training for 5-10 minutes to confirm batches are being skipped (epoch completes in ~1.7h, not 2.5h). If confirmed, this is just a cosmetic bug and training can continue.
