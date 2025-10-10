# Checkpoint Resume Bug: Epoch Re-Training Issue

**Status:** CRITICAL BUG - Blocks efficient auto-restart implementation
**Created:** October 10, 2025
**Impact:** Wastes 6-14 GPU hours per restart (~$24-56 per restart on A100)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current Checkpoint System](#current-checkpoint-system)
3. [The Bug](#the-bug)
4. [Impact Analysis](#impact-analysis)
5. [Root Cause](#root-cause)
6. [Proposed Fix](#proposed-fix)
7. [Testing Plan](#testing-plan)
8. [Relation to Auto-Restart Strategy](#relation-to-auto-restart-strategy)

---

## Executive Summary

**Problem:** When training is resumed from a checkpoint, it re-trains the epoch that was just completed, wasting 6-14 hours of GPU time per restart.

**Root Cause:** The checkpoint saves the **current epoch index** (the epoch that just finished), but the training loop interprets this as the **next epoch to train**.

**Fix:** Change `epoch` to `epoch + 1` in the `last.pt` checkpoint save after validation completes.

**Why Critical:** With 12 auto-restarts planned over 12 days, this bug could waste 72-168 hours of A100 time ($288-672).

---

## Current Checkpoint System

### Three Types of Checkpoints

**1. Mid-Epoch Checkpoints (train_step.py:524-554)**
```python
# Saved every 30 minutes during training
mid_path = checkpoint_dir / f"mid_epoch_{epoch_index + 1:03d}_{batch_idx:06d}.pt"
save_checkpoint(
    model, optimizer, epoch_index, 0.0, mid_path,
    extra={"batch_idx": batch_idx, "kind": "mid_epoch"}
)
```
- **Purpose:** Crash recovery
- **Frequency:** Every 30 minutes (configurable via `mid_checkpoint_interval_s`)
- **Retention:** Last 3 checkpoints (configurable via `mid_epoch_keep`)
- **Contains:** Model state at specific batch within epoch

**2. Best Checkpoint (loop.py:418-430)**
```python
# Saved when new best metric achieved
save_checkpoint(
    model, optimizer, epoch, current_metric,
    checkpoint_dir / CHECKPOINT_BEST, ...
)
```
- **Purpose:** Keep best model for deployment
- **Frequency:** Only when metric improves
- **Contains:** Best model state

**3. Last Checkpoint (loop.py:457-469) ← BUG IS HERE**
```python
# Saved after every epoch completes (training + validation)
if config.training.resume or config.experiment.save_model:
    save_checkpoint(
        model, optimizer, epoch, best_metric,
        checkpoint_dir / CHECKPOINT_LAST, ...
    )
```
- **Purpose:** Resume training from last completed epoch
- **Frequency:** After every epoch
- **Contains:** Model state after validation completes

### Resume Priority (loop.py:145-178)

```python
# Priority: mid_epoch_*.pt > last.pt > best.pt
if mid_epoch_checkpoints and config.training.resume:
    # Load mid-epoch checkpoint (with batch_idx metadata)
    start_epoch, best_metric = load_checkpoint(latest_mid, ...)
    logger.info(f"Resumed from epoch {start_epoch + 1}, batch {ckpt.get('batch_idx', '?')}")
    # NOTE: Resumes from START of epoch, not the saved batch!
elif (checkpoint_dir / CHECKPOINT_LAST).exists() and config.training.resume:
    # Load last.pt checkpoint
    start_epoch, best_metric = load_checkpoint(checkpoint_dir / CHECKPOINT_LAST, ...)
    logger.info(f"Resumed from epoch {start_epoch + 1}")
```

**Training Loop (loop.py:235):**
```python
for epoch in range(start_epoch, config.training.epochs):
    # epoch = start_epoch on first iteration
    ...
```

---

## The Bug

### Current Behavior (INCORRECT)

**File:** `src/brain_brr/train/loop.py:457-469`

```python
# Always save last checkpoint for resume capability (even if save_model=False)
if config.training.resume or config.experiment.save_model:
    save_checkpoint(
        model,
        optimizer,
        epoch,              # ← BUG! This is the epoch that JUST FINISHED
        best_metric,
        checkpoint_dir / CHECKPOINT_LAST,
        scheduler,
        config,
        scaler=scaler,
        save_rng=True,
    )
```

**What happens:**
1. Training completes Epoch 2 (batches 0-3088) → model weights updated ✅
2. Validation runs on Epoch 2 → metrics computed ✅
3. Checkpoint saved with `epoch=1` (0-indexed, so Epoch 2 internally is `epoch=1`) ✅
4. **Training killed during validation or after checkpoint save** ⚠️
5. **Resume loads checkpoint:** `start_epoch = 1` ❌
6. **Training loop:** `range(start_epoch, 100)` → `range(1, 100)` → **STARTS AT EPOCH 2 AGAIN!** ❌❌❌

### Timeline Example (Epoch 2 was in progress)

```
T=0h:      Start Epoch 2 training (3088 batches)
T=0.5h:    Mid-epoch checkpoint saved (batch 200)
T=1.0h:    Mid-epoch checkpoint saved (batch 400)
...
T=13.5h:   Epoch 2 training COMPLETES (batch 3088)
T=13.5h:   Start Epoch 2 VALIDATION (1832 batches)
T=14.2h:   Validation at batch 431/1832
T=14.2h:   Modal kills training (23h timeout)
           ❌ last.pt was saved AFTER Epoch 2 training with epoch=1
           ✅ But mid-epoch checkpoints exist from ~T=13h
```

**On resume (loading mid-epoch checkpoint):**
- Loads model state from T=13h (batch ~3000 of Epoch 2 training)
- Sets `start_epoch = 1` (Epoch 2 in human terms)
- **Restarts training from batch 0 of Epoch 2** ← WASTES ~13 hours!

**On resume (loading last.pt if no mid-epoch):**
- Loads model state from T=13.5h (END of Epoch 2 training)
- Sets `start_epoch = 1` (Epoch 2 in human terms)
- **Restarts training from batch 0 of Epoch 2** ← WASTES ~14 hours!

---

## Impact Analysis

### Single Restart Impact

**Scenario:** Training runs for 23 hours, completes Epoch 2, killed during validation.

**Current (buggy):**
- Resume loads Epoch 2 model state
- Re-trains Epoch 2 from scratch: **14 hours wasted**
- Cost: 14h × $4/h = **$56 wasted per restart**

**With fix:**
- Resume starts at Epoch 3
- No wasted time: **$0 wasted**

### Auto-Restart Impact (12 restarts over 12 days)

**Current (buggy):**
- 12 restarts × 14h wasted = **168 hours wasted**
- Cost: 168h × $4/h = **$672 wasted**
- Total training time: 280h (actual) + 168h (wasted) = 448h
- Total cost: $1,792 (vs. $1,120 without bug)

**With fix:**
- 12 restarts × 0h wasted = **0 hours wasted**
- Cost: **$0 wasted**
- Total training time: 280h
- Total cost: $1,120

**Savings: $672 and 168 GPU hours** 💰

---

## Root Cause

### Semantic Confusion: "Current Epoch" vs "Next Epoch"

The bug stems from confusion about what the `epoch` variable means in the checkpoint:

**During training loop (loop.py:235):**
```python
for epoch in range(start_epoch, config.training.epochs):
    # epoch = 0 → Train Epoch 1, validate, save
    # epoch = 1 → Train Epoch 2, validate, save
    # epoch = 2 → Train Epoch 3, validate, save
```

**When saving checkpoint after Epoch 2 completes:**
```python
epoch = 1  # Loop variable (0-indexed)
# This means "Epoch 2 just finished training and validation"
# We INTEND to mean "start_epoch should be 2 (Epoch 3) on resume"
# But we're saving epoch=1, which means "start_epoch=1 (Epoch 2) on resume"
```

**The fix is simple:** Save `epoch + 1` to mean "next epoch to train".

### Why Mid-Epoch Checkpoints Also Have This Bug

Mid-epoch checkpoints save `epoch_index` (not `epoch_index + 1`), so they also resume from the START of the epoch, not the saved batch.

**From loop.py:168:**
```python
# Note: This resumes from start of epoch, not exact batch
```

**Why this is acceptable for mid-epoch:**
- Mid-epoch checkpoints are for **crash recovery** (rare)
- Wasting a few hours is better than losing ALL progress
- Implementing batch-level resume is complex (breaks reproducibility)

**Why this is NOT acceptable for last.pt:**
- `last.pt` is used for **every normal resume** (frequent with auto-restart)
- The epoch is FULLY COMPLETE - no reason to re-train it!
- Fix is trivial: just save `epoch + 1`

---

## Proposed Fix

### Change 1: Fix `last.pt` Save After Validation (loop.py:457-469)

**BEFORE (buggy):**
```python
# Always save last checkpoint for resume capability (even if save_model=False)
if config.training.resume or config.experiment.save_model:
    save_checkpoint(
        model,
        optimizer,
        epoch,              # ← BUG: Saves completed epoch index
        best_metric,
        checkpoint_dir / CHECKPOINT_LAST,
        scheduler,
        config,
        scaler=scaler,
        save_rng=True,
    )
```

**AFTER (fixed):**
```python
# Always save last checkpoint for resume capability (even if save_model=False)
if config.training.resume or config.experiment.save_model:
    save_checkpoint(
        model,
        optimizer,
        epoch + 1,          # ✅ FIX: Save next epoch to train
        best_metric,
        checkpoint_dir / CHECKPOINT_LAST,
        scheduler,
        config,
        scaler=scaler,
        save_rng=True,
    )
```

### Change 2: Fix Timeout Guard Checkpoint (loop.py:253-264)

**BEFORE (buggy):**
```python
# Save final checkpoint before exit
save_checkpoint(
    model,
    optimizer,
    epoch,  # ← BUG: Could be in middle of epoch
    best_metric if "best_metric" in locals() else 0.0,
    checkpoint_dir / "timeout_exit.pt",
    scheduler,
    config,
    scaler=scaler,
    save_rng=True,
)
```

**AFTER (fixed):**
```python
# Save final checkpoint before exit
save_checkpoint(
    model,
    optimizer,
    epoch,  # ✅ CORRECT: Timeout happens BEFORE epoch starts, so epoch is next to train
    best_metric if "best_metric" in locals() else 0.0,
    checkpoint_dir / "timeout_exit.pt",
    scheduler,
    config,
    scaler=scaler,
    save_rng=True,
)
```

**Note:** The timeout guard triggers BEFORE starting an epoch (loop.py:239-265), so `epoch` is already the next epoch to train. No change needed here, but adding comment for clarity.

### Change 3: Fix Signal Handler Checkpoint (loop.py:208-218)

**BEFORE (needs review):**
```python
save_checkpoint(
    model,
    optimizer,
    epoch_val,  # ← REVIEW: Depends on when signal is caught
    float(best_val),
    checkpoint_dir / f"signal_exit_{sig_name.lower()}.pt",
    scheduler,
    config,
    scaler=scaler,
    save_rng=True,
)
```

**AFTER (needs analysis):**
```python
# ANALYSIS NEEDED: When does SIGTERM/SIGINT arrive?
# - If during training: epoch is current epoch (in progress) → save epoch (correct)
# - If after validation: epoch is completed epoch → save epoch + 1 (like last.pt)
# For now, keep as-is since signals are rare and this is a secondary checkpoint
```

**Decision:** Leave signal handler as-is for now. Signals are rare (manual kills), and user can always use `last.pt` instead.

---

## Testing Plan

### Phase 1: Smoke Test with Resume (15 minutes)

**Objective:** Verify fix doesn't break existing functionality.

**Steps:**
1. Apply fix to `loop.py:462` (change `epoch` → `epoch + 1`)
2. Run smoke test (3 files, 1 epoch):
   ```bash
   modal run deploy/modal/app.py --action train --config configs/modal/smoke_bimamba.yaml
   ```
3. Let it complete Epoch 1
4. Check `last.pt`:
   ```python
   import torch
   ckpt = torch.load("last.pt")
   print(f"Saved epoch: {ckpt['epoch']}")  # Should be 1 (next epoch to train)
   ```
5. Resume and verify it starts Epoch 2:
   ```bash
   modal run deploy/modal/app.py --action train --config configs/modal/smoke_bimamba.yaml --resume true
   ```
6. Check logs: Should say "Resumed from epoch 2" and start training Epoch 2

### Phase 2: Resume Test (30 minutes)

**Objective:** Verify resume logic with the fix.

**Steps:**
1. Start smoke test (let run for 2 epochs)
2. After Epoch 1 completes, stop training manually:
   ```bash
   modal app stop <app-id>
   ```
3. Resume training:
   ```bash
   modal run deploy/modal/app.py --action train --config configs/modal/smoke_bimamba.yaml --resume true
   ```
4. **Expected:** Logs show "Resumed from epoch 2", starts Epoch 2
5. **Failure case (old bug):** Logs show "Resumed from epoch 1", re-trains Epoch 1

### Phase 3: Production Validation (1 hour)

**Objective:** Ensure fix works with full training.

**Steps:**
1. Start full training:
   ```bash
   modal run --detach deploy/modal/app.py --action train --config configs/modal/train_bimamba.yaml
   ```
2. Wait for Epoch 1 to complete (~14 hours)
3. Stop training:
   ```bash
   modal app stop <app-id>
   ```
4. Resume:
   ```bash
   modal run --detach deploy/modal/app.py --action train --config configs/modal/train_bimamba.yaml --resume true
   ```
5. **Expected:** Training starts at Epoch 2, NOT Epoch 1
6. Monitor W&B: Verify Epoch 2 metrics are new (not duplicate of previous Epoch 2)

---

## Relation to Auto-Restart Strategy

### Why This Fix is Critical for Auto-Restart

**From `MODAL_AUTO_RESTART_STRATEGY.md`:**

The auto-restart strategy will restart training every 23 hours via `modal.Period(hours=23)`. This means:
- **12 restarts** over 12 days (for 20 epochs at 14h/epoch)
- Each restart calls `train.remote(resume=True)`
- Each restart loads `last.pt` or `timeout_exit.pt`

**Without this fix:**
- Every restart re-trains the completed epoch
- 12 restarts × 14h wasted = **168 hours wasted** ($672)

**With this fix:**
- Every restart starts at the next epoch
- 12 restarts × 0h wasted = **0 hours wasted** ($0)

### Timeline: Fix First, Then Implement Auto-Restart

**Current plan:**
1. ✅ Phase 1: Validate manual resume workflow (DONE)
2. ✅ Phase 2.1: Read existing train() function (DONE)
3. **→ FIX THIS BUG FIRST** ← **WE ARE HERE**
4. Test fix with smoke test + resume
5. Resume full training with fixed code
6. **THEN** implement auto-restart (Phase 2.2-2.3)
7. Test auto-restart
8. Deploy for production

**Why this order matters:**
- Fixing now saves debugging later
- One training run will validate the fix (14h)
- Auto-restart implementation assumes resume works correctly

---

## Code Reference Summary

### Files Affected

| File | Line | Change | Reason |
|------|------|--------|--------|
| `src/brain_brr/train/loop.py` | 462 | `epoch` → `epoch + 1` | Fix last.pt to save next epoch |
| `src/brain_brr/train/loop.py` | 256 | Add comment | Clarify timeout_exit.pt is correct |

### Related Code (No Changes Needed)

| File | Line | Description |
|------|------|-------------|
| `src/brain_brr/train/loop.py` | 150-168 | Resume logic (loads correctly) |
| `src/brain_brr/train/loop.py` | 235 | Training loop (works correctly) |
| `src/brain_brr/train/train_step.py` | 530-554 | Mid-epoch checkpoints (acceptable behavior) |
| `src/brain_brr/train/checkpoint.py` | 122-123 | Load/save functions (work correctly) |

---

## Decision: Fix Now or Later?

### Arguments for Fixing Now ✅

1. **Cost:** Saves $672 over 12 days of training
2. **Time:** Saves 168 GPU hours (7 full days)
3. **Simplicity:** One-line change, low risk
4. **Testing:** Can validate with one training run (14h)
5. **Auto-restart:** Blocks efficient implementation

### Arguments for Fixing Later ❌

1. None - this is a critical bug with a trivial fix

### Recommendation: **FIX NOW** ✅

**Action plan:**
1. User kills current training run (ap-vSWIaqOTB65Nu3zmI92NpB)
2. Apply fix to `loop.py:462`
3. Run smoke test to validate
4. Resume full training with fixed code
5. Proceed with auto-restart implementation

**Time cost of fixing now:** ~1 hour (testing) vs. $672 saved

---

## Status Tracking

- [x] Bug identified (October 10, 2025)
- [x] Root cause analysis complete
- [x] Fix proposed and documented
- [ ] Fix implemented
- [ ] Smoke test passed
- [ ] Full training validation
- [ ] Auto-restart implementation can proceed

---

**NEXT STEPS:**
1. Kill current training (user action)
2. Implement fix (1-line change)
3. Run smoke test with resume (15 min)
4. Resume full training (14h to validate)
5. Proceed with auto-restart implementation

**Questions for review:**
1. Is the fix correct? (Change `epoch` → `epoch + 1` in line 462)
2. Are there other places where this pattern appears?
3. Should we fix mid-epoch checkpoint behavior too? (More complex, defer?)
4. Should we add a test to prevent regression?
