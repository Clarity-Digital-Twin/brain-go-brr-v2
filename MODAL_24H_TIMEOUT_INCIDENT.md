# Modal 24-Hour Timeout Incident Analysis

**Date**: October 12, 2025
**Version**: v4.0.0
**Incident**: Training exceeded 23h timeout guard and hit Modal's 24h hard limit
**Status**: Under Investigation

---

## 🔍 Executive Summary

A manual training run (`modal run --detach --action train`) exceeded the 23-hour timeout guard and hit Modal's 24-hour hard limit, but **somehow resumed automatically** without using the scheduled `train_auto_restart` function. This document investigates:

1. **Why timeout guard didn't fire** at 23h (expected behavior)
2. **How training auto-resumed** without scheduled restart (unexpected behavior)
3. **Differences** between manual vs scheduled restart mechanisms
4. **Duplicate validation** anomaly during resume transition
5. **Fixes needed** to prevent recurrence

---

## 📊 Timeline of Events

### First Training Session (Started: Oct 11, ~12:49 EDT)

| Time (UTC) | Event | Notes |
|------------|-------|-------|
| 16:29:40 | ✅ Last mid-epoch checkpoint saved | `mid_epoch_004_001258.pt` (batch 1258/1284) |
| 16:38:39 | Training phase completed | Epoch 4 training finished, validation started |
| 16:40:02 | Validation began (normal) | Batch 1132, 1150, 1168, 1186, 1204... |
| 16:41:41 | 🚨 ANOMALY: Duplicate validation started | Batch 0, 17, 35... CONCURRENT with first validation! |
| 16:49:36 | Modal cancellation signal | `[modal-client] Received a cancellation signal` |
| 16:49:51 | 💀 Modal hard timeout | `Runner has been running for too long (max runtime: 86430 seconds)` |
| 16:51:54 | ⚠️ Validation continued (!) | Process kept running 11+ minutes after timeout |
| 17:22:37 | Logs stop | Last validation heartbeat (batch 342) |

**Analysis**:
- Timeout guard **never triggered** (expected at 82800s = 23h)
- Modal killed at **86430s = 24h + 30s**
- Training started Oct 11 12:49 EDT → 24h later = Oct 12 12:49 EDT
- UTC timestamp 16:49:51 = Oct 12 16:49:51 UTC = 12:49:51 EDT ✅ **Correct 24h timeout**

### Second Training Session (Auto-Resumed?!)

| Time (UTC) | Event | Notes |
|------------|-------|-------|
| 15:59:34 | 🔄 Resume detected | `[CHECKPOINT] Loading mid_epoch_004_001258.pt` |
| 15:59:36 | Epoch 4 training resumed | Batch 1203, `global_step 2245` |
| 16:01:08 | Validation started (clean) | Batch 796 (no duplicates this time) |
| 16:08:31+ | Training progressing | Batch 1203→1209→1215... |
| 17:34:58+ | Validation ongoing | Batch 444/3088 (current as of this analysis) |

**Mystery**: How did training resume WITHOUT using `schedule-training` action?

---

## 🤔 Critical Questions & Answers

### Q1: Why didn't timeout guard fire at 23h?

**Root Cause**: Timeout guard only checks **between epochs**, NOT during validation!

**Evidence from code** (`src/brain_brr/train/loop.py`):

```python
# Line 265: Check before starting epoch
if timeout_guard.check():
    logger.warning("[TIMEOUT] Wall-clock limit approaching...")
    # Save timeout_exit.pt and exit

# ... training happens ...
# ... validation happens ... <- NO TIMEOUT CHECK HERE!

# Line 505: Check after validation completes
if timeout_guard.check():
    logger.warning("[TIMEOUT] Wall-clock limit approaching...")
    # Save timeout_exit.pt and exit
```

**Timeline reconstruction**:
- **T=22h 50m**: Start of Epoch 4 training → timeout guard checked → **not triggered yet** (still 10 min margin)
- **T=23h 10m**: Epoch 4 training finishes, validation starts → **no check during validation!**
- **T=24h 00m**: Still in validation (batch ~1204/3088, 39% through) → **Modal kills the process**

**The Bug**: Validation runs for **~6-7 hours** per epoch, but timeout guard doesn't check during this phase!

### Q2: How did training auto-resume without scheduled function?

**Theory 1: Modal's Built-in Retry Mechanism**

Modal has platform-level retry logic for failed functions. When a function times out or raises an exception, Modal may automatically retry it based on:
- Function configuration (e.g., `retries` parameter)
- Platform defaults for detached runs
- Error type (timeout vs exception)

**Checking our code** (`deploy/modal/app.py:702-712`):
```python
@app.function(
    gpu="A100-80GB",
    timeout=86400,  # 24h timeout
    # NO explicit 'retries' parameter = Modal uses defaults
    volumes={...},
    memory=98304,
    cpu=24,
)
def train(...):
    ...
```

**Result**: We don't explicitly set `retries=0`, so Modal may be retrying failed runs!

**Theory 2: Scheduled Function Was Already Running**

Possible the user had deployed the scheduled function earlier and forgot?

**Check**: `modal app list` shows only 1 app (`ap-yxXph1bCmDI8tBMFB5CL3V`) with status "ephemeral (detached)" → This suggests manual run, not scheduled.

**Most Likely Answer**: **Modal's default retry mechanism** restarted the failed training run automatically.

### Q3: What's the difference between manual retry vs scheduled auto-restart?

| Aspect | Manual Run + Modal Retry | Scheduled `train_auto_restart` |
|--------|--------------------------|-------------------------------|
| **Trigger** | Reactive (after failure) | Proactive (`modal.Period(hours=23)`) |
| **Timing** | Immediate after failure | Every 23h from start (regardless of failure) |
| **Guarantee** | **Not guaranteed** (depends on Modal retry policy) | **Guaranteed** (explicit schedule) |
| **Overlap protection** | None (could run multiple instances) | **Yes** (`max_containers=1`) |
| **Timeout behavior** | Hits 24h hard limit first | Exits gracefully at 23h via timeout guard |
| **Control** | User has no control over retries | User controls schedule explicitly |

**Key Insight**: We got lucky this time, but **Modal retry is NOT reliable** for hands-free 100-epoch training!

### Q4: Why were there duplicate validation batches?

**Log evidence**:
```
16:40:02 - [VAL HEARTBEAT] Batch 1132/3088
16:41:41 - [VAL HEARTBEAT] Batch 0/3088      <- Second validation started!
16:42:03 - [VAL HEARTBEAT] Batch 1150/3088   <- First validation continues
16:43:42 - [VAL HEARTBEAT] Batch 17/3088     <- Second validation continues
```

**Theory**: Modal retry mechanism started the new training process BEFORE the old one fully died, causing temporary concurrent execution.

**Why it happened**:
1. Modal sends SIGTERM at 24h (`16:49:36`)
2. Process doesn't exit immediately (validation loop doesn't check cancellation)
3. Modal starts retry while old process still logging
4. Both processes write to same log stream briefly
5. Old process finally dies after ~11 minutes

**Evidence for this**:
- Cancellation signal at `16:49:36`
- Old validation logs continue until `17:22:37` (11 minutes later!)
- New validation starts cleanly at `16:01:08` (next session)

---

## 🐛 Bugs Identified

### Bug #1: Timeout Guard Doesn't Check During Validation ⚠️ HIGH PRIORITY

**Severity**: HIGH
**Impact**: Training hits 24h hard limit instead of graceful 23h exit

**Current behavior**:
- Timeout check at **start of epoch**
- Timeout check **after validation completes**
- **NO check during validation** (which can take 6-7 hours!)

**Fix needed**:
Add timeout check **inside validation loop** (`src/brain_brr/train/val_step.py`):

```python
def validate_epoch(..., timeout_guard=None):
    for batch_idx, batch in enumerate(val_loader):
        # Add this check every N batches
        if timeout_guard and batch_idx % 100 == 0:
            if timeout_guard.check():
                logger.warning("[TIMEOUT] Validation interrupted by timeout guard")
                # Save what we have so far and exit
                break

        # ... rest of validation ...
```

**Alternative fix** (simpler):
Add timeout check in validation heartbeat logger (every ~10 batches):

```python
if batch_idx % 10 == 0:  # Heartbeat logging
    logger.info(f"[VAL HEARTBEAT] Batch {batch_idx}/{total_batches}")

    # Add timeout check here
    if timeout_guard and timeout_guard.check():
        logger.warning("[TIMEOUT] Validation interrupted by timeout guard")
        raise TimeoutError("Wall-clock timeout during validation")
```

### Bug #2: Validation Loop Ignores Cancellation Signals 🟡 MEDIUM PRIORITY

**Severity**: MEDIUM
**Impact**: Process continues 11 minutes after SIGTERM, wastes compute

**Current behavior**:
- SIGTERM sent at `16:49:36`
- Validation continues until `17:22:37` (11 min later!)
- Signal handlers exist but validation loop doesn't check them

**Fix needed**:
Add signal flag check in validation loop:

```python
# In loop.py, track cancellation flag
class CancellationFlag:
    def __init__(self):
        self.cancelled = False

    def set(self):
        self.cancelled = True

cancellation_flag = CancellationFlag()

def signal_handler(sig, frame):
    logger.warning(f"[SIGNAL] Received {sig}, setting cancellation flag")
    cancellation_flag.set()

# In val_step.py
def validate_epoch(..., cancellation_flag=None):
    for batch_idx, batch in enumerate(val_loader):
        if cancellation_flag and cancellation_flag.cancelled:
            logger.warning("[SIGNAL] Validation cancelled by signal")
            break
        # ... rest of validation ...
```

### Bug #3: No Explicit Retry Control for Modal Function 🟢 LOW PRIORITY

**Severity**: LOW
**Impact**: Unpredictable retry behavior, but worked in our favor this time

**Current behavior**:
- No `retries` parameter set → Modal uses platform defaults
- We got lucky with auto-retry, but it's not guaranteed

**Fix needed** (for clarity, not urgency):
Explicitly set retry policy in `deploy/modal/app.py`:

```python
@app.function(
    gpu="A100-80GB",
    timeout=86400,
    retries=0,  # Explicitly disable retries (use schedule-training instead)
    # OR
    retries=modal.Retries(max_retries=1, initial_delay=60),  # One retry with delay
    volumes={...},
)
def train(...):
    ...
```

**Recommendation**: Set `retries=0` for manual runs, rely on `schedule-training` for auto-restart.

---

## ✅ What Worked Correctly

1. **Checkpoint system**: `mid_epoch_004_001258.pt` saved 20 minutes before crash ✅
2. **Resume logic**: Picked up from mid-epoch checkpoint on retry ✅
3. **StatefulDataLoader**: Restored exact batch position (1258→1203 after adjustment) ✅
4. **Atomic saves**: Checkpoint not corrupted despite hard kill ✅
5. **Modal retry**: Automatically restarted training (even though unintended) ✅

---

## 📋 Recommendations & Action Plan

### Immediate Actions (Do Right Now)

1. ✅ **Let current validation finish** (~5 hours remaining)
2. ✅ **Deploy scheduled auto-restart** after Epoch 4 checkpoints:
   ```bash
   modal deploy deploy/modal/app.py
   modal run --detach deploy/modal/app.py --action schedule-training \
     --config configs/modal/train_bimamba.yaml
   ```
3. ✅ **Monitor for proper 23h exits** in future runs

### Short-Term Fixes (v4.0.1 - Next Week)

1. **Add timeout check during validation** (Bug #1)
   - Priority: HIGH
   - Time: 2 hours
   - Location: `src/brain_brr/train/val_step.py`

2. **Add cancellation flag to validation** (Bug #2)
   - Priority: MEDIUM
   - Time: 1 hour
   - Location: `src/brain_brr/train/val_step.py` + `loop.py`

3. **Explicit retry policy** (Bug #3)
   - Priority: LOW
   - Time: 15 minutes
   - Location: `deploy/modal/app.py:702`

### Long-Term Improvements (v4.1.0 - Future)

1. **Validation optimization**: 3088 batches × 2 min/batch = 6.2 hours is SLOW
   - Consider larger batch sizes for validation
   - Parallelize validation across multiple GPUs
   - Use subset sampling for quick validation checks

2. **Smarter timeout guard**: Predictive timeout based on epoch progress
   - Calculate "can we finish this epoch in remaining time?"
   - Skip validation if it would exceed timeout

3. **Better Modal integration**: Custom health checks and graceful shutdown hooks

---

## 📚 Documentation Updates Needed

1. **Update `docs/05-training/modal.md`**:
   - Add warning about validation timeout gap
   - Clarify difference between manual retry vs scheduled restart
   - Document Modal's retry behavior

2. **Update `docs/05-training/checkpoint-strategy.md`**:
   - Add note about mid-validation interruptions
   - Document timeout guard limitations

3. **Update `CLAUDE.md`**:
   - Add to troubleshooting: "Training hit 24h limit instead of 23h"
   - Solution: "Validation was longer than safety margin"

---

## 🎯 Key Takeaways

### What We Learned

1. **Timeout guard is not perfect**: Gaps during validation phase
2. **Modal has hidden retry logic**: Can help or hurt depending on situation
3. **Scheduled restart is superior**: Proactive vs reactive, guaranteed vs probabilistic
4. **Validation is expensive**: 6-7 hours per epoch is a bottleneck

### Why This Happened

- Validation takes 6-7 hours
- Timeout guard only checks before/after epochs
- Epoch 4 validation started at T=23h 10m
- Modal killed at T=24h 00m while still in validation (39% through)
- Timeout guard never had a chance to trigger

### Why We're Not Panicked

- Checkpoint saved 20 min before crash
- Only 26 batches to replay (~30 min)
- Training resumed automatically (lucky!)
- Current validation is healthy (no more duplicates)
- We learned a lot about Modal's behavior

---

## 🔗 Related Files & Cross-References

**Code files**:
- `src/brain_brr/train/timeout_guard.py` - TimeoutGuard class
- `src/brain_brr/train/loop.py:265,505` - Timeout check locations
- `src/brain_brr/train/val_step.py` - Validation loop (needs timeout check)
- `deploy/modal/app.py:702-712` - Train function definition

**Documentation**:
- `docs/05-training/modal.md` - Modal training guide
- `docs/05-training/checkpoint-strategy.md` - Checkpoint behavior
- `docs/05-training/resume.md` - Resume mechanics

**Issues to file**:
- [ ] GitHub Issue: "Timeout guard doesn't check during validation"
- [ ] GitHub Issue: "Validation loop ignores SIGTERM signal"
- [ ] GitHub Issue: "Document Modal's retry behavior"

---

**Next Steps**:
1. Cross-reference this document with another AI agent for accuracy
2. File GitHub issues for the 3 bugs
3. Implement Bug #1 fix in v4.0.1
4. Deploy scheduled auto-restart for current training run
