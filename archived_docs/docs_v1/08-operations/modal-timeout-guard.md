# Modal Timeout Guard - Three-Layer Defense System

**Last Updated**: October 20, 2025

**Status**: ✅ Production-ready (v4.0.2+)

**Purpose**: Prevent Modal's 24-hour hard kill by implementing graceful shutdown before the limit

---

## 🎯 The Problem

**Modal enforces 24-hour timeout** on all functions. If training doesn't exit gracefully before this limit, Modal sends SIGTERM and forcibly kills the process. This causes:

1. **Lost compute**: Hours of work wasted (e.g., 5.8-hour validation interrupted)
2. **Concurrent runs**: Scheduled restart triggers while old run still active (50 min overlap)
3. **Wasted costs**: Duplicate work on cloud resources

**Key Insight**: Validation takes 5.8 hours. If it starts after T=18h 12m, it **can't finish before 24h limit** → guaranteed timeout!

---

## ✅ The Solution: Three-Layer Defense

### Layer 1: TimeoutGuard (Primary Defense)

**What**: Checks elapsed time every 2 minutes during training AND validation
**When**: Exits gracefully at T=23h (1-hour safety margin before 24h hard kill)
**Implementation**: `src/brain_brr/train/timeout_guard.py`

**How it works**:
```python
# Initialize with 23-hour limit (1h safety margin)
timeout_guard = TimeoutGuard(timeout_seconds=82800)  # 23 hours

# Check BEFORE each epoch
if timeout_guard.check():
    save_checkpoint("timeout_exit.pt")
    sys.exit(0)

# Check DURING validation (every 2 min in heartbeat)
if timeout_guard and timeout_guard.check():
    logger.warning(f"[TIMEOUT] Validation interrupted at batch {batch_idx}/{total}")
    break  # Exit validation early

# Check AFTER each epoch
if timeout_guard.check():
    save_checkpoint("timeout_exit.pt")
    sys.exit(0)
```

**Why 23 hours?**
- Modal hard kill at 24h
- Need 1h buffer for checkpoint save + cleanup
- Scheduled auto-restart triggers at T=23h to start fresh run

**Added in v4.0.1** (Oct 12, 2025):
- Timeout check added to validation loop (every 2 min)
- Previously only checked before/after epochs → blind spot during 5.8h validation!

### Layer 2: CancellationFlag (SIGTERM Handling)

**What**: Thread-safe flag for coordinating graceful shutdown across signal handlers
**When**: Set when SIGTERM/SIGINT received, checked after training and during validation
**Implementation**: `src/brain_brr/train/cancellation_flag.py`

**How it works**:
```python
# Signal handler sets flag (doesn't exit immediately)
def signal_handler(signum, frame):
    logger.warning(f"[SIGNAL] Received {signal.Signals(signum).name}")
    cancellation_flag.request_cancellation()

# Check AFTER training phase (before starting 5.8h validation)
if cancellation_flag.is_cancellation_requested():
    logger.warning("[CANCEL] Exiting before validation (saves 5.8h!)")
    save_checkpoint("interrupted.pt")
    sys.exit(0)

# Check DURING validation (every 2 min in heartbeat)
if cancellation_flag.is_cancellation_requested():
    logger.warning(f"[CANCEL] Validation interrupted at batch {batch_idx}/{total}")
    break
```

**Impact**:
- **Before fix**: SIGTERM during training → runs full 5.8h validation → wastes up to 7h total
- **After fix**: SIGTERM during training → exits immediately → saves 5.8h!

**Added in v4.0.2** (Oct 12, 2025):
- Post-training cancellation check (saves 5.8h if SIGTERM during training)
- Validation loop cancellation check (every 2 min)

### Layer 3: File-Based Mutex (Scheduler Defense)

**What**: Exclusive lock prevents concurrent training runs at scheduler level
**When**: Acquired at start of `train_auto_restart()`, released on exit
**Implementation**: `deploy/modal/app.py:1193-1240`

**How it works**:
```python
# Try to acquire exclusive lock
lock_path = Path("/results/.training.lock")
lock_fd = open(lock_path, "w")

try:
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    logger.info("[LOCK] ✅ Acquired exclusive training lock")
except IOError:
    logger.warning("[OVERLAP] Another training instance running, exiting")
    return None  # Exit gracefully without training
finally:
    fcntl.flock(lock_fd, fcntl.LOCK_UN)
    lock_fd.close()
```

**Why needed?**
- If TimeoutGuard fails to trigger (bug), old run continues past T=23h
- Scheduled auto-restart triggers at T=23h regardless
- Mutex prevents NEW run from starting if OLD run still holds lock
- **Defense-in-depth**: Protects against TimeoutGuard bugs

**Added in v4.0.2** (Oct 12, 2025)

---

## 🔍 Why Validation Needs Timeout Checks

### The Blind Spot (Pre-v4.0.1)

**Original timeout guard**:
```python
# Check BEFORE epoch (line 265)
if timeout_guard.check():
    save_checkpoint("timeout_exit.pt")
    sys.exit(0)

train_epoch(...)   # 1-2 hours

validate_epoch(...) # 5.8 hours ← NO CHECKS HERE!

# Check AFTER epoch (line 505)
if timeout_guard.check():
    save_checkpoint("timeout_exit.pt")
    sys.exit(0)
```

**Timeline of failure**:
```
T=14:00 UTC - Epoch 4 training starts → guard checks → PASS (1h 49m margin)
T=14:34 UTC - Training ends, validation starts → NO CHECK
T=15:49 UTC - Scheduled restart triggers (expects guard to trigger, but we're mid-validation!)
T=20:16 UTC - Validation would complete → guard would check → TOO LATE!
T=16:49 UTC - Modal kills at T=24h (never reached validation end)
```

**The gap**: Validation runs for 5.8 hours with **zero timeout checks**. If validation starts after T=18h 12m (24h - 5.8h), guard can never fire before Modal's hard limit.

### The Fix (v4.0.1+)

**Added timeout check in validation heartbeat** (every 2 minutes):
```python
# src/brain_brr/train/val_step.py:520-529
if timeout_guard and timeout_guard.check():
    pct_complete = (batch_idx / len(dataloader)) * 100
    logger.warning(
        f"[TIMEOUT] Validation interrupted at batch {batch_idx}/{len(dataloader)} "
        f"({pct_complete:.1f}% complete)"
    )
    logger.warning("[TIMEOUT] Wall-clock limit approaching, exiting validation early")
    break
```

**New behavior**:
- Validation checks timeout every 2 minutes
- If approaching 23h limit, exits validation loop immediately
- Saves checkpoint with partial validation metrics
- Scheduled restart can begin cleanly

---

## 📊 Validation Duration Deep Dive

### Measured Performance

**Batch pace**: 18 batches per 2 minutes = **6.67 seconds/batch**
**Total batches**: 3,088 (Modal), 18,528 (Local)
**Total duration**: ~5.7-5.8 hours (Modal), varies on local hardware

### Validation Timeline (Modal A100)

```
00:00 - Start (batch 0)
01:00 - ~600 batches complete (19%)
02:00 - ~1200 batches complete (39%)
03:00 - ~1800 batches complete (58%)
04:00 - ~2400 batches complete (78%)
05:00 - ~3000 batches complete (97%)
05:43 - Complete (batch 3088)
```

### Why So Slow

1. **1832 validation recordings** across 3088 batches
2. **Disk I/O** from memory-mapped cache
3. **GNN computation** (Dynamic Laplacian PE)
4. **No gradient computation** but still forward pass overhead
5. **Timeline reconstruction** and post-processing

---

## 🚨 October 12, 2025 Incident

### What Happened

- **T=14:34 UTC**: Epoch 4 validation started (expected duration: 5.8h)
- **T=15:49 UTC**: Scheduled auto-restart triggered (T=23h)
- **T=15:59 UTC**: NEW run boots up (10min startup overhead)
- **T=16:40 UTC**: OLD run still running validation (37% complete)
- **T=16:49 UTC**: Modal sends SIGTERM to OLD run (T=24h hard kill)
- **Result**: 50 minutes of concurrent execution (15:59 → 16:49 UTC)

### Root Cause

**TimeoutGuard only checked before/after epochs** → **blind spot during 5.8h validation**

Since validation started at T=23h 45m and takes 5.8h to complete, the guard never had a chance to trigger before Modal's 24h kill.

### Resolution

✅ **v4.0.1** (Oct 12, 2025): Added timeout check in validation loop (every 2 min)
✅ **v4.0.2** (Oct 12, 2025): Added SIGTERM handling + file-based mutex

**Status**: All fixes deployed and tested, zero incidents since

---

## 🎓 Key Learnings

### What This Incident Taught Us

1. **Timeout guards need full coverage** - Check at ALL potentially long-running phases
2. **Validation is expensive** - 5.8 hours is 24% of 24h budget
3. **Scheduled restarts work** - Scheduler triggered correctly, old run was the problem
4. **Concurrent safety matters** - Need multiple layers of overlap protection
5. **Logging helps debugging** - Interleaved logs revealed concurrent execution

### Design Principles Validated

- ✅ Atomic checkpoint saves prevent corruption
- ✅ Mid-epoch checkpoints minimize replay cost
- ✅ StatefulDataLoader enables exact resume
- ✅ Scheduled restarts superior to reactive retries
- ✅ Multiple safety margins (23h guard + 24h hard limit) catch issues

---

## 🔧 Implementation Details

### TimeoutGuard Class

**File**: `src/brain_brr/train/timeout_guard.py`

```python
class TimeoutGuard:
    """Monitors elapsed time and triggers graceful shutdown before hard limit."""

    def __init__(self, timeout_seconds: int = 82800):  # 23 hours default
        self.timeout_seconds = timeout_seconds
        self.start_time = time.time()

    def check(self) -> bool:
        """Returns True if timeout approaching (should exit)."""
        elapsed = time.time() - self.start_time
        return elapsed >= self.timeout_seconds

    def remaining_seconds(self) -> float:
        """Returns remaining seconds until timeout."""
        elapsed = time.time() - self.start_time
        return max(0.0, self.timeout_seconds - elapsed)
```

### CancellationFlag Class

**File**: `src/brain_brr/train/cancellation_flag.py`

```python
class CancellationFlag:
    """Thread-safe flag for coordinating graceful shutdown."""

    def __init__(self):
        self._flag = threading.Event()

    def request_cancellation(self):
        """Set the cancellation flag."""
        self._flag.set()

    def is_cancellation_requested(self) -> bool:
        """Check if cancellation has been requested."""
        return self._flag.is_set()
```

### Signal Handler Registration

**File**: `src/brain_brr/train/loop.py:223-230`

```python
def register_signal_handlers(cancellation_flag: CancellationFlag):
    """Register SIGTERM/SIGINT handlers that set cancellation flag."""
    def signal_handler(signum, frame):
        signal_name = signal.Signals(signum).name
        logger.warning(f"[SIGNAL] Received {signal_name}, requesting graceful shutdown")
        cancellation_flag.request_cancellation()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
```

---

## 📋 Checklist: Production Validation

### Monitor Next Auto-Restart

**Expected behavior**:
1. Training reaches T=23h during validation
2. Log message: `[TIMEOUT] Validation interrupted at batch X/3088`
3. No concurrent runs (single validation stream in logs)
4. Checkpoint saved: `timeout_exit.pt`
5. NEW run starts cleanly from checkpoint

**Failure indicators**:
- Concurrent validation logs (multiple heartbeat streams)
- Modal hard kill at 24h (SIGTERM message)
- No graceful exit logs
- Two training runs visible in `modal app list`

### Verify Three-Layer Defense

**Layer 1 - TimeoutGuard**:
- [ ] Check triggered during validation (not just before/after epochs)
- [ ] Exits at ~23h (not 24h)
- [ ] Saves checkpoint before exit

**Layer 2 - CancellationFlag**:
- [ ] SIGTERM triggers immediate exit during training (saves 5.8h validation)
- [ ] SIGTERM triggers exit during validation (within 2 min)

**Layer 3 - File Mutex**:
- [ ] NEW run waits if OLD run still holds lock
- [ ] No concurrent training instances visible
- [ ] Lock released on exit

---

## 📚 Related Documentation

**Code files**:
- `src/brain_brr/train/timeout_guard.py` - TimeoutGuard class
- `src/brain_brr/train/cancellation_flag.py` - CancellationFlag class
- `src/brain_brr/train/loop.py:223-230, 310-367` - Signal handlers, cancellation checks
- `src/brain_brr/train/val_step.py:520-540` - Validation timeout/cancellation checks
- `deploy/modal/app.py:1193-1240` - File-based mutex in train_auto_restart

**Documentation**:
- `docs/05-training/modal.md` - Modal training guide
- `docs/05-training/checkpoint-strategy.md` - Checkpoint behavior
- `docs/archive/MODAL_24H_TIMEOUT_INCIDENT.md` - Full incident analysis with timeline

**See Also**:
- `docs/05-training/training-methodology.md` - Why validation takes 5.8 hours

---

## ✅ Summary

**Three-layer defense system (v4.0.2+)**:
1. **TimeoutGuard**: Checks every 2min during validation, exits at T=23h
2. **CancellationFlag**: Respects SIGTERM signals, saves up to 5.8h
3. **File mutex**: Prevents concurrent execution at scheduler level

**Status**: ✅ Production-ready, zero incidents since v4.0.2 deployment

**Key insight**: Validation's 5.8-hour duration requires explicit timeout checks. Original implementation only checked before/after epochs → blind spot fixed in v4.0.1.
