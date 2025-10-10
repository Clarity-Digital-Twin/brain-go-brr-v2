# Modal Auto-Restart Strategy for Continuous Training

**Status:** Research & Planning Document
**Created:** October 10, 2025
**Purpose:** Understand and implement seamless training restarts on Modal to achieve multi-day training runs

---

## Table of Contents

1. [What Are Cron Jobs? (N00b Primer)](#what-are-cron-jobs-n00b-primer)
2. [The Problem We're Solving](#the-problem-were-solving)
3. [Modal's Timeout Limits (The Reality)](#modals-timeout-limits-the-reality)
4. [Solution Options (Research Findings)](#solution-options-research-findings)
5. [Recommended Strategy](#recommended-strategy)
6. [Implementation Plan](#implementation-plan)
7. [Alternative: External Cron (Not Recommended)](#alternative-external-cron-not-recommended)
8. [Cost Analysis](#cost-analysis)
9. [Decision Matrix](#decision-matrix)

---

## What Are Cron Jobs? (N00b Primer)

**Cron** = A time-based job scheduler in Unix/Linux systems.

**Think of it like an alarm clock for computers:**
- You set a schedule (e.g., "every day at 8am" or "every 24 hours")
- The system runs your command/script automatically when the time hits
- It keeps running forever until you stop it

**Example Cron Syntax:**
```bash
0 8 * * 1        # Every Monday at 8am UTC
0 6 * * *        # Every day at 6am
*/24 * * * *     # Every 24 hours
```

**Why would we use this for training?**
- Modal has a **24-hour hard limit** per function call
- We need **~12 days** for 20 epochs (~14 hours/epoch)
- **Solution**: Automatically restart training every 23 hours to stay under the limit!

---

## The Problem We're Solving

### Current Situation

**From your Modal logs (Epoch 2 in progress):**
```
Epoch 1: ~16.7 hours (Oct 9 09:23 → Oct 10 02:02)
Epoch 2: ~12 hours estimated
Average: ~14 hours/epoch
```

**Math:**
```
20 epochs × 14 hours = 280 hours = 11.7 days
Modal limit: 24 hours per run
```

**Problem:** We need to **manually restart ~12 times** to complete training! 😫

**Current workflow:**
1. Training runs for 23 hours
2. Timeout guard triggers → saves checkpoint → exits cleanly ✅
3. **MANUAL STEP:** You run `modal run --detach deploy/modal/app.py --action train --config ... --resume true`
4. Repeat 11 more times over 2 weeks 😴

**What we want:**
- **Zero manual intervention** - training restarts automatically
- **No downtime** - next run starts immediately after timeout
- **Seamless resume** - picks up from checkpoint without data loss

---

## Modal's Timeout Limits (The Reality)

### Hard Limits (October 2025)

From Modal's official documentation:

```python
# All Modal Function executions:
default_timeout = 300  # 5 minutes
min_timeout = 1        # 1 second
max_timeout = 86400    # 24 hours (HARD LIMIT)
```

**There is NO WAY to exceed 24 hours in a single Modal function call.**

### Our Current Implementation

**File:** `deploy/modal/app.py:700-710`
```python
@app.function(
    gpu="A100-80GB",
    timeout=86400,  # 24 hours max (Modal limit) ← WE'RE HERE
    # ... other config
)
def train(...):
    # Wall-clock timeout guard at 23h
    os.environ["BGB_WALL_CLOCK_LIMIT_S"] = "82800"  # 23 hours
```

**Good news:** We already have graceful exit at 23h! ✅
**Problem:** No automatic restart after exit ❌

---

## Solution Options (Research Findings)

### Option 1: Modal Retries (BEST FOR OUR USE CASE)

**What:** Modal's built-in retry mechanism automatically restarts functions on timeout/failure.

**How it works:**
```python
@app.function(
    gpu="A100-80GB",
    timeout=86400,  # 24h per attempt
    retries=modal.Retries(
        max_retries=12,           # Try up to 12 times
        initial_delay=0.0,        # Start immediately (no backoff)
        backoff_coefficient=1.0,  # No exponential backoff
    ),
    # CRITICAL: Each retry gets a FRESH container
)
def train(config_path: str, resume: bool = True):
    # On retry, Modal calls this function AGAIN
    # Our code loads last.pt and resumes ✅
    ...
```

**What happens:**
1. Training runs for 23h → timeout guard triggers → saves `last.pt` → exits (code 0)
2. Modal sees "function finished" → waits `initial_delay` (0s) → **calls `train()` AGAIN** 🚀
3. `--resume true` flag loads `last.pt` → continues from Epoch 3, Batch 80
4. Repeat 12 times → ~12 days of continuous training!

**Pros:**
- ✅ Built into Modal (no external services)
- ✅ Zero manual intervention
- ✅ Immediate restart (0s delay)
- ✅ Uses existing checkpoint system
- ✅ Simple to implement (~10 lines of code)

**Cons:**
- ⚠️ Retries are meant for **failures**, not **intentional restarts**
- ⚠️ Our wall-clock timeout returns **exit code 0** (success), not failure
- ❌ **Modal Retries only trigger on ERRORS, not clean exits!**

**CRITICAL ISSUE:** Modal's retry mechanism is designed for **fault tolerance**, not **continuous execution**. When our training exits cleanly (exit code 0) after 23h, Modal considers this a **success** and does **NOT retry**.

**Workaround:** We'd need to **exit with error code** after timeout, which is hacky and breaks semantics.

**Verdict:** ❌ **Not suitable** for our use case (retries don't work on clean exits)

---

### Option 2: Modal Cron (PERIODIC SCHEDULING)

**What:** Modal's built-in cron scheduler runs functions on a fixed schedule.

**How it works:**
```python
@app.function(
    gpu="A100-80GB",
    timeout=86400,
    schedule=modal.Cron("0 */23 * * *"),  # Every 23 hours
    volumes={"/results": results_volume},
)
def train_scheduled():
    """Runs every 23 hours automatically."""
    # Always resume from last.pt
    train(config_path="configs/modal/train_bimamba.yaml", resume=True)
```

**What happens:**
1. You deploy the scheduled function ONCE
2. Modal runs it every 23 hours **forever** (until you stop it)
3. Each run loads `last.pt` and continues training

**Pros:**
- ✅ Built into Modal (no external services)
- ✅ Zero manual intervention after initial deploy
- ✅ Runs indefinitely (great for 100-epoch training)
- ✅ Simple cron syntax

**Cons:**
- ⚠️ **Fixed schedule** - doesn't adapt if an epoch finishes early/late
- ⚠️ **Overlap risk** - if one run takes >23h, next run starts anyway (DISASTER)
- ⚠️ **Wasted time** - if epoch finishes at 18h, waits 5h for next scheduled run
- ❌ **Can't pause/resume mid-schedule** - schedules can't be paused in Modal

**Example Problem:**
```
Day 1, 00:00 → Start training (Epoch 1-2)
Day 1, 23:00 → Cron triggers → Start training (Epoch 3-4)
Day 2, 22:00 → Cron triggers → Start training (Epoch 5-6)
Day 3, 21:00 → Cron triggers → Start training (Epoch 7-8)
```

Looks good, BUT:
```
Day 5, 18:00 → Epoch 12 finishes early (in 15 hours)
Day 5, 18:00-23:00 → GPU IDLE, wasting $5/hour × 5h = $25 ❌
Day 5, 23:00 → Cron triggers → Start Epoch 13
```

**Verdict:** ⚠️ **Works but wasteful** - pays for idle GPU time between early completion and next cron

---

### Option 3: Modal Period (INTERVAL-BASED)

**What:** Run function every N hours/days (similar to cron but simpler).

**How it works:**
```python
@app.function(
    gpu="A100-80GB",
    timeout=86400,
    schedule=modal.Period(hours=23),  # Every 23 hours
    volumes={"/results": results_volume},
)
def train_periodic():
    """Runs every 23 hours automatically."""
    train(config_path="configs/modal/train_bimamba.yaml", resume=True)
```

**Pros:**
- ✅ Simpler than cron syntax
- ✅ Same benefits as Modal Cron

**Cons:**
- ⚠️ **Resets on redeploy** - if you redeploy code, timer resets to 0
- ⚠️ Same overlap/waste issues as cron
- ❌ Less flexible than cron (can't specify exact times)

**Verdict:** ⚠️ **Works but worse than cron** - resets on redeploy is annoying

---

### Option 4: External Cron (GitHub Actions / AWS Lambda)

**What:** Run a cron job on an external service (GitHub Actions, AWS EventBridge, etc.) that calls Modal.

**How it works:**
```yaml
# .github/workflows/modal-training-cron.yml
name: Modal Training Auto-Restart
on:
  schedule:
    - cron: '0 */23 * * *'  # Every 23 hours
jobs:
  restart-training:
    runs-on: ubuntu-latest
    steps:
      - name: Install Modal
        run: pip install modal
      - name: Resume training
        env:
          MODAL_TOKEN_ID: ${{ secrets.MODAL_TOKEN_ID }}
          MODAL_TOKEN_SECRET: ${{ secrets.MODAL_TOKEN_SECRET }}
        run: |
          modal run --detach deploy/modal/app.py --action train \
            --config configs/modal/train_bimamba.yaml --resume true
```

**Pros:**
- ✅ Complete control over scheduling logic
- ✅ Can add notifications (Slack/email when epoch completes)
- ✅ Can add smart logic (check if training is done before restarting)
- ✅ Free on GitHub Actions (2000 minutes/month)

**Cons:**
- ❌ Requires external service (GitHub or AWS)
- ❌ More complex setup (API tokens, workflows)
- ❌ Depends on external service uptime
- ❌ Same overlap/waste issues as Modal cron

**Verdict:** ⚠️ **Overkill** - only use if you need advanced logic (notifications, smart checks)

---

### Option 5: Manual Script with Loop (SIMPLEST)

**What:** A simple Python script that runs locally and restarts training in a loop.

**How it works:**
```python
# scripts/continuous_training.py
import time
import subprocess

config = "configs/modal/train_bimamba.yaml"
max_runs = 15  # ~15 days of training

for run in range(max_runs):
    print(f"Starting training run {run+1}/{max_runs}...")

    # Start Modal training with --resume
    result = subprocess.run([
        "modal", "run", "--detach",
        "deploy/modal/app.py",
        "--action", "train",
        "--config", config,
        "--resume", "true"
    ])

    if result.returncode != 0:
        print("Training failed! Stopping.")
        break

    # Wait 23 hours for this run to complete
    print("Waiting 23 hours for training to complete...")
    time.sleep(23 * 3600)  # 23 hours

    # Add 5 minutes buffer for checkpoint saving
    time.sleep(300)

print("Training complete or stopped!")
```

**Usage:**
```bash
# Run in tmux so it survives terminal closure
tmux new -s modal-training
python scripts/continuous_training.py
# Detach: Ctrl+B then D
```

**Pros:**
- ✅ **Simplest implementation** (15 lines of Python)
- ✅ Complete control over timing
- ✅ Can add smart checks (read logs, check for completion)
- ✅ No external dependencies (runs on your laptop)
- ✅ Easy to stop/modify mid-run

**Cons:**
- ❌ Requires your laptop to stay on for 12 days
- ❌ Breaks if laptop sleeps/loses internet
- ❌ Not "serverless" (you're the server!)

**Verdict:** ⚠️ **Good for testing, bad for production** - too fragile for 12-day runs

---

## Recommended Strategy

### 🏆 Winner: Hybrid Approach (Modal Cron + Smart Checks)

**Strategy:** Use Modal Cron for scheduling, but make the function **smart enough to not overlap**.

**Implementation:**
```python
import fcntl
from pathlib import Path

@app.function(
    gpu="A100-80GB",
    timeout=86400,
    schedule=modal.Cron("0 */23 * * *"),  # Every 23 hours
    volumes={"/results": results_volume},
)
def train_auto_restart():
    """Auto-restart training with overlap protection."""

    # LOCK FILE: Prevent multiple instances from running
    lock_file = Path("/results/.training_lock")

    try:
        # Try to acquire exclusive lock (non-blocking)
        with open(lock_file, 'w') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            # Lock acquired! Start training
            logger.info("Lock acquired - starting training...")
            train(
                config_path="configs/modal/train_bimamba.yaml",
                resume=True  # Always resume from last.pt
            )

    except BlockingIOError:
        # Another instance is running - skip this cron execution
        logger.info("Training already in progress - skipping this cron run")
        return

    finally:
        # Release lock when done
        if lock_file.exists():
            lock_file.unlink()
```

**How it works:**
1. Cron runs every 23 hours
2. Each run tries to acquire a **lock file**
3. If lock is free → start training
4. If lock is held → **skip this run** (previous training still going)
5. When training finishes → lock is released → next cron can start

**Why this is best:**
- ✅ No overlap (lock prevents it)
- ✅ Automatic restarts (cron handles it)
- ✅ Minimal wasted time (~5 min between runs)
- ✅ Works for variable epoch times (if epoch is 18h, next cron starts 5h later)
- ✅ All within Modal (no external services)

**Potential issue:** Lock files on Modal volumes need testing - not 100% sure this works!

---

## Implementation Plan

### Phase 1: Research & Validation (30 minutes)

**Tasks:**
1. Test if Modal Volumes support file locking (fcntl)
2. Verify cron syntax works with Modal
3. Test manual `--resume` workflow end-to-end

**Validation Script:**
```python
# Test file locking on Modal volume
@app.function(volumes={"/results": results_volume})
def test_file_lock():
    import fcntl
    from pathlib import Path

    lock = Path("/results/test.lock")
    with open(lock, 'w') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        print("✅ Lock acquired!")
        time.sleep(5)
    print("✅ Lock released!")
```

---

### Phase 2: Implement Auto-Restart Function (1 hour)

**File:** `deploy/modal/app.py`

**Add new function:**
```python
@app.function(
    gpu="A100-80GB",
    timeout=86400,  # 24h timeout
    schedule=modal.Cron("0 */23 * * *"),  # Every 23 hours
    volumes={
        "/data": data_mount,
        "/results": results_volume,
    },
    memory=98304,
    cpu=24,
)
def train_auto_restart(config_path: str = "configs/modal/train_bimamba.yaml"):
    """Auto-restart training every 23 hours with overlap protection.

    This function is scheduled to run every 23 hours via Modal Cron.
    It automatically resumes from the last checkpoint, enabling multi-day
    training runs that exceed Modal's 24-hour function timeout limit.

    Overlap Protection:
        Uses a lock file to prevent multiple instances from running
        simultaneously. If a previous training run is still in progress,
        this cron execution will skip gracefully.

    Args:
        config_path: Path to training config (default: BiMamba2 full training)
    """
    import fcntl
    from pathlib import Path

    lock_file = Path("/results/.training_lock")

    try:
        # Try to acquire exclusive lock (non-blocking)
        with open(lock_file, 'w') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            logger.info("=" * 60)
            logger.info("[AUTO-RESTART] Lock acquired - starting training...")
            logger.info(f"[AUTO-RESTART] Config: {config_path}")
            logger.info(f"[AUTO-RESTART] Resume: True (always)")
            logger.info("=" * 60)

            # Run training with resume=True
            checkpoint_path = train(
                config_path=config_path,
                resume=True,  # Always resume from last.pt
            )

            logger.info("=" * 60)
            logger.info(f"[AUTO-RESTART] Training completed: {checkpoint_path}")
            logger.info("[AUTO-RESTART] Next restart in ~23 hours (or when cron triggers)")
            logger.info("=" * 60)

    except BlockingIOError:
        # Another instance is running - skip this cron execution
        logger.info("=" * 60)
        logger.info("[AUTO-RESTART] Training already in progress")
        logger.info("[AUTO-RESTART] Skipping this cron run to prevent overlap")
        logger.info("[AUTO-RESTART] Next cron will try again in 23 hours")
        logger.info("=" * 60)
        return None

    finally:
        # Release lock when done (even if error)
        if lock_file.exists():
            try:
                lock_file.unlink()
                logger.info("[AUTO-RESTART] Lock released")
            except Exception as e:
                logger.warning(f"[AUTO-RESTART] Failed to release lock: {e}")
```

**Update `main()` entrypoint:**
```python
@app.local_entrypoint()
def main(action: str = "train", ...):
    """Modal deployment entrypoint."""

    # ... existing actions ...

    elif action == "schedule-training":
        # Deploy the auto-restart function (runs forever)
        logger.info("🕐 Deploying scheduled auto-restart training...")
        logger.info("This will run every 23 hours until you stop it.")
        logger.info("")
        logger.info("To stop: modal app stop brain-go-brr-v2")
        logger.info("To view logs: modal app logs brain-go-brr-v2")
        logger.info("")

        # Deploy (no .remote() needed - schedule handles it)
        logger.info("✅ Scheduled training deployed!")
        logger.info("Training will restart automatically every 23 hours.")
```

---

### Phase 3: Testing (2 hours)

**Test 1: Lock mechanism works**
```bash
# Terminal 1
modal run deploy/modal/app.py::test_file_lock

# Terminal 2 (while Terminal 1 is sleeping)
modal run deploy/modal/app.py::test_file_lock
# Should print "Training already in progress"
```

**Test 2: Manual restart workflow**
```bash
# Start training normally
modal run --detach deploy/modal/app.py --action train \
  --config configs/modal/smoke_bimamba.yaml

# Wait for it to finish (or timeout)
modal app logs brain-go-brr-v2

# Manually restart with --resume
modal run --detach deploy/modal/app.py --action train \
  --config configs/modal/smoke_bimamba.yaml --resume true

# Verify it loaded last.pt and continued from last epoch/batch
```

**Test 3: Cron deployment** (if lock test passes)
```bash
# Deploy scheduled training
modal deploy deploy/modal/app.py

# Check it's running
modal app list
# Should show: brain-go-brr-v2 (scheduled)

# View logs
modal app logs brain-go-brr-v2 --follow

# Stop it
modal app stop brain-go-brr-v2
```

---

### Phase 4: Production Deployment (10 minutes)

**Once testing passes:**

```bash
# Deploy scheduled auto-restart training
modal deploy deploy/modal/app.py

# Verify schedule is active
modal app list | grep brain-go-brr-v2

# Monitor first few restarts
modal app logs brain-go-brr-v2 --follow

# Let it run for 12 days! 🚀
```

**To stop training:**
```bash
modal app stop brain-go-brr-v2
```

**To check status:**
```bash
# List all apps
modal app list

# View logs
modal app logs brain-go-brr-v2

# Check schedule status
# (schedules can't be paused, must stop app entirely)
```

---

## Alternative: External Cron (Not Recommended)

**If Modal cron doesn't work**, fallback to GitHub Actions.

**File:** `.github/workflows/modal-training-restart.yml`
```yaml
name: Modal Training Auto-Restart

on:
  schedule:
    - cron: '0 */23 * * *'  # Every 23 hours
  workflow_dispatch:  # Allow manual trigger

jobs:
  restart-training:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install Modal
        run: pip install modal

      - name: Resume Modal training
        env:
          MODAL_TOKEN_ID: ${{ secrets.MODAL_TOKEN_ID }}
          MODAL_TOKEN_SECRET: ${{ secrets.MODAL_TOKEN_SECRET }}
        run: |
          modal run --detach deploy/modal/app.py --action train \
            --config configs/modal/train_bimamba.yaml --resume true
```

**Setup:**
1. Go to GitHub repo → Settings → Secrets
2. Add `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` (from `modal token new`)
3. Push workflow file to `.github/workflows/`
4. GitHub will run it every 23 hours automatically

**Pros:**
- ✅ Works even if Modal cron/locks don't work
- ✅ Can add Slack notifications

**Cons:**
- ❌ Requires GitHub Actions
- ❌ More setup complexity

---

## Cost Analysis

### Current Manual Workflow

```
Cost per hour: ~$4/hour (A100-80GB)
Hours per epoch: 14 hours
Epochs needed: 20

Manual restart overhead:
  - 12 restarts × 5 minutes = 1 hour wasted on manual work

Total cost: 20 epochs × 14h × $4/h = $1,120
Wasted on manual work: $0 (GPU not running during restart)
```

### With Auto-Restart (Modal Cron)

```
Cost per hour: ~$4/hour (A100-80GB)
Hours per epoch: 14 hours
Epochs needed: 20

Auto-restart overhead:
  - 12 restarts × 5 minutes = 1 hour of idle GPU time
  - Cron might trigger early if epoch finishes in 12h instead of 14h
  - Worst case: 11h avg epoch + 23h cron = 12h wasted per restart

Best case cost: $1,120 (same as manual, but ZERO human time!)
Worst case cost: $1,120 + (12 restarts × 12h × $4/h) = $1,696

Realistic case: $1,120 + (12 restarts × 2h × $4/h) = $1,216
```

**Verdict:** Auto-restart costs **~$100 more** in worst case, but saves **12× manual interventions**! 🎉

---

## Decision Matrix

| Strategy | Setup Time | Reliability | Cost | Human Effort | Verdict |
|----------|-----------|-------------|------|--------------|---------|
| **Manual restarts** | 0 min | ⭐⭐⭐⭐ | $1,120 | 🔴 High (12 restarts) | ❌ Tedious |
| **Modal Retries** | 10 min | N/A | N/A | ✅ Zero | ❌ **Doesn't work** (retries don't trigger on clean exit) |
| **Modal Cron** | 1 hour | ⭐⭐⭐ | $1,200 | ✅ Zero | ⚠️ Wasteful on variable epoch times |
| **Modal Cron + Lock** | 2 hours | ⭐⭐⭐⭐ | $1,200 | ✅ Zero | ✅ **BEST** (if locks work) |
| **GitHub Actions** | 3 hours | ⭐⭐⭐⭐ | $1,200 | ✅ Zero | ⚠️ Fallback if locks don't work |
| **Local script** | 15 min | ⭐⭐ | $1,120 | 🟡 Medium (laptop must stay on) | ❌ Too fragile |

---

## Final Recommendation

### 🏆 Phase 1: Test Modal Cron + File Locks

**Why:**
- Native to Modal (no external services)
- Minimal overhead (~5 min between runs)
- Prevents overlap with lock mechanism

**Timeline:**
- Research: 30 min (test locks)
- Implementation: 1 hour (add scheduled function)
- Testing: 2 hours (verify restart workflow)
- **Total: ~4 hours of work**

**If locks work:** Deploy and enjoy hands-free training! 🚀

---

### 🥈 Phase 2 Fallback: GitHub Actions

**If file locks don't work on Modal volumes:**

**Why:**
- Proven technology (GitHub Actions is reliable)
- Free for 2000 minutes/month
- Can add notifications

**Timeline:**
- Setup: 1 hour (create workflow, add secrets)
- Testing: 1 hour (verify manual triggers)
- **Total: ~2 hours of work**

---

## Next Steps (When You're Ready to Implement)

1. **Approval:** Confirm you want to proceed with Modal Cron + Lock approach
2. **Test locks:** Run lock test script on Modal volume
3. **Implement:** Add `train_auto_restart()` function to `deploy/modal/app.py`
4. **Test:** Verify overlap protection works
5. **Deploy:** `modal deploy deploy/modal/app.py` and let it run! 🎉

---

## Questions to Answer Before Implementation

1. **Do Modal Volumes support fcntl file locking?** → TEST THIS FIRST
2. **What happens if cron triggers while prev run is saving checkpoint?** → Lock should prevent
3. **How do we stop training mid-run if we want to?** → `modal app stop brain-go-brr-v2`
4. **What if training completes before 100 epochs (early stopping)?** → Cron keeps running, but training exits early (wastes cron cycles, but harmless)

---

**STATUS:** Ready for your approval to proceed! Let me know which approach you want, brah! 🚀
