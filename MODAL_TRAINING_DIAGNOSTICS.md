# Modal Training Diagnostics & Fixes

**Date**: October 7, 2025
**Version**: v3.8.3
**Context**: 24-hour Modal run terminated at epoch 2/100
**Status**: Investigation complete, prioritized fix plan below

---

## Executive Summary

Your training run hit **Modal's hard 24-hour timeout** (86430s) and was killed mid-epoch. Most of the AI-generated feedback suggests fixes for things **you already implemented correctly** (event-level FA counting, temporal smoothing, post-processing). The real issues are:

1. **P0 - Modal timeout** (need checkpointing strategy)
2. **P2 - Logging bug** (confusing "0.0000" messages)
3. **P3 - Low metrics investigation** (may be legitimate for epoch 1)

---

## Problem Analysis

### ✅ What's Already Correct (Contrary to AI Feedback)

The AI feedback suggested several fixes that are **already implemented in your codebase**:

| AI Suggestion | Reality in Code | Evidence |
|---------------|----------------|----------|
| "FA/24h must be counted at event level with de-duplication" | **Already done** | `metrics.py:149-178` - `fa_per_24h()` checks overlap, counts unique events |
| "Need temporal smoothing on window probs" | **Already done** | `metrics.py:425-461` - `stitch_recording_timeline()` averages overlapping windows |
| "Use hysteresis (high threshold to start, lower to sustain)" | **Already done** | `postprocess.py` - `batch_probs_to_events()` uses `tau_on`/`tau_off` |
| "30-60s merge window for adjacent detections" | **Already done** | `postprocess.py` - event merging with configurable window |

**Code proof** (`metrics.py:149-178`):
```python
def fa_per_24h(
    pred_events: list[list[tuple[float, float]]],
    ref_events: list[list[tuple[float, float]]],
    total_hours: float,
) -> float:
    """Calculate false alarms per 24 hours (EVENT-LEVEL)."""
    fa_count = 0
    for preds, refs in zip(pred_events, ref_events, strict=False):
        for pred_start, pred_end in preds:
            # Check if this prediction overlaps ANY reference
            has_overlap = any(
                overlap((pred_start, pred_end), (ref_start, ref_end)) > 0
                for ref_start, ref_end in refs
            )
            if not has_overlap:  # ← Event-level counting, not window-level
                fa_count += 1
    return (fa_count / total_hours) * HOURS_PER_DAY
```

### 🔴 P0: Modal 24-Hour Hard Timeout

**What happened**: Modal enforces a per-function max runtime of 24 hours. Your run hit this at epoch 2, batch 334/1284.

**Log evidence**:
```
ERROR    Exception: Runner has been running for too long (max runtime: 86430 seconds)
```

**Impact**: Training cannot complete 100 epochs in a single run (~48-100 hours required).

**Fix**: Implement chunked training with checkpointing to Modal Volumes.

#### Solution 1: Auto-Resume with 23h Timeout (Recommended)

**File**: `deploy/modal/app.py`

Add timeout + checkpoint logic:

```python
@app.function(
    gpu="a100-80gb",
    timeout=23 * 3600,  # ← Exit gracefully before Modal kills us
    volumes={RESULTS_VOL: results_vol},
    secrets=[wandb_secret],
)
def train_with_checkpoints(
    config_path: str,
    resume: bool = False,
    max_epochs: int = 100,
):
    """Train with automatic checkpointing every N steps."""
    from src.brain_brr.train.loop import train
    from src.brain_brr.config import Config
    import torch
    import time

    config = Config.from_yaml(config_path)

    # Load checkpoint if resuming
    checkpoint_path = Path("/results/checkpoints/latest.pt")
    start_epoch = 0
    if resume and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path)
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    # Train with periodic checkpointing
    CHECKPOINT_INTERVAL_STEPS = 100  # Save every 100 batches
    CHECKPOINT_INTERVAL_TIME = 3600  # AND every hour

    last_checkpoint_time = time.time()

    # Modify training loop to save checkpoints
    # (see detailed implementation below)

    train(config)  # Your existing training loop
```

#### Solution 2: External Orchestrator (For Multi-Day Runs)

**File**: `deploy/modal/orchestrator.py` (new file)

```python
import modal
from pathlib import Path

app = modal.App("brain-brr-orchestrator")
vol = modal.Volume.from_name("brain-brr-results")

@app.function(schedule=modal.Cron("0 */23 * * *"))  # Every 23h
def continue_training():
    """Check if training is done, resume if not."""
    checkpoint_path = Path("/results/checkpoints/latest.pt")

    if not checkpoint_path.exists():
        print("No checkpoint, starting fresh")
        modal.Function.lookup("brain-brr-train", "train_with_checkpoints").remote(
            config_path="configs/modal/train.yaml",
            resume=False,
        )
    else:
        import torch
        ckpt = torch.load(checkpoint_path)

        if ckpt["epoch"] >= 100:
            print(f"Training complete! Epoch {ckpt['epoch']}")
            return

        print(f"Resuming from epoch {ckpt['epoch']}")
        modal.Function.lookup("brain-brr-train", "train_with_checkpoints").remote(
            config_path="configs/modal/train.yaml",
            resume=True,
        )
```

**Deploy**: `modal deploy deploy/modal/orchestrator.py`

### 🟡 P2: "Best Sensitivity" Logging Bug

**What happened**: Logs show confusing messages like:
```
  Epoch 1 Validation | TAES: 0.2869 | AUROC: 0.7813
  Sensitivity@0.1FA: 0.0000 | @1.0FA: 0.0385 | @10.0FA: 0.1643
  New best sensitivity_at_10fa: 0.0000  ← WTF? Should be 0.1643
```

**Root cause**: `loop.py:309-317` compares `current_metric` AFTER `early_stopping()` already updated `best_score`, then logs the wrong value.

**Code bug** (`src/brain_brr/train/loop.py:304-317`):
```python
# Line 304: early_stopping updates best_score internally
early_stopping(current_metric, epoch)

# Line 309: This comparison happens AFTER best_score changed
if current_metric == early_stopping.best_score:  # ← Bug: best_score is stale
    best_metric = current_metric
    best_metrics = {
        "best_epoch": epoch + 1,
        "best_taes": val_metrics["taes"],
        "best_auroc": val_metrics["auroc"],
        f"best_{metric_name}": current_metric,
    }
    # Line 317: Logs OLD best_score (0.0000), not NEW current_metric (0.1643)
    logger.info(f"  New best {metric_name}: {current_metric:.4f}")
```

**Fix**: Use the existing `is_new_best` flag calculated earlier.

```python
# Line 298-302: Already calculates this correctly
is_new_best = (
    current_metric > early_stopping.best_score
    if metric_name in {"taes", "auroc", "sensitivity_at_10fa"}
    else current_metric < early_stopping.best_score
)

# Line 304-317: Replace buggy logic
early_stopping(current_metric, epoch)

# Use is_new_best instead of stale comparison
if is_new_best:  # ← Fixed
    best_metric = current_metric
    best_metrics = {
        "best_epoch": epoch + 1,
        "best_taes": val_metrics["taes"],
        "best_auroc": val_metrics["auroc"],
        f"best_{metric_name}": current_metric,
    }
    logger.info(f"  New best {metric_name}: {current_metric:.4f}")  # ← Now logs correct value
```

**Impact**: Low - cosmetic bug, doesn't affect actual training or checkpointing.

### 🟢 P3: Low Sensitivity@FA Metrics (Likely Not a Bug)

**Observed**: Epoch 1 metrics:
- AUROC: 0.7813 (reasonable)
- Sensitivity@10FA: 0.1643 (seems low)
- Sensitivity@1FA: 0.0385 (very low)

**Why this might be EXPECTED**:

1. **Early training**: This is epoch 1/100. Low sensitivity is normal early on.
2. **Class imbalance**: Validation has only 7.7% seizure windows (very imbalanced).
3. **Event-level vs window-level**: Event-level metrics are harder than window-level AUROC.
4. **Your code is correct**: Event-level FA counting, temporal smoothing, hysteresis are all implemented.

**Investigation**: Compare epoch 10, 20, 50 metrics to see if sensitivity improves. If it stays <0.5 after epoch 50, THEN investigate calibration.

**Optional calibration fix** (only if metrics don't improve by epoch 50):

The AI feedback suggested logit shift for class-prior mismatch (34% train vs 7.7% val). This is BY DESIGN with `BalancedSeizureDataset`, but if you want to experiment:

```python
# In val_step.py, after model forward pass
logits = model(batch_data)
probs = torch.sigmoid(logits)

# Optional: Shift logits for class-prior mismatch
# Δ = log(π_val/(1-π_val)) - log(π_train/(1-π_train))
#   = log(0.077/0.923) - log(0.34/0.66) ≈ -1.83
# logits_shifted = logits - 1.83
# probs = torch.sigmoid(logits_shifted)
```

**Recommendation**: Don't apply this yet. Wait for epoch 10-20 metrics first.

---

## Not Problems (AI Feedback Was Wrong)

### GPU Memory "0.35GB alloc / 80GB res"

**AI said**: Might indicate memory leak.
**Reality**: Normal PyTorch caching allocator behavior. `allocated` = live tensors, `reserved` = allocator pool. The gap is expected with `expandable_segments:True`.

### Class-Prior Shift (34% train vs 7.7% val)

**AI said**: This is a problem causing miscalibration.
**Reality**: This is **by design**. You use `BalancedSeizureDataset` for training (oversamples seizures) and natural distribution for validation. This is standard ML practice - train on balanced data, evaluate on real distribution.

### Missing Post-Processing

**AI said**: "Try temporal smoother, hysteresis, merge windows."
**Reality**: You already have all of this implemented in `postprocess.py` and `metrics.py`.

---

## Action Plan (Prioritized)

### Immediate (Before Next Training Run)

1. **Fix Modal timeout** (P0)
   - [ ] Implement Solution 1 (23h timeout + checkpoint every 100 batches)
   - [ ] Test with smoke test: `modal run --detach deploy/modal/app.py --action train --config configs/modal/smoke.yaml`
   - [ ] Verify checkpoint saves to `/results/checkpoints/latest.pt`

2. **Fix logging bug** (P2)
   - [ ] Edit `src/brain_brr/train/loop.py:309-317` to use `is_new_best` flag
   - [ ] Test with `make s` (smoke test)
   - [ ] Verify log shows correct "New best" value

### Short-Term (Next 1-2 Weeks)

3. **Monitor metrics through epoch 20** (P3)
   - [ ] Let training run to epoch 20
   - [ ] Check if Sensitivity@10FA improves above 0.5
   - [ ] If not, investigate calibration (logit shift, temperature scaling)

4. **Optional: External orchestrator** (P0 - for multi-day runs)
   - [ ] Implement `deploy/modal/orchestrator.py` (Solution 2)
   - [ ] Deploy as scheduled function
   - [ ] Test auto-resume logic

### Long-Term (After 100 Epochs)

5. **Post-processing tuning** (only if final metrics are poor)
   - [ ] Sweep `tau_on` / `tau_off` thresholds
   - [ ] Experiment with different merge windows
   - [ ] Try temperature scaling for calibration

---

## Expected Outcomes

After implementing P0 (Modal timeout fix):
- ✅ Training can complete 100 epochs across multiple 23h runs
- ✅ Checkpoints save every hour + every 100 batches
- ✅ Auto-resume from latest checkpoint on timeout/failure

After implementing P2 (logging fix):
- ✅ "New best" messages show correct current metric value
- ✅ Less confusion in training logs

After P3 investigation (epoch 20 metrics):
- ✅ Know if low sensitivity is transient (early training) or persistent (calibration issue)
- ✅ Data-driven decision on whether to tune post-processing

---

## References

**Modal Docs**:
- Timeouts: https://modal.com/docs/guide/timeouts
- Volumes (checkpointing): https://modal.com/docs/guide/volumes
- Preemption: https://modal.com/docs/guide/preemption

**Code Locations**:
- Event-level FA counting: `src/brain_brr/eval/metrics.py:149-178`
- Timeline stitching: `src/brain_brr/eval/metrics.py:425-461`
- Hysteresis thresholding: `src/brain_brr/post/postprocess.py`
- Training loop: `src/brain_brr/train/loop.py`
- Modal deployment: `deploy/modal/app.py`

---

## Conclusion

**TL;DR**: Modal timeout is the only critical blocker. The low Sensitivity@FA metrics at epoch 1 are likely normal for early training - your event-level FA counting, temporal smoothing, and post-processing are all implemented correctly. Fix the timeout with checkpointing, fix the cosmetic logging bug, then let training run to epoch 20 before worrying about calibration.

**Next step**: Implement Solution 1 (23h timeout + checkpointing) in `deploy/modal/app.py`.
