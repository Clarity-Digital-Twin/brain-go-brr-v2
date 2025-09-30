# Gradient Behavior Guide

**Status**: Reference guide for interpreting training logs
**Last Updated**: 2025-09-30
**Applies to**: PyTorch 2.5.0 + mamba-ssm 2.2.5 stack

## TL;DR - What's Normal vs Alarming

| Log Message | Status | Action |
|-------------|--------|--------|
| `Large grad norm: 2.52e+00 (clipped to 0.1)` | ✅ Normal | None - system working |
| `Sanitized NaN gradients at batch X` | ⚠️ Handled | Monitor - should decrease |
| `NaN loss detected at batch X` | 🚨 Problem | Check debug logs |
| `Non-finite minimum in edge features` | 🚨 Critical | Training will crash |

## Understanding "Large Grad Norm" Messages

### What You See
```
[2025-09-30 06:36:07.002][src.brain_brr.train.loop][DEBUG] Large grad norm at batch 17: 2.52e+00 (clipped to 0.1)
```

### What It Means

**This is EXPECTED in early training!**

1. **Detected**: Gradient norm (2.52) exceeds threshold (0.1)
2. **Clipped**: Automatically scaled down to 0.1
3. **Safe**: Training continues without NaN
4. **Temporary**: Frequency decreases as model learns

### Why It Happens

**Early Training (batches 0-100):**
- Random weight initialization → large errors
- Large errors → large gradients
- Gradient clipping prevents explosion
- **Frequency**: Every few batches (normal)

**Mid Training (batches 100-1000):**
- Weights stabilizing
- Gradients shrinking
- **Frequency**: Occasionally (good progress)

**Late Training (batches 1000+):**
- Model converged
- Gradients small and stable
- **Frequency**: Rare (excellent)

### Configuration

**Local (RTX 4090):**
```yaml
training:
  gradient_clip: 0.1  # Conservative for 24GB VRAM
```

**Modal (A100-80GB):**
```yaml
training:
  gradient_clip: 0.5  # Less aggressive for larger memory
```

### When to Worry

**Normal Pattern:**
```
Batch 7:  grad_norm = 1.59 ✅
Batch 12: grad_norm = 1.53 ✅
Batch 20: grad_norm = 1.42 ✅  <- Decreasing trend
Batch 50: grad_norm = 0.85 ✅  <- Stabilizing
```

**Problem Pattern:**
```
Batch 50:  grad_norm = 1.5 ⚠️
Batch 100: grad_norm = 3.2 ⚠️
Batch 150: grad_norm = 8.7 🚨  <- Increasing trend
Batch 175: NaN loss!    🚨  <- Explosion
```

If grad norms **increase** after batch 100, investigate:
- Learning rate too high
- Data quality issues
- Architecture instability

## The 3-Tier Protection System

### Tier 1: Gradient Clipping (First Line)
```python
# loop.py - Prevents gradient explosion
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
if grad_norm > gradient_clip * 10:
    logger.debug(f"Large grad norm: {grad_norm:.2e}")
```

**Triggers**: When grad_norm > 1.0 (for gradient_clip=0.1)
**Effect**: Scales gradients down, logs message
**Result**: Training continues safely

### Tier 2: Gradient Sanitization (Backup)
```python
# Enabled with BGB_SANITIZE_GRADS=1
if not torch.isfinite(param.grad).all():
    param.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
```

**Triggers**: When NaN/Inf slips through clipping
**Effect**: Replaces with zeros
**Result**: Prevents optimizer corruption

### Tier 3: Loss Validation (Last Resort)
```python
# Catches NaN losses before backward pass
if not torch.isfinite(loss):
    consecutive_nans += 1
    if consecutive_nans >= 50:
        raise RuntimeError("Model corrupted")
```

**Triggers**: When loss becomes NaN
**Effect**: Skips update, terminates if persistent
**Result**: Prevents wasted GPU hours

## Debug Verbosity Control

**Reduce noise if it's distracting:**
```bash
# Default (shows large grads)
export BGB_NAN_DEBUG=1

# Quieter (only shows actual NaN)
unset BGB_NAN_DEBUG

# Ultra-verbose (for debugging)
export BGB_NAN_DEBUG=1
export BGB_DEBUG_FINITE=1
```

**Log levels explained:**
- `DEBUG`: Large grad norms (informational)
- `WARNING`: Sanitized gradients (handled issue)
- `ERROR`: NaN loss (training problem)
- `CRITICAL`: Consecutive NaNs (training terminated)

## PyTorch 2.5.0 Changes

**Why you see more messages after upgrade:**

1. **Different initialization seeds**
   - PyTorch 2.5 uses updated random number generation
   - Slightly different initial weights → different gradient landscape

2. **CUDA kernel changes**
   - Optimized matmul/conv implementations
   - Numerically equivalent but different floating-point rounding

3. **mamba-ssm 2.2.2 → 2.2.5**
   - Fixed int64 indexing bug (critical for A100)
   - Slight numerical differences in state-space computations

**None of these are bugs!** Just slightly different paths through the same loss landscape.

## Monitoring Checklist

**Healthy Training:**
- ✅ Grad norm messages in first 50 batches
- ✅ Frequency decreases over time
- ✅ Loss is finite and decreasing
- ✅ No "NaN loss detected" warnings

**Unhealthy Training:**
- 🚨 Grad norm messages increasing after batch 100
- 🚨 "Sanitized NaN gradients" every batch
- 🚨 "NaN loss detected" warnings
- 🚨 Loss not decreasing

## Quick Commands

**Check current training status:**
```bash
# View live training
tmux attach -t train

# Check for actual problems (not normal clipping)
grep -E "(NaN loss|Non-finite minimum)" /tmp/local-train-*.log

# Count grad norm messages (should decrease)
grep "Large grad norm" /tmp/local-train-*.log | wc -l
```

**Monitor gradient health:**
```bash
# Extract grad norms over time
grep "Large grad norm at batch" /tmp/local-train-*.log | \
  awk '{print $9, $11}' | \
  sed 's/://g'
# Should show decreasing trend: 2.52 → 1.84 → 1.42 → ...
```

## References

- NaN protection system: `docs/10-final-refactor-NAN/NAN_CANONICAL.md`
- Training config: `configs/local/train.yaml`
- Implementation: `src/brain_brr/train/loop.py:861-927`

---

**Key Takeaway**: Seeing "Large grad norm" messages in early training is **expected behavior**, not a bug. The protection system is working correctly. Only worry if the pattern persists or worsens after 100+ batches.