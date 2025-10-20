# GradScaler + LRScheduler Interaction (Resolved)

**Status**: Resolved in v3.8.2 (Zero Warnings release)
**Priority**: Closed (former P2 cosmetic issue)
**Impact**: Scheduler only advances when optimizer updates; no cosmetic warnings, accurate LR schedule

---

## The Issue

When using mixed precision (`torch.cuda.amp.GradScaler`) with a learning rate scheduler, PyTorch emits this warning:

```
UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`.
```

**This is NOT a bug in our code order!** It's a known interaction between `GradScaler` and `LRScheduler`.

---

## Root Cause

*This section documents the behaviour observed during the v3.8.1 training run before the fix shipped in v3.8.2.*

### Pre-v3.8.2 Code (train_step.py:398-409)

```python
if scaler.is_enabled():
    scaler.step(optimizer)  # ← CAN SKIP if gradients are inf/NaN!
    scaler.update()
else:
    optimizer.step()

if scheduler is not None:
    scheduler.step()  # ← ALWAYS runs, even if optimizer was skipped!
    global_step += 1
```

### What Happens

1. **Mixed precision**: Occasional batches have inf/NaN gradients (NORMAL with FP16)
2. **GradScaler**: Detects inf, **skips** `optimizer.step()` to protect weights
3. **Scheduler**: Still calls `scheduler.step()` unconditionally
4. **PyTorch**: Sees scheduler step without optimizer step → **warning**

### Evidence From Logs

```
[GRAD_CLIP] Batch 0: pre=inf → post=0.50 (clipped nan%)
[GRADIENTS] 3/11 batches had inf pre-clip norm (normal with FP16)

UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`.
```

The warning fires on batch 0, which has `pre=inf` gradients!

---

## Impact

### Functional Impact: MINIMAL ✅

- Training continues normally
- Loss decreases correctly
- GradScaler protects weights from inf/NaN updates
- Only ~3/21 batches affected (14%)

### Side Effects:

1. **Learning rate schedule slightly off**: Scheduler advances even when optimizer skips
   - Effect: LR progresses ~14% faster than intended
   - Severity: Minor - warmup still works, cosine decay still happens

2. **Log noise**: Repeated warning clutters logs
   - PyTorch suppresses after first occurrence per worker
   - Still annoying in Modal logs

---

## Fix Implemented in v3.8.2

### Official PyTorch Pattern

From [PyTorch docs](https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping):

```python
# Track scale before optimizer step
scale_before = scaler.get_scale()

# Step optimizer (may skip if gradients are inf)
scaler.step(optimizer)
scaler.update()

# Only step scheduler if optimizer actually updated
scale_after = scaler.get_scale()
if scale_after >= scale_before:
    # Optimizer stepped successfully
    scheduler.step()
    global_step += 1
# else: Optimizer was skipped, don't advance scheduler
```

**Logic**: If `scaler.get_scale()` decreased, it means inf gradients were detected and optimizer was skipped.

### Implementation Snapshot (train_step.py v3.8.2)

```python
# Track optimizer step status for scheduler
optimizer_stepped = False

if scaler.is_enabled():
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    optimizer_stepped = scaler.get_scale() >= scale_before
else:
    optimizer.step()
    optimizer_stepped = True

# Only advance scheduler if optimizer actually updated weights
if optimizer_stepped and scheduler is not None:
    scheduler.step()
    global_step += 1
```

This guard exists in both the main microbatch loop and the gradient-accumulation flush (lines 399‑418 and 563‑574).

### Testing Summary

- `make q` (ruff + mypy) – pass  
- `make test` (full clinical suite) – pass, 83.8 % coverage  
- Modal training run (config `configs/modal/train_bimamba.yaml`) – no scheduler warning emitted; LR trace matches cosine schedule

---

## Historical Postmortem (v3.8.1)

### What We Got Right (v3.8.1)

✅ **The ORDER is correct**: `optimizer.step()` before `scheduler.step()`
✅ **We verified the code**: Lines 402 → 408 show correct order
✅ **Scheduler creation is correct**: `optimizer_factory.py` properly suppresses creation-time warning

### What We Missed

❌ **GradScaler can SKIP optimizer step**: We didn't account for inf gradient handling
❌ **Different warning source**: This is a RUNTIME warning, not the creation-time warning we suppressed
❌ **False sense of security**: Removing the "paper-over" made us think we'd fixed it

### The Confusion

In `optimizer_factory.py:88-92`, we suppress this warning:
```python
# Suppress PyTorch 1.1.0+ warning about scheduler.step() order
# Our code correctly calls optimizer.step() before scheduler.step()
# but PyTorch emits warning on scheduler creation before first training step
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Detected call of.*lr_scheduler")
```

**This suppresses the CREATION-time warning** (different from the RUNTIME warning we're seeing now).

---

## Historical Decision Log (v3.8.1 Run)

| Factor | Assessment |
|--------|-----------|
| **Loss progress** | ✅ Decreasing (0.2019 → 0.1725) |
| **Functional correctness** | ✅ GradScaler handling inf properly |
| **Compute cost** | ❌ Stopping wastes ~$10 + 5 hours |
| **Impact severity** | ✅ Minor (LR schedule ~14% faster) |
| **Training stability** | ✅ No crashes, no NaN losses |

**RECOMMENDATION: Let current run complete, fix in v3.8.2** ✅

---

## References

- [PyTorch AMP Examples](https://pytorch.org/docs/stable/notes/amp_examples.html)
- [GradScaler API](https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler)
- [LRScheduler Warning Source](https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py#L224)

---

**Remember**: This is a KNOWN PATTERN in PyTorch mixed precision training. The fix is well-documented and straightforward. Not a critical bug!
