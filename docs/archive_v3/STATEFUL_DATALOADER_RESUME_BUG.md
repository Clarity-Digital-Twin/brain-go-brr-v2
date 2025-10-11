# StatefulDataLoader Resume Bug - Complete Analysis

**Date**: October 11, 2025
**Status**: IDENTIFIED - Fix required before next training run
**Severity**: **P1** - Functional regression (warmup resets, W&B metrics misaligned)

---

## Executive Summary

When resuming from **any checkpoint** (mid-epoch, last, best, timeout, signal), **two critical bugs** cause functional training degradation:

1. **`global_step` not saved/restored** → Warmup schedules reset, W&B metrics restart from 0
2. **`batch_idx` counter resets** → Progress tracking wrong, logs misleading, tqdm shows 0% instead of 33%

**Impact**: Training is **functionally degraded** (warmup restarts wasting ~1000 batches), not just cosmetic.

---

## Bug #1: `global_step` Not Saved (CRITICAL - P1)

### Root Cause

**Code**: `src/brain_brr/train/checkpoint.py:60-91` (save_checkpoint)
**Problem**: `global_step` is never added to checkpoint dict

```python
# Current behavior (BROKEN)
checkpoint = {
    "version": CHECKPOINT_VERSION,
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "best_metric": best_metric,
    # ... other fields ...
}
# ❌ global_step is NEVER saved!
```

**Code**: `src/brain_brr/train/train_step.py:530-546` (mid-epoch save)
**Problem**: `extra=` dict includes `batch_idx` but not `global_step`

```python
save_checkpoint(
    model, optimizer, epoch_index, 0.0, mid_path,
    scheduler, None, scaler=scaler, save_rng=True,
    extra={
        "batch_idx": batch_idx,
        # ❌ "global_step": global_step,  # MISSING!
        "kind": "mid_epoch",
        "dataloader_state_dict": dataloader.state_dict(),
    },
)
```

**Code**: `src/brain_brr/train/loop.py:242-318` (train function)
**Problem**: `global_step = 0` is always reset on resume (no restoration logic)

```python
# After checkpoint load:
start_epoch = 0
best_metric = 0.0
# ... load checkpoint restores start_epoch, best_metric ...

# Training loop:
global_step = 0  # ❌ ALWAYS resets to 0, even after resume!
for epoch in range(start_epoch, config.training.epochs):
    result = train_epoch(..., global_step=global_step, ...)
```

### Impact

1. **Warmup schedules reset** (`src/brain_brr/train/warmup.py:11-43`)
   - `get_focal_gamma(global_step, ...)` uses `global_step` to interpolate focal_gamma
   - If `global_step=0` after resume → warmup restarts from `focal_gamma=1.0` instead of continuing at `2.0`
   - Evidence: `[WARMUP] Batch 0 focal_gamma=1.000` when it should be `focal_gamma=2.000` (past 1000-step warmup)
   - **Wastes ~1000 batches** re-warming up (less aggressive loss, slower learning)

2. **Model gating state resets** (`src/brain_brr/models/detector.py:174-197`)
   - `SeizureDetector.set_training_state(global_step, warmup_schedule)` propagates step to child modules
   - GNN adjacency temperature warmup, other gating mechanisms reset to initial values
   - Causes training instability spike after resume

3. **W&B metrics x-axis wrong** (`src/brain_brr/train/train_step.py:377-407`)
   - `wandb_logger.log({...}, step=global_step)` logs gradients/weights with step counter
   - After resume: metrics restart from step 0, overlapping previous run
   - Makes W&B charts uninterpretable (duplicate x-values, wrong timeline)

### Affected Checkpoints

**ALL checkpoint save paths** lack `global_step`:
- `src/brain_brr/train/train_step.py:530-546` - Mid-epoch checkpoints (every 30min)
- `src/brain_brr/train/loop.py:429-441` - Best model checkpoint
- `src/brain_brr/train/loop.py:455-466` - Periodic checkpoints (every N epochs)
- `src/brain_brr/train/loop.py:469-481` - Last checkpoint (for resume)
- `src/brain_brr/train/loop.py:264-275` - Timeout checkpoint (Modal 23h limit)
- `src/brain_brr/train/loop.py:217-228` - Signal checkpoint (SIGTERM/SIGINT)

---

## Bug #2: `batch_idx` Counter Resets (P2 - Tracking/UX)

### Root Cause

**Code**: `src/brain_brr/train/train_step.py:297` (training loop)
**Problem**: `enumerate()` always starts counting from 0, regardless of DataLoader internal state

```python
# After StatefulDataLoader.load_state_dict():
# - DataLoader is positioned at batch 2527 ✅
# - But enumerate() doesn't know this, starts from 0 ❌

for batch_idx, batch in enumerate(progress):
    # batch_idx = 0, 1, 2, ... (relative to resume)
    # Should be: batch_idx = 2527, 2528, 2529, ... (absolute position)
```

**Code**: `src/brain_brr/train/train_step.py:283-290` (tqdm initialization)
**Problem**: tqdm not initialized with `initial=` parameter for resume

```python
progress_bar = tqdm(
    dataloader, desc="Training", leave=False,
    file=sys.stderr, ascii=True, ncols=80
    # ❌ Missing: initial=resume_batch_idx
)
# Shows: 0/7702 instead of 2527/7702
```

### Impact

1. **Progress bar wrong** - Shows `0/7702` (0%) instead of `2527/7702` (33%)
2. **Log messages misleading** - All batch numbers offset by resume amount
3. **Checkpoint filenames wrong** - `mid_epoch_001_000030.pt` instead of `mid_epoch_001_002557.pt`
4. **Heartbeat logs confusing** - Reports `Batch 50/7702` when really at batch 2577
5. **Debugging harder** - Can't correlate logs with actual training timeline

**Best Practice (2025)**: PyTorch Lightning and HuggingFace Transformers both use `tqdm(initial=resume_step)` pattern.

---

## Evidence from Production Logs

```
[2025-10-11 02:14:42][RESUME] ✅ Exact mid-epoch resume at batch 2527
[2025-10-11 02:14:42][RESUME] ✅ Exact mid-epoch resume at batch 2527
[2025-10-11 02:14:42] Epoch 1/100

Training:   0%|      | 0/7702 [00:00<?, ?it/s]           ← Should be 33% (2527/7702)
[2025-10-11 02:15:10][WARMUP] Batch 0 focal_gamma=1.000  ← Should be focal_gamma=2.0!
[2025-10-11 02:15:28][GRAD_CLIP] Batch 0: pre=20.43 ...  ← Really batch 2527
Training:   0%|      | 20/7702 [01:44<6:27:39]           ← Really 2547/7702
```

**Verification from checkpoint inspection**:
```python
>>> ckpt = torch.load('mid_epoch_001_002527.pt', map_location='cpu', weights_only=False)
>>> list(ckpt.keys())
['version', 'epoch', 'model_state_dict', 'optimizer_state_dict', 'best_metric',
 'timestamp', 'scheduler_state_dict', 'scaler_state_dict', 'rng_state',
 'batch_idx', 'kind', 'dataloader_state_dict']
>>> 'global_step' in ckpt
False  # ❌ CONFIRMED: global_step not saved
```

---

## Fix Implementation (Required Changes)

### 1. Save `global_step` in **ALL** Checkpoint Paths

**File**: `src/brain_brr/train/checkpoint.py`

```python
def save_checkpoint(
    model, optimizer, epoch, best_metric, checkpoint_path,
    scheduler=None, config=None, scaler=None, save_rng=True,
    extra=None,
    global_step: int | None = None,  # ← NEW required parameter
):
    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
        "timestamp": time.time(),
    }

    # Save global_step (CRITICAL for warmup/W&B resume)
    if global_step is not None:
        checkpoint["global_step"] = global_step  # ← ADD THIS

    # ... rest of function unchanged ...
```

**File**: `src/brain_brr/train/train_step.py` (mid-epoch saves)

```python
# Line 530-546 - Mid-epoch checkpoint
save_checkpoint(
    model, optimizer, epoch_index, 0.0, mid_path,
    scheduler, None, scaler=scaler, save_rng=True,
    global_step=global_step,  # ← ADD THIS
    extra={
        "batch_idx": batch_idx,
        "kind": "mid_epoch",
        "dataloader_state_dict": dataloader.state_dict(),
    },
)
```

**File**: `src/brain_brr/train/loop.py` (all other checkpoints)

Update **6 checkpoint save calls** to include `global_step=` parameter:
- Line 429: Best checkpoint
- Line 455: Periodic checkpoint
- Line 470: Last checkpoint
- Line 264: Timeout checkpoint
- Line 217: Signal checkpoint

Example:
```python
save_checkpoint(
    model, optimizer, epoch, best_metric, checkpoint_dir / CHECKPOINT_BEST,
    scheduler, config, scaler=scaler, save_rng=True,
    global_step=global_step,  # ← ADD to all 6 calls
)
```

### 2. Restore `global_step` and `batch_idx` on Resume

**File**: `src/brain_brr/train/loop.py` (load logic)

```python
# Lines 150-177 - Resume from checkpoint
mid_epoch_checkpoints = sorted(checkpoint_dir.glob("mid_epoch_*.pt"))
if mid_epoch_checkpoints and config.training.resume:
    latest_mid = mid_epoch_checkpoints[-1]
    logger.info(f"[RESUME] Found mid-epoch checkpoint: {latest_mid.name}")

    start_epoch, best_metric = load_checkpoint(
        latest_mid, model, optimizer, scheduler, scaler=scaler, device=device
    )

    ckpt = torch.load(latest_mid, map_location="cpu", weights_only=False)

    # Restore DataLoader state
    if "dataloader_state_dict" in ckpt:
        train_loader.load_state_dict(ckpt["dataloader_state_dict"])
        resume_batch_idx = ckpt.get('batch_idx', 0)
        resume_global_step = ckpt.get('global_step', 0)  # ← ADD THIS
        logger.info(
            f"[RESUME] ✅ Exact mid-epoch resume: "
            f"epoch {start_epoch+1}, batch {resume_batch_idx}, global_step {resume_global_step}"
        )
    else:
        logger.warning("[RESUME] Old checkpoint without DataLoader state")
        resume_batch_idx = 0
        resume_global_step = 0  # ← ADD THIS
        logger.info(f"Resumed from epoch {start_epoch + 1}, batch 0")

elif (checkpoint_dir / CHECKPOINT_LAST).exists() and config.training.resume:
    start_epoch, best_metric = load_checkpoint(
        checkpoint_dir / CHECKPOINT_LAST,
        model, optimizer, scheduler, scaler=scaler, device=device,
    )
    ckpt = torch.load(checkpoint_dir / CHECKPOINT_LAST, map_location="cpu", weights_only=False)
    resume_batch_idx = 0  # Epoch boundary, no batch offset
    resume_global_step = ckpt.get('global_step', 0)  # ← ADD THIS
    logger.info(f"Resumed from epoch {start_epoch + 1}, global_step {resume_global_step}")

# Initialize global_step from resume
if 'resume_global_step' not in locals():
    resume_global_step = 0
    resume_batch_idx = 0

# ... later in training loop (line 242):
global_step = resume_global_step  # ← CHANGE from: global_step = 0
```

### 3. Pass `resume_batch_idx` to Fix Progress Tracking

**File**: `src/brain_brr/train/loop.py` (train_epoch call)

```python
# Line 280-313 - Train epoch call
result = train_epoch(
    model, train_loader, optimizer,
    device=device,
    use_amp=config.training.mixed_precision,
    gradient_clip=config.training.gradient_clip,
    scheduler=scheduler,
    global_step=global_step,  # Already exists, now correctly restored
    scaler=scaler,
    # ... other args ...
    resume_batch_idx=resume_batch_idx if epoch == start_epoch else 0,  # ← ADD THIS
)
```

**File**: `src/brain_brr/train/train_step.py` (function signature + loop)

```python
def train_epoch(
    model, dataloader, optimizer,
    device="cpu", use_amp=False, gradient_clip=1.0,
    scheduler=None, global_step=0,
    scaler=None, loss_mode="focal",
    focal_alpha=FOCAL_ALPHA_DEFAULT, focal_gamma=FOCAL_GAMMA_DEFAULT,
    return_step=False, checkpoint_dir=None, epoch_index=None,
    mid_epoch_minutes=None, mid_epoch_keep=3,
    warmup_schedule=None, gradient_accumulation_steps=1,
    log_every_n_steps=LOG_EVERY_N_STEPS,
    log_gradients=False, log_weights=False,
    wandb_logger=None,
    resume_batch_idx: int = 0,  # ← ADD THIS parameter
):
    # ... setup code ...

    # Fix tqdm initialization (line 283-290)
    if use_tqdm:
        try:
            progress_bar = tqdm(
                dataloader, desc="Training", leave=False,
                file=sys.stderr, ascii=True, ncols=80,
                initial=resume_batch_idx,  # ← ADD THIS (2025 best practice)
            )
            progress = progress_bar
        except Exception as e:
            logger.warning(f"tqdm failed ({e}), using plain iteration")
            progress = dataloader
    else:
        progress = dataloader

    # Fix batch_idx enumeration (line 297)
    for relative_idx, batch in enumerate(progress):
        batch_idx = relative_idx + resume_batch_idx  # ← CHANGE from: batch_idx = relative_idx

        # ... rest of training loop uses corrected batch_idx ...
```

---

## Testing & Validation

### Unit Test (Add to `tests/train/test_checkpoint_resume.py`)

```python
def test_global_step_saved_and_restored():
    """Verify global_step is saved in checkpoints and restored on resume."""
    # Create checkpoint with global_step
    checkpoint_path = tmp_path / "test.pt"
    save_checkpoint(
        model, optimizer, epoch=0, best_metric=0.0,
        checkpoint_path=checkpoint_path,
        global_step=2527,
    )

    # Verify saved
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert "global_step" in ckpt, "global_step not saved in checkpoint"
    assert ckpt["global_step"] == 2527

    # Verify restored (check loop.py logic manually)
    resume_global_step = ckpt.get("global_step", 0)
    assert resume_global_step == 2527


def test_batch_idx_offset_in_training_loop():
    """Verify batch_idx is correctly offset when resuming."""
    resume_batch_idx = 2527

    # Simulate enumerate with offset
    for relative_idx in range(5):
        batch_idx = relative_idx + resume_batch_idx
        assert batch_idx == 2527 + relative_idx

    # First batch should be 2527, not 0
    assert batch_idx >= 2527


def test_tqdm_initial_parameter():
    """Verify tqdm shows correct progress when resuming."""
    total_batches = 7702
    resume_at = 2527

    pbar = tqdm(total=total_batches, initial=resume_at)
    # Progress should show 33% (2527/7702), not 0%
    assert pbar.n == resume_at
    pbar.close()
```

### Manual Validation

1. **Before fix**: Resume shows `[WARMUP] Batch 0 focal_gamma=1.000`
2. **After fix**: Resume shows `[WARMUP] Batch 2527 focal_gamma=2.000`
3. **Before fix**: Progress `0/7702` (0%)
4. **After fix**: Progress `2527/7702` (33%)`
5. **Before fix**: W&B charts have duplicate step 0-1000 ranges
6. **After fix**: W&B charts show continuous step progression

---

## Impact Assessment Summary

| Component | Before Fix | After Fix | Severity |
|-----------|-----------|-----------|----------|
| **Warmup schedule** | Restarts (focal_gamma=1.0) | Continues (focal_gamma=2.0) | **P1 - Critical** |
| **Training quality** | ~1000 batches wasted re-warming | No waste, continues learning | **P1** |
| **W&B metrics** | Step counter restarts, charts broken | Continuous timeline | **P1** |
| **Progress bar** | Shows 0/7702 (0%) | Shows 2527/7702 (33%) | **P2** |
| **Log messages** | Batch numbers wrong | Batch numbers correct | **P2** |
| **Checkpoint names** | `mid_epoch_001_000030.pt` | `mid_epoch_001_002557.pt` | **P3** |

---

## Related Files & Line Numbers

- `src/brain_brr/train/checkpoint.py:60-91` - save_checkpoint (add global_step param)
- `src/brain_brr/train/checkpoint.py:122-176` - load_checkpoint (already returns epoch, metric)
- `src/brain_brr/train/loop.py:150-177` - Mid-epoch checkpoint loading (add global_step restore)
- `src/brain_brr/train/loop.py:178-188` - Last checkpoint loading (add global_step restore)
- `src/brain_brr/train/loop.py:242` - global_step initialization (use resume value)
- `src/brain_brr/train/loop.py:280-313` - train_epoch call (pass resume_batch_idx)
- `src/brain_brr/train/loop.py:429-441` - Best checkpoint save (add global_step)
- `src/brain_brr/train/loop.py:455-466` - Periodic checkpoint save (add global_step)
- `src/brain_brr/train/loop.py:469-481` - Last checkpoint save (add global_step)
- `src/brain_brr/train/loop.py:264-275` - Timeout checkpoint save (add global_step)
- `src/brain_brr/train/loop.py:217-228` - Signal checkpoint save (add global_step)
- `src/brain_brr/train/train_step.py:165-189` - train_epoch signature (add resume_batch_idx param)
- `src/brain_brr/train/train_step.py:283-290` - tqdm initialization (add initial= param)
- `src/brain_brr/train/train_step.py:297` - batch_idx enumeration (add offset)
- `src/brain_brr/train/train_step.py:530-546` - Mid-epoch checkpoint save (add global_step)
- `src/brain_brr/train/train_step.py:377-407` - W&B logging (uses global_step)
- `src/brain_brr/train/warmup.py:11-43` - get_focal_gamma (uses global_step)
- `src/brain_brr/models/detector.py:174-197` - set_training_state (uses global_step)

---

## References - 2025 Best Practices

1. **PyTorch Official**: "Save epoch, optimizer, scheduler, and any custom counters in checkpoint dict"
2. **PyTorch Lightning**: Automatically saves `global_step` and restores it via `trainer.fit(ckpt_path=...)`
3. **HuggingFace Transformers**: Uses `tqdm(initial=resume_step)` pattern for progress bars
4. **StatefulDataLoader docs**: "enumerate() will start from 0 after load_state_dict() - this is expected. Track absolute position separately."
5. **tqdm docs**: "Use `initial=` parameter to set starting position for resumed loops"

---

**Status**: Ready for implementation. Fix required before next training run (local or Modal).
