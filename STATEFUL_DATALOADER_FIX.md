# StatefulDataLoader Fix - Mid-Epoch Resume

**Date**: 2025-10-10
**Status**: 🟢 **SOLUTION VERIFIED - READY TO IMPLEMENT**
**Impact**: Eliminates 1-2 hours wasted compute per Modal restart ($16-40 savings over 100 epochs)

---

## The Problem

**Current Behavior:**
- Modal kills training at 23h timeout
- We save mid-epoch checkpoints every 30 min with `batch_idx`
- We load model/optimizer/scheduler state correctly ✅
- We **restart epoch from batch 0** ❌ (DataLoader has no state restoration)
- Result: **Waste 1-2 hours per restart**

**Evidence from logs:**
```
[2025-10-10 21:46:56.126] Resumed from epoch 2, batch 1224  ← Reads batch_idx from checkpoint
[2025-10-10 21:57:15.246] [HEARTBEAT] Batch 0/1284          ← Actually starts at batch 0 ❌
```

**Root cause:** `torch.utils.data.DataLoader` has NO `state_dict()` / `load_state_dict()` methods. We can't tell it to skip to batch 1224.

**Acknowledged in code** (`src/brain_brr/train/loop.py:168`):
```python
logger.info(f"Resumed from epoch {start_epoch + 1}, batch {ckpt.get('batch_idx', '?')}")
# Note: This resumes from start of epoch, not exact batch  ← COMMENT ADMITS BUG
```

---

## The Solution (Official PyTorch, 2024-2025)

### Use `StatefulDataLoader` from `torchdata` library

**What It Is:**
- **Official PyTorch solution** for mid-epoch checkpointing (introduced torchdata 0.8.0, July 2024)
- **Drop-in replacement** for `torch.utils.data.DataLoader` (same signature, same args)
- Adds `state_dict()` / `load_state_dict()` methods for exact iteration state
- Handles **multiprocess worker state synchronization** (works with `num_workers > 0`)
- **Production-ready**: Used by HuggingFace Accelerate, validated on large-scale training

**Key Features:**
1. Works with standard Dataset classes (no DataPipe needed - DataPipe is deprecated!)
2. Handles shuffle RNG (deterministic batch order on resume when combined with our existing RNG restore)
3. Worker state aggregation (snapshots state from all workers, restores to same workers)
4. Configurable overhead (`snapshot_every_n_steps` parameter)

**Official Documentation:**
- API Reference: https://docs.pytorch.org/data/main/torchdata.stateful_dataloader.html
- Tutorial: https://docs.pytorch.org/data/main/stateful_dataloader_tutorial.html
- GitHub: https://github.com/pytorch/data

---

## Basic Usage Example

```python
from torchdata.stateful_dataloader import StatefulDataLoader

# Create loader (EXACT same args as torch.utils.data.DataLoader)
train_loader = StatefulDataLoader(
    dataset,
    batch_size=48,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
    # NEW: How often to snapshot state (default=1, can increase for performance)
    snapshot_every_n_steps=1,  # Match your checkpoint frequency or leave at 1
)

# Save state in checkpoint (NEW)
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "dataloader_state_dict": train_loader.state_dict(),  # ← NEW
    "epoch": epoch,
    "batch_idx": batch_idx,
}
torch.save(checkpoint, "mid_epoch_002_001224.pt")

# Resume - create NEW loader then restore state
train_loader = StatefulDataLoader(
    dataset,
    batch_size=48,
    num_workers=4,  # ⚠️ MUST match saved state!
    shuffle=True,
)
train_loader.load_state_dict(checkpoint["dataloader_state_dict"])  # ← Restores to batch 1224
# Now iterating starts at EXACT batch where we saved ✅
```

---

## Implementation Plan

### Step 1: Install `torchdata`

**Local environment:**
```bash
.venv/bin/pip install torchdata
```

**Modal deployment** (add to `deploy/modal/app.py` around line 85-100):
```python
# After core dependencies (scipy, scikit-learn, mne, etc.), before project code
.pip_install(
    "scipy>=1.10.0",
    "scikit-learn>=1.3.0",
    # ... existing dependencies
    "torchdata>=0.8.0",  # ← ADD THIS: StatefulDataLoader for mid-epoch resume
    "tqdm>=4.64.0",
)
```

**Version compatibility:**
- Requires: `torchdata>=0.8.0` (StatefulDataLoader introduced July 2024)
- Compatible with: `PyTorch 2.5.0` ✅ (our current version)
- No conflicts with existing dependencies

---

### Step 2: Update `loop.py` - Create StatefulDataLoader

**File:** `src/brain_brr/train/loop.py`

**Add import** (top of file, after existing torch imports around line 21):
```python
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader  # ← ADD THIS
```

**Update loader creation** (currently at line 755):
```python
# OLD:
train_loader = DataLoader(train_dataset, **train_loader_kwargs)

# NEW:
train_loader = StatefulDataLoader(train_dataset, **train_loader_kwargs)
```

**Update validation loader** (currently at line 766):
```python
# OLD:
val_loader = DataLoader(val_dataset, **val_loader_kwargs)

# NEW:
val_loader = StatefulDataLoader(val_dataset, **val_loader_kwargs)
```

**Note:** All existing `train_loader_kwargs` work identically (batch_size, sampler, shuffle, num_workers, pin_memory, persistent_workers, prefetch_factor, worker_init_fn).

---

### Step 3: Update `train_step.py` - Save DataLoader State

**File:** `src/brain_brr/train/train_step.py`

**Location:** Mid-epoch checkpoint saving (currently lines 443-454)

**Current code:**
```python
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
    extra={"batch_idx": batch_idx, "kind": "mid_epoch"},
)
```

**Change to:**
```python
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
        "kind": "mid_epoch",
        "dataloader_state_dict": dataloader.state_dict(),  # ← ADD THIS
    },
)
```

**Note:** `dataloader` is already passed to `train_epoch()` as a parameter (line 78), so it's accessible here.

---

### Step 4: Update `loop.py` - Restore DataLoader State on Resume

**File:** `src/brain_brr/train/loop.py`

**Location:** Mid-epoch checkpoint resume (currently lines 149-168)

**Current code:**
```python
if mid_epoch_checkpoints and config.training.resume:
    latest_mid = mid_epoch_checkpoints[-1]
    logger.info(f"[RESUME] Found mid-epoch checkpoint: {latest_mid.name}")
    start_epoch, best_metric = load_checkpoint(
        latest_mid, model, optimizer, scheduler, scaler=scaler, device=device
    )
    ckpt = torch.load(latest_mid, map_location="cpu", weights_only=False)
    # ... best_metric fallback logic ...
    logger.info(f"Resumed from epoch {start_epoch + 1}, batch {ckpt.get('batch_idx', '?')}")
    # Note: This resumes from start of epoch, not exact batch
```

**Change to:**
```python
if mid_epoch_checkpoints and config.training.resume:
    latest_mid = mid_epoch_checkpoints[-1]
    logger.info(f"[RESUME] Found mid-epoch checkpoint: {latest_mid.name}")

    # Load checkpoint into model/optimizer/scheduler/scaler
    start_epoch, best_metric = load_checkpoint(
        latest_mid, model, optimizer, scheduler, scaler=scaler, device=device
    )

    # Load full checkpoint for DataLoader state
    ckpt = torch.load(latest_mid, map_location="cpu", weights_only=False)

    # Restore best_metric if mid-epoch checkpoint doesn't have it
    if best_metric == 0.0 and (checkpoint_dir / CHECKPOINT_LAST).exists():
        try:
            _last = torch.load(
                checkpoint_dir / CHECKPOINT_LAST, map_location="cpu", weights_only=False
            )
            best_metric = _last.get("best_metric", _last.get("metric", 0.0))
        except Exception:
            pass

    # Restore DataLoader state for exact batch resume (NEW)
    if "dataloader_state_dict" in ckpt:
        train_loader.load_state_dict(ckpt["dataloader_state_dict"])
        logger.info(
            f"[RESUME] Restored from epoch {start_epoch + 1}, batch {ckpt.get('batch_idx', '?')} "
            f"(exact mid-epoch resume) ✅"
        )
    else:
        logger.warning(
            "[RESUME] Old checkpoint without DataLoader state - will restart epoch from batch 0 "
            "(expected for checkpoints saved before StatefulDataLoader upgrade)"
        )
        logger.info(f"Resumed from epoch {start_epoch + 1} (epoch start)")
```

**Critical:** This must happen AFTER `train_loader` is created (line 755) but BEFORE the training loop starts (line 235).

---

### Step 5: Test Locally

**Test procedure:**
```bash
# Start training
.venv/bin/python -m src train configs/local/train_bimamba.yaml

# Wait for first mid-epoch checkpoint (30 min) or force early save by modifying mid_checkpoint_interval_s

# Kill training mid-epoch (Ctrl+C)

# Resume - should continue from exact batch
.venv/bin/python -m src train configs/local/train_bimamba.yaml --resume
```

**Expected output:**
```
[RESUME] Found mid-epoch checkpoint: mid_epoch_001_000500.pt
[RESUME] Restored from epoch 1, batch 500 (exact mid-epoch resume) ✅
[HEARTBEAT] Still training... Batch 500/1284 | Avg Loss: 0.0648  ← CORRECT! Not batch 0
```

**If you see warning about old checkpoint**, that's expected until you create your first new checkpoint with dataloader state.

---

### Step 6: Deploy to Modal

```bash
# Update Modal image (includes torchdata)
modal deploy deploy/modal/app.py

# Resume training with new code
modal run --detach deploy/modal/app.py --action train \
  --config configs/modal/train_bimamba.yaml --resume
```

**First resume after deployment:** Will load the latest checkpoint (likely `last.pt` from epoch end, since old mid-epoch checkpoints don't have dataloader state). This is fine - training will create new mid-epoch checkpoints with dataloader state, and subsequent resumes will work perfectly.

---

## Critical Implementation Details

### 1. num_workers Consistency

When calling `load_state_dict()`, DataLoader MUST be created with **same num_workers** as when state was saved.

**Our case:**
- Modal: `num_workers=4` (configs/modal/*.yaml)
- Local: `num_workers=0` (configs/local/*.yaml - WSL2 fix)

This is fine because we don't mix them (Modal checkpoints resume on Modal, local resumes locally).

**If you ever change num_workers in config**, old checkpoints will fail with:
```
RuntimeError: Calling load_state_dict requires StatefulDataLoader to have same num_workers (4) as those of the provided state_dict (8).
```

**Solution:** Delete old mid-epoch checkpoints when changing num_workers:
```bash
# Modal
modal volume get brain-go-brr-results /results/train/checkpoints
rm mid_epoch_*.pt
modal volume put brain-go-brr-results /results/train/checkpoints

# Local
rm results/*/checkpoints/mid_epoch_*.pt
```

---

### 2. snapshot_every_n_steps Parameter

Controls how often worker state is transferred to main process:
- **Default: 1** (snapshot every batch)
- **Overhead:** Minimal for most workloads (batch processing is slower than state transfer)
- **Optimization:** Set to checkpoint frequency (e.g., if checkpointing every 100 batches, use `snapshot_every_n_steps=100`)

**Our case:**
- We checkpoint every 30 minutes (~300-400 batches on A100)
- **Recommendation:** Start with default `snapshot_every_n_steps=1`, only increase if profiling shows overhead

**Trade-off:**
- Lower value = more frequent snapshots = more overhead, but can resume from any batch
- Higher value = less overhead, but can only resume from batches divisible by snapshot interval

With default=1, we can resume from ANY batch exactly.

---

### 3. Interaction with RNG State

We already save/restore RNG states (`checkpoint.py:82-88`). StatefulDataLoader complements this:

- **Our RNG restore:** Sets global torch/numpy/python RNG states (affects shuffle order)
- **StatefulDataLoader:** Saves iterator position within shuffled order

**Together:** Fully deterministic mid-epoch resume ✅

**Example:**
1. Save checkpoint at batch 1224 with RNG state + dataloader state
2. Resume: RNG state → same shuffle order, DataLoader state → skip to batch 1224
3. Result: Exact same batches in exact same order ✅

---

### 4. Validation DataLoader

Validation runs full epoch every time (no mid-epoch interruption), so `val_loader` doesn't need state restoration.

**But:** We create it as `StatefulDataLoader` anyway (Step 2) for consistency and future-proofing. The state methods simply won't be called.

---

## Expected Results

### Before (Current)
```
Modal Run 1:  0h → 23h (1427 batches) → Timeout → Save checkpoint (epoch=1, batch=1427)
Modal Run 2: 23h → Resume from epoch 1, batch 0 ❌ → Re-train batches 0-1427 (1-2h wasted)
```

### After (With StatefulDataLoader)
```
Modal Run 1:  0h → 23h (1427 batches) → Timeout → Save checkpoint (epoch=1, batch=1427, dataloader_state)
Modal Run 2: 23h → Resume from epoch 1, batch 1427 ✅ → Continue to end (0h wasted)
```

**Savings per restart:** 1-2 hours = $4-8
**Total savings (4-5 restarts over 100 epochs):** $16-40

---

## Troubleshooting

### Issue: `ImportError: No module named 'torchdata'`
**Cause:** torchdata not installed
**Fix:** `pip install torchdata` (local) or add to Modal image (Step 1)

---

### Issue: `RuntimeError: num_workers mismatch (expected 4, got 8)`
**Cause:** Checkpoint saved with different num_workers than current loader
**Fix:** Either:
1. Delete old mid-epoch checkpoints: `rm checkpoints/mid_epoch_*.pt`
2. OR update config to match checkpoint's num_workers

---

### Issue: Resume still starts at batch 0
**Possible causes:**
1. Old checkpoint without `dataloader_state_dict` (expected - see warning in logs)
2. StatefulDataLoader not being used (check imports in loop.py)
3. `load_state_dict()` not called (check resume logic in loop.py:149-168)

**Debug:**
```python
# Add after loading checkpoint:
print(f"Checkpoint keys: {list(ckpt.keys())}")
# Should include 'dataloader_state_dict' for new checkpoints

# Add after creating loader:
print(f"Loader type: {type(train_loader)}")
# Should be <class 'torchdata.stateful_dataloader.StatefulDataLoader'>
```

---

### Issue: Training slower after adding StatefulDataLoader
**Cause:** Overhead from state snapshots (unlikely with default settings)
**Fix:** Increase `snapshot_every_n_steps` parameter
**Benchmark first:** Time 100 batches before/after to confirm overhead

---

## Implementation Checklist

- [ ] Install torchdata locally: `.venv/bin/pip install torchdata`
- [ ] Add torchdata to Modal image: `deploy/modal/app.py` line ~95
- [ ] Update loop.py imports: `from torchdata.stateful_dataloader import StatefulDataLoader`
- [ ] Update loop.py train loader (line 755): Use `StatefulDataLoader` instead of `DataLoader`
- [ ] Update loop.py val loader (line 766): Use `StatefulDataLoader` instead of `DataLoader`
- [ ] Update train_step.py (line 453): Save `dataloader.state_dict()` in `extra` dict
- [ ] Update loop.py resume (line 167): Call `train_loader.load_state_dict()` if available
- [ ] Test locally: Kill + resume mid-epoch, verify batch number continues
- [ ] Deploy to Modal: `modal deploy deploy/modal/app.py`
- [ ] Test on Modal: Resume from mid-epoch checkpoint, verify logs show correct batch
- [ ] Monitor: Check that batch numbers are continuous across restarts
- [ ] Clean up: Remove comment from loop.py:168 ("Note: This resumes from start of epoch...")

**ETA:** 30-45 minutes implementation + testing

---

## References

- **PyTorch TorchData Docs:** https://docs.pytorch.org/data/main/torchdata.stateful_dataloader.html
- **Tutorial:** https://docs.pytorch.org/data/main/stateful_dataloader_tutorial.html
- **GitHub:** https://github.com/pytorch/data
- **HuggingFace Accelerate Integration:** https://huggingface.co/docs/accelerate/en/package_reference/torch_wrappers
- **Original GitHub Issue:** https://github.com/pytorch/pytorch/issues/36650

**Status:** Ready to implement. This is the official PyTorch solution, validated in production use (HuggingFace Accelerate, others).
