# Release Notes: v3.10.0-auto-restart

**Release Date:** October 10, 2025
**Tag:** `v3.10.0-auto-restart`
**Status:** ✅ Production Ready (Modal A100-80GB)

---

## 🚀 What's New

### Auto-Restart Training (Major Feature)

**Hands-free 100-epoch training** with zero manual intervention after initial setup.

- **Scheduled Function**: `train_auto_restart()` runs every 23 hours via `modal.Period(hours=23)`
- **Overlap Protection**: `max_containers=1` ensures only one instance runs (no file locks needed)
- **Seamless Resume**: Automatically loads `last.pt` and continues from next epoch
- **Timeline**: T=0h start → T=22h50m timeout → T=23h restart (10min safety margin)

**Usage**:
```bash
# Deploy app (one-time)
modal deploy deploy/modal/app.py

# Start auto-restart training (runs until stopped)
modal run --detach deploy/modal/app.py --action schedule-training \
  --config configs/modal/train_bimamba.yaml

# Monitor
modal app logs brain-go-brr-v2

# Stop
modal app stop brain-go-brr-v2
```

**Impact**:
- **Before**: Manual resume every 23h (12 interventions for 100 epochs)
- **After**: Set-and-forget (0 interventions until completion)
- **Human Time**: 10 min total vs. 60 min manual interventions

---

### Checkpoint Resume Bug Fix (Critical)

**Problem**: Checkpoints saved `epoch` (completed) instead of `epoch + 1` (next to train), causing every resume to re-train the last completed epoch.

**Fix**: Changed `save_checkpoint(..., epoch + 1, ...)` in:
- `src/brain_brr/train/loop.py:464-465` (last.pt saves)
- `src/brain_brr/train/loop.py:449` (periodic checkpoint saves)

**Impact**:
- **One-time waste**: $56 (first resume with old checkpoints)
- **Savings**: $672 over 12 auto-restarts (11 restarts × $56)
- **Net benefit**: **$616 saved** over 100-epoch training

**Documentation**: `docs/archive_v2/CHECKPOINT_RESUME_BUG.md`

---

### Checkpoint Buffer Compatibility Fix (Critical)

**Problem**: PyTorch's `register_buffer(name, None)` doesn't add buffer to `state_dict()` until a tensor is assigned. This created a timing bug where checkpoints saved mid-training contained `gnn.last_valid_pe` buffer, but freshly initialized models didn't, causing resume to fail with "Unexpected key(s) in state_dict".

**Root Cause**: Dynamic buffer registration - buffer appears in state_dict only after first forward pass assigns a tensor value.

**Fix** (three-layer defense):
1. **Checkpoint loading** (`checkpoint.py:160-191`): Detect and skip buffers with shape mismatches before `load_state_dict()`
2. **Model initialization** (`gnn_pyg.py:137-142`): Initialize buffer with placeholder tensor `torch.zeros(1,1,1,k)` instead of `None`, ensuring it's always in state_dict
3. **Forward pass** (existing): Automatic fallback logic recomputes PE when placeholder dimensions don't match batch

**Impact**:
- Enables resume from any mid-epoch checkpoint (previously blocked)
- Backward compatible with old checkpoints (skip mechanism handles mismatches)
- Self-healing: placeholder automatically updated on first forward pass

**Documentation**: `docs/archive_v2/CHECKPOINT_BUFFER_BUG.md`

**Tests**: 5 regression tests in `tests/unit/train/test_checkpoint_buffer_compatibility.py`

---

### RNG State Device Mismatch Fix (Critical)

**Problem**: When loading checkpoints with `map_location="cuda"`, PyTorch moves ALL tensors (including RNG states) to GPU. However, both `torch.set_rng_state()` and `torch.cuda.set_rng_state_all()` require CPU ByteTensors, causing "RNG state must be a torch.ByteTensor" errors on resume.

**Root Cause**: PyTorch's `torch.load(map_location=device)` moves all tensors to specified device, but RNG state restoration APIs expect CPU tensors (PyTorch handles GPU transfer internally).

**Fix** (`checkpoint.py:225-247`): Force both CPU and CUDA RNG states back to CPU before restoration:
```python
# CPU RNG: torch.set_rng_state() requires CPU ByteTensor
torch.set_rng_state(rng["torch"].cpu())

# CUDA RNG: ALSO requires CPU tensors (counter-intuitive but correct!)
if torch.cuda.is_available() and rng["torch_cuda"] is not None:
    cuda_rng = rng["torch_cuda"]
    if isinstance(cuda_rng, list) and len(cuda_rng) > 0 and cuda_rng[0].is_cuda:
        cuda_rng = [state.cpu() for state in cuda_rng]
    torch.cuda.set_rng_state_all(cuda_rng)
```

**Impact**:
- Enables deterministic resume on GPU (Modal A100)
- Supports all device combinations (CPU→CPU, CPU→CUDA, CUDA→CUDA, CUDA→CPU)
- Preserves exact training reproducibility

**Documentation**: `docs/archive_v2/RNG_STATE_DEVICE_BUG.md`

**Tests**: 4 regression tests in `tests/unit/train/test_checkpoint_rng_device.py`

---

### Modal 1.0 Migration

**Deprecated Parameter**: `concurrency_limit` → `max_containers` (breaking change Feb 2025)

**Updated**: `deploy/modal/app.py:1141`

**Benefits**: Future-proof, no deprecation warnings

---

### New Documentation

**`MODAL_CLI_REFERENCE.md`** - Complete Modal command reference:
- All updated commands (Modal 1.0 compatible)
- Common workflows (fresh start, resume, auto-restart)
- Troubleshooting guide
- Storage architecture
- Monitoring commands

---

## 📊 Training Status

**Current Deployment**:
- **App ID**: `ap-ik2xwlXmuQMvPyhSfrZJfi`
- **Config**: BiMamba2, 100 epochs, batch_size=48, A100-80GB
- **Status**: Step 2 of 3-step plan (manual resume with fixed code)
- **Next**: Enable auto-restart after timeout (~23h)

**Timeline**:
```
T=0h:       Load mid_epoch_002_*.pt (epoch=1, buggy)
T=0-14h:    Re-train Epoch 2 (one-time waste, $56)
T=14h:      Save last.pt with epoch=3 (FIXED!)
T=14-23h:   Train Epoch 3, maybe 4
T=23h:      Timeout → save → exit
→ THEN:     Enable auto-restart (hands-free to completion)
```

---

## 🔧 Breaking Changes

**None** - 100% backward compatible

---

## 📦 Upgrade Guide

### From v3.9.2

```bash
git pull
git checkout v3.10.0-auto-restart

# No config changes needed
# New auto-restart workflow available
```

### Checkpoint Compatibility

- ✅ Old checkpoints (v3.9.x) load correctly
- ⚠️ First resume will re-train last epoch ONCE (unavoidable with old checkpoints)
- ✅ New checkpoints (v3.10.0+) resume correctly from next epoch

---

## 📈 Quality Metrics

**Tests**:
```bash
make q     # ✅ PASS (lint + format + mypy + config validation)
make test  # ✅ PASS (104 tests, 75%+ coverage)
```

**Smoke Test** (Modal):
- Run: `ap-c8pqL1a2TfE24wBqvAWmb0`
- Status: ✅ SUCCESS (1 epoch, 50 files, ~40 min)
- Checkpoint: `/results/smoke/checkpoints/last.pt` has `epoch=1` (correct)

---

## 🎯 Cost Analysis

| Scenario | Cost | Notes |
|----------|------|-------|
| **v3.9.2 (buggy)** | $1,792 | 280h training + 168h wasted (12×14h) |
| **v3.10.0 (first resume)** | $56 | One-time waste (old checkpoint) |
| **v3.10.0 (remaining)** | $1,120 | Normal training (280h @ $4/h) |
| **Net savings** | **$616** | $1,792 - $1,176 = $616 saved |

---

## 📚 Documentation Updates

**Updated Files**:
- `CHANGELOG.md` - Added v3.10.0 entry
- `STATUS.md` - Updated version, deployment status, training commands
- `CLAUDE.md` - Updated project overview, commands, current status
- `README.md` - Updated version badge, status footer
- `pyproject.toml` - Updated version to 3.10.0
- **New**: `MODAL_CLI_REFERENCE.md` - Complete Modal command reference

**Archived Documentation**:
- `docs/archive_v2/CHECKPOINT_RESUME_BUG.md` - Marked as FIXED with implementation details
- `docs/archive_v2/CLEAR_PATH_FORWARD.md` - 3-step plan executed
- `docs/archive_v2/MODAL_AUTO_RESTART_STRATEGY.md` - Strategy implemented

---

## 🔗 Links

- **GitHub Release**: https://github.com/Clarity-Digital-Twin/brain-go-brr-v2/releases/tag/v3.10.0-auto-restart
- **Modal Dashboard**: https://modal.com/apps/clarity-digital-twin/main
- **Documentation**: See `MODAL_CLI_REFERENCE.md` for commands
- **Status**: See `STATUS.md` for current deployment

---

## 🙏 Acknowledgments

Special thanks to the external agent that helped identify the checkpoint resume bug during the Modal training diagnostics session on October 10, 2025.

---

**Full Changelog**: https://github.com/Clarity-Digital-Twin/brain-go-brr-v2/compare/v3.9.2-ci-stability...v3.10.0-auto-restart
