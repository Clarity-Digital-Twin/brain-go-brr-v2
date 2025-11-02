# WSL2 GPU Training Crash Analysis

**Date**: Oct 28, 2025
**Status**: CRITICAL - Training crashes every ~6 hours during epoch 22
**Impact**: Cannot complete training runs beyond epoch 21

---

## Crash Pattern

### Incident 1: Oct 27-28, ~10:52 AM
- **Location**: Epoch 22 validation, batch 8886/18528 (48%)
- **Error**: `CUDA error: unknown error`
- **Duration**: Unknown (found crashed)
- **Checkpoint**: epoch_021.pt saved

### Incident 2: Oct 28, 12:22 PM - 6:17 PM
- **Started**: 12:22 PM (epoch 22, resumed from batch 3307)
- **Crashed**: 6:17 PM (~6 hours runtime)
- **Location**: Epoch 22 training (batch ~7456/7702)
- **Error**: Process killed, no Python traceback
- **Checkpoint**: mid_epoch_022_007456.pt saved

---

## Root Cause: WSL2 DirectX Graphics Driver

### Evidence
```bash
# dmesg shows repeated GPU allocation failures
[1157159.622631] misc dxg: dxgk: dxgvmb_send_create_allocation: send_create_allocation failed ffffffb5
[1157159.625676] misc dxg: dxgk: dxgkio_create_allocation: Ioctl failed: -75
```

### Technical Analysis
- **Error Code**: `-75` = `EOVERFLOW` (Value overflow, not "bad file descriptor")
- **Component**: WSL2 DirectX Graphics Kernel (`dxg`)
- **Operation**: GPU memory allocation via Windows-Linux bridge
- **Pattern**: Failures accumulate over time, then CUDA crashes
- **Note**: Earlier analysis incorrectly stated -75 = EBADF (which is errno 9)

### Why It's NOT a Code Bug
1. ✅ **GPU is healthy**: `nvidia-smi` shows normal operation (43°C, 70W)
2. ✅ **Checkpoints save**: Mid-epoch checkpoints written successfully
3. ✅ **Consistent location**: Always crashes during epoch 22
4. ✅ **System-level**: Error in WSL2 kernel module, not PyTorch/CUDA
5. ✅ **No Python traceback**: Process killed by system, not exception

---

## Impact Summary

### Completed Epochs
- ✅ Epochs 0-21: Successfully completed
- ❌ Epoch 22: Crashes during training/validation

### Data Loss
- **WandB Synced**: Epochs 0-15, 17-20 (19 data points)
- **WandB Lost**: Epochs 16, 21, 22 (not logged before crashes)
- **Local Checkpoints**: All epochs 0-21 saved ✅

---

## Workarounds Attempted

### ✅ Successful
1. Resume from checkpoints - works perfectly
2. Manual WandB sync - recovered most data
3. Mid-epoch checkpoints - saves progress

### ❌ Not Attempted Yet
1. Reduce batch size (currently 8)
2. Reduce validation frequency
3. Enable CUDA memory caching limits
4. Try native Linux (not WSL2)
5. Update WSL2 kernel
6. Update NVIDIA drivers

---

## Recommended Actions

### Immediate (Can try now)
1. **Restart with smaller batch size**: `batch_size: 6` (reduce GPU memory pressure)
2. **Add CUDA env vars**:
   ```bash
   export CUDA_LAUNCH_BLOCKING=1  # Synchronous CUDA for better errors
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Limit fragmentation
   ```

3. **Increase mid-epoch checkpoint frequency**: Save every 15 min (currently 30 min)

### Long-term (Requires setup)
1. **Update WSL2 kernel**:
   ```bash
   wsl --update
   wsl --shutdown
   # Restart WSL2
   ```

2. **Update NVIDIA drivers** (Windows side):
   - Current: 581.42
   - Check for latest GeForce driver

3. **Try Modal training**: A100 won't have WSL2 issues (but costs $4.40/hour)

4. **Bare metal Linux**: Eliminate WSL2 entirely

---

## Current Status (Oct 28, 6:18 PM)

```
✅ GPU: Healthy (43°C, 1GB used, no processes)
✅ Checkpoints: epoch_021.pt (Epoch 21 @ 161,742 steps)
✅ WandB: Synced up to epoch 20
❌ Training: Crashed (epoch 22, batch ~7456)
🔄 Next: Investigate workarounds, restart training
```

---

## Technical Details

### System Info
- **OS**: WSL2 (Windows Subsystem for Linux)
- **Kernel**: Linux 5.15.167.4-microsoft-standard-WSL2
- **GPU**: NVIDIA GeForce RTX 4090 (24GB VRAM)
- **Driver**: 581.42 (Windows) / 580.95.02 (WSL2)
- **CUDA**: 13.0
- **PyTorch**: 2.5.0+cu124

### Training Config
- **Batch size**: 8
- **Mixed precision**: False
- **Workers**: 0 (WSL2 fix)
- **Validation**: Every epoch, 18528 batches

### Crash Signature
```
Training: 100%|##############| 7456/7702
<process killed, no traceback>
```

---

**Last Updated**: Oct 28, 2025 6:18 PM EDT
