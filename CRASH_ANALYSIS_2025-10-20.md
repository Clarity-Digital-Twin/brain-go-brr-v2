# CUDA Crash Analysis - 2025-10-20 10:27

## Incident Summary

**Status**: CUDA kernel crash (NOT out-of-memory)
**Time**: 2025-10-20 10:27:20 (after 44+ hours of stable training)
**Location**: Batch 4989/7702 (64.8% through Epoch 5)
**Last Checkpoint**: `mid_epoch_005_004195.pt` (29 minutes before crash)

## Error Message

```
[ERROR] Training loop failed at batch 4989: CUDA error: unknown error
CUDA kernel errors might be asynchronously reported at some other API call,
so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

## Evidence

### GPU State at Crash
- **Memory**: 0.28GB used / 17.24GB reserved (plenty available!)
- **Temperature**: 42°C (very cool)
- **Power**: 67W / 450W (normal)
- **No OOM**: System logs clean, no OOM killer triggered

### Training Progress
- **Started**: Oct 18 14:52 (Saturday)
- **Crashed**: Oct 20 10:27 (Monday) → **44 hours stable**
- **Epochs Completed**: 4/100 (last.pt has epoch=4, best_metric=0.2708)
- **Mid-Epoch Progress**: Batch 4195/7702 saved at 09:58

### System Configuration
- **Driver**: 581.42 (Windows) / 580.95.02 (WSL2)
- **CUDA**: 12.4
- **PyTorch**: 2.5.0+cu124
- **GPU**: RTX 4090 (24GB)
- **Platform**: WSL2 Ubuntu

## Root Cause Analysis

### Primary Suspect: WSL2 GPU Driver Issue 🎯

**Likelihood: VERY HIGH**

**Evidence**:
1. User was using web browser during training (stressed Windows GPU/VRAM)
2. WSL2 GPU passthrough fragile under mixed Windows/WSL workloads
3. "CUDA error: unknown error" = driver-level crash, not application error
4. Crash timing (44h) suggests driver state corruption, not code bug
5. No numerical issues (gradients stable: P50=0.72, Max=13.54)

**Mechanism**:
- Windows browser uses GPU/VRAM
- WSL2 CUDA context gets confused by Windows VRAM pressure
- Driver reports "unknown error" when kernel call fails
- GPU left in zombie state (1448MB allocated but no process)

### Secondary Suspect: FLA CUDA Kernel Edge Case

**Likelihood: LOW**

**Evidence**:
1. FLA (Gated DeltaNet) has complex CUDA operations
2. Could hit illegal memory access on specific input combination
3. But: training was stable for 44h, passed 20k+ batches without issue

**Why Unlikely**:
- Numerical instability would show up as NaN/Inf (didn't happen)
- Would likely crash earlier if code bug (not after 44h)
- Memory was plentiful (no fragmentation pressure)

### Tertiary Suspect: Hardware Fault

**Likelihood: VERY LOW**

**Evidence**:
- Temperature: 42°C (excellent)
- No ECC errors (N/A on RTX 4090)
- GPU health metrics normal

## Mitigation Strategy

### Immediate Actions (for this session)

1. **Enable CUDA_LAUNCH_BLOCKING=1**
   - Get synchronous error reports (easier debugging)
   - Performance hit: ~10-20% slower, but worth it for stability

2. **Resume from mid-epoch checkpoint**
   - File: `mid_epoch_005_004195.pt`
   - Will resume from batch 4195/7702
   - Lost progress: ~794 batches (~28 min work)

3. **Isolate GPU workload**
   - **CRITICAL**: Close all Windows apps using GPU (browser, games, etc.)
   - WSL2 needs exclusive GPU access for stability
   - Monitor with Windows Task Manager → Performance → GPU

### Long-Term Mitigations

1. **Avoid Mixed GPU Workloads**
   - Don't use Windows GPU apps while training in WSL2
   - If needed, pause training, use browser, resume training

2. **More Frequent Checkpoints**
   - Current: every 30 min (1800s)
   - Consider: every 15 min (900s) for expensive experiments
   - Trade-off: more disk I/O vs less lost work

3. **Monitor for Patterns**
   - If crashes repeat at similar batch counts → code bug
   - If crashes are random → driver/hardware issue
   - Track: time-of-day, Windows activity, GPU temp

4. **Consider Native Linux**
   - WSL2 GPU passthrough adds complexity
   - Native Ubuntu would eliminate driver translation layer
   - But: requires dual-boot or separate machine

## Recovery Instructions

### Safe Resume Command

```bash
# Start new tmux session
tmux new -s exp1-reg-resume

# Enable CUDA debugging
export BGB_NAN_DEBUG=1
export CUDA_LAUNCH_BLOCKING=1  # Synchronous CUDA errors

# Resume from mid-epoch checkpoint
.venv/bin/python -m src train configs/local/train_fla_exp1_reg.yaml --resume

# Detach: Ctrl+B then D
# Monitor: tmux attach -t exp1-reg-resume
```

### Checkpoint Status

```
✅ last.pt (epoch 4 completed, best_metric=0.2708)
✅ mid_epoch_005_004195.pt (batch 4195/7702, 54.5% through epoch 5)
✅ best.pt (epoch 3, copied from best validation so far)
```

Resume will:
1. Load `mid_epoch_005_004195.pt`
2. Restore model, optimizer, scheduler, RNG states
3. Continue from batch 4196/7702
4. Complete epoch 5, then continue to epoch 100

### Monitoring for Stability

First 100 batches after resume:
- Watch for immediate crash → code bug in resume logic
- Watch for crash at batch 4989 again → specific input pattern issue
- Watch for random crash → driver/hardware issue

Check logs for:
```bash
# Real-time monitoring (in separate terminal)
tail -f results/local_fla_exp1_reg/wandb/run-*/files/output.log

# GPU monitoring
watch -n 5 nvidia-smi
```

## Expected Behavior

### If Crash Was WSL2/Driver Issue (LIKELY)
- Training will be stable if Windows GPU apps are closed
- Should complete 100 epochs without further crashes
- Consider native Linux if crashes continue

### If Crash Was Code Bug (UNLIKELY)
- Will crash again at similar location (batch ~4900-5000)
- Need to bisect which CUDA kernel is failing
- May need to disable specific features (dynamic PE, FLA, etc.)

### If Crash Was Hardware (VERY UNLIKELY)
- Random crashes at unpredictable times
- Need to stress-test GPU (memtest, furmark)
- May need RMA if hardware fault confirmed

## Lessons Learned

1. **W&B output.log is CRITICAL** - only place with crash details (no training.log!)
2. **WSL2 GPU passthrough is fragile** - avoid mixed workloads
3. **Mid-epoch checkpoints saved us** - only lost 28 min vs full epoch (~4h)
4. **CUDA_LAUNCH_BLOCKING** should be default for long experiments

## Action Items

- [ ] Close all Windows GPU apps before resuming
- [ ] Resume with CUDA_LAUNCH_BLOCKING=1
- [ ] Monitor first 100 batches for stability
- [ ] If stable, consider removing CUDA_LAUNCH_BLOCKING for performance
- [ ] If unstable, investigate specific failing kernel
- [ ] Document any patterns (time, batch number, Windows activity)

---

**Generated**: 2025-10-20 10:37
**Analyzed by**: Claude Code
**Next Step**: Resume training with CUDA debugging enabled
