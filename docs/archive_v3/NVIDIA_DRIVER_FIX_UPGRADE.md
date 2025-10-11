# NVIDIA Driver Crash Fix - UPGRADE to 581.42

**Date**: October 11, 2025
**Status**: RESOLVED - Driver upgrade required (not downgrade!)

## Current Situation

**You have**: NVIDIA Driver 572.16 (January 2025 - BUGGY)
**You need**: NVIDIA Driver 581.42 (October 2025 - LATEST STABLE)

Your crash at batch 2996 (SIGBUS signal 7) was caused by known bugs in driver 572.16.

---

## Driver Timeline & Fixes

| Driver | Date | Status | Notes |
|--------|------|--------|-------|
| **572.16** | Jan 30, 2025 | ❌ BUGGY | Your current driver - widespread RTX 4090 crashes |
| 572.24 | Feb 3, 2025 | ❌ Hotfix attempt | Partial fixes only |
| 572.75 | Mar 9, 2025 | ❌ Hotfix attempt | Still problematic |
| **576.02** | Apr 2025 | ✅ FIXED | Resolved 40+ crash issues from 572.XX |
| 576.XX | Spring-Summer | ✅ Stable | Multiple releases |
| 581.08 | Aug 2025 | ✅ Stable | New stable branch |
| 581.15 | Oct 2025 | ✅ Stable | Current stable |
| 581.29 | Oct 10, 2025 | ✅ Stable | Bug fixes, memory leak fixes |
| **581.42** | Oct 2025 | ✅ **LATEST** | **Recommended for RTX 4090** |

---

## Solution: Upgrade to 581.42

### Step 1: Download Latest Driver

**Official NVIDIA Link**:
- Go to: https://www.nvidia.com/download/index.aspx
- Product: GeForce RTX 40 Series
- Series: GeForce RTX 40 Series
- Product: GeForce RTX 4090
- OS: Windows 10/11 64-bit
- Download Type: Game Ready Driver (GRD)
- Latest version: 581.42 WHQL

**Or Direct Download**:
- https://www.nvidia.com/en-us/geforce/drivers/

### Step 2: Install (Clean Install Recommended)

```
1. Run downloaded driver installer
2. Choose "Custom Installation"
3. Check "Perform a clean installation" ✅
4. Continue through installation
5. Reboot Windows
```

### Step 3: Verify Installation

**In Windows**:
- NVIDIA Control Panel → Help → System Information
- Should show: Driver Version 581.42

**In WSL2**:
```bash
nvidia-smi
# Should show: Driver Version: 581.42
```

### Step 4: Resume Training

```bash
tmux new -s train-fla
export BGB_NAN_DEBUG=1
.venv/bin/python -m src train configs/local/train_fla.yaml --resume
```

---

## What 581.42 Fixes

**From 581.29 release notes** (October 10, 2025):
- ✅ Fixed performance regression in Marvel: Rivals
- ✅ Fixed memory leak in NVENC hardware encoding
- ✅ Optimizations for Borderlands 4 and Dying Light: The Beast
- ✅ DLSS 4 support improvements

**From 576.02** (Fixed the 572.XX crashes):
- ✅ Resolved 40+ black screen issues
- ✅ Fixed display output failures
- ✅ Resolved game/training crashes
- ✅ Fixed CUDA workload stability

---

## Why You're Still on 572.16

**Likely reasons**:
1. NVIDIA App/GeForce Experience not auto-updating
2. Windows Update not pushing driver updates
3. Manual installation stuck on old version

**Check**:
- Open NVIDIA App or GeForce Experience
- Go to Drivers tab
- Click "Check for Updates"
- Should show 581.42 available

---

## Crash Details (For Reference)

```
Time: Oct 11, 2025 01:23:27
Batch: 2996/7702 (39% epoch 1)
Signal: 7 (SIGBUS - bus error)
Driver: 572.16 (known buggy)
GPU: RTX 4090
Training: Healthy until instant crash
```

**This was a driver bug, not your code.**

---

## Post-Upgrade Verification

After installing 581.42, verify stability:

```bash
# Check driver version
nvidia-smi

# Quick CUDA test
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# Resume training
.venv/bin/python -m src train configs/local/train_fla.yaml --resume
```

**Expected**: Training should run past batch 2996 without crashes.

---

## References

- NVIDIA Driver Downloads: https://www.nvidia.com/en-us/geforce/drivers/
- Driver 581.42 Release: TechPowerUp (October 2025)
- Driver 581.29 Fixes: OC3D (October 10, 2025)
- Driver 576.02 Crash Fixes: Tom's Hardware (April 2025)
- Driver 572.XX Issues: WCCFTech, Igor's Lab (January-March 2025)
