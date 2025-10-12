# Archive v2: Checkpoint System Fixes

**Status**: ✅ All issues RESOLVED
**Archive Date**: October 12, 2025
**Context**: Checkpoint robustness work from v3.10.0-v3.11.0 development

## Contents

This archive documents the **three critical checkpoint bugs** discovered and fixed during Modal A100 training:

### Major Fixes

1. **CHECKPOINT_RESUME_BUG.md** - Epoch re-training waste (14h per restart)
   - **Bug**: Saved `epoch` instead of `epoch+1` after completion
   - **Fix**: Changed to `epoch+1` in loop.py:464-465
   - **Impact**: Eliminated $672 wasted compute over 12 restarts
   - **Status**: ✅ Fixed in v3.10.0

2. **CHECKPOINT_BUFFER_BUG.md** - Buffer registration timing issue
   - **Bug**: `register_buffer(None)` doesn't add to state_dict until tensor assigned
   - **Fix**: Initialize with placeholder tensor + skip logic in checkpoint.py
   - **Impact**: Mid-epoch checkpoints now fully compatible
   - **Status**: ✅ Fixed in v3.10.0

3. **RNG_STATE_DEVICE_BUG.md** - RNG state device mismatch
   - **Bug**: `map_location` moved RNG states to CUDA (requires CPU)
   - **Fix**: Force RNG states to CPU before restoration
   - **Impact**: Deterministic resume now works on GPU
   - **Status**: ✅ Fixed in v3.10.0

### Strategy Documents

4. **MODAL_AUTO_RESTART_STRATEGY.md** - Auto-restart implementation guide
   - **Strategy**: modal.Period(hours=23) + concurrency_limit=1
   - **Result**: Hands-free 100-epoch training
   - **Status**: ✅ Implemented in v3.11.0

5. **MODAL_SSD_CHECKPOINT_STATUS.md** - Storage audit and path forward
   - **Analysis**: No contamination between smoke/full training
   - **Decision**: Accept one-time 14h waste, proceed with fixed code
   - **Status**: ✅ Resolved

6. **CLEAR_PATH_FORWARD.md** - Quick reference for proceeding
   - **Purpose**: Decision matrix for resuming training
   - **Result**: Training resumed successfully
   - **Status**: ✅ Complete

## Impact Summary

These fixes enabled:
- ✅ **Zero wasted compute**: Mid-epoch resume works perfectly
- ✅ **Auto-restart training**: Hands-free 100-epoch runs on Modal
- ✅ **Deterministic resume**: RNG state restoration works on GPU
- ✅ **$150+ savings**: Eliminated checkpoint-related compute waste

## Current Status

All checkpoint issues are **resolved**. The system now has:
- Atomic saves (temp + fsync + rename)
- AMP scaler + RNG + DataLoader state capture
- Mid-epoch resume with exact batch position
- Auto-restart via modal.Period
- Backward compatibility with old checkpoints

## Current Documentation

For up-to-date checkpoint information, see:
- Checkpoint strategy: `docs/05-training/checkpoint-strategy.md`
- Resume guide: `docs/05-training/resume.md`
- Modal training: `docs/05-training/modal.md`
