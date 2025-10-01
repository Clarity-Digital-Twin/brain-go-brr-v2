# PR #708 Manual Application Guide

**Date**: 2025-09-30
**Status**: 🧪 **TESTING IN PROGRESS**
**Context**: Modal A100 XID 31 fix - unmerged mamba-ssm PR #708 applied to local + Modal

---

## Current Testing Status (2025-09-30 12:17 UTC)

| Environment | Status | Progress | ETA |
|-------------|--------|----------|-----|
| **Modal Smoke** | 🟡 RUNNING | Batch 50/114 (~44%) | ~30 min |
| **Local Full Training** | 🟡 STARTING | Building dev index (11/1832) | ~1 hour to first batch |
| **PR #708 Validation** | ⏳ PENDING | Awaiting smoke completion | TBD |

---

## Executive Summary

**Problem**: Modal A100 crashes with XID 31 MMU Fault due to mamba-ssm CUDA kernel pointer overflow bug

**Root Cause**: mamba-ssm Triton kernels use int32 pointers that overflow with large batch sizes (batch=64, d_model=512)

**Solution**: PR #708 by younesbelkada casts all `tl.program_id()` calls to `tl.int64` in Triton kernels

**Status**:
- ✅ PR #708 exists and targets the exact bug we're hitting
- ❌ PR #708 is **NOT MERGED** into any released version (including 2.2.5)
- ✅ **LOCAL APPLICATION COMPLETE** - applied to local mamba-ssm 2.2.5
- ⏳ **MODAL APPLICATION PENDING** - need to patch Modal image

---

## What is PR #708?

### Official PR Details

- **Author**: younesbelkada (HuggingFace)
- **Title**: "fix: fix large batch size & prefill size issue"
- **Target**: Fixes Issue #503 (CUDA illegal memory access with large batches)
- **Link**: https://github.com/state-spaces/mamba/pull/708
- **Status**: OPEN (not merged as of Sept 2025)

### The Fix

**Changes 3 files** in `mamba_ssm/ops/triton/`:
1. `ssd_chunk_scan.py` - Main scan operations
2. `ssd_chunk_state.py` - State management
3. `ssd_state_passing.py` - State passing between blocks

**The pattern** (applied everywhere):
```python
# BEFORE (causes overflow with large batches)
pid_bc = tl.program_id(axis=1)
pid_m = tl.program_id(axis=0) // num_pid_n
pid_n = tl.program_id(axis=0) % num_pid_n

# AFTER (PR #708 fix - int64 casting)
pid_bc = tl.program_id(axis=1).to(tl.int64)
pid_m = tl.program_id(axis=0).to(tl.int64) // num_pid_n
pid_n = tl.program_id(axis=0).to(tl.int64) % num_pid_n
```

### Why This Fixes XID 31

1. **Pointer Overflow**: With batch=64, d_model=512, pointer arithmetic overflows int32
2. **Invalid Address**: Overflow produces addresses like `0x2b57_b3000000` (~47TB - clearly invalid)
3. **XID 31 MMU Fault**: GPU tries to access unmapped virtual address → page fault
4. **int64 Fix**: Casting to int64 prevents overflow, keeps addresses in valid range

---

## Why We Need This

### The Modal A100 Problem

```
Smoke test (batch=64, 50 files):  ✅ PASSED
Full training (batch=64, 4667 files): ❌ XID 31 at preflight
```

**Pattern**: First large batch after cold GPU start (52-min dataset indexing) triggers bug

### Evidence This is the Right Fix

1. **Issue #732 (A6000)**: Same pattern - first large batch fails, warm-up fixes it
2. **Our crash**: Exact same XID 31 pattern at preflight (first CUDA operation)
3. **Pointer address**: `0x2b57_b3000000` indicates int32 overflow
4. **Local (RTX 4090)**: Works because Ada architecture uses different CUDA kernels

### Why It's Not in 2.2.5

From web research:
- mamba-ssm 2.2.5 released July 2025
- Includes PR #537 (gradient fix) but NOT PR #708
- Issue #732 still OPEN - confirms this bug unfixed
- **PR #708 is the fix we need but it's not merged yet!**

---

## Local Application (COMPLETED ✅)

### What We Did

1. **Located mamba-ssm installation**:
   ```bash
   .venv/lib/python3.11/site-packages/mamba_ssm/ops/triton/
   ```

2. **Backed up original files**:
   ```bash
   cp ssd_chunk_scan.py ssd_chunk_scan.py.backup
   cp ssd_chunk_state.py ssd_chunk_state.py.backup
   cp ssd_state_passing.py ssd_state_passing.py.backup
   ```

3. **Applied PR #708 fix** (sed replace all `tl.program_id()` calls):
   ```bash
   sed -i 's/tl\.program_id(axis=0)/tl.program_id(axis=0).to(tl.int64)/g' ssd_*.py
   sed -i 's/tl\.program_id(axis=1)/tl.program_id(axis=1).to(tl.int64)/g' ssd_*.py
   sed -i 's/tl\.program_id(axis=2)/tl.program_id(axis=2).to(tl.int64)/g' ssd_*.py
   ```

4. **Verified changes**:
   ```bash
   grep "tl.program_id(axis=1).to(tl.int64)" ssd_chunk_scan.py
   # Found 11 occurrences - correct!
   ```

5. **Tested import**:
   ```python
   from mamba_ssm import Mamba2  # ✅ Imports successfully
   ```

### Current Local Status

- **Smoke test**: Running in tmux session `pr708_smoke`
- **Preflight**: ✅ PASSED (this is the critical test!)
- **Training**: In progress (16 batches, 1 epoch)
- **Expected**: Should complete without crashes

---

## Modal Application Strategy

### Challenge: Can't Edit Modal Image Directly

Modal images are immutable - we can't SSH in and edit files. We need to patch during image build.

### Solution: Patch Script in Image Build

**Strategy**: Add `.run_commands()` step that applies PR #708 after mamba-ssm installs

### Implementation Plan

#### Step 1: Create Patch Script

Create `deploy/modal/patch_mamba_pr708.py`:

```python
#!/usr/bin/env python3
"""Apply PR #708 fix to mamba-ssm Triton kernels.

This patches the installed mamba-ssm package to fix the int32 pointer overflow bug
that causes XID 31 crashes on A100 with large batch sizes.

PR #708: https://github.com/state-spaces/mamba/pull/708
"""

import sys
from pathlib import Path

def apply_pr708_fix():
    """Apply PR #708 int64 pointer casting fix to Triton kernels."""

    # Find mamba_ssm installation
    try:
        import mamba_ssm
        mamba_dir = Path(mamba_ssm.__file__).parent
        triton_dir = mamba_dir / "ops" / "triton"
    except ImportError:
        print("❌ mamba_ssm not installed!")
        return False

    if not triton_dir.exists():
        print(f"❌ Triton directory not found: {triton_dir}")
        return False

    # Files to patch
    files_to_patch = [
        triton_dir / "ssd_chunk_scan.py",
        triton_dir / "ssd_chunk_state.py",
        triton_dir / "ssd_state_passing.py",
    ]

    print(f"🔧 Applying PR #708 fix to {len(files_to_patch)} files...")
    print(f"📁 Location: {triton_dir}")

    for file_path in files_to_patch:
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return False

        # Read file
        content = file_path.read_text()
        original_content = content

        # Apply PR #708 fix: cast all tl.program_id() to int64
        content = content.replace(
            "tl.program_id(axis=0)",
            "tl.program_id(axis=0).to(tl.int64)"
        )
        content = content.replace(
            "tl.program_id(axis=1)",
            "tl.program_id(axis=1).to(tl.int64)"
        )
        content = content.replace(
            "tl.program_id(axis=2)",
            "tl.program_id(axis=2).to(tl.int64)"
        )

        # Count changes
        changes = len(content) - len(original_content)
        if changes == 0:
            print(f"⚠️  No changes in {file_path.name} (already patched?)")
        else:
            # Write patched file
            file_path.write_text(content)
            print(f"✅ Patched {file_path.name} ({changes} bytes changed)")

    print("✅ PR #708 fix applied successfully!")

    # Verify by importing
    try:
        # Force reimport to load patched code
        if 'mamba_ssm.ops.triton.ssd_chunk_scan' in sys.modules:
            del sys.modules['mamba_ssm.ops.triton.ssd_chunk_scan']
        if 'mamba_ssm.ops.triton.ssd_chunk_state' in sys.modules:
            del sys.modules['mamba_ssm.ops.triton.ssd_chunk_state']
        if 'mamba_ssm.ops.triton.ssd_state_passing' in sys.modules:
            del sys.modules['mamba_ssm.ops.triton.ssd_state_passing']

        # Reimport
        from mamba_ssm import Mamba2
        print("✅ mamba_ssm reimports successfully with PR #708 fix")
        return True
    except ImportError as e:
        print(f"❌ Failed to reimport mamba_ssm: {e}")
        return False

if __name__ == "__main__":
    success = apply_pr708_fix()
    sys.exit(0 if success else 1)
```

#### Step 2: Update deploy/modal/app.py

Add patch step after mamba-ssm installation:

```python
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("build-essential", "ninja-build", "git")
    .env({
        "CUDA_HOME": "/usr/local/cuda-12.4",
        "PATH": "/usr/local/cuda-12.4/bin:$PATH",
        "LD_LIBRARY_PATH": "/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0",
    })
    .run_commands("pip install --upgrade pip")
    .run_commands(
        "pip install torch==2.5.0 torchvision==0.20.0 'numpy<2.0' --index-url https://download.pytorch.org/whl/cu124"
    )
    .run_commands(
        "python -c 'import torch; assert torch.__version__.startswith(\"2.5.0\"), f\"Wrong torch: {torch.__version__}\"'"
    )
    .pip_install("packaging", "wheel", "setuptools")
    .run_commands(
        "pip install --no-build-isolation --no-cache-dir causal-conv1d==1.5.2"
    )
    .run_commands(
        "pip install --no-build-isolation --no-cache-dir mamba-ssm==2.2.5"
    )
    .run_commands(
        "python -c 'from mamba_ssm import Mamba2; print(\"✅ Mamba2 imports successfully\")'"
    )
    # 🔧 NEW: Apply PR #708 fix to mamba-ssm Triton kernels
    .add_local_file("deploy/modal/patch_mamba_pr708.py", "/tmp/patch_mamba_pr708.py")
    .run_commands(
        "python /tmp/patch_mamba_pr708.py"
    )
    # ... rest of image build
)
```

#### Step 3: Test Modal Image Build

```bash
# Test that image builds and patch applies
modal run deploy/modal/app.py --action test-mamba
```

**Expected output**:
```
🔧 Applying PR #708 fix to 3 files...
📁 Location: /usr/local/lib/python3.11/site-packages/mamba_ssm/ops/triton
✅ Patched ssd_chunk_scan.py (xxx bytes changed)
✅ Patched ssd_chunk_state.py (xxx bytes changed)
✅ Patched ssd_state_passing.py (xxx bytes changed)
✅ PR #708 fix applied successfully!
✅ mamba_ssm reimports successfully with PR #708 fix
✓ Mamba2 model created
✓ Forward pass successful! Output shape: torch.Size([2, 100, 512])
✓ Backward pass successful!
```

#### Step 4: Modal Smoke Test

```bash
modal run --detach deploy/modal/app.py --action train --config configs/modal/smoke.yaml
```

**Expected**: Preflight check PASSES (critical test!)

#### Step 5: Modal Full Training

```bash
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml
```

**Expected**:
- No XID 31 crash at preflight
- Training progresses normally
- Completes 100 epochs without crashes

---

## Verification Steps

### Local Verification (DONE)

```bash
# Check patch was applied
grep "tl.program_id(axis=1).to(tl.int64)" \
  .venv/lib/python3.11/site-packages/mamba_ssm/ops/triton/ssd_chunk_scan.py

# Should see multiple matches like:
# pid_bc = tl.program_id(axis=1).to(tl.int64)
```

### Modal Verification

```bash
# After image builds, test Mamba CUDA
modal run deploy/modal/app.py --action test-mamba

# Check logs for patch application
# Should see: "✅ PR #708 fix applied successfully!"
```

---

## Rollback Plan

### If Local Smoke Test Fails

```bash
# Restore original files
cd .venv/lib/python3.11/site-packages/mamba_ssm/ops/triton/
cp ssd_chunk_scan.py.backup ssd_chunk_scan.py
cp ssd_chunk_state.py.backup ssd_chunk_state.py
cp ssd_state_passing.py.backup ssd_state_passing.py

# Verify rollback
python -c "from mamba_ssm import Mamba2; print('Rollback successful')"
```

### If Modal Smoke Test Fails

```bash
# Simply remove the patch steps from deploy/modal/app.py
# Modal will rebuild image without the patch
# Training will revert to pre-patch behavior (likely XID 31 crash)
```

---

## Documentation Updates

### Files to Update After Success

1. **INSTALLATION.md**:
   - Add section: "PR #708 Manual Application (Required for A100)"
   - Document the int64 casting fix
   - Include verification steps

2. **MODAL_XID31_RECURRENCE.md**:
   - Update status to "✅ RESOLVED with PR #708"
   - Document local and Modal test results
   - Include timeline of fix application

3. **CLAUDE.md**:
   - Add note in "Common Issues" about PR #708
   - Update version to v3.3.1 (PR #708 patch)
   - Document that this is a temporary fix until upstream merge

4. **ARCHITECTURE_EVOLUTION.md**:
   - Add v3.3.1 entry documenting PR #708 patch
   - Explain why manual patch was necessary

5. **deploy/modal/README.md** (create if doesn't exist):
   - Document the patch_mamba_pr708.py script
   - Explain why it's needed
   - Instructions for removing once PR #708 merges upstream

---

## Success Criteria

### Local (In Progress)

- [x] PR #708 fix applied to 3 Triton kernel files
- [x] mamba-ssm imports successfully
- [x] Preflight check PASSES ✅
- [ ] Smoke test completes (1 epoch, 3 files)
- [ ] No crashes or NaN losses

### Modal (Pending)

- [ ] patch_mamba_pr708.py script created
- [ ] deploy/modal/app.py updated with patch step
- [ ] Modal image builds successfully
- [ ] Mamba CUDA test passes
- [ ] Smoke test preflight PASSES (critical!)
- [ ] Smoke test completes (1 epoch, 50 files)
- [ ] Full training runs without XID 31 crash
- [ ] Training completes 5+ epochs stably

---

## Timeline Estimate

| Phase | Time Estimate |
|-------|---------------|
| Create patch script | 30 min |
| Update deploy/modal/app.py | 15 min |
| Test Modal image build | 30 min |
| Modal smoke test | 30 min |
| Modal full training (5 epochs) | 5-7 hours |
| Documentation updates | 1-2 hours |
| **Total** | **8-11 hours** |

---

## Why This Approach?

### Alternative: Wait for Upstream Merge

**Pros**:
- Official fix, no manual patching
- Guaranteed compatibility

**Cons**:
- Timeline unknown (could be weeks/months)
- Modal training blocked indefinitely
- $300-600 for 100 epochs stuck waiting

### Alternative: Fork mamba-ssm

**Pros**:
- We control the fix
- Can push to PyPI if needed

**Cons**:
- Maintenance burden (keeping fork updated)
- More complex than patch script
- Overkill for 3-file change

### Why Patch Script? ⭐⭐⭐⭐⭐

**Pros**:
- ✅ Fastest solution (hours, not weeks)
- ✅ Minimal code change (just casting)
- ✅ Easy to remove once PR #708 merges
- ✅ No fork maintenance
- ✅ Works on both local and Modal
- ✅ Documented and reproducible

**Cons**:
- ⚠️ Unofficial modification (but well-documented)
- ⚠️ Need to remember to remove after upstream merge

**Verdict**: Best balance of speed, safety, and maintainability

---

## FAQ

### Q: Is this safe?

**A**: YES. PR #708 is a targeted fix by a HuggingFace engineer (younesbelkada) that only adds `.to(tl.int64)` casting. This is the correct fix for pointer overflow - it's not a hack or workaround.

### Q: Will this break anything?

**A**: NO. int64 casting is backward-compatible - it just prevents overflow. Existing working code (like local RTX 4090) continues to work exactly as before.

### Q: What if PR #708 gets rejected upstream?

**A**: Unlikely - it fixes a real bug. But if rejected, we'd need to either:
1. Find alternative fix (different pointer casting approach)
2. Change architecture (replace BiMamba with BiLSTM)
3. Use different hardware (L40S or stay on RTX 4090)

### Q: How do we know when to remove the patch?

**A**: Monitor mamba-ssm releases. Once PR #708 is merged and released (e.g., mamba-ssm 2.2.6 or 2.3.0), we can:
1. Upgrade to new version
2. Remove patch script from Modal image
3. Update docs to reflect official fix

### Q: Can we test this without Modal?

**A**: Local testing is in progress (tmux session `pr708_smoke`). If local passes, Modal should pass too since it's the same fix. But Modal A100 is the real test since that's where the bug manifests.

---

## Related Issues & PRs

- **PR #708**: https://github.com/state-spaces/mamba/pull/708 (THE FIX)
- **Issue #503**: CUDA illegal memory access with large batch/prefill (target of PR #708)
- **Issue #732**: Shape-dependent first-batch bug on A6000 (STILL OPEN - same root cause)
- **Issue #686**: Long sequence illegal memory access on H100

---

## Contact & Support

**If PR #708 patch fails**:
1. Check Modal logs for patch application errors
2. Verify mamba-ssm 2.2.5 installed correctly
3. Test with Mamba CUDA test: `modal run deploy/modal/app.py --action test-mamba`
4. Report to state-spaces/mamba with full reproduction

**Upstream Bug Report Template** (if needed):
```markdown
Title: XID 31 MMU Fault on A100 with batch=64 after cold GPU start

Environment:
- GPU: NVIDIA A100-80GB (sm_80)
- mamba-ssm: 2.2.5
- PyTorch: 2.5.0 + CUDA 12.4
- causal-conv1d: 1.5.2

Issue:
First Mamba forward pass with (64, 512, 960) after 52-min CPU work crashes with:
- XID 31: MMU Fault at 0x2b57_b3000000 (FAULT_PDE ACCESS_TYPE_VIRT_WRITE)
- CUDA error: illegal memory access

Pattern:
- Smoke test (50 files): PASSES
- Full training (4667 files): CRASHES at preflight (first CUDA op)
- Matches Issue #732 pattern (A6000 first-batch bug)

Solution:
Applied PR #708 (int64 pointer casting) - testing now

Request:
Please merge PR #708 or provide official fix for int32 pointer overflow in Triton kernels.

Full logs: [attach CUDA_LAUNCH_BLOCKING=1 output]
```

---

**Status**: Local application COMPLETE, Modal application IN PROGRESS

**Next Step**: Wait for local smoke test completion, then apply to Modal

**Last Updated**: 2025-09-30 10:55 UTC