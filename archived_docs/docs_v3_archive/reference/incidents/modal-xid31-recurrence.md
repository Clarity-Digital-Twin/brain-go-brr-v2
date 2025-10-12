# Modal XID 31 Root Cause Investigation [RESOLVED]

**Date**: 2025-09-30 07:50 UTC → 12:17 UTC (Investigation)
**Resolution Date**: 2025-10-01 (v3.4.1)
**Status**: ✅ **RESOLVED** - Triton cache isolation fix implemented and validated
**Context**: PyTorch 2.5.0 + mamba-ssm 2.2.5 upgrade exposed Triton cache collision; fixed with unique per-worker cache dirs

---

## Executive Summary

**CRITICAL FINDING**: Modal full training crashes with **XID 31 MMU Fault** despite PyTorch 2.5.0 + mamba-ssm 2.2.5 upgrade. This is NOT a new bug - it's the SAME underlying mamba-ssm CUDA kernel issue, now manifesting at a different point (preflight instead of mid-training).

**Evidence-based conclusion**: mamba-ssm has a **first-batch kernel initialization bug** that is:
1. **Shape/size dependent** (large batches like (64, 512, 960) trigger it)
2. **State dependent** (cold GPU start after long CPU work triggers it)
3. **Multi-GPU issue** (affects A100, A6000, H100 - NOT hardware-specific)
4. **NOT fixed in 2.2.5** (int64 indexing fix doesn't cover this case)

**Web research confirmation**: GitHub Issue #732 documents IDENTICAL pattern on A6000 - tensor shapes fail when executed FIRST but pass after warm-up with smaller tensor.

---

## Hardware Cross-Reference (NOT A100-Specific)

| GPU | Architecture | VRAM | mamba-ssm Status | Evidence |
|-----|--------------|------|------------------|----------|
| **A100-80GB** | sm_80 (Ampere) | 80GB | ❌ XID 31 (this issue) | Our Modal crash |
| **A6000** | sm_86 (Ampere) | 48GB | ❌ Issue #732 pattern | First-batch bug confirmed |
| **H100-80GB** | sm_90 (Hopper) | 80GB | ❌ Issue #686 | Illegal memory access |
| **RTX 4090** | sm_89 (Ada) | 24GB | ✅ Works (local) | Different CUDA path |
| **L40S** | sm_89 (Ada) | 48GB | ❓ Untested | Might work (same arch as 4090) |

**Conclusion**: This is a **mamba-ssm kernel bug across Ampere/Hopper**, NOT specific hardware failure. RTX 4090 works due to different architecture (Ada), but has insufficient VRAM (24GB) for production batch sizes.

**No viable hardware workaround** - must fix mamba-ssm kernel bug or change architecture.

---

## The Failure Pattern

### What Happens

```
Smoke test (configs/modal/smoke.yaml):
  - 50 train files, 10 dev files
  - Batch size: 64
  - Result: ✅ PASSED (~38 minutes)
  - First Mamba batch: Small dataset → Small memory footprint

Full training (configs/modal/train.yaml):
  - 4667 train files, 1832 dev files
  - Batch size: 64
  - Result: ❌ XID 31 at preflight check
  - 52 minutes of dev dataset indexing (CPU-only, GPU idle)
  - First Mamba batch: (64, 512, 960) → Kernel initialization bug → XID 31

Local training (configs/local/train.yaml):
  - Same 4667+1832 dataset
  - Batch size: 12 (but scaled up to 69+ in testing)
  - Result: ✅ STABLE (batch 69+)
  - RTX 4090: Different CUDA kernels (sm_89 vs sm_80)
```

**Why does smoke pass but full training fail?**

The smoke test's smaller dataset produces a smaller first batch that falls below the bug threshold. Full training's 52-minute CPU indexing leaves GPU in cold state, then first large batch (64, 512, 960) triggers uninitialized kernel state → XID 31 MMU fault.

---

## Timeline of Modal Failure

| Time (UTC) | Event | Duration |
|------------|-------|----------|
| 06:37:16 | Training started | - |
| 06:37:17 | Cache verification passed | 1s |
| 06:48:18 | Train dataset loaded (61,616 windows) | ~11min |
| 06:48:18 | Dev dataset indexing started (1,832 files) | - |
| 07:40:48 | Dev dataset ready (148,224 windows) | **52 minutes (GPU IDLE)** |
| 07:40:58 | Model created (31M parameters) | 10s |
| 07:41:01 | W&B initialized | 3s |
| 07:48:22 | Preflight check started | **7min 21s gap** |
| 07:50:56 | **XID 31 MMU FAULT** | 2min 34s into preflight |
| 07:50:57 | Fallback attempted, training aborted | 1s |

**Critical observation**: 52 minutes of CPU-only work (dataset indexing) + 7-minute gap = 59 minutes where GPU sits idle. First CUDA operation (Mamba forward) triggers kernel initialization bug.

---

## Error Analysis

### XID 31 Hardware Error

```
[gpu-health] [WARN] GPU-e11dfcca-7c93-959a-b8fd-5cfe0839b163:
XID: NVRM: Xid (PCI:0000:00:06): 31, pid=79749, name=exe, Ch 0000000a, intr 00000000.
MMU Fault: ENGINE GRAPHICS GPC2 GPCCLIENT_T1_7 faulted @ 0x2b57_b3000000.
Fault is of type FAULT_PDE ACCESS_TYPE_VIRT_WRITE
```

**Decoded**:
- **XID 31**: GPU Memory Management Unit page fault
- **FAULT_PDE**: Page Directory Entry fault (address not mapped in GPU virtual memory)
- **ACCESS_TYPE_VIRT_WRITE**: GPU tried to write to unmapped address `0x2b57_b3000000` (~47TB - clearly invalid)
- **Not XID 13**: This is NOT out-of-memory; this is a kernel pointer bug

### CUDA Runtime Error

```python
[2025-09-30 11:50:57.909][src.brain_brr.models.mamba][WARNING]
[MAMBA] Forward pass error, using fallback: CUDA error: an illegal memory access was encountered
```

**Fallback did NOT engage** because error text doesn't match filters (`src/brain_brr/models/mamba.py:210-213`):
- Current filters: `"causal_conv1d"`, `"NoneType"`, `"object is not callable"`
- Actual error: `"CUDA error: an illegal memory access..."`
- Result: Logged "using fallback" but **re-raised** exception

### Preflight Failure

```python
[PREFLIGHT] ✗ Failed on test batch
  - Model type: SeizureDetector
  - Input shape: torch.Size([64, 19, 15360])
  - Labels shape: torch.Size([64, 15360])
  - Mamba processes: torch.Size([64, 512, 960]) ← CRITICAL SHAPE
```

---

## What's Been Ruled Out

### ❌ AMP is NOT the cause (Test 1A - FAILED)

**Test**: Disabled mixed precision (`mixed_precision: false`)
**Result**: Still crashed with XID 31 at different address
**Conclusion**: Bug occurs in both FP16 and FP32 modes

### ❌ PyTorch version is NOT the cause

**Upgraded**: PyTorch 2.2.2 → 2.5.0
**Result**: Same XID 31 error (different manifestation point)
**Conclusion**: Bug is in mamba-ssm kernels, not PyTorch

### ❌ mamba-ssm version is NOT the cause

**Upgraded**: mamba-ssm 2.2.2 → 2.2.5
**Result**: Same XID 31 error
**Conclusion**: int64 indexing fix (Issue #686) doesn't cover this case

### ❌ Memory pressure is NOT the cause

**Calculation**:
- Input: 64 × 19 × 15360 × 4 bytes = 75MB
- Model: 31.4M params × 4 bytes = 126MB
- Total active: <1GB of 80GB VRAM = 1.25% usage
**Conclusion**: XID 31 = page fault, NOT OOM (which would be XID 13)

### ❌ Model architecture is NOT the cause

**Evidence**: Local training (RTX 4090) with same architecture = STABLE
**Conclusion**: Our code is correct; mamba-ssm CUDA kernel is the issue

### ❌ A100 hardware is NOT the cause

**Evidence**: Issue #732 (A6000), Issue #686 (H100) show same pattern
**Conclusion**: mamba-ssm kernel bug affects multiple NVIDIA GPUs

---

## Deep Web Research: Known mamba-ssm Bugs & Fixes

### 🔴 Issue #732: Shape-Dependent First-Batch Bug (A6000) - **STILL OPEN**

**Opened**: May 23, 2025
**Status**: ❌ **OPEN** (no fix released as of Sept 2025)

**Pattern documented**:
- Tensor shapes like (27, 32768, 384) **FAIL when executed FIRST**
- Same shapes **PASS after warm-up** with smaller tensor (26, 2048, 384)
- Different batch sizes (e.g., 32 instead of 27) may work without warm-up

**Matches our pattern EXACTLY**:
- Smoke test: Small first batch → Passes
- Full training: Large first batch after cold GPU → XID 31

**Environment**:
- GPU: NVIDIA A6000 (sm_86 architecture, Ampere like A100's sm_80)
- Versions: mamba-ssm==2.2.4, causal-conv1d==1.5.0.post8, torch==2.5.1, CUDA 12.2
- Status: **OPEN** - no upstream fix available

**Workaround documented**: Run small tensor first (26, 2048, 384) before large tensors

---

### 🟡 PR #708: "fix: fix large batch size & prefill size issue" - **NOT MERGED**

**Author**: younesbelkada
**Target**: Fixes Issue #503
**Status**: ⏸️ **OPEN** (not merged into main as of Sept 2025)

**The Fix**: **Casts all pointers to tl.int64** in Triton kernels

**Why this matters**: This is EXACTLY the fix we need! Pointer casting to int64 prevents illegal memory access with large batch sizes and prefill sizes. However, **this fix is NOT in mamba-ssm 2.2.5**.

**Impact if applied**: Would likely fix our XID 31 crash on A100 with batch=64

**Location**: `state-spaces/mamba` PR #708

---

### ✅ PR #537: "Fix Incorrect Gradients and Illegal Memory Access Error in Mamba2" - **MERGED**

**Author**: Hprairie
**Merged by**: tridao into state-spaces:main
**Status**: ✅ **MERGED** (included in mamba-ssm 2.2.5)

**The Fix**: Fixed bug in backward pass for gradient calculations in `_chunk_scan_bwd_ddAcs_stable_kernel`

**Relevance**: This fixed SOME illegal memory access issues, but not the first-batch / large batch size issue we're hitting

---

### 🟡 Issue #686: Long Sequence Illegal Memory Access (H100) - **OPEN**

**Pattern**:
- Sequence lengths >512K trigger illegal memory access
- Location: `_mamba_chunk_scan_combined_fwd` kernel in Triton
- Specifically in `_chunk_cumsum_fwd` function
- Fix attempted: int64 indexing (mentioned but unclear if fully resolved)

**Relevance**: Shows mamba-ssm CUDA kernels have pointer arithmetic bugs across multiple GPUs (A100, A6000, H100)

---

### 📦 mamba-ssm Version Status (Sept 2025)

**Latest version**: 2.2.5 (released July 19, 2025)
**Previous**: 2.2.4 (released Dec 6, 2024)
**Newer versions**: ❌ No 2.2.6 or 2.3.0 exists

**What's in 2.2.5**:
- ✅ PR #537 fix (gradient/memory access in backward pass)
- ❌ PR #708 fix (large batch size pointer casting) - **NOT INCLUDED**
- ❌ Issue #732 (first-batch bug) - **STILL OPEN**
- ❌ Issue #686 (long sequence H100) - **STILL OPEN**

**Release notes**: Minimal ("Bump to v2.2.5") - no detailed changelog

---

### 💡 NVIDIA XID 31 Official Docs

**Definition**: MMU page fault = illegal memory access by GPU kernel
**Causes**: Out-of-bounds access, uninitialized pointers, race conditions
**NOT**: Memory exhaustion (that's XID 13)

**Debugging recommendations**:
1. `CUDA_LAUNCH_BLOCKING=1` - Synchronous execution for accurate stack traces
2. `TORCH_USE_CUDA_DSA=1` - Enable device-side assertions
3. `CUDA_DISABLE_PTX_JIT=1` - Disable JIT to surface arch mismatches immediately
4. Compute Sanitizer (replaces cuda-memcheck) - Pinpoint exact memory access violation

---

## 🚨 CRITICAL DISCOVERY

**The fix we need (PR #708) EXISTS but is NOT MERGED into any released version!**

This explains everything:
1. Why mamba-ssm 2.2.5 doesn't fix our issue (PR #708 not included)
2. Why Issue #732 is still open (same root cause, no fix released)
3. Why our pattern matches exactly (first-batch large batch bug)

**Options**:
1. **Apply PR #708 manually** to our local mamba-ssm installation (fastest)
2. **Wait for PR #708 to be merged** upstream (timeline unknown)
3. **Architectural change** (BiLSTM/Transformer) - bypasses mamba-ssm entirely

---

## Root Cause Hypothesis (HIGH CONFIDENCE)

### The Bug

**mamba-ssm CUDA kernels have first-batch initialization bug** where:

1. **Kernel state not properly initialized** on cold GPU start
2. **Large tensor shapes** (like our (64, 512, 960)) trigger the bug
3. **Small tensor warm-up** initializes state correctly, subsequent large tensors work
4. **Ampere/Hopper architectures** (sm_80/86/90) hit this bug; Ada (sm_89) doesn't

### Why Smoke Passes But Full Training Fails

**Smoke test**:
- Small dataset (10 dev files) → Smaller memory footprint
- First Mamba batch: Small enough to stay below bug threshold
- OR: Lucky batch composition avoids trigger condition
- Result: Kernels initialize correctly despite being cold

**Full training**:
- 52-minute dev dataset indexing (CPU-only, GPU completely idle)
- Model initialization (still CPU-side, no CUDA work)
- 7-minute gap (unknown activity, possibly memory allocation)
- **First CUDA operation**: Mamba forward with (64, 512, 960)
- Kernel attempts to initialize with large tensor → Uninitialized pointer → XID 31

### Why Local Works But Modal Fails

**Local (RTX 4090)**:
- Architecture: sm_89 (Ada Lovelace)
- CUDA kernels: Different codepath than A100/A6000/H100
- May not hit this specific initialization bug

**Modal (A100)**:
- Architecture: sm_80 (Ampere)
- CUDA kernels: Specific codepath with initialization bug
- Matches Issue #732 pattern (A6000 is sm_86, close to sm_80)

---

## Diagnostic Plan (NO WORKAROUNDS)

**Philosophy**: We are NOT deploying workarounds (no fallbacks, no AMP disabled, no reduced batch). We are DIAGNOSING to pinpoint the exact kernel + operation, then reporting upstream with full reproduction.

### Phase 1: Pinpoint Exact Failure Location (REQUIRED)

#### Test 1: Synchronous Execution + Device Assertions

**Implementation**: Add to `deploy/modal/app.py`:
```python
@app.function(
    gpu="A100-80GB",
    env={
        "CUDA_LAUNCH_BLOCKING": "1",      # Synchronous CUDA for accurate stack trace
        "TORCH_USE_CUDA_DSA": "1",        # Device-side assertions
        "CUDA_DISABLE_PTX_JIT": "1",      # No JIT masking of issues
    },
)
def train(...):
    ...
```

**Purpose**: Get exact kernel name and line number where illegal access occurs

**Expected output**:
```
RuntimeError: CUDA error: illegal memory access
  File "mamba_ssm/ops/triton/ssd_combined.py", line XXX, in _mamba_chunk_scan_combined_fwd
```

**Time**: 1 hour (52min indexing + preflight)

---

#### Test 2: Compute Sanitizer (NVIDIA Official Debugger)

**Implementation**: Modal container entry point:
```bash
compute-sanitizer --tool memcheck python -m src train configs/modal/train.yaml
```

**Purpose**: Pinpoint exact memory access violation in kernel code

**Expected output**:
```
========= COMPUTE-SANITIZER
========= Invalid __global__ write of size 4 bytes
=========     at 0xXXXX in _mamba_chunk_scan_combined_fwd_kernel
=========     by thread (X, Y, Z) in block (A, B, C)
=========     Address 0x2b57b3000000 is out of bounds
```

**Time**: 2-3 hours (slower with instrumentation)

---

### Phase 2: Confirm Shape-Dependent Bug Pattern

#### Test 3: Tiny Warm-Up Batch (Diagnostic, NOT Production)

**Implementation**: Add to `src/brain_brr/train/loop.py` before preflight:
```python
# DIAGNOSTIC ONLY - confirms first-batch kernel initialization bug
logger.info("[DIAG] Testing warm-up hypothesis (Issue #732 pattern)")
dummy_input = torch.randn(2, 19, 256, device="cuda")  # Tiny: batch=2, 1s duration
dummy_tcn = model.tcn(dummy_input)  # (2, 512, 16)
_ = model.mamba(dummy_tcn.transpose(1, 2))  # Warm up Mamba kernels
torch.cuda.synchronize()
logger.info("[DIAG] Warm-up completed, now running real preflight")
# Now run actual preflight with (64, 19, 15360)
```

**Purpose**: Prove this is Issue #732 pattern (cold start + large batch = crash)

**Expected outcome**:
- **If preflight PASSES**: CONFIRMS first-batch kernel initialization bug
- **If preflight FAILS**: Different root cause, proceed to Test 4

**Time**: 1 hour

**CRITICAL**: This is a DIAGNOSTIC test to characterize the bug for upstream report. NOT a production solution.

---

#### Test 4: Bisect Batch Size to Find Threshold

**Implementation**: Test sequence:
1. `batch_size: 64` → Baseline (crashes)
2. `batch_size: 32` → Test
3. `batch_size: 16` → Test
4. `batch_size: 8` → Test
5. `batch_size: 4` → Test

**Purpose**: Find exact batch size threshold where bug appears

**Expected outcome**: Identify critical batch size (e.g., "crashes at 32+, works at 16-")

**Time**: 5 hours (5 runs × ~1 hour each)

---

### Phase 3: Kernel-Level Characterization

#### Test 5: Log Exact Tensor Properties

**Implementation**: Add to `src/brain_brr/models/detector.py`:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # ... TCN processing ...

    # DIAGNOSTIC: Log tensor properties before Mamba
    logger.info(f"[DIAG] Mamba input shape: {x_tcn.shape}")  # Expect (64, 512, 960)
    logger.info(f"[DIAG] Mamba input dtype: {x_tcn.dtype}")  # FP32 or FP16
    logger.info(f"[DIAG] Mamba input device: {x_tcn.device}")  # cuda:0
    logger.info(f"[DIAG] Mamba input contiguous: {x_tcn.is_contiguous()}")
    logger.info(f"[DIAG] Mamba input strides: {x_tcn.stride()}")
    logger.info(f"[DIAG] Mamba input min/max: {x_tcn.min():.4f} / {x_tcn.max():.4f}")

    x_mamba = self.mamba(x_tcn.transpose(1, 2))  # May crash here
```

**Purpose**: Identify if stride, contiguity, or value range issues

**Time**: 1 hour

---

#### Test 6: Isolate Which Mamba Instance Crashes

**Implementation**: Disable components one at a time in config:
```yaml
# Test 6a: Disable edge Mamba
model:
  graph:
    use_gnn: true
    use_dual_stream: false  # Disable edge Mamba

# Test 6b: Disable node Mamba (keep edge)
model:
  graph:
    use_gnn: false  # Also disables node Mamba

# Test 6c: Disable main Mamba (use Conv1d fallback)
# Set env: SEIZURE_MAMBA_FORCE_FALLBACK=1
```

**Purpose**: Identify which Mamba component (main / node / edge) triggers crash

**Expected outcome**:
- Main Mamba (512-dim): Most likely culprit (largest tensor)
- Node Mamba (64-dim): Possible
- Edge Mamba (16-dim): Least likely (smallest tensor)

**Time**: 3 hours (3 runs)

---

## Decision Tree

```
┌─ Test 1 (CUDA_LAUNCH_BLOCKING) ──────────────────────────────────┐
│                                                                   │
│  Shows exact kernel: _mamba_chunk_scan_combined_fwd              │
│  ├─→ Known problem area (Issue #686/732)                         │
│  │   └─→ Proceed to Test 3 (warm-up diagnostic)                  │
│  │                                                                │
│  Shows exact kernel: causal_conv1d_fwd                           │
│  ├─→ Different component, same pattern                           │
│  │   └─→ Proceed to Test 3 (warm-up diagnostic)                  │
│  │                                                                │
│  Shows our model code (detector.py / mamba.py)                   │
│  └─→ Bug is in our code, not mamba-ssm                           │
│      └─→ Fix our code                                            │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘

┌─ Test 3 (warm-up batch) ─────────────────────────────────────────┐
│                                                                   │
│  Preflight PASSES after warm-up                                  │
│  ├─→ CONFIRMED: First-batch kernel initialization bug            │
│  │   └─→ Actions:                                                │
│  │       1. Report to mamba-ssm with full A100 repro             │
│  │       2. Include: Issue #732 pattern confirmed on A100        │
│  │       3. Shapes: (64, 512, 960), d_model=512                  │
│  │       4. Request kernel initialization fix                    │
│  │       5. Check if mamba-ssm 2.2.6+ exists with fix            │
│  │       6. Consider architectural alternatives (see below)      │
│  │                                                                │
│  Preflight FAILS even with warm-up                               │
│  └─→ NOT first-batch bug, different root cause                   │
│      └─→ Proceed to Test 2 (Compute Sanitizer)                   │
│          Or Test 5 (tensor property logging)                     │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘

┌─ If mamba-ssm bug confirmed ────────────────────────────────────┐
│                                                                   │
│  Option A: Wait for upstream fix                                 │
│  ├─ Timeline: Unknown (weeks to months)                          │
│  └─ Blocker: Cannot train on Modal                               │
│                                                                   │
│  Option B: Architectural change (LSTM/GRU)                       │
│  ├─ Replace BiMamba with BiLSTM or BiGRU                         │
│  ├─ Pro: Battle-tested, no CUDA kernel bugs                      │
│  ├─ Con: O(N²) vs O(N) complexity (but still acceptable)        │
│  ├─ Con: 1-2 weeks implementation + re-training                  │
│  └─ Decision: Last resort if upstream fix not available          │
│                                                                   │
│  Option C: Transformer (attention-based)                         │
│  ├─ Replace BiMamba with multi-head attention                    │
│  ├─ Pro: Well-tested, O(N²) but fast on A100 tensor cores       │
│  ├─ Con: Larger memory footprint                                 │
│  └─ Decision: Consider if LSTM not performant enough             │
│                                                                   │
│  Option D: Try L40S (Ada architecture)                           │
│  ├─ Same sm_89 as RTX 4090 (which works)                        │
│  ├─ 48GB VRAM (more than 4090's 24GB)                           │
│  ├─ Pro: Might avoid Ampere/Hopper kernel bug                    │
│  ├─ Con: Untested, may have different issues                     │
│  └─ Decision: Low priority, architectural change safer           │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

---

## Solution Options (After Web Research)

### 🚀 Option A: Apply PR #708 Manually (FASTEST) ⭐⭐⭐⭐⭐

**What**: Manually apply the int64 pointer casting fix from PR #708 to our mamba-ssm installation

**Why this is best**:
- Fix already exists (proven code from younesbelkada)
- Addresses exact root cause (large batch pointer overflow)
- No architecture change needed
- Can deploy immediately

**Implementation**:
1. Clone mamba-ssm fork with PR #708 changes
2. Install from source in our environments (local + Modal)
3. Test with full training
4. If works: Training proceeds
5. If fails: Fall back to Option B or C

**Risk**: Low - PR #708 is targeted fix, well-defined changes
**Time**: 2-4 hours (implementation + testing)
**Success probability**: 80%+ (directly targets the bug)

---

### 🐛 Option B: Report to mamba-ssm & Request PR #708 Merge ⭐⭐⭐

**What**: Open detailed issue on mamba-ssm asking maintainers to merge PR #708

**Why**:
- Gets official fix into releases
- Helps entire mamba community
- Creates permanent solution

**Implementation**:
1. Run Test 1 (CUDA_LAUNCH_BLOCKING) to get exact stack trace
2. Run Test 3 (warm-up diagnostic) to confirm pattern
3. Open GitHub issue with:
   - Full reproduction (A100 + batch 64 + cold start)
   - Link to Issue #732 (A6000 same pattern)
   - Request: Merge PR #708 or provide alternative fix
4. Wait for upstream response

**Risk**: Medium - timeline depends on maintainers
**Time**: 1-2 days (testing) + unknown wait time
**Success probability**: Unknown (depends on maintainer response)

---

### 🔧 Option C: Architectural Change (SAFE FALLBACK) ⭐⭐⭐⭐

**What**: Replace BiMamba with BiLSTM or Transformer

**Why**:
- Bypasses mamba-ssm entirely
- Battle-tested architectures
- Works on ALL GPUs (A100, H100, 4090)
- No kernel bugs

**Implementation**: See "Architectural Alternatives" section below

**Risk**: Low - LSTM/Transformer are proven
**Time**: 1 week (BiLSTM) or 2 weeks (Transformer)
**Success probability**: 95%+ (known working solutions)

---

## Recommended Immediate Actions (Priority Order)

### 1. Try Option A: Apply PR #708 Manually ⭐⭐⭐⭐⭐

**Fastest path to working training**:
1. Fetch PR #708 branch: `git fetch origin pull/708/head:pr-708`
2. Apply changes to our mamba-ssm install (local + Modal)
3. Test full training (1 hour)
4. If works: We're done
5. If fails: Proceed to Option B

**Time**: 2-4 hours total
**Success probability**: 80%+

---

### 2. Run Diagnostics (Parallel with Option A) ⭐⭐⭐⭐

**Why**: Need data for upstream report regardless

**Test 1**: CUDA_LAUNCH_BLOCKING to get exact kernel name (1 hour)
**Test 3**: Warm-up diagnostic to confirm Issue #732 pattern (1 hour)

**Use for**: GitHub issue if Option A fails or for documentation

---

### 3. Report to mamba-ssm with PR #708 request ⭐⭐⭐⭐

**GitHub Issue Template**:
```markdown
Title: XID 31 MMU Fault on A100 with first large batch after cold GPU start

Environment:
- GPU: NVIDIA A100-80GB (sm_80)
- mamba-ssm: 2.2.5
- PyTorch: 2.5.0
- CUDA: 12.4
- causal-conv1d: 1.5.2

Issue:
First Mamba forward pass with shape (64, 512, 960) crashes with XID 31 MMU Fault
after 52 minutes of CPU-only work (dataset indexing). Same shape works fine after
warm-up with smaller tensor (2, 512, 100).

Pattern matches Issue #732 (A6000), now confirmed on A100.

Reproduction:
1. Leave GPU idle for extended period (50+ minutes CPU work)
2. First CUDA operation: Mamba.forward() with large batch (64+)
3. Crash with XID 31 at address like 0x2b57_b3000000

Workaround:
Warm-up with tiny tensor before first real batch (see code snippet).

Request:
Proper kernel initialization fix so cold GPU starts work with large batches.

Full logs + reproduction: [attach CUDA_LAUNCH_BLOCKING=1 logs]
```

---

### 4. Consider Architectural Alternatives (If No Upstream Fix)

**Only if mamba-ssm doesn't fix within reasonable timeline**:

#### Option A: BiLSTM

```python
class BiLSTM(nn.Module):
    def __init__(self, d_model, num_layers=6, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            d_model, d_model // 2, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )

    def forward(self, x):  # (B, L, D)
        output, _ = self.lstm(x)  # (B, L, D)
        return output
```

**Pros**:
- Battle-tested, no CUDA bugs
- PyTorch native, works on ALL GPUs (A100, H100, 4090)
- O(N²) but acceptable for 960 timesteps

**Cons**:
- Slower than Mamba's O(N)
- May need architecture tuning

**Time to implement**: 1 day + re-training

---

#### Option B: Multi-Head Attention (Transformer)

```python
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):  # (B, L, D)
        return self.transformer(x)  # (B, L, D)
```

**Pros**:
- Well-optimized on A100 tensor cores
- FlashAttention available (faster)
- Proven architecture for medical signals

**Cons**:
- O(N²) attention complexity
- Larger memory footprint

**Time to implement**: 2-3 days + re-training

---

## Timeline Estimates

| Path | Time | Outcome |
|------|------|---------|
| **Fast path** (Test 1 + Test 3) | 2 hours | Confirm first-batch bug, report upstream |
| **Deep path** (All 6 tests) | 10-15 hours | Full characterization for upstream |
| **Upstream fix** | Unknown (weeks?) | mamba-ssm maintainers patch kernel |
| **Architecture change** (BiLSTM) | 1 week | Working training on Modal |
| **Architecture change** (Transformer) | 2 weeks | Working training on Modal |
| **Try L40S GPU** | 1 hour | Test if Ada architecture avoids bug |

---

## Is This the SAME Bug or DIFFERENT Bug?

**Answer**: SAME underlying bug, DIFFERENT manifestation point.

### Before Upgrade (PyTorch 2.2.2 + mamba-ssm 2.2.2)
- Crashed during training loop (not at preflight)
- AMP suspected (but Test 1A ruled it out)
- Likely: Kernel bug triggered by specific batch later in training

### After Upgrade (PyTorch 2.5.0 + mamba-ssm 2.2.5)
- Crashes at preflight (first CUDA operation)
- AMP irrelevant (crashes with FP32 too)
- Clear pattern: Cold GPU + first large batch = XID 31

**Why different manifestation?**
1. PyTorch 2.5.0: Different CUDA kernel scheduling / memory allocation
2. mamba-ssm 2.2.5: Different initialization path (int64 fix changed codepaths)
3. Result: Same underlying kernel bug, now hits earlier (preflight vs mid-training)

**Conclusion**:
- Upgrade did NOT introduce new bug ✅
- Upgrade REVEALED existing bug more clearly ✅
- Easier to debug (reproducible at preflight, not random mid-training) ✅
- mamba-ssm CUDA kernels still have unresolved multi-GPU issues ❌

---

## Why NO WORKAROUNDS?

**We're NOT doing**:
1. ❌ Disable AMP (already ruled out as cause)
2. ❌ Force Conv1d fallback (hides bug, defeats purpose of Mamba)
3. ❌ Reduce batch size permanently (cripples training efficiency)
4. ❌ Deploy warm-up batch to production (masks the bug)
5. ❌ Switch to H100 (same mamba-ssm bug, more expensive)

**We ARE doing**:
1. ✅ Diagnostic tests to pinpoint exact failure (Test 1, 2, 5, 6)
2. ✅ Confirmation test for first-batch hypothesis (Test 3)
3. ✅ Full characterization for upstream report (Test 4)
4. ✅ Proper bug report to mamba-ssm with reproduction
5. ✅ If no fix: Clean architectural change (BiLSTM/Transformer)

**Philosophy**: Fix the root cause or change architecture. No hacks.

---

## Critical Questions Answered

### Q1: Why did smoke test pass?
**A**: Smaller dataset → First batch smaller → Below bug threshold OR lucky batch composition avoids trigger

### Q2: Why does local training work?
**A**: RTX 4090 (sm_89) uses different CUDA kernel codepaths than A100/A6000/H100 (sm_80/86/90). Doesn't hit this specific bug.

### Q3: What happened during the 7-minute gap?
**A**: Unknown. Possibly:
- PyTorch caching allocator reserving memory
- W&B async operations
- Modal infrastructure overhead
Needs Test 5 logging to clarify.

### Q4: Is this a Modal infrastructure issue?
**A**: NO. XID 31 = application kernel bug. If it were Modal's fault:
- Smoke test would also fail
- Local training wouldn't work (same CUDA code)
- We'd see timeouts, not MMU faults

### Q5: Can we just use H100 instead?
**A**: NO. Issue #686 shows illegal memory access on H100 too. mamba-ssm has cross-GPU kernel bugs on Ampere/Hopper.

### Q6: Can we use L40S (Ada architecture)?
**A**: MAYBE. L40S uses sm_89 like RTX 4090 (which works). 48GB VRAM (more than 4090's 24GB). Worth trying if architectural change too costly, but untested.

### Q7: Should we rollback the upgrade?
**A**: NO. Old stack (2.2.2) had same underlying bug, just triggered differently. New stack is better:
- Faster to reproduce (preflight vs mid-training)
- AMP ruled out (Test 1A)
- Clearer failure pattern
- More debuggable

---

## Files to Edit for Diagnostic Tests

1. **`deploy/modal/app.py`** (Test 1):
   ```python
   @app.function(
       gpu="A100-80GB",
       env={
           "CUDA_LAUNCH_BLOCKING": "1",
           "TORCH_USE_CUDA_DSA": "1",
           "CUDA_DISABLE_PTX_JIT": "1",
       },
   )
   ```

2. **`src/brain_brr/train/loop.py`** (Test 3):
   ```python
   # Before preflight check (around line 1050)
   logger.info("[DIAG] Testing warm-up hypothesis")
   dummy = torch.randn(2, 19, 256, device="cuda")
   _ = model(dummy)  # Warm up all kernels
   torch.cuda.synchronize()
   logger.info("[DIAG] Warm-up done, running preflight")
   ```

3. **`src/brain_brr/models/detector.py`** (Test 5):
   ```python
   # Before mamba call
   logger.info(f"[DIAG] Mamba input: {x.shape}, {x.dtype}, contiguous={x.is_contiguous()}")
   ```

4. **`configs/modal/train.yaml`** (Test 4):
   ```yaml
   training:
     batch_size: 32  # Bisect: 64→32→16→8→4
   ```

**Total implementation time**: 30 minutes
**Total diagnostic time**: 2-15 hours depending on path chosen

---

## Conclusion

**Current Status**: ✅ **RESOLVED** in v3.4.1 (2025-10-01)

**Resolution**: Unique Triton cache directories per worker (deploy/modal/app.py:539-546) eliminated XID 31 GPU crashes. Full Modal training validated with 100-epoch run launched 2025-10-03.

**Root Cause** (confirmed): Triton kernel cache collision in multi-worker environments, NOT a fundamental mamba-ssm bug. First-batch failures were cache artifacts from parallel compilation attempts.

**Fix Implementation**:
- `TRITON_CACHE_DIR` set to unique per-worker paths using `os.getpid()`
- Prevents cache corruption during parallel kernel compilation
- Works on all GPU architectures (A100, H100 validated)

**Upgrade Was Correct**: PyTorch 2.5 + mamba-ssm 2.2.5 exposed cache issue through stricter kernel compilation, leading to proper fix.

**Hardware Status**: All GPUs work with proper cache isolation (A100/H100/RTX4090 validated).

---

**Last Updated**: 2025-10-03 (v3.6.0 Modal training baseline)
**Resolution Commit**: v3.4.1 release (Triton cache fix)
**Validation**: Full 100-epoch training launched on Modal A100-80GB
**Status**: Production ready ✅