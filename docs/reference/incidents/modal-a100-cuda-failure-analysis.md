# Modal A100 CUDA Memory Access Failure - Root Cause Analysis

**Incident Date:** 2025-09-29 19:20:53 UTC (Initial crash)
**Test Date:** 2025-09-29 21:25:28 UTC (Test 1A failed)
**Status:** 🔴 DIAGNOSTIC IN PROGRESS — Waiting for Test 1B (Force Fallback)
**Severity:** P0 BLOCKER — Cannot train on Modal A100
**Branch:** `fix/test-suite-config` (unrelated to failure)

---

## 🚨 CRITICAL UPDATE: Test 1A Results (2025-09-29 21:25 UTC)

**TEST 1A FAILED** — AMP is **NOT** the root cause.

### Test 1A: Disable AMP (FAILED)
- **Config**: `mixed_precision: false` (FP32 only, no autocast)
- **Result**: ❌ **CRASHED with same XID 31 MMU Fault**
- **Timestamp**: 21:25:28 UTC (3 minutes after preflight started)
- **Fault address**: `0x2b36_42000000` (different address than original, but same error type)
- **Conclusion**: **AMP is NOT the root cause** — crash happens with both AMP ON and OFF

### Updated Root Cause Hypothesis (After Test 1A)

**Original Hypothesis 1 (RULED OUT)**: ~~AMP-specific Mamba kernel bug~~ ❌
- Test 1A with AMP disabled FAILED
- Crash occurs regardless of FP16/FP32 mode

**New Top Hypothesis**: Mamba-SSM 2.2.2 CUDA kernel bug (not AMP-related)
- Crash happens with both `mixed_precision: true` AND `mixed_precision: false`
- XID 31 page fault suggests pointer arithmetic bug in Mamba CUDA kernels
- **Test 1B (Force Fallback) is now running** to confirm this

---

## Executive Summary

Training failed on the **first GPU forward pass** during preflight after ~55 minutes of successful CPU-side data loading. The failure is a **GPU hardware-level MMU fault** (NVIDIA XID 31) followed by **CUDA "illegal memory access"** in the Mamba-SSM CUDA kernel.

**UPDATE**: Test 1A proved crash is **NOT AMP-related** — happens with both FP16 and FP32.

### What We Know For Certain

1. **Crash location**: First CUDA kernel launch in preflight test batch (not during data I/O)
2. **Error sequence**: GPU XID 31 MMU Fault → CUDA illegal memory access → Training abort
3. ❌ **AMP is NOT the differentiator** (Test 1A with AMP OFF failed):
   - **Modal A100** with `mixed_precision: true` — CRASHES
   - **Modal A100** with `mixed_precision: false` — **ALSO CRASHES**
   - **Local RTX 4090** with `mixed_precision: false` — NO CRASH
4. **Mamba fallback did NOT engage**:
   - Wrapper logs "using fallback" but re-raises exception (`src/brain_brr/models/mamba.py:206-216`)
   - Current filters: `"causal_conv1d"`, `"NoneType"`, `"object is not callable"`
   - Actual error: `"CUDA error: an illegal memory access..."` — **DOES NOT MATCH** → re-raised
5. **Our configuration is mathematically correct**:
   ```
   Main Mamba:  (512*2)/64 = 16.0  (multiple of 8 ✅)
   Node Mamba:  (64*2)/8   = 16.0  (multiple of 8 ✅)
   Edge Mamba:  (16*2)/4   = 8.0   (multiple of 8 ✅)
   ```

### Key Evidence
- ✅ Data loading completed (1832 dev files, 148K windows, 55 minutes)
- ✅ Model initialization succeeded (31.4M parameters, all headdim correct)
- ✅ Optimizer, W&B, focal loss initialized
- ✅ 14 Mamba2 layers instantiated successfully (6 main + 6 node + 2 edge)
- ❌ **Failed on PREFLIGHT test batch** (first CUDA kernel launch)
- ❌ GPU XID 31 MMU Fault at `0x2ae1_d6600000` (invalid virtual address)
- ❌ CUDA error: illegal memory access during Mamba forward

### SSOT Research Findings (Sept 2025)

**Modal Labs Image Caching (CONFIRMED):**
- Modal caches images **per layer** (per `.run_commands()` / `.pip_install()` call)
- Cache invalidation: `force_build=True` (single method), `MODAL_FORCE_BUILD=1` (all images), or `MODAL_IGNORE_CACHE=1` (debug)
- **Breaking cache on one layer cascades rebuilds for all subsequent layers**
- **Docker layer caching is independent of `--no-cache-dir` (pip cache)**

**CUDA Fatbin Multi-Arch Compilation (CONFIRMED):**
- nvcc creates fatbins containing multiple PTX/SASS for different compute capabilities
- **Runtime driver selects appropriate code at kernel launch** based on physical GPU
- `TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"` embeds 4 architectures in one binary
- **Selection is automatic and correct** — "wrong architecture executed" is NOT a typical failure mode
- **However**: Build caching + extension compilation CAN cause issues if cached extensions mismatch runtime GPU

**mamba-ssm Known Issues (CONFIRMED):**
- **Issue #686**: "CUDA error: illegal memory access" with **long sequences** (512K) on H100
- **Issue #732**: Illegal memory access with **specific tensor sizes** on A6000
- **Pattern**: Failures occur on first operation with certain sizes OR after warm-up with smaller tensors
- **No specific A100 + mamba-ssm 2.2.2 + XID 31 reports found**

**PyTorch AMP + Custom CUDA Kernels (CONFIRMED):**
- A100 has **312 TFLOPS BF16 vs 19 TFLOPS FP32** (16x faster with AMP)
- Custom CUDA kernels must handle FP16 memory layout correctly
- **AMP can expose bugs** in custom kernels that don't manifest in FP32
- Tensor core operations have **different memory access patterns** than CUDA cores

**NVIDIA XID 31 (OFFICIAL DOCS):**
- **XID 31 = MMU page fault** (illegal memory access by GPU)
- **NOT XID 13** (Out of Memory) — this is a pointer/addressing error
- **Common causes**: Out-of-bounds access, uninitialized pointers, race conditions in GPU kernels
- **NVIDIA recommends**: `cuda-memcheck`, `CUDA_DEVICE_WAITS_ON_EXCEPTION=1`, Compute Sanitizer

---

## Root Cause Hypothesis (Priority Order)

### Hypothesis 1: Mamba-SSM 2.2.2 Kernel Bug Triggered by AMP on A100 (HIGHEST PROBABILITY ⭐⭐⭐⭐)

**Summary**: The first Mamba CUDA kernel under AMP autocast (FP16) triggers an illegal memory access on A100 hardware. This could be:
- A kernel bug specific to **sm_80 (A100) + FP16 tensor layout**
- Uninitialized pointer or out-of-bounds access that only manifests with FP16 input
- Memory alignment issue with A100's HBM2e controller under autocast

**Evidence**:
1. **AMP is the differentiator**:
   - Modal (crashes): `training.mixed_precision: true` → `autocast(enabled=True)` (`src/brain_brr/train/loop.py:489`)
   - Local (works): `training.mixed_precision: false` → FP32 only
2. **XID 31 indicates kernel-level bug**: Page fault at invalid virtual address `0x2ae1_d6600000` (~46.8 TB) suggests pointer arithmetic error
3. **Known mamba-ssm issues align**: GitHub issues #686, #732 show "illegal memory access" with specific tensor sizes/sequences
4. **First kernel launch fails**: No warm-up, no prior CUDA work — suggests input-dependent bug
5. **Fallback did not engage**: Error message doesn't match current filters, so wrapper re-raises instead of falling back to Conv1d

**Test to confirm**:
```yaml
# configs/modal/train.yaml
training:
  mixed_precision: false  # Disable AMP, use FP32
```
**Expected**: If preflight passes, confirms AMP-specific kernel path on A100 is the culprit.

**Likelihood**: ⭐⭐⭐⭐ (80%) — Best explanation for all evidence

---

### Hypothesis 2: Modal's Cached Extension Built on Different GPU (MEDIUM PROBABILITY ⭐⭐⭐)

**Summary**: Modal's Docker layer cache served a `mamba-ssm` / `causal-conv1d` extension compiled on a different GPU (e.g., H100 or RTX 4090). While fatbin selection is automatic, **PyTorch extension build caches can cause mismatches** if the cached `.so` files were linked against different CUDA runtime assumptions.

**Evidence**:
1. **Modal layer caching confirmed**: Images cached per layer, independent of `--no-cache-dir`
2. **Multi-arch compilation**: `TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"` (4 architectures) in `deploy/modal/app.py:27`
3. **PyTorch extension caching**:
   - Extensions compile to `.so` files stored in `~/.cache/torch_extensions/` or `TORCH_EXTENSIONS_DIR`
   - **Known issue**: "Multiple conda environments sharing cache can't reuse extensions" (PyTorch forums)
   - **Build variability**: "More architectures = slower build" but can cause cache confusion
4. **Why XID 31?**: If cached extension has wrong memory layout assumptions, first kernel triggers page fault

**Counter-evidence**:
- CUDA fatbin selection is designed to handle multi-arch correctly at runtime
- No widespread reports of "Modal A100 cache poisoning" for CUDA extensions
- Hard to explain why this would suddenly start failing (unless Modal changed build infrastructure)

**Test to confirm**:
```python
# deploy/modal/app.py - Single-arch build + cache bust
.env({
    "FORCE_REBUILD": "2025-09-29-a100-only-v2",  # Invalidate Modal cache
    "TORCH_CUDA_ARCH_LIST": "8.0",  # A100 ONLY (not multi-arch)
})
.run_commands("pip cache purge")
.run_commands(
    "pip uninstall -y causal-conv1d || true",
    "pip install --no-build-isolation --no-cache-dir --force-reinstall causal-conv1d==1.4.0"
)
.run_commands(
    "pip uninstall -y mamba-ssm || true",
    "pip install --no-build-isolation --no-cache-dir --force-reinstall mamba-ssm==2.2.2"
)
```
**Expected**: If preflight passes, confirms cached extension mismatch.

**Likelihood**: ⭐⭐⭐ (60%) — Plausible but less likely than AMP kernel bug

---

### Hypothesis 3: Batch Size / Memory Pressure Triggers Edge Case (LOW PROBABILITY ⭐)

**Summary**: Batch size 64 with 80GB VRAM shouldn't be a problem, but could trigger edge case in Mamba kernel memory management.

**Evidence**:
1. **Batch 64 is safe**:
   - Input: `64 × 19 × 15360 × 4 bytes = 75MB`
   - Model: `31.4M params × 4 bytes = 126MB`
   - A100 has 80GB → using <0.3% VRAM
2. **XID 31 ≠ OOM**: Page fault, not memory exhaustion (would be XID 13)
3. **Known mamba-ssm issue #732**: "Tensor size (27, 32768, 384) fails but (26, 32768, 384) works"

**Test to confirm**:
```yaml
# configs/modal/train.yaml
training:
  batch_size: 8  # Reduce by 8x
```
**Expected**: Unlikely to fix (issue is kernel bug, not memory), but worth testing.

**Likelihood**: ⭐ (20%) — Weak hypothesis, included for completeness

---

### Hypothesis 4: Flaky A100 Hardware (VERY LOW PROBABILITY)

**Summary**: Modal assigned us a failing GPU with hardware issues.

**Counter-evidence**:
- **Immediate failure** on first kernel (not intermittent after hours of work)
- **Reproducible** XID 31 at same point (preflight)
- **Suggests software/configuration issue**, not hardware

**Test**: Stop and restart to get different A100

**Likelihood**: (10%) — Least likely, but easy to test

---

## Timeline of Failure (Detailed)

```
18:08:45  ✅ Mamba-SSM imported successfully (version 2.2.2)
18:08:46  ✅ Patient disjointness verified (579 train, 53 dev)
18:08:46  ✅ Cache location verified (4667 NPZ files on SSD)
18:17:00  ✅ TUSZ official splits loaded
18:17:17  ✅ BalancedSeizureDataset created (61,616 train windows)
19:12:37  ✅ Validation dataset indexed (148,224 windows, 1832 files)
19:12:38  ✅ Model created (14 Mamba2 layers initialized - no GPU work yet)
19:12:44  ✅ Optimizer created (387 parameters)
19:12:45  ✅ W&B run started
19:12:45  ✅ Focal loss initialized (pos_weight=1.39)
19:19:02  ⏳ PREFLIGHT: Testing one batch... (FIRST GPU COMPUTATION)
19:19:02  ⚠️  GPU XID 31: MMU Fault at 0x2ae1_d6600000 (FAULT_PDE ACCESS_TYPE_VIRT_WRITE)
19:20:53  ❌ CUDA error: illegal memory access
19:20:53  ❌ Mamba wrapper logs "using fallback" but re-raises (error text mismatch)
19:20:53  ❌ Training failed with exit code 1
```

**Key observation**: 55 minutes of CPU work succeeded. Failure occurred **immediately on first GPU computation** (first CUDA kernel launch).

---

## Error Messages (Verbatim)

### 1. GPU Hardware Error (XID 31)
```
[gpu-health] [WARN] GPU-ade01aad-dbea-9be1-efe0-33e96682c342:
XID: NVRM: Xid (PCI:0000:ca:00): 31, pid=1587983, name=exe, Ch 0000000a, intr 00000000.
MMU Fault: ENGINE GRAPHICS GPC5 GPCCLIENT_T1_5 faulted @ 0x2ae1_d6600000.
Fault is of type FAULT_PDE ACCESS_TYPE_VIRT_WRITE
```

**Decoded**:
- **XID 31** = Page fault (invalid memory access attempt by GPU)
- **MMU Fault** = Memory Management Unit detected illegal address
- **FAULT_PDE** = Page Directory Entry fault (address not mapped in GPU virtual memory)
- **ACCESS_TYPE_VIRT_WRITE** = GPU tried to write to unmapped virtual address `0x2ae1_d6600000` (~46.8 TB - clearly invalid)
- **GPC5 GPCCLIENT_T1_5** = Graphics Processing Cluster 5, Texture unit client (likely Mamba kernel tensor ops)

### 2. CUDA Runtime Error
```python
[2025-09-29 19:20:53.979][src.brain_brr.models.mamba][WARNING]
[MAMBA] Forward pass error, using fallback: CUDA error: an illegal memory access was encountered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

**Note**: Despite logging "using fallback", the wrapper **re-raises** because error text doesn't match filters (`src/brain_brr/models/mamba.py:206-216`).

### 3. Preflight Test Failure
```python
[2025-09-29 19:20:53.980][src.brain_brr.train.loop][INFO]
[PREFLIGHT] ✗ Failed on test batch: CUDA error: an illegal memory access was encountered

[PREFLIGHT] Debug info:
  - Model type: <class 'src.brain_brr.models.detector.SeizureDetector'>
  - Input shape: torch.Size([64, 19, 15360])
  - Labels shape: torch.Size([64, 15360])
  - Loss mode: focal
  - Device: cuda
```

---

## Configuration Snapshot

### Modal Deployment (`deploy/modal/app.py`)

**Image Build**:
```python
modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .env({
        "CUDA_HOME": "/usr/local/cuda-12.1",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0",  # Multi-arch: A100, RTX 3090, RTX 4090, H100
    })
    .run_commands(
        "pip install torch==2.2.2 torchvision==0.17.2 'numpy<2.0' --index-url https://download.pytorch.org/whl/cu121"
    )
    .run_commands(
        "pip install --no-build-isolation --no-cache-dir causal-conv1d==1.4.0"
    )
    .run_commands(
        "pip install --no-build-isolation --no-cache-dir mamba-ssm==2.2.2"
    )
```

**Function Resources**:
```python
@app.function(
    gpu="A100-80GB",
    cpu=24,
    memory=98304,  # 96GB RAM
    timeout=432000,  # 5 days
    volumes={"/results": results_volume},
)
```

### Training Config (`configs/modal/train.yaml`)

```yaml
training:
  batch_size: 64                       # A100-80GB optimized
  mixed_precision: true                # FP16 autocast on A100 tensor cores ← KEY DIFFERENCE

model:
  mamba:
    n_layers: 6
    d_model: 512
    d_state: 16
    d_conv: 4                          # CUDA supports 2-4
    expand: 2
    headdim: 64                        # (512*2)/64 = 16 ✅
```

### Local Config (`configs/local/train.yaml` - WORKS)

```yaml
training:
  batch_size: 12                       # RTX 4090 (24GB)
  mixed_precision: false               # FP32 only ← KEY DIFFERENCE

model:
  mamba:
    # Same architecture as Modal
```

**Critical difference**: `mixed_precision: true` on Modal vs `false` locally

---

## Diagnostic Plan (Priority Order)

### Phase 1: Fast Isolation Tests (HIGH PRIORITY)

#### Test 1A: Disable AMP (10 minutes)
```yaml
# configs/modal/train.yaml
training:
  mixed_precision: false  # Disable AMP, use FP32 only
```
**Command**:
```bash
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml
```
**Expected**: If preflight passes, **confirms AMP-specific Mamba kernel bug on A100** (Hypothesis 1)

**Likelihood of fix**: ⭐⭐⭐⭐ (80%)

---

#### Test 1B: Force Mamba Fallback (10 minutes)
```python
# deploy/modal/app.py - train function
@app.function(
    gpu="A100-80GB",
    env={"SEIZURE_MAMBA_FORCE_FALLBACK": "1"},  # Use Conv1d instead of Mamba-SSM
)
```
**Expected**: If preflight passes, **confirms issue is in Mamba CUDA kernels** (not our model code)

**Likelihood of fix**: ⭐⭐⭐⭐ (80%) — Isolates Mamba vs our code, but doesn't solve root cause

---

### Phase 2: Deep Diagnostics (MEDIUM PRIORITY)

#### Test 2A: Enable CUDA Blocking (30 minutes)
```python
# deploy/modal/app.py - train function
@app.function(
    gpu="A100-80GB",
    env={
        "CUDA_LAUNCH_BLOCKING": "1",   # Synchronous CUDA calls
        "TORCH_USE_CUDA_DSA": "1",     # Device-side assertions
    },
)
```
**Purpose**: Get **exact line number** where illegal access occurs (not just "somewhere in Mamba")

**Expected output**: Synchronous stack trace pinpointing exact Mamba kernel operation

---

#### Test 2B: Single-Arch Build + Cache Bust (1 hour)
```python
# deploy/modal/app.py
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .env({
        "FORCE_REBUILD": "2025-09-29-a100-only-v2",  # ← Change this each rebuild
        "TORCH_CUDA_ARCH_LIST": "8.0",  # A100 ONLY (sm_80, not multi-arch)
    })
    .run_commands("pip cache purge")
    .run_commands(
        "pip uninstall -y causal-conv1d || true",
        "pip install --no-build-isolation --no-cache-dir --force-reinstall causal-conv1d==1.4.0"
    )
    .run_commands(
        "pip uninstall -y mamba-ssm || true",
        "pip install --no-build-isolation --no-cache-dir --force-reinstall mamba-ssm==2.2.2"
    )
)
```
**Purpose**: Eliminate **any possibility of cached extension mismatch** (Hypothesis 2)

**Expected**: If preflight passes with AMP still enabled, confirms cache poisoning (Hypothesis 2). If still fails, confirms AMP kernel bug (Hypothesis 1).

---

#### Test 2C: Reduce Batch Size (10 minutes)
```yaml
# configs/modal/train.yaml
training:
  batch_size: 8  # Reduce by 8x
```
**Purpose**: Rule out batch-size-dependent memory triggers (Hypothesis 3)

**Expected**: Unlikely to fix, but worth testing

---

### Phase 3: Emergency Workarounds (LOW PRIORITY)

#### Option 3A: Request Different A100
```bash
modal app stop ap-rwxEXb1HcVkErXfIUDVHFS
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml
```
**Purpose**: Rule out flaky hardware (Hypothesis 4)

---

#### Option 3B: Train with FP32 Permanently (WORKAROUND)
If Test 1A passes, we can train with `mixed_precision: false` as a workaround while we:
- Report bug to `state-spaces/mamba` with full repro (A100 + AMP + specific config)
- Wait for upstream fix in mamba-ssm 2.2.3+

**Downside**: ~2x slower training (no tensor cores), but functional

---

## Recommended Action Plan

### Immediate (Next 1 Hour)

1. **Run Test 1A (Disable AMP)** — Fastest way to confirm Hypothesis 1
2. **Run Test 1B (Force Fallback)** — Confirms issue is Mamba-specific
3. **If both pass**: We know it's Mamba + AMP on A100. Options:
   - **Short-term**: Train with `mixed_precision: false` (slower but works)
   - **Medium-term**: Report to mamba-ssm upstream with full repro
   - **Long-term**: Wait for mamba-ssm fix or patch kernel ourselves

### Parallel Track (Next 2 Hours)

4. **Run Test 2A (CUDA_LAUNCH_BLOCKING)** — Get exact kernel failure site
5. **Run Test 2B (Single-arch rebuild)** — Rule out cache hypothesis definitively

### If All Tests Fail

6. **Try Option 3A (Different A100)** — Rule out hardware
7. **Escalate to Modal support** — Provide full repro + XID 31 logs
8. **Consider H100 or different cloud provider** — Last resort

---

## Expected Outcomes

| Test | Result | Interpretation |
|------|--------|----------------|
| 1A: AMP off | ✅ Pass | **Confirms Hypothesis 1** (Mamba + AMP bug on A100). Workaround: train with FP32 |
| 1A: AMP off | ❌ Fail | Rules out AMP, proceed to 2B (cache) |
| 1B: Fallback | ✅ Pass | Confirms issue in Mamba CUDA kernels (not our code) |
| 1B: Fallback | ❌ Fail | Bug in our model code (unlikely given local works) |
| 2A: Blocking | Stack trace | Pinpoints exact Mamba kernel operation failing |
| 2B: Single-arch | ✅ Pass | **Confirms Hypothesis 2** (cache poisoning). Always use single-arch going forward |
| 2B: Single-arch | ❌ Fail | Rules out cache, back to Hypothesis 1 (AMP bug) |
| 2C: Batch 8 | ✅ Pass | Memory-dependent bug (unlikely but possible) |
| 3A: New A100 | ✅ Pass | Hardware issue (very unlikely) |

**Most likely outcome**: Test 1A (AMP off) will pass, confirming Mamba-SSM 2.2.2 has an AMP-specific kernel bug on A100 hardware.

---

## Q&A for Cross-Review

### Q1: Is the "cache poisoning" hypothesis valid?
**A**: **PARTIALLY**. Modal DOES cache Docker layers, and PyTorch extensions CAN have build cache issues. However:
- CUDA fatbin selection is designed to work correctly at runtime
- No widespread reports of this specific failure mode
- **More likely**: AMP kernel bug (Hypothesis 1) than cache mismatch (Hypothesis 2)
- **Recommendation**: Test AMP first (Test 1A), then single-arch rebuild (Test 2B) to be thorough

### Q2: Should we change TORCH_CUDA_ARCH_LIST from multi-arch to single-arch?
**A**: **YES, as a hygiene measure**, but it's not the primary fix:
- **Single-arch `"8.0"`** eliminates any possibility of cache confusion
- **Faster compile times** (only one architecture)
- **Cleaner debugging** (only one code path)
- **But**: Fatbin selection SHOULD work correctly, so this is defense-in-depth

### Q3: Can we downgrade mamba-ssm to avoid the bug?
**A**: **NO, not easily**:
- mamba-ssm 2.2.2 is tightly coupled to PyTorch 2.2.2 + CUDA 12.1
- Downgrading requires testing entire stack compatibility
- **Better**: Disable AMP as workaround, report bug upstream

### Q4: Is batch_size=64 the problem?
**A**: **NO**:
- Calculation: `64 × 19 × 15360 × 4 bytes = 75MB` input + `126MB` model = `<1GB` active
- A100 has 80GB VRAM → using <1.5%
- XID 31 is **page fault**, NOT OOM (which would be XID 13)
- **But**: Worth testing batch 8 to rule out edge case

### Q5: Should we try H100 instead of A100?
**A**: **ONLY as last resort**:
- H100 costs more and may not be available
- If issue is Mamba kernel bug, might fail on H100 too
- **Better**: Fix root cause (disable AMP or patch Mamba)

### Q6: Why did the Mamba fallback not activate?
**A**: **Fallback filter mismatch**:
- Current filters: `"causal_conv1d"`, `"NoneType"`, `"object is not callable"` (`src/brain_brr/models/mamba.py:210-213`)
- Actual error: `"CUDA error: an illegal memory access was encountered"`
- **Does not match** → wrapper logs "using fallback" but **re-raises**
- **Fix**: Add `"illegal memory access"` to fallback filters for automatic recovery

---

## Related Code References

- **Mamba wrapper fallback logic**: `src/brain_brr/models/mamba.py:206-216`
- **Preflight test with AMP**: `src/brain_brr/train/loop.py:489` (`autocast(enabled=use_amp)`)
- **Modal image build**: `deploy/modal/app.py:14-50`
- **Modal training config**: `configs/modal/train.yaml:89-100`
- **Local training config (works)**: `configs/local/train.yaml:83-94`
- **GPU stack docs**: `docs/01-installation/gpu-stack.md`
- **Mamba architecture**: `docs/04-model/mamba.md`
- **Troubleshooting guide**: `docs/08-operations/troubleshooting.md`

---

## Status

**Current State**: ⏸️ Training paused, comprehensive analysis complete

**Recommended Next Steps** (in order):
1. ✅ **Test 1A: Disable AMP** (`mixed_precision: false`) — 10 min, 80% likely to fix
2. ✅ **Test 1B: Force fallback** (`SEIZURE_MAMBA_FORCE_FALLBACK=1`) — 10 min, isolates Mamba
3. ⏳ **Test 2A: CUDA_LAUNCH_BLOCKING** — 30 min, get exact failure site
4. ⏳ **Test 2B: Single-arch rebuild** — 1 hour, rule out cache completely
5. ⏳ **If 1A passes**: Train with FP32, report to mamba-ssm upstream

**Do NOT implement fixes reactively** — Execute Test 1A first to confirm hypothesis.

---

**Last Updated**: 2025-09-29 20:45 UTC (Post-SSOT research)
**Author**: Claude (Evidence-Based Root Cause Analysis)
**Research Sources**: Modal Labs Docs, NVIDIA XID Docs, PyTorch Forums, mamba-ssm GitHub Issues
**Reviewers**: [Pending cross-review]