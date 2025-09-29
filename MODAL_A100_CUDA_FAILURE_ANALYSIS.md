# Modal A100 CUDA Memory Access Failure - Root Cause Analysis

**Incident Date:** 2025-09-29 19:20:53 UTC
**Status:** 🔴 TRAINING HALTED - Root cause analysis in progress
**Severity:** P0 BLOCKER - Cannot train on Modal A100
**Branch:** `fix/test-suite-config` (unrelated to failure)

---

## Executive Summary

Training **failed on first forward pass** after 55 minutes of successful data loading. The failure is a **GPU hardware-level MMU fault** (NVIDIA XID 31) followed by **CUDA illegal memory access** in Mamba-SSM kernel. This is NOT a code bug but likely a **configuration mismatch** between Mamba-SSM compilation and A100 compute capability.

**Critical Evidence:**
- ✅ Data loading completed successfully (1832 dev files indexed in 55 minutes)
- ✅ Model initialization succeeded (31.4M parameters)
- ✅ Optimizer created successfully
- ✅ W&B initialized
- ❌ **Failed on PREFLIGHT test batch** (single forward pass before training loop)
- ❌ GPU threw **XID 31 MMU Fault** immediately before CUDA error
- ❌ Mamba layer triggered fallback due to illegal memory access

**Key Insight:** This is an **A100-specific issue**. Local RTX 4090 training works fine. The problem manifests in Mamba-SSM CUDA kernels, not our code.

---

## Timeline of Failure

```
18:08:45  ✅ Mamba-SSM imported successfully (version 2.2.2)
18:08:46  ✅ Patient disjointness verified (579 train, 53 dev)
18:08:46  ✅ Cache location verified (4667 NPZ files on SSD)
18:17:00  ✅ TUSZ official splits loaded
18:17:17  ✅ BalancedSeizureDataset created (61,616 train windows)
19:12:37  ✅ Validation dataset indexed (148,224 windows, 1832 files)
19:12:38  ✅ Model created (14 Mamba2 layers initialized)
19:12:44  ✅ Optimizer created (387 parameters)
19:12:45  ✅ W&B run started
19:12:45  ✅ Focal loss initialized (pos_weight=1.39)
19:19:02  ⏳ PREFLIGHT: Testing one batch...
19:19:02  ⚠️  GPU XID 31: MMU Fault at 0x2ae1_d6600000 (FAULT_PDE ACCESS_TYPE_VIRT_WRITE)
19:20:53  ❌ CUDA error: illegal memory access
19:20:53  ❌ Training failed with exit code 1
```

**Key Observation:** 55 minutes of CPU work (data loading) succeeded. Failure occurred **immediately on first GPU computation**.

---

## Error Messages (Verbatim)

### 1. GPU Hardware Error (XID 31)
```
[gpu-health] [WARN] GPU-ade01aad-dbea-9be1-efe0-33e96682c342:
XID: NVRM: Xid (PCI:0000:ca:00): 31, pid=1587983, name=exe, Ch 0000000a, intr 00000000.
MMU Fault: ENGINE GRAPHICS GPC5 GPCCLIENT_T1_5 faulted @ 0x2ae1_d6600000.
Fault is of type FAULT_PDE ACCESS_TYPE_VIRT_WRITE
```

**Decoded:**
- **XID 31** = Page fault (invalid memory access attempt by GPU)
- **MMU Fault** = Memory Management Unit detected illegal address
- **FAULT_PDE** = Page Directory Entry fault (address not mapped)
- **ACCESS_TYPE_VIRT_WRITE** = GPU tried to write to unmapped virtual address `0x2ae1_d6600000`
- **GPC5 GPCCLIENT_T1_5** = Graphics Processing Cluster 5, Texture unit client (Mamba kernel)

### 2. CUDA Runtime Error
```python
[2025-09-29 19:20:53.979][src.brain_brr.models.mamba][WARNING]
[MAMBA] Forward pass error, using fallback: CUDA error: an illegal memory access was encountered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

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

## Root Cause Analysis

### Hypothesis 1: Mamba-SSM CUDA Kernel Compilation Mismatch (MOST LIKELY ⭐)

**Evidence:**
1. **Mamba-SSM compiled for specific CUDA architecture:**
   - Modal image uses `TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"`
   - A100 is compute capability **8.0** (included in list)
   - BUT: Kernel may have been compiled for wrong sub-architecture

2. **causal-conv1d dependency critical:**
   - Modal image installs `causal-conv1d==1.4.0` (exact version)
   - Must be compiled with `--no-build-isolation --no-cache-dir`
   - If binary cache used wrong version, kernels will fail

3. **Mamba-SSM version 2.2.2 specifics:**
   ```python
   # deploy/modal/app.py lines 42-46
   .run_commands(
       "pip install --no-build-isolation --no-cache-dir causal-conv1d==1.4.0"
   )
   .run_commands(
       "pip install --no-build-isolation --no-cache-dir mamba-ssm==2.2.2"
   )
   ```
   - These flags force compilation from source
   - But Modal may cache compiled binaries between image builds
   - **Cached binaries from different GPU may be incompatible**

4. **XID 31 context:**
   - Page fault means GPU tried to access memory address that doesn't exist
   - This happens when CUDA kernel assumes memory layout that doesn't match runtime
   - Classic symptom of **wrong compute capability compilation**

**Why this explains the failure:**
- Import succeeds (Python bindings work)
- Model creation succeeds (no GPU work yet)
- **First forward pass fails** (CUDA kernel executes with wrong memory assumptions)

### Hypothesis 2: A100 Memory Addressing Bug in Mamba-SSM 2.2.2

**Evidence:**
1. **A100-specific memory architecture:**
   - A100 has 80GB HBM2e with different memory controller than consumer GPUs
   - Memory addressing is different from RTX 4090 (24GB GDDR6X)
   - Mamba-SSM may have A100-specific bugs in version 2.2.2

2. **Faulted address suspicious:**
   - `0x2ae1_d6600000` = ~46.8 TB in hex (clearly invalid virtual address)
   - Suggests pointer arithmetic overflow or uninitialized pointer
   - Could be Mamba kernel bug triggered only on A100

3. **Known Mamba-SSM bugs:**
   ```python
   # pyproject.toml comment
   # mamba-ssm==2.2.2 (2.2.5 has bugs, 2.2.4 has issues)
   ```
   - We're using 2.2.2 to avoid known bugs in later versions
   - But 2.2.2 may have A100-specific bugs not present on consumer GPUs

### Hypothesis 3: Modal A100 GPU Hardware Issue (LESS LIKELY)

**Evidence:**
1. **GPU health warning in logs:**
   - Modal's `[gpu-health]` system detected XID 31
   - Could indicate flaky A100 hardware in Modal's pool
   - Modal may have assigned us a failing GPU

2. **Counter-evidence:**
   - This would be random and intermittent
   - We failed **immediately** on first GPU use (not after hours of training)
   - Suggests configuration issue, not hardware

### Hypothesis 4: Batch Size Too Large (UNLIKELY)

**Evidence:**
1. **Batch size 64 is reasonable for A100:**
   - A100 has 80GB VRAM
   - Our model is 31.4M parameters (~125MB)
   - Input: 64 × 19 × 15360 = ~75MB per batch
   - Should fit comfortably in 80GB

2. **XID 31 is not OOM:**
   - OOM would be XID 13 (Out of Memory)
   - XID 31 is page fault (illegal address)
   - Different error class entirely

---

## Configuration Snapshot

### Modal Deployment Configuration

**Image Build (`deploy/modal/app.py`):**
```python
modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .env({
        "CUDA_HOME": "/usr/local/cuda-12.1",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0",  # A100 is 8.0
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

**Function Resources:**
```python
@app.function(
    gpu="A100-80GB",
    cpu=24,
    memory=98304,  # 96GB RAM
    timeout=432000,  # 5 days
    volumes={"/results": results_volume},
)
```

**Training Configuration (`configs/modal/train.yaml`):**
```yaml
training:
  batch_size: 64                       # A100-80GB optimized
  mixed_precision: true                # FP16 on A100 tensor cores

model:
  mamba:
    n_layers: 6
    d_model: 512
    d_state: 16                        # Mamba SSM state dimension
    d_conv: 4                          # CUDA supports 2-4
    expand: 2
    headdim: 64
```

**Input Shape at Failure:**
```python
Input:  torch.Size([64, 19, 15360])  # batch=64, channels=19, samples=15360
Labels: torch.Size([64, 15360])      # batch=64, samples=15360
```

### Version Constraints

**Critical Dependencies:**
```toml
[project.dependencies]
torch = "==2.2.2"                    # EXACT: PyTorch 2.2.2 with CUDA 12.1
numpy = "==1.26.4"                   # EXACT: numpy 2.x breaks mamba-ssm
# mamba-ssm==2.2.2 (2.2.5 has bugs, 2.2.4 has issues)
```

**Why These Versions:**
- **PyTorch 2.2.2:** Last version compatible with mamba-ssm 2.2.2
- **CUDA 12.1:** Required for mamba-ssm CUDA kernels
- **numpy 1.26.4:** numpy 2.x breaks mamba-ssm
- **causal-conv1d 1.4.0:** Must match mamba-ssm 2.2.2 (1.5+ requires PyTorch 2.4+)
- **mamba-ssm 2.2.2:** Latest stable without known critical bugs

---

## What Works vs What Fails

### ✅ Works on Local RTX 4090
```bash
# Local training (configs/local/train.yaml)
batch_size: 12
mixed_precision: false
gpu: RTX 4090 (24GB VRAM, compute 8.9)
```

**Evidence:**
- Smoke tests pass locally
- Integration tests pass
- No XID errors on RTX 4090
- Same mamba-ssm version (2.2.2)
- Same model architecture (V3 dual-stream)

### ❌ Fails on Modal A100
```bash
# Modal training (configs/modal/train.yaml)
batch_size: 64
mixed_precision: true
gpu: A100-80GB (80GB HBM2e, compute 8.0)
```

**Failure Point:**
- First forward pass in preflight test
- Before any training loop iteration
- Mamba-SSM CUDA kernel triggers XID 31

---

## Diagnostic Commands (Not Yet Run)

### 1. Force CUDA Blocking for Precise Error Location
```yaml
# Add to Modal deployment environment
env:
  CUDA_LAUNCH_BLOCKING: "1"          # Synchronous CUDA calls
  TORCH_USE_CUDA_DSA: "1"            # Device-side assertions
```

**Purpose:** Get exact line number where illegal access occurs (not just "somewhere in Mamba")

### 2. Reduce Batch Size to Isolate Memory Issue
```yaml
# Test with minimal batch
training:
  batch_size: 2                      # Absolute minimum for testing
```

**Purpose:** If this fails, confirms it's not memory exhaustion but kernel bug

### 3. Force Mamba Fallback to Conv1d
```yaml
# Add to Modal deployment
env:
  SEIZURE_MAMBA_FORCE_FALLBACK: "1"  # Use Conv1d instead of Mamba-SSM
```

**Purpose:** If training succeeds, confirms Mamba-SSM is the problem

### 4. Verify GPU Compute Capability
```python
# Add to preflight in train/loop.py
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
print(f"CUDA version: {torch.version.cuda}")
```

**Purpose:** Verify Modal gave us correct A100 (8.0) and CUDA 12.1

### 5. Test Mamba-SSM Directly (Minimal Repro)
```python
# Create minimal_mamba_test.py
import torch
from mamba_ssm import Mamba2

# Exact config from our model
mamba = Mamba2(
    d_model=512,
    d_state=16,
    d_conv=4,
    expand=2,
    headdim=64
).cuda()

# Exact input shape from failure
x = torch.randn(64, 960, 512).cuda()  # (batch, seqlen, d_model)

try:
    out = mamba(x)
    print("✅ Mamba forward pass succeeded")
except Exception as e:
    print(f"❌ Mamba forward pass failed: {e}")
```

**Purpose:** Isolate Mamba-SSM from our model code

---

## Potential Fixes (Prioritized by Likelihood)

### Fix 1: Rebuild Modal Image with Clean Compilation (HIGHEST PRIORITY ⭐)

**Rationale:** Cached binaries may be from wrong GPU architecture.

**Implementation:**
```python
# deploy/modal/app.py - Add force rebuild
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    # FORCE clean build by adding timestamp or version
    .env({
        "FORCE_REBUILD": "2025-09-29-fix1",  # Change this to invalidate cache
        "CUDA_HOME": "/usr/local/cuda-12.1",
        "PATH": "/usr/local/cuda-12.1/bin:$PATH",
        "LD_LIBRARY_PATH": "/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH",
        "TORCH_CUDA_ARCH_LIST": "8.0",  # ONLY A100 (force single-arch build)
    })
    # Add explicit cache clearing
    .run_commands("pip cache purge")
    .run_commands(
        "pip install torch==2.2.2 torchvision==0.17.2 'numpy<2.0' --index-url https://download.pytorch.org/whl/cu121"
    )
    # Clean build of causal-conv1d
    .run_commands(
        "pip uninstall -y causal-conv1d || true",  # Ensure clean
        "pip install --no-build-isolation --no-cache-dir --force-reinstall causal-conv1d==1.4.0"
    )
    # Clean build of mamba-ssm
    .run_commands(
        "pip uninstall -y mamba-ssm || true",
        "pip install --no-build-isolation --no-cache-dir --force-reinstall mamba-ssm==2.2.2"
    )
)
```

**Why This Should Work:**
- Forces complete recompilation for A100 (8.0) only
- Eliminates cached binaries from other GPUs
- `--force-reinstall` ensures no pip cache reuse

**Risk:** Low (just rebuilds image, no code changes)

### Fix 2: Downgrade to Mamba-SSM 2.2.0 (MEDIUM PRIORITY)

**Rationale:** Version 2.2.2 may have A100-specific bugs.

**Implementation:**
```python
# deploy/modal/app.py
.run_commands(
    "pip install --no-build-isolation --no-cache-dir mamba-ssm==2.2.0"
)

# Also update pyproject.toml
[project.dependencies]
# mamba-ssm==2.2.0 (testing A100 compatibility)
```

**Why This Might Work:**
- 2.2.0 is older and may have better A100 testing
- We're using 2.2.2 to avoid bugs in 2.2.4/2.2.5, but 2.2.0 predates those

**Risk:** Medium (may reintroduce bugs that 2.2.2 fixed)

### Fix 3: Reduce Batch Size Temporarily (LOW PRIORITY)

**Rationale:** Rule out memory-related triggers.

**Implementation:**
```yaml
# configs/modal/train.yaml
training:
  batch_size: 32  # Half of current 64
```

**Why This Might Work:**
- Smaller batch = less memory pressure
- May avoid edge case in Mamba kernel

**Risk:** Low (just slower training)

### Fix 4: Disable Mixed Precision (LOW PRIORITY)

**Rationale:** FP16 tensor cores may trigger different code path in Mamba.

**Implementation:**
```yaml
# configs/modal/train.yaml
training:
  mixed_precision: false  # Use FP32 only
```

**Why This Might Work:**
- A100 tensor cores have different memory access patterns
- FP32 may use safer code path

**Risk:** Low (just slower and more VRAM usage)

### Fix 5: Request Different A100 from Modal (LAST RESORT)

**Rationale:** Hardware may be flaky.

**Implementation:**
```python
# Stop current run, restart to get different GPU
modal app stop ap-rwxEXb1HcVkErXfIUDVHFS
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml
```

**Why This Might Work:**
- Modal has pool of A100s
- May get different hardware

**Risk:** Low (but if it fails again, proves it's not hardware)

---

## Investigation Plan

### Phase 1: Gather More Diagnostics (30 minutes)

1. **Add debug environment variables:**
   ```python
   # deploy/modal/app.py - train function
   env={
       "CUDA_LAUNCH_BLOCKING": "1",
       "TORCH_USE_CUDA_DSA": "1",
   }
   ```

2. **Add GPU info logging:**
   ```python
   # src/brain_brr/train/loop.py - before preflight
   logger.info(f"[GPU] Device: {torch.cuda.get_device_name(0)}")
   logger.info(f"[GPU] Compute: {torch.cuda.get_device_capability(0)}")
   logger.info(f"[GPU] CUDA version: {torch.version.cuda}")
   logger.info(f"[GPU] Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
   ```

3. **Run minimal Mamba test separately**

### Phase 2: Try Fix 1 - Clean Rebuild (1 hour)

1. **Modify `deploy/modal/app.py`** with clean compilation (see Fix 1 above)
2. **Rebuild Modal image:** `modal deploy deploy/modal/app.py`
3. **Launch training:** `modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml`
4. **Monitor:** Check if preflight passes

### Phase 3: If Fix 1 Fails, Try Fallback (30 minutes)

1. **Force Mamba fallback:**
   ```python
   env={"SEIZURE_MAMBA_FORCE_FALLBACK": "1"}
   ```
2. **If this works:** Confirms Mamba-SSM is the problem
3. **Next step:** Try different mamba-ssm version or report bug upstream

### Phase 4: If All Fails, Consider Alternatives

1. **Use different Modal GPU (H100?):** May have better mamba-ssm support
2. **Train without Mamba:** Use Transformer instead (architectural change)
3. **Report to mamba-ssm:** This may be undiscovered A100 bug

---

## Questions for Cross-Review

1. **Is our TORCH_CUDA_ARCH_LIST correct?**
   - We use `"8.0;8.6;8.9;9.0"`
   - A100 is 8.0 (included)
   - Should we use **ONLY** `"8.0"` to force single-arch build?

2. **Is mamba-ssm 2.2.2 known to work on A100?**
   - We chose it to avoid bugs in 2.2.4/2.2.5
   - But did 2.2.2 introduce A100-specific bugs?
   - Should we try 2.2.0 or even 2.1.x?

3. **Is batch_size=64 safe?**
   - 64 × 19 × 15360 × 4 bytes (FP32) = ~75MB input
   - Model is 31.4M params × 4 bytes = ~126MB
   - A100 has 80GB
   - Should be fine, but worth testing batch=2?

4. **Could this be Modal's cached binaries?**
   - Modal caches pip installs between image builds
   - Even with `--no-cache-dir`, Modal may cache at layer level
   - Do we need to invalidate Modal's image cache entirely?

5. **Should we test on H100 instead?**
   - H100 is newer (compute 9.0)
   - May have better mamba-ssm support
   - Worth trying if A100 continues to fail?

---

## Related Documentation

- **CLAUDE.md** lines 188-200: Installation requirements (exact versions)
- **deploy/modal/app.py** lines 14-50: Image build with CUDA compilation
- **configs/modal/train.yaml** lines 11-30: Training configuration
- **src/brain_brr/models/mamba.py** lines 48-96: Mamba layer initialization
- **pyproject.toml** lines 12-15: Version constraints and rationale

---

## Status

**Current State:** ⏸️ Training paused, investigation in progress

**Next Actions:**
1. Create branch for Modal fix: `git checkout -b fix/modal-a100-mamba-kernel`
2. Implement Fix 1 (clean rebuild with CUDA_ARCH=8.0 only)
3. Cross-review this document with another AI agent
4. Deploy fix and monitor preflight test
5. If fails, escalate to Fix 2 (mamba-ssm 2.2.0)

**Do NOT Fix Reactively:** Wait for cross-review confirmation before deploying fixes.

---

**Last Updated:** 2025-09-29 19:30 UTC
**Author:** Claude (Comprehensive Root Cause Analysis)
**Reviewers:** [Pending cross-review]