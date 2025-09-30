# Modal XID 31 Recurrence Investigation

**Date**: 2025-09-30 07:50 UTC
**Status**: 🔴 P0 BLOCKER - Full training failed with same XID 31 error
**Context**: PyTorch 2.5.0 + mamba-ssm 2.2.5 upgrade supposedly fixed this

---

## Executive Summary

**CRITICAL**: Modal full training crashed with **XID 31 MMU Fault** despite:
- ✅ Smoke test passed (50 files, 1 epoch)
- ✅ Local training stable (RTX 4090, batch 69+, no issues)
- ✅ PyTorch 2.5.0 + mamba-ssm 2.2.5 installed (should fix int64 indexing bug)
- ✅ Preflight test ran successfully during smoke

**The failure pattern**:
```
Smoke test (configs/modal/smoke.yaml):
  - 50 files, 1 epoch
  - Batch size: 64
  - Result: ✅ PASSED (~38 minutes)

Full training (configs/modal/train.yaml):
  - 4667 files, 100 epochs
  - Batch size: 64
  - Result: ❌ XID 31 at preflight check
```

**Why did smoke pass but full training fail?**

---

## Error Analysis

### The Crash (2025-09-30 07:50 UTC)

```
[gpu-health] [WARN] GPU-e11dfcca-7c93-959a-b8fd-5cfe0839b163: XID: NVRM: Xid (PCI:0000:00:06): 31, pid=79749, name=exe, Ch 0000000a, intr 00000000. MMU Fault: ENGINE GRAPHICS GPC2 GPCCLIENT_T1_7 faulted @ 0x2b57_b3000000. Fault is of type FAULT_PDE ACCESS_TYPE_VIRT_WRITE

[MAMBA] Forward pass error, using fallback: CUDA error: an illegal memory access was encountered
[PREFLIGHT] ✗ Failed on test batch: CUDA error: an illegal memory access was encountered
  - Model type: SeizureDetector
  - Input shape: torch.Size([64, 19, 15360])
  - Labels shape: torch.Size([64, 15360])
  - Loss mode: focal
  - Device: cuda

Training error: CUDA error: an illegal memory access was encountered
RuntimeError: Training failed with exit code 1
```

### Timeline of Events

| Time (UTC) | Event | Duration |
|------------|-------|----------|
| 06:37:16 | Training started | - |
| 06:37:17 | Cache verification passed | 1s |
| 06:48:18 | Train dataset loaded (61,616 windows) | ~11min |
| 06:48:18 | Dev dataset indexing started (1,832 files) | - |
| 07:40:48 | Dev dataset ready (148,224 windows) | **52 minutes** |
| 07:40:58 | Model created (31M parameters) | 10s |
| 07:41:01 | W&B initialized | 3s |
| 07:48:22 | Preflight check started | **7min 21s gap!** |
| 07:50:56 | **XID 31 MMU FAULT** | 2min 34s into preflight |
| 07:50:57 | Fallback attempted, training aborted | 1s |

### Suspicious Observations

1. **52-minute dev dataset indexing** (07:48 to 07:40)
   - Smoke test: Indexing was fast (<5 min for 50 files)
   - Full training: 52 minutes for 1,832 files
   - **Rate**: ~35 files/minute (smoke was ~10 files/minute)

2. **7-minute gap before preflight** (07:41 to 07:48)
   - What was happening during this time?
   - No log output between W&B init and preflight start

3. **Preflight failed, but smoke's preflight passed**
   - Same batch size (64)
   - Same model architecture
   - Same mamba-ssm version
   - **Different**: More files loaded into dev dataset

---

## Hypothesis: Dataset Size Triggers Memory Corruption

### Theory

**The XID 31 fault is NOT fixed by PyTorch 2.5 + mamba-ssm 2.2.5.**

**New hypothesis**: The bug is **data-dependent** or **cumulative**:
- Smoke test: Small dataset (50 files) → Memory state OK → Preflight passes
- Full training: Large dataset (1,832 dev files) → Memory corruption during indexing → Preflight fails

**Possible causes**:
1. **Dataset indexing corrupts GPU memory**
   - 52 minutes of file I/O + NPZ loading
   - Builds large index structure in memory
   - Possible memory leak or fragmentation

2. **Memory allocator state corrupted**
   - PyTorch caching allocator gets into bad state
   - First forward pass triggers illegal access
   - Preflight catches it before training starts

3. **CUDA context initialization race**
   - Large dataset triggers different memory layout
   - Mamba kernels initialized with wrong pointers
   - Crash on first actual use

4. **Batch sampler creates pathological batch**
   - BalancedSeizureDataset with 148K windows
   - First batch after indexing has specific pattern
   - Mamba kernel can't handle it

---

## Evidence Review

### What We Know

| Fact | Implication |
|------|-------------|
| Smoke test passed | Mamba CUDA works with small datasets |
| Local training stable | Architecture is sound, RTX 4090 has no issue |
| PyTorch 2.5 + mamba 2.2.5 | int64 indexing fix present |
| XID 31 at preflight | Crash happens BEFORE training loop |
| 52-minute dev indexing | Unusually long for SSD cache |
| No crash during smoke indexing | Indexing itself may not be the issue |

### What We Don't Know

| Unknown | Investigation Needed |
|---------|---------------------|
| Why 7-minute gap before preflight? | Check Modal logs for hidden processes |
| What is CPU/RAM usage during indexing? | Modal metrics dashboard |
| Is GPU memory fragmented after indexing? | Add `torch.cuda.memory_summary()` logging |
| Does batch 0 have specific pattern? | Log first batch contents before preflight |
| Is there a file that triggers corruption? | Bisect dev dataset (1832 → 916 → 458...) |

---

## Comparison: Smoke vs Full Training

### Config Differences

| Setting | Smoke | Full | Impact |
|---------|-------|------|--------|
| **Train files** | 50 | 4,667 | 93× more data |
| **Dev files** | ~10 | 1,832 | 183× more data |
| **Epochs** | 1 | 100 | Irrelevant (crashed before epoch 1) |
| **Batch size** | 64 | 64 | Same |
| **Mixed precision** | true | true | Same |
| **Gradient clip** | 0.5 | 0.5 | Same |
| **Model architecture** | v3 | v3 | Same |

**Key difference**: **Dev dataset size (10 vs 1,832 files)**

### Timing Differences

| Phase | Smoke | Full | Ratio |
|-------|-------|------|-------|
| Train index | <5 min | ~11 min | 2× slower |
| Dev index | <5 min | **52 min** | **10× slower** |
| Model init | ~10s | 10s | Same |
| Preflight | ✅ Pass | ❌ XID 31 | - |

**Anomaly**: Dev indexing 10× slower than expected

---

## Root Cause Candidates

### Candidate 1: mamba-ssm 2.2.5 Does NOT Fix A100 XID 31 ⭐⭐⭐⭐⭐

**Evidence FOR**:
- XID 31 still happens with mamba-ssm 2.2.5
- Same error message as before upgrade
- Same GPU (A100-80GB)
- GitHub Issue #686 fix may not cover all cases

**Evidence AGAINST**:
- Smoke test passed (but with smaller dataset)
- Local training stable (but different GPU)

**Conclusion**: **LIKELY** - The int64 fix may not cover the specific pattern we hit with large datasets.

---

### Candidate 2: Dev Dataset Indexing Corrupts GPU Memory ⭐⭐⭐⭐

**Evidence FOR**:
- 52-minute indexing is unusually long
- Crash happens immediately after indexing completes
- Preflight uses first batch from newly-indexed dataset

**Evidence AGAINST**:
- Indexing should be CPU-only (NPZ loading)
- No GPU operations during indexing phase

**Test**: Add `torch.cuda.empty_cache()` and `torch.cuda.reset_peak_memory_stats()` after dev dataset initialization.

---

### Candidate 3: Batch Size 64 + Dev Dataset Size Triggers Bug ⭐⭐⭐

**Evidence FOR**:
- Smoke used batch 64 with small dataset → OK
- Full uses batch 64 with large dataset → Crash
- Mamba kernels may allocate buffers based on dataset size

**Evidence AGAINST**:
- Batch size should only affect per-batch memory, not global state

**Test**: Reduce batch size to 32 for full training, see if crash still happens.

---

### Candidate 4: Specific File in Dev Dataset Triggers Corruption ⭐⭐⭐

**Evidence FOR**:
- Crash happens after processing all 1,832 dev files
- One file may have pathological data (extreme values, NaN, inf)
- Indexing loads file into memory, corruption persists

**Evidence AGAINST**:
- Cache validation passed (manifest.json exists)
- Preprocessing should have sanitized data

**Test**: Bisect dev dataset - run with dev files 1-916, then 917-1832, find which half triggers crash.

---

### Candidate 5: Modal A100 GPU Hardware Issue ⭐⭐

**Evidence FOR**:
- Consistent XID 31 (MMU fault at specific address)
- Same GPU instance ID may have hardware defect

**Evidence AGAINST**:
- Smoke test passed on same instance
- XID 31 is a software error (illegal memory access)

**Test**: Request different A100 instance from Modal, retry training.

---

## Critical Questions

### Q1: Why did smoke test pass?

**Possible answers**:
1. **Dataset size below threshold**: Smoke's 10 dev files don't trigger the bug
2. **Luck**: First batch in smoke test happened to avoid bad memory pattern
3. **Timing**: Short runtime means corrupted state doesn't manifest

### Q2: Why does local training work?

**Possible answers**:
1. **Different GPU** (RTX 4090 vs A100): Different CUDA kernel codepaths
2. **Different PyTorch build** (local vs Modal): Slightly different compilation
3. **No dataset indexing phase locally**: Using pre-built cache, different memory state

### Q3: What happened during the 7-minute gap?

**Log shows**:
```
07:41:01 - W&B initialized
07:48:22 - Preflight check started
```

**Possible explanations**:
1. **Waiting for resources**: Modal container startup lag
2. **Background operations**: Disk I/O, cache warming
3. **Memory allocation**: Large tensors being allocated
4. **Logging suppressed**: Operations happened but didn't log

**Action**: Add debug logging before preflight to capture this phase.

### Q4: Is this the same bug or a new bug?

**Comparison**:

| Aspect | Old Bug (Pre-upgrade) | Current Bug |
|--------|----------------------|-------------|
| **Error** | XID 31 MMU Fault | XID 31 MMU Fault |
| **Location** | During training loop | During preflight check |
| **Trigger** | AMP + batch processing | Dev dataset indexing? |
| **Fix attempted** | PyTorch 2.5 + mamba 2.2.5 | Not yet fixed |

**Verdict**: **SAME BUG, different manifestation** - Upgrade didn't fix root cause.

---

## Immediate Next Steps

### Step 1: Verify Local Still Works ✅

**Status**: User confirmed local training at batch 69+, no issues

**Conclusion**: Bug is Modal/A100-specific, not architecture issue.

---

### Step 2: Review Upgrade Validation

**What we validated**:
- ✅ Smoke test passed (50 files, 1 epoch)
- ✅ Modal Mamba CUDA test passed (isolated test)

**What we DID NOT validate**:
- ❌ Full training with 1,832 dev files
- ❌ Long dataset indexing phase
- ❌ Memory state after large dataset load

**Lesson**: Smoke test is NOT sufficient to validate full training.

---

### Step 3: Add Diagnostic Logging

**Proposal**: Update `deploy/modal/app.py` and `src/brain_brr/train/loop.py` to log:

```python
# After dev dataset initialization
logger.info(f"[GPU] Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
logger.info(f"[GPU] Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f}GB")
torch.cuda.empty_cache()
logger.info(f"[GPU] After empty_cache: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

# Before preflight
logger.info("[PREFLIGHT] Starting preflight check...")
logger.info(f"[PREFLIGHT] First batch device: {next(iter(train_loader))[0].device}")
logger.info(f"[PREFLIGHT] Model device: {next(model.parameters()).device}")
```

---

### Step 4: Bisect Dev Dataset

**Plan**:
1. Run with dev files 1-916 (half)
2. If passes → bug in files 917-1832
3. If fails → bug in files 1-916
4. Repeat until single file identified

**Time**: ~2 hours per bisection (52min indexing + 10min preflight)

---

### Step 5: Try Workarounds

#### Workaround A: Reduce Batch Size

```yaml
# configs/modal/train.yaml
training:
  batch_size: 32  # Was 64
```

**Rationale**: Smaller batch = less GPU memory pressure per forward pass

**Time**: 1 hour to test

---

#### Workaround B: Clear GPU Cache Before Preflight

```python
# loop.py before preflight
torch.cuda.empty_cache()
torch.cuda.synchronize()
gc.collect()
```

**Rationale**: Clean slate before first forward pass

**Time**: 5 minutes to implement + 1 hour to test

---

#### Workaround C: Disable Mixed Precision

```yaml
# configs/modal/train.yaml
training:
  mixed_precision: false  # Was true
```

**Rationale**: AMP may still be involved (smoke test ran briefly, full training stresses AMP)

**Time**: 1 hour to test

**Cost**: 2× slower, 2× more expensive (~$600 vs $300)

---

#### Workaround D: Use Mamba Fallback for First Batch

```python
# mamba.py - force fallback on first call
if not hasattr(self, '_first_call_done'):
    self._first_call_done = True
    return self._conv1d_fallback(x)
```

**Rationale**: Let CUDA context stabilize before using Mamba kernels

**Time**: 10 minutes to implement + 1 hour to test

---

## Long-Term Solutions

### Solution 1: Report to mamba-ssm Maintainers

**Create GitHub issue**:
- Title: "XID 31 MMU Fault on A100 with PyTorch 2.5.0 + mamba-ssm 2.2.5 after large dataset indexing"
- Include: Full logs, config, dataset size, reproducible test case (smoke passes, full fails)
- Ask: "Is there a known issue with dataset size triggering memory corruption?"

---

### Solution 2: Consider Alternative Architectures

**If Mamba CUDA is fundamentally broken on A100**:
1. Replace BiMamba with Transformer (attention-based)
2. Use Conv1D fallback permanently (slower but stable)
3. Switch to GRU/LSTM for sequential modeling

**Impact**: May require retraining + architecture changes

---

### Solution 3: Request Modal Support

**Contact Modal**:
- Report XID 31 pattern (hardware MMU fault)
- Ask if specific A100 instances have known issues
- Request different GPU allocation

---

## Decision Tree

```
Is local training still stable?
├─ YES → Bug is Modal/A100-specific
│   ├─ Try Workaround B (clear cache) [5min impl + 1hr test]
│   ├─ Try Workaround C (disable AMP) [1hr test, 2× cost]
│   └─ Try Workaround D (fallback first batch) [10min impl + 1hr test]
│
└─ NO → Bug is architectural
    ├─ Rollback to PyTorch 2.2.2 + mamba-ssm 2.2.2
    └─ Investigate NaN protection system

Did any workaround succeed?
├─ YES → Document + proceed with training
│   └─ Report to mamba-ssm as non-blocking issue
│
└─ NO → P0 BLOCKER
    ├─ Bisect dev dataset to find trigger file [~10 hours]
    ├─ Contact Modal support [1-2 days]
    └─ Consider architecture change [1 week+]
```

---

## Recommended Action Plan

### Phase 1: Quick Workarounds (Next 2 hours)

1. **Workaround B** (5 min impl): Clear GPU cache before preflight
2. **Test** (1 hour): Run full training, see if preflight passes
3. **If fails**: Try Workaround C (disable AMP, 1 hour test)

### Phase 2: Diagnostic Investigation (If Phase 1 fails)

4. **Add logging** (10 min impl): GPU memory stats, dataset stats
5. **Bisect dataset** (10 hours): Find which dev files trigger bug
6. **Report to mamba-ssm** (30 min): Create detailed GitHub issue

### Phase 3: Escalation (If Phase 2 inconclusive)

7. **Contact Modal support**: Request different A100 instance
8. **Consider fallback architecture**: Replace Mamba with Transformer
9. **Document P0 blocker**: Full training impossible on Modal

---

## Open Questions for Discussion

1. **Should we try workarounds before investigation?**
   - Pro: Faster path to working training
   - Con: Don't understand root cause

2. **Should we bisect the dev dataset?**
   - Pro: May find specific trigger file
   - Con: 10+ hours of testing, may not be file-specific

3. **Should we disable AMP as workaround?**
   - Pro: Smoke test passed with AMP, so may not help
   - Con: 2× cost increase (~$600 for 100 epochs)

4. **Should we rollback PyTorch 2.5 upgrade?**
   - Pro: Known working state (local training)
   - Con: Doesn't help - old stack also had XID 31 on Modal

5. **Should we switch away from Mamba?**
   - Pro: Transformer is battle-tested
   - Con: Architecture change = re-training + validation

---

## Conclusion

**Current Status**: 🔴 **P0 BLOCKER** - Cannot run full training on Modal

**Root Cause**: **UNKNOWN** - XID 31 persists despite PyTorch 2.5 + mamba-ssm 2.2.5

**Leading Theory**: Dataset size triggers memory corruption not caught by smoke test

**Recommended Next Step**: **Try Workaround B (clear GPU cache)** - 5 min impl + 1 hour test

**If Workaround B fails**: **Disable AMP (Workaround C)** - Last resort before deep investigation

**Critical Insight**: **Smoke test validation was insufficient** - Need full-scale validation before declaring upgrade successful

---

**Last Updated**: 2025-09-30
**Requires Decision**: Workaround strategy vs deep investigation
**Blocking**: All Modal training (local unaffected)