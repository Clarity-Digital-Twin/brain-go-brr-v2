# Modal A100 CUDA Failure - Diagnostic Test Status

**Started:** 2025-09-29 20:31 UTC
**Expected completion:** ~60 minutes per test (55min data load + 5min preflight)

---

## Tests Running

### Test 1A: Disable AMP ⭐⭐⭐⭐ (HIGHEST PRIORITY)
**Hypothesis:** Mamba-SSM 2.2.2 AMP-specific kernel bug on A100
**Config:** `configs/modal/diag_1a_amp_off.yaml`
**Change:** `mixed_precision: false` (FP32 only)
**Expected:** If preflight passes, **CONFIRMS** AMP is the root cause
**Modal App:** https://modal.com/apps/clarity-digital-twin/main/ap-bzlS4m8MF3DV7SjtwgavPq
**Status:** 🟡 RUNNING - Data loading phase
**Bash ID:** 7b474a

**If this passes:** We have the root cause (AMP + Mamba on A100). Fix = disable AMP or wait for upstream mamba-ssm patch.

---

### Test 1B: Force Mamba Fallback ⭐⭐⭐⭐ (ISOLATION)
**Hypothesis:** Issue is in Mamba CUDA kernels, not our code
**Config:** `configs/modal/diag_1b_fallback.yaml`
**Change:** `force_fallback=true` (use Conv1d instead of Mamba CUDA)
**Expected:** If preflight passes with AMP ON, **CONFIRMS** Mamba CUDA is the culprit
**Status:** 🟡 RUNNING
**Bash ID:** 79dbf0

**If this passes:** Isolates Mamba vs our PyG/TCN code. Confirms Mamba CUDA path is broken.

---

### Test 2A: CUDA Synchronous Execution ⭐⭐⭐ (DIAGNOSTIC)
**Purpose:** Get exact line number where illegal memory access occurs
**Config:** `configs/modal/diag_2a_blocking.yaml`
**Change:** `CUDA_LAUNCH_BLOCKING=1` + `TORCH_USE_CUDA_DSA=1`
**Expected:** Synchronous stack trace pinpointing exact Mamba kernel operation
**Status:** 🟡 RUNNING
**Bash ID:** b48372

**If this fails (expected):** We get exact kernel failure site for upstream bug report.

---

### Test 2B: Single-Arch A100 Rebuild ⭐⭐⭐ (CACHE HYGIENE)
**Hypothesis:** Modal cached extension mismatch
**Config:** `configs/modal/train.yaml` (full config with AMP)
**Change:** `TORCH_CUDA_ARCH_LIST="8.0"` (A100 only), cache purge, force reinstall
**Expected:** If preflight passes, **CONFIRMS** cache poisoning hypothesis
**Status:** 🟡 RUNNING - Building single-arch image
**Bash ID:** 541997

**If this passes:** Cache mismatch was the issue. Always use single-arch builds.
**If this fails:** Rules out cache, back to Hypothesis 1 (AMP bug).

---

## Monitoring via Modal Web UI

**Test 1A (AMP OFF):** https://modal.com/apps/clarity-digital-twin/main/ap-bzlS4m8MF3DV7SjtwgavPq

**Key phrases to search for in logs:**
- `[PREFLIGHT] ✓` = **PASS** (root cause confirmed!)
- `[PREFLIGHT] ✗` = **FAIL** (not AMP, move to Test 1B)
- `XID 31` = GPU MMU fault (same as before)
- `illegal memory access` = CUDA error (same as before)
- `CUDA error` = Kernel failure

---

## Expected Timeline

| Time | Event |
|------|-------|
| +0 min | Tests launched in parallel |
| +5 min | Image pulls complete, data loading starts |
| +55 min | Data loading complete (4667 train files, 1832 dev files) |
| +56 min | **PREFLIGHT TEST** - First GPU forward pass |
| +57 min | **RESULTS KNOWN** |

---

## Decision Tree

```
Test 1A (AMP OFF):
├─ ✅ PASS → ROOT CAUSE = AMP bug in Mamba-SSM
│   └─ FIX: Train with FP32, report to mamba-ssm upstream
│
├─ ❌ FAIL → Test 1B (Fallback):
    ├─ ✅ PASS → Issue in Mamba CUDA (not AMP-specific)
    │   └─ FIX: Use fallback permanently, report to mamba-ssm
    │
    └─ ❌ FAIL → Test 2B (Single-arch):
        ├─ ✅ PASS → ROOT CAUSE = Cache poisoning
        │   └─ FIX: Always use single-arch builds
        │
        └─ ❌ FAIL → Issue in PyG/TCN or hardware
            └─ Investigate PyG CUDA kernels next
```

---

## Next Steps (After Results)

1. **Collect all logs** from Modal
2. **Analyze failure patterns**
3. **Update MODAL_A100_CUDA_FAILURE_ANALYSIS.md** with confirmed root cause
4. **Implement clean fix** (not hacky workaround)
5. **Remove diagnostic cruft** (test configs, app_single_arch.py)
6. **Resume production training** with fix applied

---

**Status Last Updated:** 2025-09-29 20:33 UTC