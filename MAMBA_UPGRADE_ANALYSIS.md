# Mamba-SSM Upgrade Analysis (2.2.2 → 2.2.5)

**Date**: 2025-09-29
**Context**: Modal A100 XID 31 crash investigation revealed potential AMP bug in mamba-ssm 2.2.2

---

## Executive Summary

**Current Status**: STUCK on mamba-ssm 2.2.2 due to dependency constraints
**Can We Upgrade?**: YES, but requires full stack upgrade (PyTorch 2.2.2 → 2.4+)
**Should We Upgrade?**: WAIT for Test 1A results first

---

## Current Stack Constraints

### The Dependency Chain

```
causal-conv1d 1.4.0 (LOCKED)
    ↓
PyTorch 2.2.2 (LOCKED - causal-conv1d 1.5+ requires PyTorch 2.4+)
    ↓
mamba-ssm 2.2.2 (LOCKED - we chose this for PyTorch 2.2.2 compatibility)
    ↓
torch-geometric 2.6.1 (LOCKED - latest stable for PyTorch 2.2.2)
    ↓
CUDA 12.1 (LOCKED - PyTorch 2.2.2 build target)
```

**Key constraint**: `causal-conv1d 1.4.0` is the anchor preventing upgrades.

---

## Why We're on mamba-ssm 2.2.2

### Documentation Claims

From `INSTALLATION.md:106`, `pyproject.toml:62`:
```
mamba-ssm==2.2.2  # 2.2.5 has bugs, 2.2.4 has issues
```

**Last updated**: 2025-09-28 (based on git log)

### Web Research Findings (2025-09-29)

| Version | Released | Notes |
|---------|----------|-------|
| **2.2.2** | 2024-07-03 | Fix: "varlen generation by passing seq_idx to causal_conv1d" |
| 2.2.3 | 2024-12-06 | Version bump, no specific fixes listed |
| 2.2.4 | 2024-12-06 | Version bump, no specific fixes listed |
| **2.2.5** | 2025-07-19 | **Supports PyTorch 2.4/2.5** + CUDA 11/12 |

**Critical finding**: Issue #686 (illegal memory access with long sequences on H100) was **fixed by changing indexing to int64 in kernels** — this fix may be in 2.2.3+.

---

## What Would an Upgrade Require?

### Option A: Upgrade mamba-ssm Only (NOT POSSIBLE)

```
mamba-ssm 2.2.2 → 2.2.5
```

**Blocker**: mamba-ssm 2.2.5 requires PyTorch 2.4+, we're on 2.2.2

---

### Option B: Full Stack Upgrade (POSSIBLE)

```
PyTorch 2.2.2 → 2.4.0 (or 2.5.0)
    ↓
causal-conv1d 1.4.0 → 1.5.0+
    ↓
mamba-ssm 2.2.2 → 2.2.5
    ↓
torch-geometric 2.6.1 → 2.7.0+ (check compatibility with PyTorch 2.4+)
    ↓
CUDA 12.1 → 12.4 (PyTorch 2.4+ build target)
```

**Required changes**:
1. Upgrade CUDA Toolkit 12.1 → 12.4 (local + Modal image)
2. Upgrade PyTorch 2.2.2 → 2.4.0
3. Upgrade causal-conv1d 1.4.0 → 1.5.0+
4. Upgrade mamba-ssm 2.2.2 → 2.2.5
5. Upgrade torch-geometric 2.6.1 → latest for PyTorch 2.4
6. Rebuild ALL cache (4667 train + 1832 dev NPZ files)
7. Re-test entire pipeline (unit + integration + smoke tests)
8. Re-validate trained models (checkpoint compatibility likely broken)

**Estimated effort**: 2-3 days + full re-training (100 epochs ≈ $300-600 on Modal)

---

## Risk Analysis

### Risks of Upgrading

| Risk | Severity | Mitigation |
|------|----------|------------|
| PyTorch 2.4 API changes break our code | 🟡 MEDIUM | Test suite should catch most issues |
| PyG 2.7+ incompatibility | 🟡 MEDIUM | Check PyG docs for PyTorch 2.4 support |
| New mamba-ssm bugs (2.2.5 is recent) | 🟠 MEDIUM-HIGH | July 2025 release, limited production testing |
| Cache incompatibility requires rebuild | 🔴 HIGH | 4667+1832 files ≈ 1-2 hours rebuild time |
| Checkpoint incompatibility | 🔴 HIGH | May need to restart training from scratch |
| Local + Modal drift if one fails | 🔴 HIGH | Must upgrade both simultaneously |

### Risks of NOT Upgrading (Staying on 2.2.2)

| Risk | Severity | Mitigation |
|------|----------|------------|
| AMP bug on A100 (suspected) | 🔴 HIGH | **Workaround: Disable AMP** (2x slower training) |
| Missing kernel fixes from 2.2.3-2.2.5 | 🟡 MEDIUM | Fallback to Conv1d works locally |
| Technical debt accumulates | 🟢 LOW | Can upgrade later when stable |

---

## Current Web Search Findings

### mamba-ssm Known Issues

**Issue #686** (Long sequences, H100):
- **Root cause**: int32 overflow in Triton kernel indexing with 512K token sequences
- **Fix**: "Changed indexing to int64 in kernels"
- **Status**: Fixed (version unknown, likely 2.2.3+)

**Issue #732** (A6000, specific tensor sizes):
- **Root cause**: Illegal memory access with certain (batch, seq_len, d_model) combinations
- **Status**: Open, no definitive fix

**No reports of**: "A100 + mamba-ssm 2.2.2 + AMP + XID 31" specifically

### PyTorch AMP + Custom CUDA Kernels

- **Known pattern**: AMP can expose pointer arithmetic bugs in custom kernels
- **A100 specifics**: 312 TFLOPS FP16 vs 19 TFLOPS FP32 (16x speedup with AMP)
- **Recommendation**: "Use AMP properly: parameters in FP32, autocast during compute"

---

## Recommended Action Plan

### Phase 1: Confirm Root Cause (IN PROGRESS)

**Test 1A** (AMP OFF on mamba-ssm 2.2.2):
- **Status**: Running (ETA 30-40 minutes)
- **If PASSES**: Confirms AMP bug in 2.2.2, proceed to Phase 2
- **If FAILS**: Rules out AMP, investigate other causes

---

### Phase 2: Choose Fix Strategy (DECISION POINT)

#### Strategy A: Quick Fix (Disable AMP) ⭐⭐⭐⭐
**If Test 1A passes**:

```yaml
# configs/modal/train.yaml
training:
  mixed_precision: false  # Disable AMP due to Mamba 2.2.2 kernel bug on A100
```

**Pros**:
- ✅ 10 minutes to implement and test
- ✅ Zero risk (local already uses FP32)
- ✅ No cache rebuild required
- ✅ No dependency changes

**Cons**:
- ❌ ~2x slower training (FP32 vs FP16)
- ❌ ~2x more expensive (A100 tensor cores unused)
- ❌ Doesn't fix underlying bug

**Cost**: ~$600 for 100 epochs (vs $300 with AMP)

**Recommendation**: **DO THIS FIRST**. Get training working, evaluate later.

---

#### Strategy B: Upgrade Full Stack ⭐⭐
**If Test 1A passes AND we want to keep AMP**:

**Phase B1: Research & Planning (2-4 hours)**
1. Check PyTorch 2.4/2.5 release notes for breaking changes
2. Verify torch-geometric 2.7+ compatibility with PyTorch 2.4
3. Check if mamba-ssm 2.2.5 has AMP fixes (search GitHub issues/PRs)
4. Document exact new version matrix

**Phase B2: Local Upgrade (4-8 hours)**
1. Install CUDA 12.4 toolkit locally
2. Upgrade PyTorch 2.2.2 → 2.4.0
3. Upgrade causal-conv1d, mamba-ssm, PyG
4. Run full test suite
5. Rebuild cache (4667+1832 files)
6. Run smoke test + verify training works

**Phase B3: Modal Upgrade (2-4 hours)**
1. Update `deploy/modal/app.py` image with new versions
2. Test Mamba CUDA: `modal run deploy/modal/app.py --action test-mamba`
3. Run Modal smoke test
4. If passes, run full training

**Phase B4: Validation (24-48 hours)**
1. Run full training on Modal (100 epochs)
2. Compare metrics to baseline (if we have one)
3. Monitor for new NaN/crash issues

**Total time**: 1-2 weeks (including training validation)
**Risk**: 🟠 MEDIUM-HIGH (many moving parts)

**Recommendation**: **DEFER until after successful training run with AMP disabled**.

---

#### Strategy C: Report Bug to mamba-ssm (Parallel Track)

**Create GitHub issue**:
- Title: "XID 31 MMU Fault with AMP on A100 (mamba-ssm 2.2.2 + PyTorch 2.2.2)"
- Include: Full config, logs, headdim calculations, Test 1A results
- Ask: "Is this fixed in 2.2.5? Should we upgrade to PyTorch 2.4?"

**Expected outcome**: Upstream guidance on whether 2.2.5 fixes this specific issue

**Recommendation**: **DO THIS AFTER Test 1A confirms AMP is the cause**.

---

## Decision Matrix

| Scenario | Action |
|----------|--------|
| **Test 1A PASSES** | 1. Disable AMP (Strategy A) <br> 2. Train successfully <br> 3. Report bug to mamba-ssm (Strategy C) <br> 4. DEFER full upgrade (Strategy B) until urgent need |
| **Test 1A FAILS** | 1. Rule out AMP <br> 2. Analyze Test 1B (force fallback) <br> 3. Check cache/batch size <br> 4. Upgrade NOT relevant to current crash |

---

## Why We're NOT Upgrading Right Now

1. **Test 1A is running**: Need to confirm AMP is the root cause before deciding on fix
2. **Quick workaround exists**: Disabling AMP is 10 minutes vs 1-2 weeks for full upgrade
3. **Risk is high**: New stack = new bugs, cache rebuild, re-training
4. **2.2.5 is recent** (July 2025): Limited production testing, unclear if it fixes THIS specific issue
5. **Local training works**: Proves our model architecture is sound on 2.2.2 with FP32

---

## When SHOULD We Upgrade?

**Triggers for full stack upgrade**:
1. ✅ mamba-ssm maintainers confirm 2.2.5 fixes AMP+A100 bug
2. ✅ PyTorch 2.4/2.5 has critical features we need
3. ✅ torch-geometric 2.7+ has critical features we need
4. ✅ We have 1-2 weeks for careful testing + re-training
5. ✅ FP32 training speed becomes unacceptable cost bottleneck

**Current status**: NONE of these triggers are met.

---

## Conclusion

### Current Constraints (Why mamba-ssm 2.2.2)

```
causal-conv1d 1.4.0 → Requires PyTorch <2.4
                   ↓
                PyTorch 2.2.2 → Compatible with mamba-ssm 2.2.2
                               ↓
                            mamba-ssm 2.2.2 (STUCK HERE)
```

### Can We Upgrade?

**Technical**: YES (PyTorch 2.2.2 → 2.4+ → mamba-ssm 2.2.5)
**Practical**: NO (high risk, 1-2 weeks effort, unclear if fixes our specific bug)

### Recommendation

1. ✅ **WAIT for Test 1A results** (30-40 minutes)
2. ✅ **If Test 1A passes**: Disable AMP, train successfully
3. ✅ **Report bug to mamba-ssm** with full details
4. ⏸️ **DEFER upgrade** until:
   - Upstream confirms 2.2.5 fixes this issue, OR
   - We have urgent need for newer PyTorch features, OR
   - FP32 training cost becomes prohibitive

### Bottom Line

**We're not constrained by architectural incompatibilities** — we're constrained by:
1. **causal-conv1d 1.4.0** requiring PyTorch <2.4
2. **Risk/effort trade-off**: 10 minutes (disable AMP) vs 1-2 weeks (full upgrade)
3. **Uncertainty**: No guarantee 2.2.5 fixes our specific AMP+A100 bug

**The documentation saying "2.2.5 has bugs" is outdated or refers to issues we haven't hit.** The REAL constraint is the PyTorch version dependency chain.

---

**Status**: Waiting for Test 1A preflight results to confirm AMP is the root cause before implementing any fix.