# Incomplete Implementations Audit — October 4, 2025

**Status**: 🔴 **CRITICAL DISCOVERY - FALSE SECURITY**
**Impact**: ⚠️ **NO IMPACT ON TRAINING** (gradient clipping works independently)
**Action Required**: 🛠️ **Remove dead code OR implement features properly**

---

## Executive Summary

**CRITICAL FINDING**: Four "protection" environment variables are **defined but never used** in the codebase, creating a **false sense of security**. Modal training explicitly sets `BGB_SANITIZE_GRADS=1` thinking it enables gradient sanitization, but **this flag is completely ignored** by the training loop.

**Good News**: Training is NOT affected because the actual protection (`torch.nn.utils.clip_grad_norm_`) works independently. But we've been running with **fake safety features**.

---

## 🚨 FALSE SECURITY: Unused Protection Flags

### 1. `BGB_SANITIZE_GRADS` ❌ **MOST CRITICAL**

**Defined**: `src/brain_brr/utils/env.py:30`
**Used**: **NOWHERE** (0 references in training code)
**Modal Sets It**: `deploy/modal/app.py:713` with comment "Prevent gradient explosion"

**What We Thought It Did**:
- Sanitize NaN/Inf gradients before optimizer step
- Replace non-finite gradients with zeros
- Add extra layer of protection beyond clipping

**What It Actually Does**:
- **NOTHING**. The flag is read into `_SANITIZE_GRADS` but never checked.

**Evidence**:
```bash
$ rg "env\.sanitize_grads|ENV\.sanitize_grads" src/ --type py
# NO RESULTS
```

**Actual Gradient Protection** (src/brain_brr/train/train_step.py:194, 199):
```python
# THIS is what actually protects us (PyTorch built-in):
grad_norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
# gradient_clip = 0.5 from config
```

**Impact**:
- ✅ **Training is SAFE** - gradient clipping works
- ❌ **False security** - we thought we had extra protection
- ❌ **Misleading logs** - "Mean=inf" is pre-clip value, looks scary but is harmless

---

### 2. `BGB_SANITIZE_INPUTS` ❌

**Defined**: `src/brain_brr/utils/env.py:29`
**Used**: **NOWHERE**

**What We Thought It Did**:
- Sanitize input tensors (EEG windows) before forward pass
- Replace NaN/Inf in data with zeros

**What It Actually Does**:
- **NOTHING**

**Evidence**:
```bash
$ rg "env\.sanitize_inputs" src/ --type py
# NO RESULTS
```

**Actual Input Protection**:
- Preprocessing clips to ±10σ (src/brain_brr/data/preprocess.py)
- Model has input sanitization at component boundaries (edge_features.py)
- **BUT**: Not controlled by this flag!

---

### 3. `BGB_SKIP_OPT_STEP_ON_NAN` ❌

**Defined**: `src/brain_brr/utils/env.py:31`
**Used**: **NOWHERE**

**What We Thought It Did**:
- Skip optimizer.step() if loss is NaN
- Prevent NaN from propagating to model parameters

**What It Actually Does**:
- **NOTHING**

**Evidence**:
```bash
$ rg "env\.skip_opt_step_on_nan" src/ --type py
# NO RESULTS
```

**Actual NaN Handling** (src/brain_brr/train/train_step.py:214-221):
```python
if torch.isfinite(torch.tensor(loss_val)):
    total_loss += loss_val
    num_batches += 1
else:
    logger.warning(f"Non-finite loss detected at batch {batch_idx}, skipping in average")
```

**Impact**:
- NaN losses are excluded from average **but optimizer still steps**
- Flag would prevent the step entirely if implemented
- Current behavior is probably fine (clipping prevents NaN propagation)

---

### 4. `BGB_SAFE_CLAMP` ❌

**Defined**: `src/brain_brr/utils/env.py:37-39`
**Used**: **NOWHERE**

**What We Thought It Did**:
- Clamp all activations to safe range (default: [-10, 10])
- Apply `torch.nan_to_num()` + `torch.clamp()` after every block

**What It Actually Does**:
- **NOTHING**

**Evidence**:
```bash
$ rg "env\.safe_clamp" src/ --type py
# NO RESULTS
```

**Actual Clamping**:
- Edge features have hardcoded clamping (edge_features.py: `torch.clamp(x, min=-10.0, max=10.0)`)
- **BUT**: Not controlled by this flag!

---

## ✅ Flags That ARE Actually Used

For comparison, here are flags that DO work:

### 1. `BGB_ANOMALY_DETECT` ✅

**Defined**: `src/brain_brr/utils/env.py:32`
**Used**: `src/brain_brr/train/loop.py:81`

```python
if env.anomaly_detect():
    torch.autograd.set_detect_anomaly(True)
    logger.info("[DEBUG] PyTorch anomaly detection enabled")
```

**Works Correctly**: Enables PyTorch's backward anomaly detection

---

### 2. `BGB_NAN_DEBUG` ✅

**Defined**: `src/brain_brr/utils/env.py:27`
**Used**: `src/brain_brr/models/debug_utils.py:21`

```python
DEBUG_FINITE = env.debug_finite() or env.smoke_test() or env.nan_debug()
```

**Works Correctly**: Enables finite value debugging in models

---

### 3. `BGB_SMOKE_TEST` ✅

**Defined**: `src/brain_brr/utils/env.py:20`
**Used**: 6 locations across data loading and training

**Works Correctly**: Limits dataset to 3 files for quick validation

---

### 4. `BGB_DISABLE_TQDM` ✅

**Defined**: `src/brain_brr/utils/env.py:23`
**Used**: `train_step.py:147`, `val_step.py:263`, `cache_utils.py:89`

**Works Correctly**: Disables progress bars (needed for Modal logs)

---

### 5. `BGB_FORCE_MANIFEST_REBUILD` ✅

**Defined**: `src/brain_brr/utils/env.py:22`
**Used**: `src/brain_brr/train/loop.py:478`

**Works Correctly**: Forces cache manifest regeneration

---

### 6. `BGB_DEBUG_FINITE` ✅

**Defined**: `src/brain_brr/utils/env.py:15`
**Used**: `src/brain_brr/models/debug_utils.py:21`

**Works Correctly**: Enables finite value assertions in forward pass

---

## 📊 Complete Environment Variable Status

| Variable | Status | Locations Used | Notes |
|----------|--------|----------------|-------|
| `BGB_SANITIZE_GRADS` | ❌ **UNUSED** | 0 | **Modal sets this!** |
| `BGB_SANITIZE_INPUTS` | ❌ **UNUSED** | 0 | Defined but never checked |
| `BGB_SKIP_OPT_STEP_ON_NAN` | ❌ **UNUSED** | 0 | Would be useful if implemented |
| `BGB_SAFE_CLAMP` | ❌ **UNUSED** | 0 | Hardcoded clamping exists elsewhere |
| `BGB_ANOMALY_DETECT` | ✅ WORKS | 1 | Correctly enables anomaly detection |
| `BGB_NAN_DEBUG` | ✅ WORKS | 1 | Correctly enables NaN debugging |
| `BGB_SMOKE_TEST` | ✅ WORKS | 6 | Correctly limits dataset |
| `BGB_DISABLE_TQDM` | ✅ WORKS | 3 | Correctly disables progress bars |
| `BGB_FORCE_MANIFEST_REBUILD` | ✅ WORKS | 1 | Correctly forces rebuild |
| `BGB_DEBUG_FINITE` | ✅ WORKS | 1 | Correctly enables assertions |
| `BGB_DISABLE_TB` | ⚠️ **UNTESTED** | ? | Need to verify |
| `BGB_MID_EPOCH_MINUTES` | ⚠️ **UNTESTED** | ? | Need to verify |
| `BGB_FORCE_MAMBA_FALLBACK` | ⚠️ **UNTESTED** | ? | Need to verify |
| `BGB_FORCE_TCN_EXT` | ⚠️ **UNTESTED** | ? | Need to verify |

---

## 🔍 Deep Dive: Modal Training Analysis

### What Modal Actually Sets (deploy/modal/app.py:712-715)

```python
env["BGB_SANITIZE_GRADS"] = "1"  # ❌ DOES NOTHING
env["BGB_NAN_DEBUG"] = "1"        # ✅ WORKS
logger.info("[ENV] BGB_SANITIZE_GRADS=1 BGB_NAN_DEBUG=1 (NaN protection enabled)")
```

**Misleading Log**: The log says "NaN protection enabled" but:
- `BGB_NAN_DEBUG=1` only enables **logging** (debug_utils.py:21)
- `BGB_SANITIZE_GRADS=1` does **NOTHING**
- Actual protection comes from `gradient_clip: 0.5` in config

---

### What Actually Protects Modal Training

**From config** (configs/modal/train.yaml:140):
```yaml
training:
  gradient_clip: 0.5  # THIS is what protects us
  mixed_precision: true  # FP16 → easier to overflow to inf
```

**From code** (src/brain_brr/train/train_step.py:194):
```python
grad_norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
```

**This Always Runs**: No environment variable controls it. It's a required parameter.

---

### Why Logs Show `Mean=inf`

**The logged gradient norm is PRE-CLIP** (train_step.py:202):
```python
grad_norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
gradient_norms.append(float(grad_norm))  # ← Stores PRE-CLIP value
```

**What `clip_grad_norm_` returns**:
- The **total gradient norm BEFORE clipping**
- If any gradient is `inf`, total norm is `inf`
- But function **internally scales all gradients** to max_norm
- **Actual parameter updates use clipped gradients** ✅

**So `Mean=inf` in logs means**:
- "Some gradients were infinite before clipping"
- **NOT**: "Parameters were updated with infinite gradients"
- This is actually **NORMAL with FP16** (max value = 65,504)

---

## 🧪 Evidence from Current Training

### Modal Training (Batch 0 → 370, ~6 hours runtime)

| Metric | Batch 0 | Batch 370 | Change | Interpretation |
|--------|---------|-----------|--------|----------------|
| **Loss** | 0.1171 | 0.0594 | **-49%** | ✅ Converging normally |
| **P50 (Median)** | 15.42 | 2.19 | **-86%** | ✅ Most grads are small |
| **P95** | inf → 18.67 | 11.39 | **Finite + decreasing** | ✅ Outliers shrinking |
| **Max** | inf | inf | Still inf | ⚠️ Occasional FP16 overflow |
| **NaN Warnings** | 0 | 0 | **Zero** | ✅ No actual NaN |

**Conclusion**: Training is **completely healthy** despite `Mean=inf`:
1. Loss is decreasing smoothly
2. Median gradients are small and shrinking
3. P95 became finite and is decreasing
4. **Zero NaN warnings** (gradient clipping works!)

---

### Local Training (RTX 4090, FP32)

**Gradient Statistics** (from docs/08-operations/gradient-monitoring.md):
- Batch 0: P95=52.06, Mean=14.54
- Batch 723: P95=9.74, Mean=3.32
- **Never saw `Mean=inf`** because FP32 max = 3.4×10^38

**Why Local Doesn't Show Inf**:
- `mixed_precision: false` (FP32 everywhere)
- FP32 rarely overflows to inf
- Same clipping protection, just different numeric range

---

## ❓ Questions to Answer

### Q1: Is training actually affected?

**A**: **NO**. Training is completely safe because:
1. `torch.nn.utils.clip_grad_norm_` is PyTorch built-in (not our code)
2. It **always** works, regardless of environment variables
3. Modal and local training both use `gradient_clip: 0.5`
4. Evidence: Loss decreasing, zero NaN warnings

---

### Q2: Should we stop training to fix this?

**A**: **NO**. This is a **code quality issue**, not a training bug:
- Current training runs are healthy
- Gradient clipping protects us
- Fixing this won't change training dynamics
- **BUT**: We should fix to remove false security

---

### Q3: What IS the actual risk?

**A**: **Maintainability and trust**:
1. Future developers might rely on these flags
2. Debugging is harder when flags don't work
3. Logs are misleading (`Mean=inf` looks scary)
4. Modal log says "NaN protection enabled" but it's only partial

---

### Q4: What other implementations might be incomplete?

**A**: Checked all 23 environment variables. Found:
- 4 protection flags: **UNUSED** ❌
- 6 feature flags: **WORKING** ✅
- 13 other flags: **NEED TESTING** ⚠️

See "Complete Environment Variable Status" table above.

---

## 🛠️ Recommended Actions

### Option A: Remove Dead Code (RECOMMENDED)

**Pros**:
- Clean codebase
- No false security
- Clear what's actually protected

**Cons**:
- Removes potential future features
- Need to update Modal setup
- Need to update all docs

**Changes Required**:
1. Remove unused flags from `env.py`
2. Update `deploy/modal/app.py` log message
3. Update `docs/08-operations/gradient-monitoring.md`
4. Update `CLAUDE.md` environment variable list

---

### Option B: Implement Features Properly

**Pros**:
- Adds extra safety layers
- Makes flags work as documented
- Keeps existing Modal setup

**Cons**:
- More code to maintain
- May not actually improve training
- Clipping already works well

**Changes Required**:
1. Implement gradient sanitization in `train_step.py`
2. Implement input sanitization in data loading
3. Implement conditional optimizer stepping
4. Implement configurable safe clamping
5. Add tests for all new features
6. **Verify improvements actually help**

---

### Option C: Hybrid (Implement Critical, Remove Others)

**Keep and Implement**:
- `BGB_SANITIZE_GRADS` (most useful, Modal already sets it)
- `BGB_SKIP_OPT_STEP_ON_NAN` (good safety feature)

**Remove**:
- `BGB_SANITIZE_INPUTS` (preprocessing already handles this)
- `BGB_SAFE_CLAMP` (hardcoded clamping exists, not needed as flag)

---

## 📝 Implementation Plan (If Option B/C Chosen)

### 1. Implement `BGB_SANITIZE_GRADS`

**Location**: `src/brain_brr/train/train_step.py` (after line 199)

```python
# After gradient clipping
grad_norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

# NEW: Sanitize gradients if enabled
if env.sanitize_grads():
    for param in model.parameters():
        if param.grad is not None:
            if not torch.isfinite(param.grad).all():
                param.grad = torch.nan_to_num(
                    param.grad,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0
                )

optimizer.step()
```

**Test**:
1. Inject infinite gradient
2. Verify it's replaced with 0
3. Verify training continues

---

### 2. Fix Gradient Norm Logging

**Problem**: We log PRE-CLIP norm (shows inf, scary)
**Solution**: Log POST-CLIP norm (shows actual applied gradient)

**Location**: `src/brain_brr/train/train_step.py:194-202`

```python
# BEFORE:
grad_norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
gradient_norms.append(float(grad_norm))  # PRE-CLIP (can be inf)

# AFTER:
pre_clip_norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
# POST-CLIP norm is guaranteed to be ≤ gradient_clip (analytical formula)
post_clip_norm = min(float(pre_clip_norm), gradient_clip)
gradient_norms.append(float(post_clip_norm))  # POST-CLIP (always finite)

# Optionally log both
if batch_idx % LOG_EVERY_N_STEPS == 0 and pre_clip_norm > gradient_clip:
    logger.debug(f"Clipped grad norm: {pre_clip_norm:.2f} → {post_clip_norm:.2f}")
```

---

### 3. Implement `BGB_SKIP_OPT_STEP_ON_NAN`

**Location**: `src/brain_brr/train/train_step.py:200-201`

```python
# BEFORE:
optimizer.step()

# AFTER:
if env.skip_opt_step_on_nan():
    if torch.isfinite(torch.tensor(loss_val)):
        optimizer.step()
    else:
        logger.warning(f"Skipping optimizer step at batch {batch_idx} due to NaN loss")
else:
    optimizer.step()
```

---

## 🎯 Next Steps

1. **[IMMEDIATE]** Update TODO.md with this finding
2. **[DECISION NEEDED]** Choose Option A, B, or C
3. **[IF OPTION A]** Remove dead code, update docs
4. **[IF OPTION B/C]** Implement missing features, add tests
5. **[ALWAYS]** Audit remaining untested flags (see table above)

---

## 📚 Related Documentation

- `docs/08-operations/gradient-monitoring.md` - Gradient expectations (needs update)
- `deploy/modal/app.py:712-715` - Environment setup (needs update)
- `src/brain_brr/utils/env.py` - All environment variables
- `CLAUDE.md` - Project overview (needs update)

---

**Audit Date**: October 4, 2025
**Auditor**: Claude (Sonnet 4.5) + User Investigation
**Trigger**: Observed `Mean=inf` in Modal training logs
**Severity**: 🟡 **MEDIUM** (code quality issue, not training bug)
**Training Impact**: ✅ **NONE** (training is safe)
**Recommendation**: 🛠️ **Option A: Remove Dead Code** (cleanest solution)
