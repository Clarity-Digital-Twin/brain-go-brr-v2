# NaN Protection Reference - SINGLE SOURCE OF TRUTH

**Last Updated**: 2025-09-30 (POST dependency upgrade PyTorch 2.5.0)
**Status**: 🚨 **CRITICAL** - Required reading before training
**Version**: v3.3.0 (PyTorch 2.5.0 + mamba-ssm 2.2.5)

---

## 🚨 CRITICAL: What Happened

**Current training WITHOUT proper NaN protection flags → constant NaN warnings**

**Root cause**: All NaN protection flags default to DISABLED in `src/brain_brr/utils/env.py`.
NAN_CANONICAL.md recommends enabling them, but configs don't set them automatically.

**Result**: Training with:
- ❌ `BGB_DEBUG_FINITE=0` - `assert_finite()` checks bypassed
- ❌ `BGB_SANITIZE_GRADS=0` - Gradient explosion unchecked
- ❌ `BGB_NAN_DEBUG=0` - No NaN origin tracking

**This document is the AUTHORITATIVE reference for ALL NaN protection mechanisms.**

---

## Table of Contents

1. [Environment Variables Reference](#environment-variables-reference)
2. [Required Settings by Environment](#required-settings-by-environment)
3. [How NaN Protection Works](#how-nan-protection-works)
4. [Immediate Fix for Current Training](#immediate-fix-for-current-training)
5. [Long-term Configuration](#long-term-configuration)

---

## Environment Variables Reference

### Core NaN Detection & Debugging

| Variable | Default | Purpose | When to Enable | Performance Impact |
|----------|---------|---------|----------------|-------------------|
| `BGB_NAN_DEBUG` | **0** | Enable NaN debugging output | Investigation, smoke tests | Low (logging only) |
| `BGB_NAN_DEBUG_MAX` | **10** | Max NaN warnings before stopping | Always (if NAN_DEBUG=1) | None |
| `BGB_DEBUG_FINITE` | **0** | Enable `assert_finite()` checks | Investigation, smoke tests | **Medium** (9 checks per forward) |
| `BGB_ANOMALY_DETECT` | **0** | PyTorch anomaly detection | Investigation only | **HIGH** (very slow) |

**CRITICAL**: `BGB_DEBUG_FINITE=0` means ALL `assert_finite()` calls are NO-OPs!

**Code**: `src/brain_brr/models/debug_utils.py:10-24`
```python
DEBUG_FINITE = env.debug_finite() or env.smoke_test() or env.nan_debug()

def assert_finite(tag: str, x: torch.Tensor, raise_on_fail: bool = True) -> bool:
    if not DEBUG_FINITE:
        return True  # ❌ SKIPS ALL CHECKS!
```

### Gradient Protection (MOST IMPORTANT)

| Variable | Default | Purpose | When to Enable | Performance Impact |
|----------|---------|---------|----------------|-------------------|
| `BGB_SANITIZE_GRADS` | **0** | Replace NaN gradients with zeros | **ALWAYS (production)** | Low |
| `BGB_SKIP_OPT_STEP_ON_NAN` | **0** | Skip optimizer when NaN detected | Debug/investigation | None |

**🚨 CRITICAL**: `BGB_SANITIZE_GRADS=1` is **REQUIRED** for PyTorch 2.5.0 training.

**Reason**: Prevents gradient explosion → NaN activations in next batch.

**Code**: `src/brain_brr/train/loop.py:728-739`
```python
if env.sanitize_grads():
    grad_has_nan = False
    for param in model.parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            grad_has_nan = True
            param.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    if grad_has_nan:
        print(f"[WARN] Sanitized NaN gradients at batch {batch_idx}")
```

### Input Sanitization

| Variable | Default | Purpose | When to Enable | Performance Impact |
|----------|---------|---------|----------------|-------------------|
| `BGB_SANITIZE_INPUTS` | **0** | Replace NaN/Inf in batch inputs | Debug/testing | Low |

**NOTE**: Redundant with preprocessing `np.nan_to_num()` in `data/preprocess.py:71`.

### Activation Safety Clamps

| Variable | Default | Purpose | When to Enable | Performance Impact |
|----------|---------|---------|----------------|-------------------|
| `BGB_SAFE_CLAMP` | **0** | Enable extra activation clamps | Debug/investigation | **Medium** |
| `BGB_SAFE_CLAMP_MIN` | **-10.0** | Min clamp value | If SAFE_CLAMP=1 | None |
| `BGB_SAFE_CLAMP_MAX` | **10.0** | Max clamp value | If SAFE_CLAMP=1 | None |

**NOTE**: PR1-PR5 refactor makes these unnecessary. Only use for debugging.

### Model Fallbacks

| Variable | Default | Purpose | When to Enable | Performance Impact |
|----------|---------|---------|----------------|-------------------|
| `SEIZURE_MAMBA_FORCE_FALLBACK` | **0** | Force Conv1d fallback (no CUDA) | CUDA kernel issues | **HIGH** (much slower) |
| `BGB_FORCE_TCN_EXT` | **0** | Use TCNExt instead of TCN | Never (deprecated) | N/A |

---

## Required Settings by Environment

### Production Training (RTX 4090 / A100)

**REQUIRED** (must set):
```bash
export BGB_SANITIZE_GRADS=1  # ✅ CRITICAL - prevents gradient explosion
```

**RECOMMENDED** (for monitoring):
```bash
export BGB_NAN_DEBUG=1       # ✅ Shows NaN count/location if it occurs
```

**NOT NEEDED** (unless investigating):
```bash
# BGB_DEBUG_FINITE=0         # ❌ Leave disabled (performance cost)
# BGB_SANITIZE_INPUTS=0      # ❌ Redundant with preprocessing
# BGB_SAFE_CLAMP=0           # ❌ Redundant with PR1-PR5 refactor
# BGB_ANOMALY_DETECT=0       # ❌ Too slow for production
```

### Investigation / Debugging

**Full diagnostics**:
```bash
export BGB_DEBUG_FINITE=1      # Enable assert_finite() checks
export BGB_NAN_DEBUG=1         # Enable NaN warnings
export BGB_SANITIZE_GRADS=1    # Sanitize gradients
export BGB_SANITIZE_INPUTS=1   # Sanitize inputs (extra safety)
export BGB_ANOMALY_DETECT=1    # PyTorch anomaly detection (SLOW!)
```

**Performance monitoring**:
```bash
export BGB_DEBUG_FINITE=1      # Check where NaNs originate
export BGB_NAN_DEBUG=1         # Count NaN occurrences
export BGB_SANITIZE_GRADS=1    # Prevent training corruption
# Skip ANOMALY_DETECT (too slow)
```

### Smoke Tests (Quick Validation)

```bash
export BGB_SMOKE_TEST=1        # Limit to 3 files (auto-enables DEBUG_FINITE)
export BGB_NAN_DEBUG=1         # Show any NaN warnings
export BGB_SANITIZE_GRADS=1    # Production-like sanitization
```

**NOTE**: `BGB_SMOKE_TEST=1` automatically enables `DEBUG_FINITE` checks!

---

## How NaN Protection Works

### 3-Layer Defense System

**Layer 1: Data Preprocessing** (ALWAYS ACTIVE)
- **File**: `src/brain_brr/data/preprocess.py:68,71`
- **Actions**:
  - Clip outliers to ±10σ after z-score normalization
  - Replace NaN/Inf with zeros (`np.nan_to_num`)
- **Purpose**: Prevent bad data from entering pipeline

**Layer 2: Model Boundaries** (ALWAYS ACTIVE)
- **Files**: `tcn.py:236-241`, `mamba.py:177-180,329-335`, `edge_features.py:72-75`
- **Actions**:
  - Check for NaN/Inf at component inputs
  - Replace with zeros + warn
  - Clamp to safe ranges
- **Purpose**: Catch NaNs before they propagate

**Layer 3: Gradient Sanitization** (OPTIONAL, **RECOMMENDED**)
- **File**: `src/brain_brr/train/loop.py:728-739`
- **Actions**:
  - Check gradients after `loss.backward()`
  - Replace NaN/Inf with zeros
  - Warn user
- **Purpose**: Prevent gradient explosion → NaN activations

### PR1-PR5 Architectural Fixes (ALWAYS ACTIVE)

**These are ALWAYS enabled via config** (`configs/local/train.yaml:59`):

1. **PR-1: Boundary Normalization**
   - LayerNorm at 5 critical boundaries
   - LayerScale for residual connections
   - Prevents unbounded activation growth

2. **PR-2: Bounded Edge Stream**
   - Tanh activation in edge projection
   - LayerNorm after edge Mamba
   - Prevents edge feature explosion

3. **PR-3: Adjacency Conditioning**
   - Row-softmax for probability distribution
   - EMA smoothing for stability
   - Regularized Laplacian (eps=1e-3)

4. **PR-4: Fusion & Monitoring**
   - Multi-head gated fusion
   - Conservative weight initialization

5. **PR-5: Edge Similarity Clamping**
   - Safety margin from ±1 boundaries: `edge_similarity_margin: 0.01`
   - Prevents cosine similarity explosion
   - Applied at source in `edge_features.py:89,99`

**These fixes eliminate NaN at SOURCE, not just symptoms.**

---

## Immediate Fix for Current Training

### Option 1: Restart with Proper Flags (RECOMMENDED)

```bash
# Stop current training
tmux send-keys -t train C-c

# Restart with proper protection
tmux send-keys -t train "export BGB_SANITIZE_GRADS=1 BGB_NAN_DEBUG=1 && .venv/bin/python -m src train configs/local/train.yaml" Enter

# Detach
# Ctrl+B then D
```

**Expected**: NaN warnings should drastically reduce or stop.

### Option 2: Debug First (if unsure)

```bash
# Stop current training
tmux send-keys -t train C-c

# Run smoke test with full diagnostics
export BGB_DEBUG_FINITE=1 BGB_NAN_DEBUG=1 BGB_SANITIZE_GRADS=1
.venv/bin/python -m src train configs/local/smoke.yaml
```

**This will show EXACTLY where NaNs originate** (TCN? Edges? GNN?).

---

## Long-term Configuration

### Update Makefile Commands

**File**: `Makefile`

**Add production-safe training target**:
```makefile
.PHONY: train-local-safe
train-local-safe:  ## Full training with NaN protection (RECOMMENDED)
	export BGB_SANITIZE_GRADS=1 BGB_NAN_DEBUG=1 && \
	$(PYTHON) -m src train configs/local/train.yaml

.PHONY: train-modal-safe
train-modal-safe:  ## Modal training with NaN protection
	modal run --detach deploy/modal/app.py --action train \
	  --config configs/modal/train.yaml \
	  --env BGB_SANITIZE_GRADS=1 \
	  --env BGB_NAN_DEBUG=1
```

### Update Config Documentation

**File**: `configs/local/train.yaml` (add comment block)
```yaml
# NaN Protection (CRITICAL for PyTorch 2.5.0):
# export BGB_SANITIZE_GRADS=1  # REQUIRED - prevents gradient explosion
# export BGB_NAN_DEBUG=1       # RECOMMENDED - shows NaN warnings
# See: NAN-PROTECTION-REFERENCE.md
```

### Update CLAUDE.md Quick Commands

**File**: `CLAUDE.md`
```markdown
### Local Training (RTX 4090)
```bash
# Smoke test (quick validation)
export BGB_SANITIZE_GRADS=1 && make s

# Full training in tmux (PRODUCTION)
tmux new -s train
export BGB_SANITIZE_GRADS=1 BGB_NAN_DEBUG=1
make train-local
# Detach: Ctrl+B then D
```
```

---

## Why This Happened

### Root Causes

1. **Environment variables default to DISABLED**
   - All `BGB_*` flags in `env.py` default to "0"
   - No automatic enablement for production

2. **NAN_CANONICAL.md buried in docs**
   - Recommends `BGB_SANITIZE_GRADS=1` but not prominent
   - No enforcement mechanism

3. **Configs don't set flags**
   - YAML configs don't export environment variables
   - User must remember to set manually

4. **Dependency upgrade risk underestimated**
   - PyTorch 2.5.0 may have different numerical behavior
   - No regression testing after upgrade

### Lessons Learned

1. **Production-critical flags should be DEFAULT**
   - Consider changing `_SANITIZE_GRADS` default to "1"
   - Or add config-level enablement

2. **Root documentation needed**
   - NAN_CANONICAL.md is comprehensive but buried
   - Need SINGLE prominent reference

3. **Makefiles should include protection**
   - Production training commands should set flags automatically
   - Explicit `make train-unsafe` if user wants to disable

4. **Dependency upgrades need validation**
   - Should run smoke test with diagnostics BEFORE full training
   - Verify no regression in numerical behavior

---

## Decision Matrix

### When to Enable Each Flag

| Scenario | SANITIZE_GRADS | NAN_DEBUG | DEBUG_FINITE | SANITIZE_INPUTS | ANOMALY_DETECT |
|----------|----------------|-----------|--------------|-----------------|----------------|
| **Production Training** | ✅ YES | ✅ YES | ❌ NO | ❌ NO | ❌ NO |
| **Smoke Test** | ✅ YES | ✅ YES | ✅ YES (auto) | ❌ NO | ❌ NO |
| **Debugging NaNs** | ✅ YES | ✅ YES | ✅ YES | ✅ YES | ⚠️ MAYBE |
| **Performance Testing** | ❌ NO | ❌ NO | ❌ NO | ❌ NO | ❌ NO |
| **CI/CD Tests** | ❌ NO | ❌ NO | ✅ YES | ❌ NO | ❌ NO |

**Legend**:
- ✅ YES - Always enable
- ❌ NO - Leave disabled
- ⚠️ MAYBE - Only if needed (slow!)
- ✅ YES (auto) - Automatically enabled by smoke test

---

## Complete Reference: All NaN-Related Code

### Data Preprocessing

**File**: `src/brain_brr/data/preprocess.py`
- Line 68: Outlier clipping to ±10σ (ALWAYS ACTIVE)
- Line 71: `np.nan_to_num()` sanitization (ALWAYS ACTIVE)

### Model Input Boundaries

**File**: `src/brain_brr/models/tcn.py`
- Lines 236-241: Input sanitization + clamp [-10, 10]

**File**: `src/brain_brr/models/mamba.py`
- Lines 177-180: BiMamba2Layer input sanitization
- Lines 329-335: BiMamba2 input sanitization
- Lines 249,259,339,342: Output clamping

**File**: `src/brain_brr/models/edge_features.py`
- Lines 72-75: Input sanitization before cosine similarity
- Lines 81,89: Norm clamping (min=1e-6)
- Lines 89,99: Similarity clamping with margin

### Detector Forward Pass

**File**: `src/brain_brr/models/detector.py`
- Lines 225,247,263,278,282,286,290,309,312: `assert_finite()` checks (9 total)
- Lines 385-386,392-393: Output sanitization before loss

### Training Loop

**File**: `src/brain_brr/train/loop.py`
- Lines 566-571: Input sanitization (if `BGB_SANITIZE_INPUTS=1`)
- Lines 585-606: Logit sanitization + bad batch save
- Lines 639-686: Loss NaN handling (consecutive count, terminate after 50)
- Lines 694-709,728-739: Gradient sanitization (if `BGB_SANITIZE_GRADS=1`)

### Focal Loss

**File**: `src/brain_brr/train/loop.py:180-224`
- Line 205: Logit clamping [-100, 100]
- Line 212: Probability clamping [1e-6, 1-1e-6]
- Line 218: p_t stability clamp
- Line 223: Loss explosion prevention (max=100.0)

---

## FAQ

### Q: Why aren't these flags enabled by default?

**A**: Historical design - originally optional debugging tools, but PyTorch 2.5.0 makes gradient sanitization effectively required.

### Q: Will sanitization hurt model performance?

**A**: No evidence of harm. Sanitization prevents training corruption, and clean gradients → better convergence.

### Q: Should I always use `BGB_DEBUG_FINITE=1`?

**A**: No - has performance cost (~5-10% slower due to 9 finite checks per forward). Only for debugging.

### Q: What if I see NaNs even with sanitization?

**A**: Indicates deeper issue. Check:
1. Cache was rebuilt after dependency upgrade?
2. Dynamic PE causing instability? (try `use_dynamic_pe: false`)
3. Config has `edge_similarity_margin: 0.01`?

### Q: How do I know if sanitization is working?

**A**: With `BGB_NAN_DEBUG=1`, you'll see warnings like:
```
[WARN] Sanitized NaN gradients at batch 42
```

If you see this ONCE or occasionally → sanitization working.
If you see this EVERY batch → deeper problem, investigate.

---

## Summary

**CRITICAL SETTINGS FOR PRODUCTION**:
```bash
export BGB_SANITIZE_GRADS=1  # ✅ REQUIRED
export BGB_NAN_DEBUG=1       # ✅ RECOMMENDED
```

**Everything else defaults are fine for production.**

**PR1-PR5 architectural fixes are ALWAYS active via configs** - no environment variables needed.

---

**Status**: 🟢 COMPLETE - This is the authoritative reference for ALL NaN protection in v3.3.0.

**Last Verified**: 2025-09-30 against commit HEAD on `fix/upgrade-mamba` branch.

**Related Docs**:
- `NAN_CANONICAL.md` - Historical implementation reference
- `NAN-REGRESSION-POST-DEPENDENCY-UPGRADE.md` - Current investigation
- `docs/10-major-NAN-refactor/` - PR1-PR5 architectural fixes
