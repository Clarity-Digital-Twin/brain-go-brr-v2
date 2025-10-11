# NaN Regression After Dependency Upgrade - CRITICAL ANALYSIS

**Date**: 2025-09-30
**Status**: 🚨 **ACTIVE INVESTIGATION** - Training shows persistent NaN warnings
**Context**: PyTorch 2.2.2 → 2.5.0 + mamba-ssm 2.2.2 → 2.2.5 upgrade

---

## 🚨 Executive Summary

**Current training (batch 79/15404) shows CONSTANT NaN warnings at Mamba input boundaries**. This is NOT expected behavior after the PR1-PR5 NaN refactor (Sept 26). Investigation reveals:

1. **NaN warnings are persistent** - Every batch, multiple times per batch
2. **`assert_finite()` checks are DISABLED** - NaNs passing silently through pipeline
3. **Critical environment variables NOT set** - Missing `BGB_SANITIZE_GRADS=1`, `BGB_DEBUG_FINITE=1`
4. **Cache built with old PyTorch** - Potential numerical incompatibility
5. **Dependency upgrade introduces risk** - PyTorch 2.5.0 may have different numerical behavior

**This is NOT normal. The NaN refactor was supposed to prevent this.**

---

## Current Situation

### Observed Warnings (batch 79/15404)
```
[2025-09-30 12:56:55] WARNING  Non-finite values in BiMamba input replaced with zeros  mamba.py:332
[2025-09-30 12:56:55] WARNING  Non-finite values in Mamba input replaced with zeros    mamba.py:180
[2025-09-30 12:56:55] WARNING  Non-finite values in Mamba input replaced with zeros    mamba.py:180
[2025-09-30 12:56:55] WARNING  Non-finite values in Mamba input replaced with zeros    mamba.py:180
[2025-09-30 12:56:55] WARNING  Non-finite values in Mamba input replaced with zeros    mamba.py:180
[2025-09-30 12:56:55] WARNING  Non-finite values in Mamba input replaced with zeros    mamba.py:180
[2025-09-30 12:56:55] WARNING  Non-finite values in BiMamba input replaced with zeros  mamba.py:332
[2025-09-30 12:56:55] WARNING  Non-finite values in Mamba input replaced with zeros    mamba.py:180
```

**Frequency**: ~8-10 warnings per batch, EVERY batch
**Training Loss**: 0.0866 (converging from 0.0927) - loss IS converging despite NaNs
**Training Rate**: ~13-17s/batch on RTX 4090

---

## Root Cause Analysis

### 1. `assert_finite()` Checks Are DISABLED

**Code**: `src/brain_brr/models/debug_utils.py:10-24`
```python
DEBUG_FINITE = env.debug_finite() or env.smoke_test() or env.nan_debug()

def assert_finite(tag: str, x: torch.Tensor, raise_on_fail: bool = True) -> bool:
    if not DEBUG_FINITE:
        return True  # ❌ SKIPS ALL CHECKS!
```

**Current State**:
- `BGB_DEBUG_FINITE=0` (not set) ❌
- Not a smoke test (full training) ❌
- `BGB_NAN_DEBUG=0` (not set) ❌

**Impact**: All 9 `assert_finite()` calls in `detector.py` are NO-OPs:
- `assert_finite("tcn_out", features)` - SKIPPED
- `assert_finite("proj_to_electrodes", elec_flat)` - SKIPPED
- `assert_finite("node_mamba", node_processed)` - SKIPPED
- `assert_finite("edge_weights", edge_weights)` - SKIPPED
- `assert_finite("adjacency", adj)` - SKIPPED
- `assert_finite("gnn_out", elec_enhanced)` - SKIPPED
- `assert_finite("backproj", temporal)` - SKIPPED
- `assert_finite("decoder_prelogits", decoded)` - SKIPPED
- `assert_finite("final_logits", output)` - SKIPPED

**Result**: NaN values pass silently through entire pipeline until Mamba input sanitization catches them.

### 2. Mamba Input Sanitization is Last Line of Defense

**Code**: `src/brain_brr/models/mamba.py:177-180, 329-332`
```python
# Check for NaN/Inf and replace with zeros
if torch.isnan(x).any() or torch.isinf(x).any():
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    logger.warning("Non-finite values in Mamba input replaced with zeros")
```

**This is catching NaNs that should have been prevented upstream.**

### 3. Critical Environment Variables NOT Set

**Missing from current training**:
- `BGB_SANITIZE_GRADS=1` - **RECOMMENDED** per NAN_CANONICAL.md:565
  - Would sanitize gradients during backprop
  - Prevents gradient explosion causing NaN activations
- `BGB_DEBUG_FINITE=1` - Would enable `assert_finite()` checks
  - Would show EXACTLY where NaNs originate
  - Would fail fast instead of masking with zeros

**NAN_CANONICAL.md explicitly recommends**:
```bash
# Production run (with gradient sanitization recommended)
export BGB_SANITIZE_GRADS=1  # Recommended for TCN stability
python -m src train configs/local/train.yaml
```

### 4. Cache Built with Old PyTorch

**Timeline**:
- Cache built: Before Sept 30 (PyTorch 2.2.2, mamba-ssm 2.2.2)
- Dependency upgrade: Sept 30 (PyTorch 2.5.0, mamba-ssm 2.2.5)
- Training started: Sept 30 (with NEW PyTorch, OLD cache)

**Risk**: PyTorch 2.5.0 may have different:
- Float precision handling
- CUDA kernel implementations
- Numerical stability characteristics

**Cache preprocessing** (`data/preprocess.py:68`):
```python
# Clip outliers to prevent infinities during training
x = np.clip(x, -10.0, 10.0)  # ✅ This was added Sept 26
```

**Question**: Did old cache files skip this clipping?

---

## What the PR1-PR5 Refactor Fixed

### Before PR1-PR5 (Sept 26)
1. **Unbounded edge similarity** - Could reach exactly ±1.0, causing tanh(inf)
2. **No boundary normalization** - Activations could explode
3. **Missing output sanitization** - Detector output not clamped
4. **Data preprocessing gaps** - Extreme outliers (>100σ) not clipped

### After PR1-PR5 (Sept 26)
1. ✅ **Edge similarity clamping** - `edge_similarity_margin: 0.01` prevents ±1.0
2. ✅ **Boundary normalization** - LayerNorm at 5 critical boundaries
3. ✅ **Output sanitization** - Detector output clamped [-100, 100]
4. ✅ **Data preprocessing** - Outliers clipped to ±10σ

### Config Verification
```yaml
# configs/local/train.yaml:59
graph:
  edge_similarity_margin: 0.01  # ✅ PR-5: Safety margin from ±1 boundaries
```

**All PR1-PR5 fixes are ACTIVE and working correctly.**

---

## Why Are We Still Seeing NaNs?

### Hypothesis 1: Gradient Explosion (MOST LIKELY)
**Evidence**:
- NAN_CANONICAL.md recommends `BGB_SANITIZE_GRADS=1` for "TCN stability"
- Training WITHOUT this flag
- NaNs appear at Mamba INPUT (after forward pass from upstream)
- Gradients from previous batch may be causing NaN activations

**Mechanism**:
1. Batch N: Forward pass OK, loss computed
2. Batch N: Backward pass with exploding gradients
3. Batch N: Optimizer updates weights with NaN gradients
4. Batch N+1: Forward pass with corrupted weights → NaN activations
5. Mamba input sanitization catches and replaces with zeros

**Test**: Enable `BGB_SANITIZE_GRADS=1` and see if warnings stop.

### Hypothesis 2: PyTorch 2.5.0 Numerical Regression
**Evidence**:
- PR1-PR5 refactor done with PyTorch 2.2.2 (Sept 26)
- Training with PyTorch 2.5.0 (Sept 30)
- No NaN issues during PR1-PR5 testing (Sept 26-27)
- NaN warnings appear immediately with 2.5.0

**Mechanism**:
PyTorch 2.5.0 may have:
- Different CUDA kernel implementations
- Changed precision handling in certain operations
- New optimizations that introduce instability

**Test**: Downgrade to PyTorch 2.2.2 and see if warnings stop (NOT recommended due to XID 31 fix).

### Hypothesis 3: Cache Incompatibility
**Evidence**:
- Cache built with PyTorch 2.2.2
- Training with PyTorch 2.5.0
- Data preprocessing updated Sept 26 (added ±10σ clipping)
- Unclear if cache was rebuilt after preprocessing fix

**Mechanism**:
- Old cache may contain extreme outliers
- PyTorch 2.5.0 processes them differently
- Causes numerical instability in TCN/projection layers

**Test**: Rebuild cache with current code and PyTorch 2.5.0.

### Hypothesis 4: False Positive Warnings (UNLIKELY)
**Evidence**:
- Loss is converging (0.0927 → 0.0866)
- Training is stable (not crashing)
- Warnings are consistent, not escalating

**Mechanism**:
- Mamba input sanitization may be overly sensitive
- Small subnormal values detected as "non-finite"
- Replacement with zeros is inconsequential

**Test**: Add detailed logging to see NaN magnitudes and locations.

---

## Is This Acceptable?

### 🔴 NO - This is NOT acceptable behavior

**Reasons**:
1. **Masking root cause** - Replacing NaNs with zeros hides the actual problem
2. **Loss of information** - Zero-filled tensors reduce model capacity
3. **Indicates instability** - Persistent NaNs suggest architectural issues
4. **Contradicts refactor** - PR1-PR5 was supposed to eliminate NaNs at source
5. **Training integrity** - Model is learning on corrupted data

### What "Normal" Looks Like

**After successful NaN refactor** (Sept 26-27 testing):
- Zero NaN warnings for thousands of batches
- `assert_finite()` checks pass (when enabled)
- Clean forward/backward passes
- No gradient sanitization needed

**Current state** (Sept 30):
- 8-10 NaN warnings PER BATCH
- `assert_finite()` checks bypassed (disabled)
- Gradient sanitization NOT enabled (recommended but missing)
- NaNs caught only at last line of defense

---

## Recommended Actions

### IMMEDIATE (Stop Current Training)

**1. Enable Debug Logging**
```bash
# Stop current training (Ctrl+C in tmux)
tmux send-keys -t train C-c

# Restart with full debugging
tmux send-keys -t train "export BGB_DEBUG_FINITE=1 BGB_NAN_DEBUG=1 BGB_SANITIZE_GRADS=1 && .venv/bin/python -m src train configs/local/train.yaml" Enter
```

**Why**: Will show EXACTLY where NaNs originate (TCN? Projection? Edges?)

**2. Check if Gradients Are Exploding**
```bash
# In new terminal, monitor gradient norms
watch -n 1 'grep -E "grad_norm|NaN" <training_log>'
```

**3. Verify Cache Was Rebuilt**
```bash
# Check cache file timestamps
ls -lh cache/tusz/train/*.npz | head -5
# Should be AFTER Sept 26 (preprocessing fix date)
```

### SHORT-TERM (Investigation)

**4. Minimal Reproduction**
```bash
# Run smoke test with full debugging
export BGB_DEBUG_FINITE=1 BGB_NAN_DEBUG=1 BGB_SANITIZE_GRADS=1 BGB_ANOMALY_DETECT=1
.venv/bin/python -m src train configs/local/smoke.yaml
```

**Expected**: Either:
- No NaN warnings (cache issue)
- NaN warnings with location details (can fix)

**5. Test Without Dynamic PE**
```yaml
# configs/local/train.yaml - temporarily disable
model:
  graph:
    use_dynamic_pe: false  # Test without eigendecomposition
```

**Why**: Dynamic PE eigendecomposition can introduce NaNs if:
- Laplacian matrix is ill-conditioned
- Adjacency matrix has disconnected nodes
- PyTorch 2.5.0 changed eigen solver

**6. Rebuild Cache with Current Stack**
```bash
# Force rebuild with PyTorch 2.5.0 + current preprocessing
export BGB_FORCE_MANIFEST_REBUILD=1
rm -rf cache/tusz
python -m src build-cache --data-dir data_ext4/tusz/edf --cache-dir cache/tusz
```

**Why**: Ensures cache uses:
- PyTorch 2.5.0 numerical behavior
- Updated preprocessing (±10σ clipping)
- Current numpy version (1.26.4)

### LONG-TERM (If Unfixable)

**7. Fallback Options**

**Option A: Downgrade PyTorch (NOT RECOMMENDED)**
- Would require giving up PR #708 fix (XID 31)
- Modal A100 training would still crash
- Not a viable solution

**Option B: Accept Sanitization as Design**
- Keep `BGB_SANITIZE_GRADS=1` permanently
- Document that PyTorch 2.5.0 requires gradient sanitization
- Add sanitization as architectural requirement
- Monitor for performance impact

**Option C: Architectural Changes**
- Stronger gradient clipping (0.1 → 0.05)
- More aggressive weight initialization (0.2 → 0.1)
- Additional boundary normalization
- Disable problematic components (Dynamic PE?)

---

## Expected Outcomes

### Best Case (Hypothesis 1 correct)
- Enable `BGB_SANITIZE_GRADS=1`
- NaN warnings stop or drastically reduce
- Training continues normally
- Document as required for PyTorch 2.5.0

### Medium Case (Hypothesis 3 correct)
- Rebuild cache with current stack
- NaN warnings stop
- Training continues normally
- Update cache build documentation

### Worst Case (Hypothesis 2 correct)
- PyTorch 2.5.0 has fundamental regression
- Must either:
  - Accept sanitization as permanent requirement
  - Make architectural changes
  - Find workarounds for specific operations

---

## Key Questions

1. **When was cache last rebuilt?** Check timestamps vs Sept 26 preprocessing fix
2. **Is gradient sanitization sufficient?** Test with `BGB_SANITIZE_GRADS=1`
3. **Where do NaNs originate?** Run with `BGB_DEBUG_FINITE=1`
4. **Is Dynamic PE the culprit?** Test with `use_dynamic_pe: false`
5. **Can we reproduce minimally?** Smoke test with full debugging

---

## Technical Context

### PR1-PR5 NaN Refactor (Sept 26-27)
- **Goal**: Eliminate NaN at SOURCE, not just catch-and-replace
- **Method**: Boundary normalization, edge similarity clamping, output sanitization
- **Testing**: Passed with PyTorch 2.2.2 + mamba-ssm 2.2.2
- **Status**: ✅ Complete, configs updated, documented

### Dependency Upgrade (Sept 30)
- **Change**: PyTorch 2.2.2 → 2.5.0, mamba-ssm 2.2.2 → 2.2.5
- **Reason**: Fix XID 31 MMU Fault on Modal A100 (PR #708)
- **Testing**: Modal smoke test passed (NO XID 31) ✅
- **Risk**: PyTorch 2.5.0 may have different numerical behavior ⚠️

### Current Training
- **Config**: configs/local/train.yaml (batch_size=4, mixed_precision=false)
- **Environment**: RTX 4090, WSL2, PyTorch 2.5.0+cu124
- **Status**: Batch 79/15404, loss converging despite NaN warnings
- **Flags**: No debug flags set ❌

---

## Conclusion

**This situation is CONCERNING but potentially FIXABLE.**

**Most likely cause**: Gradient explosion due to missing `BGB_SANITIZE_GRADS=1` flag.

**Recommended immediate action**:
1. Stop current training
2. Enable `BGB_DEBUG_FINITE=1 BGB_NAN_DEBUG=1 BGB_SANITIZE_GRADS=1`
3. Restart training and observe
4. If warnings persist, investigate with smoke test

**If sanitization is required**, document it as:
- Architectural requirement for PyTorch 2.5.0
- Not a regression, but a dependency characteristic
- Monitor for performance impact

**If sanitization is insufficient**, deeper investigation needed:
- Cache rebuild with current stack
- Dynamic PE testing
- Potential architectural adjustments

---

**Status**: 🚨 ACTIVE - Requires immediate investigation before continuing 100-epoch training.

**Next Update**: After enabling debug flags and observing results.
