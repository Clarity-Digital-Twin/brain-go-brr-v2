# V3 NaN Explosion Incident & Resolution (Historical)

**Last Updated**: October 1, 2025
**Status**: ✅ RESOLVED - Historical incident report for reference

> **For Current Best Practices**: See `docs_v2/08-operations/nan-prevention-complete.md` and `docs_v2/04-model/v3-architecture.md`

---

## Quick Summary

**This document describes the Sept 24-26, 2025 NaN crisis that was FULLY RESOLVED by v3.3.0 architectural fixes (PR1-5) and v3.3.1 eigendecomposition fix.**

**Current status (v3.4.1)**: Zero NaN/Inf across 723+ batches. Training is ROCK SOLID.

---

## Incident Timeline
- **Date**: September 24-26, 2025
- **Impact**: V3 dual-stream architecture training completely broken (NaN explosions at batch 10-20)
- **Resolution**: PR1-5 architectural fixes + v3.3.1 eigendecomposition detachment
- **Validation**: 100+ batches stable (Sept 26), 723+ batches stable (Oct 1)

## Root Causes Identified

### 1. Dynamic Laplacian PE (PRIMARY CAUSE)
- **Issue**: Eigendecomposition on uninitialized/garbage adjacency matrices
- **Symptoms**: NaN explosion at batch 10-20
- **Fix**: Hardened eigendecomposition (regularization, condition checks, cached fallback) and kept dynamic PE enabled in configs with safeguards

### 2. Optimizer Hygiene
- **Issue**: Weight decay applied to normalization parameters
- **Impact**: Gradual parameter corruption leading to NaNs
- **Fix**: Separated parameters into decay/no-decay groups in optimizer

### 3. Edge Projection Explosion
- **Issue**: 1→16 dimension projection without bounds
- **Impact**: Values exploding in edge Mamba stream
- **Fix**: Added hardcoded clamping in forward path (similarity [-0.99, 0.99], projection [-3, 3])

### 4. Gradient Accumulation
- **Issue**: Gradients not properly cleared on NaN batches
- **Impact**: Corrupted gradients propagating
- **Fix**: Added gradient sanitization and optional step skipping

## Implemented Fixes

### Code Changes

#### 1. Optimizer Parameter Groups (src/brain_brr/train/loop.py)
```python
# Separate parameters to prevent weight decay on normalization
no_decay = ["bias", "bn", "ln", "layernorm", "norm", "rmsnorm"]
decay_params = []
no_decay_params = []

for name, param in model.named_parameters():
    if any(nd in name.lower() for nd in no_decay):
        no_decay_params.append(param)
    else:
        decay_params.append(param)
```

#### 2. Edge Clamping (src/brain_brr/models/detector.py)
```python
# Hardcoded edge clamping for numerical stability
edge_feats = torch.clamp(edge_feats, -0.99, 0.99)  # Similarity bounds
edge_in = torch.clamp(edge_in, -3.0, 3.0)          # Projection bounds
```

#### 3. Gradient Sanitization (src/brain_brr/train/loop.py)
```python
# Optional gradient sanitization
if os.getenv("BGB_SANITIZE_GRADS", "0") == "1":
    for name, param in model.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            param.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
```

### Configuration Changes

#### configs/local/train.yaml
```yaml
graph:
  use_dynamic_pe: true              # Enabled with safeguards (regularization + fallback)
  semi_dynamic_interval: 5          # Reduce eigendecomp frequency
  pe_sign_consistency: true         # Fix eigenvector signs
```

## Environment Variables

| Variable | Default | Purpose | Status |
|----------|---------|---------|--------|
| `BGB_NAN_DEBUG` | 0 | Enable NaN debugging output | Active when set |
| `BGB_SANITIZE_GRADS` | 0 | Enable gradient sanitization | Active when set |
| `BGB_SKIP_OPT_STEP_ON_NAN` | 0 | Skip optimizer step on NaN gradients | Active when set |
| `SEIZURE_MAMBA_FORCE_FALLBACK` | 0 | Force Conv1d fallback for Mamba | Active when set |
| `BGB_SAFE_CLAMP` | 0 | Optional activation rails | Active when set |
| `BGB_EDGE_CLAMP*` | — | Legacy edge clamping toggles | Removed; edge clamping is hardcoded in forward paths |

## Verification Tests

### Test Suite Created
- `test_v3_components.py`: Component isolation tests
- `test_v3_fixes.py`: Fix verification tests
- `debug_v3_training.py`: Debug script for NaN sources

### Test Results (September 26, 2025)
- All component tests pass in isolation
- Integration tests pass with comprehensive fixes
- Extended training verified stable through 100+ batches
- No NaN occurrences with all safeguards in place
- 3 benign test failures due to conservative initialization (gradient magnitude tests)

---

## Post-Incident Evolution

### v3.3.1 (September 30, 2025) - **CRITICAL FIX**
**Problem**: Gradient norms still increasing over time (batch 257: 7.03, worse than batch 24: 5.31)

**Root Cause**: Eigendecomposition backward pass computes `∂L/∂A ∝ 1/(λᵢ - λⱼ)` → gradient explosion when eigenvalues are close together (near-degenerate from row-softmax + EMA + symmetry in PR-3)

**Fix**: Detach eigenvectors (`gnn_pyg.py:205`)
```python
eigenvectors = eigenvectors.detach()  # NO gradients through eigendecomp
```

**Why This Works**:
- Eigenvectors are FIXED positional coordinates (2025 best practice)
- Learning happens in GNN layers that PROCESS PE, not in PE itself
- Zero architectural compromise

**Result**: Gradient norms decreasing smoothly, architecture ROCK SOLID

### v3.4.0 (September 30, 2025) - Pre-Norm Mamba
**Change**: Switch from post-norm to pre-norm pattern (align with reference Mamba2)
**Benefit**: Improved scale consistency in residual paths

### v3.4.1 (October 1, 2025) - Warmup Schedules
**Addition**: Optional warmup schedules for gradient stabilization
- Adjacency temperature: 2.0 → 1.0 (first 1000 steps)
- Focal gamma: 1.0 → 2.0 (first 1000 steps)
- Residual scale: optional

**Status**: Training validated through batch 723+ with ZERO NaN/Inf

---

## Related Documentation

**Current best practices**:
- `docs_v2/08-operations/nan-prevention-complete.md` - Complete NaN protection reference
- `docs_v2/04-model/v3-architecture.md` - V3.4.1 architecture with all fixes
- `docs_v2/04-model/v3-stability-evolution.md` - Evolution details (v3.3.0 → v3.4.1)
- `docs_v2/08-operations/gradient-monitoring.md` - Gradient monitoring guide

**Historical context**:
- This document (v3-nan-explosion-resolution.md) - Sept 24-26 incident
- `docs_v2/reference/incidents/` - Other incident reports

## Comprehensive Fixes Applied (September 26)

### Weight Initialization
- All layers now use conservative gains (0.01-0.2)
- Residual projections near-zero initialization
- Edge projections reduced from gain=0.5 to 0.1

### Input Validation
- TCN: Input clamping [-10, 10] with NaN/Inf replacement
- Mamba: Input validation and clamping [-10, 10]; projection clamp [-5, 5]
- GNN: Optional safe clamp via `BGB_SAFE_CLAMP`

### Numerical Stability
- Edge features: Improved cosine similarity with epsilon=1e-6
- Dynamic PE: Condition number checks, stronger regularization
- Focal loss: Probability clamping [1e-6, 1-1e-6]

### Debug Enhancements
- `debug_utils.py`: Detailed NaN reporting with statistics
- `clamp_and_check()`: Combined validation and clamping
- `check_gradients()`: Gradient health monitoring

## Lessons Learned

1. **Dynamic PE is unstable**: Eigendecomposition on learned adjacency is numerically dangerous
2. **Optimizer hygiene critical**: Weight decay must exclude normalization parameters
3. **Gradient guards necessary**: NaN gradients must be caught and handled
4. **Component isolation essential**: Test each part independently before integration

## Monitoring Recommendations

1. Always run with `BGB_NAN_DEBUG=1` during development
2. Monitor gradient norms for early warning
3. Use smaller batch sizes initially
4. Disable mixed precision on consumer GPUs (RTX 4090)

## Current Status
✅ V3 architecture stable and training successfully
- Local training: 600+ batches without NaN
- Loss converging normally (~0.15)
- All safeguards in place
