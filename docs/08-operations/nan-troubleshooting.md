# Comprehensive NaN Troubleshooting Guide

## Current Status (September 26, 2025)
**CRITICAL UPDATE**: New root causes identified and fixed:
1. **Data Outliers**: Raw EEG had extreme outliers (>100σ) causing overflow
2. **Missing Output Sanitization**: Detection head wasn't clamping final logits
3. **TCN Gradient Instability**: Gradients explode after ~30 batches

**FIXES APPLIED** (Commits: `57426ea`, `7ba8017`, `c0578f4`):
- Added outlier clipping in preprocessing (`np.clip(x, -10.0, 10.0)`)
- Added output sanitization in detector (Tier 3 clamping)
- Recommend `BGB_SANITIZE_GRADS=1` for training stability

**ACTION REQUIRED**: Must rebuild cache after preprocessing fix!

## Historical NaN Issues & Resolutions

### 1. Dynamic PE Eigendecomposition (PRIMARY)
**Symptoms**: NaN explosion at batch 7-28 consistently
**Root Cause**: Unstable eigendecomposition on ill-conditioned Laplacian matrices
**Resolution**:
- Increased regularization (1e-6 → 1e-4)
- Added condition number checking
- Implemented cached PE fallback
- Fixed eigenvector sign ambiguity

### 2. Learning Rate Near-Zero Bug
**Symptoms**: LR showing 6.62e-09 during warmup
**Root Cause**: Warmup ratio too high (0.10) with low base LR (1e-5)
**Resolution**:
- Increased base LR to 1e-4
- Reduced warmup ratio to 0.01
- Verified proper LR schedule (6.62e-07 → 1e-4)

### 3. Edge Feature Explosion
**Symptoms**: Edge stream values exploding in dual-stream V3
**Root Cause**: 1→16 dimension projection without bounds
**Resolution**:
- Added conservative initialization (gain=0.1)
- Implemented edge clamping [-3.0, 3.0]
- Improved numerical stability in cosine similarity

### 4. TCN Input Validation
**Symptoms**: NaN propagation from input layer
**Root Cause**: No input validation or clamping
**Resolution**:
- Added NaN/Inf detection and replacement
- Input clamping [-10, 10]
- Optional intermediate clamping via `BGB_SAFE_CLAMP`

### 5. Mamba State Accumulation
**Symptoms**: Gradual value explosion in Mamba layers
**Root Cause**: SSM states accumulating without bounds
**Resolution**:
- Added input/output clamping
- Intermediate checks every 2 layers
- Conservative weight initialization

## Comprehensive Fix Locations

### Model Components
```
src/brain_brr/models/gnn_pyg.py:170-220     # Dynamic PE safeguards
src/brain_brr/models/detector.py:200-220    # Edge features & clamping
src/brain_brr/models/tcn.py:196-223         # TCN input validation
src/brain_brr/models/mamba.py:128-170       # Mamba state management
src/brain_brr/models/edge_features.py:70-91 # Numerical stability
```

### Training Loop
```
src/brain_brr/train/loop.py:280-320         # Optimizer parameter groups
src/brain_brr/train/loop.py:520-580         # Gradient sanitization
```

### Debug Utilities
```
src/brain_brr/models/debug_utils.py         # Enhanced NaN detection
```

## Environment Variables for Debugging

### Always Safe
- `BGB_NAN_DEBUG=1` - Enable detailed NaN reporting
- `BGB_DEBUG_FINITE=1` - Check all tensors for finite values
- `BGB_ANOMALY_DETECT=1` - PyTorch anomaly detection

### Use During Investigation Only
- `BGB_SAFE_CLAMP=1` - Aggressive activation clamping
- `BGB_SANITIZE_INPUTS=1` - Replace NaN/Inf in inputs
- `BGB_SANITIZE_GRADS=1` - Clean gradients before optimizer step
- `BGB_SKIP_OPT_STEP_ON_NAN=1` - Skip updates on NaN detection
- `SEIZURE_MAMBA_FORCE_FALLBACK=1` - Force Conv1d instead of CUDA kernels

## Quick Diagnosis Flowchart

```
NaN Detected?
├─> Check batch number
│   └─> Consistent (e.g., batch 7)?
│       └─> Likely Dynamic PE → Check eigendecomposition
├─> Check learning rate
│   └─> Near zero (< 1e-8)?
│       └─> Warmup issue → Adjust warmup_ratio
├─> Check loss type
│   └─> Focal loss?
│       └─> Check pos_weight and class balance
└─> Check GPU
    └─> RTX 4090?
        └─> Try SEIZURE_MAMBA_FORCE_FALLBACK=1
```

## Validation Checklist

Before declaring NaN issues resolved:

- [ ] Train for 500+ batches without NaN warnings
- [ ] Learning rate follows expected schedule
- [ ] Loss decreases normally (not stuck)
- [ ] No gradient sanitization warnings after batch 50
- [ ] Can disable all BGB_* safeguards after batch 100
- [ ] Tests pass (some gradient magnitude tests may fail due to conservative init)

## Prevention Best Practices

1. **Always use parameter groups**: Separate weight decay for different param types
2. **Conservative initialization**: Use gain=0.01-0.2 for deep networks
3. **Input validation**: Check and clamp inputs at entry points
4. **Gradient clipping**: Use 0.1-0.5 for aggressive clipping
5. **Regular monitoring**: Log key statistics every 100 batches

## Links to Related Documentation

- [Dynamic PE Details](nan-logits-dynamic-pe.md)
- [V3 Architecture Resolution](v3-nan-explosion-resolution.md)
- [Performance Optimization](performance-optimization.md)
- [Model Architecture](../04-model/architecture-v3.md)
- [Training Configuration](../05-training/training.md)
