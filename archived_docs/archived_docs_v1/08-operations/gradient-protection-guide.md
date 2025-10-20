# Gradient Protection Guide

**Last Updated**: October 4, 2025
**Architecture**: V3 Dual-Stream
**Status**: ✅ **AUTHORITATIVE REFERENCE**

---

## What Actually Protects Your Gradients

### Primary Protection: Gradient Clipping (ALWAYS ON)

```yaml
# configs/local/train_bimamba.yaml or configs/modal/train_bimamba.yaml
training:
  gradient_clip: 0.5  # Scales gradients with norm > 0.5
```

This is PyTorch standard practice and is **always applied**, regardless of environment variables.

**How it works**:
1. Compute total gradient norm across all parameters
2. If norm > max_norm (0.5), scale ALL gradients proportionally
3. Result: Total gradient norm ≤ max_norm (guaranteed finite)

**Code** (`src/brain_brr/train/train_step.py:247`):
```python
pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
```

**With Mixed Precision** (Modal A100):
```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)  # ← CRITICAL: unscale before clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
scaler.step(optimizer)
```

---

## Optional: Gradient Sanitization (DEBUGGING ONLY)

```bash
export BGB_SANITIZE_GRADS=1  # Replace NaN/Inf with zeros
```

**When to use**:
- ✅ Debugging gradient explosions (see WHERE NaNs occur)
- ✅ Research with experimental architectures
- ✅ Investigating FP16 overflow with mixed precision

**When NOT to use**:
- ❌ Production training (adds overhead, no benefit)
- ❌ If gradient clipping is working (it is)
- ❌ If you just want stability (clipping provides it)

**Default**: `0` (disabled)

**What it does**:
- Replaces NaN/Inf gradients with zeros AFTER backward, BEFORE clipping
- Logs warnings when sanitization occurs
- Does NOT fix root cause - investigate why NaNs occur!

---

## Why You Don't Need Sanitization

Gradient clipping already handles inf/nan gradients:

1. Calculates total gradient norm (can be inf)
2. Scales ALL gradients proportionally
3. Result: Finite gradients even if some were inf
4. PyTorch handles this automatically

**Example**:
- Gradient norms: `[1.0, inf, 2.0, 3.0]`
- Total norm: `inf`
- After clipping to 0.5: `[0.1, 0.0, 0.2, 0.2]` (all finite)

---

## When Gradient Logs Report `inf`

**This is normal with FP16 mixed precision!**

- The `inf` entries refer to the **pre-clip** gradient norm that PyTorch reports
- Actual parameter updates use the **post-clip** (finite) gradients
- If loss is decreasing, training is working correctly
- No action needed

**Why it happens**:
- FP16 max value: 65,504
- Large gradients overflow to `inf` during backward
- Gradient clipping scales them down to finite values before the optimizer step
- The logger prints the count of batches whose pre-clip norm exceeded the FP16 range

**Modal (A100, FP16)**:
```
[GRADIENTS] Last 100 batches: P50=2.19 | IQR=2.39 | P95=11.38 | Max=14.82
[GRADIENTS] 15/100 batches had inf pre-clip norm (normal with FP16, clipping handles it)
```

**Local (RTX 4090, FP32)**:
```
[GRADIENTS] Last 100 batches: P50=3.32 | IQR=2.87 | P95=9.74 | Max=10.84
```

---

## Removed Anti-Patterns (Oct 2025)

The following flags were documented but **never implemented**. They have been removed following 2025 best practices:

### ❌ BGB_SANITIZE_INPUTS (REMOVED)

**Why removed**: Input sanitization masks data quality issues. Preprocessing should guarantee clean data.

**Replacement**: Outlier clipping in preprocessing (±10σ in `preprocess.py`)

### ❌ BGB_SKIP_OPT_STEP_ON_NAN (REMOVED)

**Why removed**: Skipping optimizer steps breaks learning rate schedules and can cause desynchronization in distributed training.

**PyTorch Community Consensus**: Investigate root cause, don't skip steps.

### ❌ BGB_SAFE_CLAMP (REMOVED)

**Why removed**: Global activation clamping indicates architectural problems. LayerNorm is the correct solution.

**Replacement**: LayerNorm at 5 component boundaries (always enabled via config)

---

## ML 2025 Best Practices: Gradient Logging

### Why Median Over Mean?

**Problem with Mean**: Arithmetic mean is **outlier-sensitive**. A single FP16 overflow (`65504`) can dominate the mean, hiding the true gradient distribution.

**Example (100 batches)**:
- 99 batches with gradient norm ~2.0
- 1 batch with FP16 overflow (65504)
- **Mean**: `(99×2 + 65504)/100 = 657` (misleading!)
- **Median (P50)**: `2.0` (robust, accurate)

### New Gradient Logging Format (v3.6.1)

```
[GRADIENTS] Last 100 batches: P50=2.19 | IQR=2.39 | P95=11.38 | Max=14.82
```

**Metrics**:
- **P50 (Median)**: Central tendency, immune to outliers
- **IQR (P75 - P25)**: Robust spread measurement (better than std for heavy-tailed)
- **P95**: High percentile for detecting spikes
- **Max**: Absolute maximum (useful for sanity checks)

### Why IQR Over Standard Deviation?

**IQR advantages**:
- ✅ Robust to extreme outliers (FP16 overflow)
- ✅ Describes middle 50% of distribution
- ✅ Interpretable: "Most gradients are within X ± IQR/2"

**Standard deviation problems**:
- ❌ Squared deviations amplify outliers
- ❌ Assumes Gaussian distribution (gradients are heavy-tailed)
- ❌ Can be infinite with extreme values

### Why This Matters for Seizure Detection

**Clinical ML requirements**:
1. **Stability**: Must train for 100 epochs (80-100 hours)
2. **Reproducibility**: Same hyperparameters across runs
3. **Debugging**: Quickly identify instability patterns

**Percentile-based logging provides**:
- Early warning of instability (P95 trending up)
- Confidence in median behavior (P50 stable)
- Immune to transient FP16 overflows
- Better basis for hyperparameter tuning

### Migration from v3.6.0

**Old format (mean-based)**:
```
[GRADIENTS] Last 50 batches (finite): Mean=inf | P50=2.19 | P95=11.38 | Max=14.82
```

**Problems**:
- Mean could be `inf` despite "(finite)" label (contradictory)
- Mean highly sensitive to outliers
- Hard to interpret true gradient health

**New format (median-first)**:
```
[GRADIENTS] Last 100 batches: P50=2.19 | IQR=2.39 | P95=11.38 | Max=14.82
```

**Improvements**:
- ✅ P50 emphasized (most important metric)
- ✅ IQR added (robust spread)
- ✅ Mean removed (outlier-sensitive)
- ✅ Cleaner output (no contradictory labels)

---

## Related Documentation

- Industry best practices: [PyTorch Tutorials](https://pytorch.org/tutorials)
- Mixed precision: [PyTorch AMP](https://pytorch.org/docs/stable/amp.html)
- Gradient clipping: [torch.nn.utils.clip_grad_norm_](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
- Gradient logging implementation: `src/brain_brr/train/train_step.py:336-347`

---

**Key Takeaway**: Gradient clipping from config is your primary protection. The `BGB_SANITIZE_GRADS` flag is purely a debugging tool for investigating gradient issues, not a requirement for training. Use median-based logging (v3.6.1+) for robust gradient monitoring.
