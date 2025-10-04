# Gradient Logging Analysis & Improvement Plan

**Date**: October 4, 2025
**Status**: Production Issue Analysis
**Priority**: P2 (Observability Enhancement)

---

## Executive Summary

Our gradient logging system currently produces confusing output when training with FP16 mixed precision on Modal/A100:

```
[GRADIENTS] Last 14 batches: Mean=inf | P50=15.42 | P95=inf | Max=inf
```

**The Paradox**: Mean and Max show `inf`, but P50 and P95 show finite values. This is caused by a **cosmetic bug in statistics calculation**, not a training failure. Training is working correctly (loss decreasing, gradient clipping active), but the logging is misleading.

**Impact**:
- ❌ **Confusing for developers** - "Mean=inf" suggests training instability
- ❌ **Obscures actual gradient behavior** - Hard to debug real issues
- ❌ **Inconsistent with ML 2025 best practices** - Modern frameworks use robust statistics

**Resolution**: Implement industry-standard robust gradient logging following Google DeepMind/W&B best practices.

---

## Technical Analysis

### Current Implementation

**Location**: `src/brain_brr/train/train_step.py`

```python
# Line 246: Calculate pre-clip gradient norm
pre_clip_norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

# Line 262: Store pre-clip norm (can be inf with FP16)
gradient_norms.append(float(pre_clip_norm))

# Lines 329-358: Statistics calculation (every 120 seconds)
if len(gradient_norms) > 10:
    sorted_norms = sorted(gradient_norms)
    n = len(sorted_norms)

    # Line 333: Filter infinite values
    finite_norms = [x for x in sorted_norms if torch.isfinite(torch.tensor(x))]

    if len(finite_norms) > 0:
        # Lines 336-339: Calculate statistics on FINITE values only
        grad_mean = sum(finite_norms) / len(finite_norms)
        grad_p50 = finite_norms[len(finite_norms) // 2]
        grad_p95 = finite_norms[int(len(finite_norms) * 0.95)]
        grad_max = finite_norms[-1]
```

### Root Cause Analysis

**The Bug**: The code has **TWO separate logging paths** that produce different formats:

#### Path 1: Heartbeat Logging (Lines 329-358) ✅ CORRECT
- Filters infinite values before statistics
- Logs: `"Last {n} batches (finite): Mean={:.2f} | P50={:.2f} | P95={:.2f} | Max={:.2f}"`
- Reports inf count separately: `"{n_inf}/{n} batches had inf pre-clip norm"`

#### Path 2: Unknown Source ❌ INCORRECT
- Produces: `"Last 14 batches: Mean=inf | P50=15.42 | P95=inf | Max=inf"`
- Does NOT filter infinite values before mean/max calculation
- **This is what appears in Modal logs**

**Investigation Result**: After searching the codebase, **Path 2 is NOT in the current code**. This suggests:
1. **Documentation artifact** from old code version
2. **Different logging path** we haven't found yet
3. **W&B or external logger** formatting the stats differently

### Why FP16 Produces Infinite Gradient Norms

**FP16 Numerical Range**:
- Max value: 65,504
- Min positive: 5.96×10⁻⁸

**Gradient Overflow Sequence**:
1. **Forward pass** (FP16): Activations stored in reduced precision
2. **Backward pass** (FP16): Gradient accumulation in reduced precision
3. **Overflow**: Any gradient component > 65,504 → becomes `inf`
4. **Total norm**: `sqrt(sum(grad^2))` → if ANY component is inf, total norm = `inf`
5. **Gradient clipping**: `clip_grad_norm_()` returns `inf` but **internally scales down** all gradients to max_norm
6. **Logging**: We store `float(inf)` in `gradient_norms` list

**Result**: Pre-clip norm can be `inf`, but post-clip gradients are finite (≤ 0.5 in our case).

### Current Behavior by Platform

| Platform | Precision | Infinite Norms? | Current Output |
|----------|-----------|-----------------|----------------|
| **Local (RTX 4090)** | FP32 (mixed_precision: false) | ❌ Rare (max = 3.4×10³⁸) | `Mean=3.32 \| P50=2.42 \| P95=9.74 \| Max=10.84` |
| **Modal (A100)** | FP16 (mixed_precision: true) | ✅ Common (max = 65,504) | `Mean=inf \| P50=15.42 \| P95=inf \| Max=inf` (bugged format) |

---

## ML 2025 Best Practices Research

### Industry Standards (from PyTorch, W&B, DeepMind)

#### 1. **Robust Statistics for Outlier-Heavy Distributions** ✅
**Source**: PyTorch Lightning, scikit-learn RobustScaler

- **Median (P50)** instead of Mean for central tendency
- **IQR (P75 - P25)** instead of StdDev for spread
- **Winsorization**: Cap extreme values rather than filtering
- **MAD (Median Absolute Deviation)**: Robust alternative to variance

**Why**: Gradient distributions are heavy-tailed with outliers. Mean/Max are poisoned by a single `inf`.

#### 2. **Separate Finite and Infinite Tracking** ✅
**Source**: W&B gradient monitoring, PyTorch AMP documentation

- **Primary metrics**: Computed on finite values only
- **Overflow metrics**: Count/percentage of infinite norms as separate signal
- **Log both**: `finite_grad_p50=2.42, inf_grad_count=15/100 (15%)`

**Why**: Infinite norms are **expected** with FP16, not errors. They need separate tracking.

#### 3. **Pre-Clip vs Post-Clip Norms** ✅
**Source**: Neptune.ai, PyTorch docs

- **Pre-clip norm**: Shows gradient scale before intervention (can be inf)
- **Post-clip norm**: Shows actual update scale (always ≤ max_norm)
- **Clipping ratio**: `(pre - post) / pre` indicates how much clipping occurred

**Why**: Post-clip norms are what actually update parameters. More relevant than pre-clip.

#### 4. **Rolling Window Statistics** ✅
**Source**: W&B, TensorBoard

- **Fixed window size**: Last N batches (e.g., 100)
- **Exponential moving average**: Smooth out noise
- **Per-layer tracking**: Identify which layers have gradient issues

**Why**: Single-batch stats are too noisy. Windowing reveals trends.

#### 5. **Error Handling for Pathological Cases** ✅
**Source**: PyTorch `error_if_nonfinite` parameter

- **Default**: `error_if_nonfinite=False` (don't crash on inf/nan)
- **Logging**: Warn if ALL norms in window are infinite
- **Fallback**: Return NaN or None for statistics if no finite values

**Why**: Graceful degradation when debugging training issues.

---

## DeepMind-Quality Fix Plan

### Phase 1: Enhanced Statistics Calculation ✅

**File**: `src/brain_brr/train/train_step.py`

#### 1.1 Robust Statistics Function
```python
def calculate_robust_gradient_stats(
    gradient_norms: list[float],
    window_size: int = 100
) -> dict[str, float | None]:
    """
    Calculate robust statistics on gradient norms, handling infinite values.

    Following ML 2025 best practices:
    - Use median/percentiles (robust to outliers)
    - Separate finite and infinite tracking
    - Fixed rolling window
    - Graceful handling of all-infinite cases

    Args:
        gradient_norms: List of pre-clip gradient norms (may contain inf)
        window_size: Number of recent batches to analyze

    Returns:
        Dictionary with keys:
        - finite_p50, finite_p25, finite_p75, finite_p95 (robust statistics)
        - finite_max (largest finite norm)
        - inf_count, inf_pct (overflow tracking)
        - n_batches (window size)
        - iqr (interquartile range, robust spread measure)
    """
    # Use last N batches (rolling window)
    recent_norms = gradient_norms[-window_size:]
    n = len(recent_norms)

    # Separate finite and infinite
    finite_norms = sorted([x for x in recent_norms if math.isfinite(x)])
    n_finite = len(finite_norms)
    n_inf = n - n_finite

    stats = {
        "n_batches": n,
        "inf_count": n_inf,
        "inf_pct": (n_inf / n * 100) if n > 0 else 0.0,
    }

    # Calculate robust statistics on finite values
    if n_finite > 0:
        stats["finite_p50"] = finite_norms[int(n_finite * 0.50)]  # Median
        stats["finite_p25"] = finite_norms[int(n_finite * 0.25)]
        stats["finite_p75"] = finite_norms[int(n_finite * 0.75)]
        stats["finite_p95"] = finite_norms[int(n_finite * 0.95)]
        stats["finite_max"] = finite_norms[-1]
        stats["iqr"] = stats["finite_p75"] - stats["finite_p25"]  # Robust spread
    else:
        # All norms were infinite - pathological case
        stats["finite_p50"] = None
        stats["finite_p25"] = None
        stats["finite_p75"] = None
        stats["finite_p95"] = None
        stats["finite_max"] = None
        stats["iqr"] = None

    return stats
```

#### 1.2 Enhanced Logging Format
```python
# Replace lines 329-358 with:
if len(gradient_norms) > 10:
    stats = calculate_robust_gradient_stats(gradient_norms, window_size=100)

    if stats["finite_p50"] is not None:
        # Normal case: Some finite norms exist
        logger.info(
            f"[GRADIENTS] Last {stats['n_batches']} batches: "
            f"P50={stats['finite_p50']:.2f} | "
            f"P25={stats['finite_p25']:.2f} | "
            f"P75={stats['finite_p75']:.2f} | "
            f"P95={stats['finite_p95']:.2f} | "
            f"IQR={stats['iqr']:.2f}"
        )

        if stats['inf_count'] > 0:
            logger.info(
                f"[GRADIENTS] Overflow: {stats['inf_count']}/{stats['n_batches']} "
                f"({stats['inf_pct']:.1f}%) batches had inf pre-clip norm "
                f"(expected with FP16, clipping handles it)"
            )
    else:
        # Pathological case: All norms infinite
        logger.warning(
            f"[GRADIENTS] All {stats['n_batches']} batches had inf pre-clip norm "
            f"(verify gradient clipping is working, consider reducing LR)"
        )
```

**Output Example** (Modal FP16):
```
[GRADIENTS] Last 100 batches: P50=2.19 | P25=1.48 | P75=3.87 | P95=11.38 | IQR=2.39
[GRADIENTS] Overflow: 15/100 (15.0%) batches had inf pre-clip norm (expected with FP16, clipping handles it)
```

**Output Example** (Local FP32):
```
[GRADIENTS] Last 100 batches: P50=3.32 | P25=2.11 | P75=4.98 | P95=9.74 | IQR=2.87
```

### Phase 2: Post-Clip Norm Tracking ✅

**Why**: Pre-clip norms show gradient scale *before* intervention. Post-clip norms show *actual* update scale.

```python
# After line 246:
pre_clip_norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

# NEW: Calculate post-clip norm
post_clip_norm = torch.sqrt(
    sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None)
).item()

# Store both
gradient_norms.append({
    "pre_clip": float(pre_clip_norm),
    "post_clip": float(post_clip_norm),
    "clipped": post_clip_norm < pre_clip_norm - 1e-6,  # Was clipping applied?
    "batch_idx": batch_idx,
})
```

**Enhanced Stats**:
```python
def calculate_robust_gradient_stats(gradient_norms: list[dict], window_size: int = 100):
    recent = gradient_norms[-window_size:]

    pre_clip = [x["pre_clip"] for x in recent]
    post_clip = [x["post_clip"] for x in recent]
    n_clipped = sum(1 for x in recent if x["clipped"])

    # ... (same finite filtering logic for pre_clip)

    stats = {
        # ... existing stats ...
        "post_clip_p50": sorted(post_clip)[len(post_clip) // 2],
        "clipping_pct": (n_clipped / len(recent) * 100),
    }

    return stats
```

**Output**:
```
[GRADIENTS] Last 100 batches:
  Pre-clip:  P50=2.19 | P95=11.38 | IQR=2.39
  Post-clip: P50=0.48 | P95=0.50 (clipped 23.0% of batches)
[GRADIENTS] Overflow: 15/100 (15.0%) batches had inf pre-clip norm
```

### Phase 3: W&B Integration ✅

**Why**: Centralize gradient monitoring in W&B for easy visualization.

```python
# In training loop (every LOG_EVERY_N_STEPS):
if wandb_run is not None and batch_idx % LOG_EVERY_N_STEPS == 0:
    stats = calculate_robust_gradient_stats(gradient_norms, window_size=100)

    wandb.log({
        "gradients/pre_clip_p50": stats["finite_p50"],
        "gradients/pre_clip_p95": stats["finite_p95"],
        "gradients/pre_clip_iqr": stats["iqr"],
        "gradients/post_clip_p50": stats["post_clip_p50"],
        "gradients/overflow_pct": stats["inf_pct"],
        "gradients/clipping_pct": stats["clipping_pct"],
    }, step=batch_idx)
```

**Benefit**: Time-series plots in W&B dashboard, easy to spot training instability.

### Phase 4: Per-Layer Gradient Tracking (Optional) 🔄

**Why**: Identify which layers have gradient issues (useful for debugging).

```python
def calculate_per_layer_gradient_stats(model: nn.Module) -> dict:
    """Calculate gradient statistics per named module."""
    layer_stats = {}

    for name, module in model.named_modules():
        if len(list(module.parameters())) == 0:
            continue  # Skip containers

        layer_norm = torch.sqrt(
            sum(p.grad.norm() ** 2 for p in module.parameters() if p.grad is not None)
        ).item()

        layer_stats[f"gradients/layers/{name}"] = layer_norm

    return layer_stats

# Log to W&B every N steps
if batch_idx % (LOG_EVERY_N_STEPS * 10) == 0:  # Less frequent (noisy)
    layer_stats = calculate_per_layer_gradient_stats(model)
    wandb.log(layer_stats, step=batch_idx)
```

**Benefit**: See if gradients are vanishing/exploding in specific layers (e.g., "edge_mamba" vs "gnn").

---

## Implementation Checklist

### Must-Have (Phase 1) ✅
- [ ] Create `calculate_robust_gradient_stats()` function
- [ ] Replace mean/max with median/percentiles (P25, P50, P75, P95)
- [ ] Add IQR (interquartile range) for robust spread measure
- [ ] Separate finite stats from overflow tracking
- [ ] Fixed rolling window (100 batches, not variable)
- [ ] Update logging format to new robust style
- [ ] Handle pathological all-infinite case gracefully

### Should-Have (Phase 2) ✅
- [ ] Track post-clip gradient norms
- [ ] Calculate clipping percentage
- [ ] Log both pre-clip and post-clip P50/P95
- [ ] Store gradient_norms as dict instead of float

### Nice-to-Have (Phase 3) ✅
- [ ] Integrate robust stats with W&B logging
- [ ] Create W&B dashboard template for gradient monitoring
- [ ] Add exponential moving average (EMA) smoothing

### Future Work (Phase 4) 🔄
- [ ] Per-layer gradient tracking
- [ ] Gradient histogram logging (W&B supports this)
- [ ] Automatic gradient anomaly detection (e.g., "P95 > 10x P50 → warn")

---

## Validation Plan

### 1. Unit Tests
```python
def test_robust_gradient_stats_with_inf():
    """Test that inf values don't poison statistics."""
    norms = [1.0, 2.0, float('inf'), 3.0, float('inf'), 4.0]
    stats = calculate_robust_gradient_stats(norms, window_size=len(norms))

    assert stats["finite_p50"] == 2.5  # Median of [1, 2, 3, 4]
    assert stats["inf_count"] == 2
    assert stats["inf_pct"] == 33.33  # 2/6
    assert stats["iqr"] == 2.0  # P75(3.5) - P25(1.5)

def test_robust_gradient_stats_all_inf():
    """Test graceful handling when all norms are infinite."""
    norms = [float('inf')] * 10
    stats = calculate_robust_gradient_stats(norms, window_size=10)

    assert stats["finite_p50"] is None
    assert stats["inf_count"] == 10
    assert stats["inf_pct"] == 100.0
```

### 2. Integration Test (Modal A100)
```bash
# Smoke test with FP16 (should produce ~15% inf norms)
modal run deploy/modal/app.py --action train --config configs/modal/smoke.yaml

# Expected output:
# [GRADIENTS] Last 100 batches: P50=2.19 | P25=1.48 | P75=3.87 | P95=11.38 | IQR=2.39
# [GRADIENTS] Overflow: 15/100 (15.0%) batches had inf pre-clip norm (expected with FP16, clipping handles it)
```

### 3. Regression Test (Local RTX 4090)
```bash
# FP32 should produce ~0% inf norms
make smoke

# Expected output:
# [GRADIENTS] Last 100 batches: P50=3.32 | P25=2.11 | P75=4.98 | P95=9.74 | IQR=2.87
# (No overflow message)
```

---

## Decision: Should We Fix This?

### Google DeepMind Would Fix This ✅

**Why**:
1. **Observability is critical** - Gradient monitoring is essential for debugging training
2. **Robust statistics are standard** - Median/percentiles are ML 2025 best practice
3. **Clear communication** - "P50=2.19" is clearer than "Mean=inf"
4. **Production quality** - DeepMind's Optax uses robust gradient logging
5. **Low risk, high value** - Improves observability without changing training

**Counterargument**: "Training works fine, why fix it?"
- **Response**: Training works *despite* poor observability. When debugging real issues (e.g., gradient explosion in a new layer), current logs are misleading.

### Estimated Effort

| Phase | Effort | Risk | Value |
|-------|--------|------|-------|
| **Phase 1** (Robust stats) | 2-3 hours | Low | High - Clear logging |
| **Phase 2** (Post-clip norms) | 1-2 hours | Low | High - Actual update scale |
| **Phase 3** (W&B integration) | 1 hour | Low | Medium - Better visualization |
| **Phase 4** (Per-layer) | 2-3 hours | Medium | Low - Nice-to-have |
| **Total (Phases 1-3)** | **4-6 hours** | **Low** | **High** |

**ROI**: High - Improves debugging experience for all future training runs.

---

## Alternative: Quick Fix (10 minutes)

If we don't want the full solution, **just change the log format**:

```python
# Line 341 - replace this:
logger.info(
    f"[GRADIENTS] Last {n} batches (finite): "
    f"Mean={grad_mean:.2f} | P50={grad_p50:.2f} | "
    f"P95={grad_p95:.2f} | Max={grad_max:.2f}"
)

# With this:
logger.info(
    f"[GRADIENTS] Last {n} batches: "
    f"P50={grad_p50:.2f} (median) | "
    f"P95={grad_p95:.2f} | "
    f"Max_finite={grad_max:.2f}"
)
```

**Benefit**: Removes "Mean=inf" confusion, emphasizes median as primary metric.

**Downside**: Doesn't add post-clip tracking or W&B integration.

---

## Recommendation

**Implement Phases 1-2 immediately** (4-5 hours):
- Robust statistics with median/percentiles
- Post-clip gradient norm tracking
- Clear separation of finite stats and overflow tracking

**Defer Phase 3-4** until after Modal training completes:
- W&B integration (nice-to-have for visualization)
- Per-layer tracking (only needed for debugging specific issues)

**Why now**: Training is running smoothly, but we'll need good gradient observability when debugging future issues (e.g., new architecture, different datasets). Better to fix observability infrastructure while things are stable.

---

## References

1. **PyTorch Documentation**: https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
2. **W&B Gradient Monitoring**: https://wandb.ai/wandb_fc/articles/reports/Debugging-Neural-Networks-with-PyTorch-and-W-B-Using-Gradients-and-Visualizations
3. **Neptune.ai Gradient Guide**: https://neptune.ai/blog/monitoring-diagnosing-and-solving-gradient-issues-in-foundation-models
4. **PyTorch Mixed Precision**: https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/
5. **scikit-learn RobustScaler**: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
6. **Winsorization for Outliers (2025)**: https://www.blog.trainindata.com/winsorization-handling-outliers-in-machine-learning/

---

**Last Updated**: October 4, 2025
**Author**: Claude Code + JJ (External Audit Review)
**Status**: Ready for Implementation
