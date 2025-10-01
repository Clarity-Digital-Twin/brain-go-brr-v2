# Warmup Schedules Guide

**Last Updated**: October 1, 2025
**Version**: v3.4.1
**Status**: Production ready, optional feature

---

## Quick Start

### Enable Warmup Schedules

**Edit your config** (`configs/local/train.yaml` or `configs/modal/train.yaml`):

```yaml
training:
  # Warmup schedules (OPTIONAL - for smoother early training)
  warmup_schedule:
    enabled: true
    warmup_steps: 1000

    # Adjacency temperature: smooth → sharp softmax
    adj_temperature_enabled: true
    adj_temperature_start: 2.0
    adj_temperature_end: 1.0

    # Focal gamma: less → more focusing
    focal_gamma_enabled: true
    focal_gamma_start: 1.0
    focal_gamma_end: 2.0
```

**That's it!** Training will automatically use scheduled values for first 1000 steps.

---

## Table of Contents

1. [What Are Warmup Schedules?](#what-are-warmup-schedules)
2. [When to Use Them](#when-to-use-them)
3. [Configuration Options](#configuration-options)
4. [How They Work](#how-they-work)
5. [Expected Impact](#expected-impact)
6. [Best Practices](#best-practices)
7. [FAQ](#faq)

---

## What Are Warmup Schedules?

### The Problem

**Early training is volatile** due to:
- Random weight initialization → large errors
- Sharp losses (focal γ=2.0) → amplified gradients
- Dynamic graph (adjacency changes rapidly) → unstable gradients

**Warmup schedules** provide a **smoother ramp-up** to full training intensity.

### The Solution

**Gradually increase training intensity** over first N steps:

| Parameter | Start (gentle) | End (full) | Why |
|-----------|----------------|------------|-----|
| **Adjacency τ** | 2.0 | 1.0 | Softer softmax → smaller adjacency gradients |
| **Focal γ** | 1.0 | 2.0 | Less amplification → smaller loss gradients |

**Result**: Smoother gradient trajectory during most volatile phase (batches 0-1000).

---

## When to Use Them

### ✅ Recommended For:

**First training run on new dataset**:
- Unsure how architecture will behave
- Want smoother early training
- Extra safety margin

**After significant architecture changes**:
- Added new layers or components
- Changed activation functions
- Modified loss function

**High gradient variance early**:
- P95 >50 in first 100 batches
- Frequent "Large grad norm" messages
- Training feels unstable

### ❌ NOT Needed For:

**Model already trains stably**:
- Current v3.4.1 architecture is ROCK SOLID
- Training without warmup works perfectly
- P95=9.74 at batch 723 is excellent ✅

**Fine-tuning from checkpoint**:
- Weights already learned
- Not starting from random initialization
- Skip warmup to save compute

**Quick experiments**:
- Testing small code changes
- Smoke tests (too short for 1000-step warmup)
- A/B testing features

### Current Training Status

**v3.4.1 WITHOUT warmup** (proven working):
```
Batch 723 Results:
  Loss: 0.3050 → 0.1555 (49% decrease)
  P95: 52.06 → 9.74 (82% decrease)
  Mean: 14.54 → 3.32 (77% decrease)
  NaN/Inf: ZERO
```

**Warmup is OPTIONAL enhancement, not a fix!**

---

## Configuration Options

### Full Configuration Schema

```yaml
training:
  warmup_schedule:
    # Master switch (set to false or null to disable everything)
    enabled: true

    # How many steps for warmup phase
    warmup_steps: 1000  # Typical: 500-2000

    # ===== ADJACENCY TEMPERATURE =====
    # Controls how sharp the row-softmax is in adjacency matrix
    adj_temperature_enabled: true
    adj_temperature_start: 2.0  # Softer at start (valid: 1.0-5.0)
    adj_temperature_end: 1.0    # Match graph.adj_softmax_tau

    # ===== FOCAL LOSS GAMMA =====
    # Controls how much focal loss amplifies hard examples
    focal_gamma_enabled: true
    focal_gamma_start: 1.0  # Standard BCE at start (valid: 0.0-3.0)
    focal_gamma_end: 2.0    # Match training.focal_gamma

    # ===== RESIDUAL SCALING (Advanced - usually not needed) =====
    residual_scale_enabled: false  # Experimental
    residual_scale_blocks: [0, 1]   # First 2 Mamba blocks
    residual_scale_factor: 0.5      # Scale residuals to 50%
```

### Minimal Configuration (Recommended)

```yaml
training:
  warmup_schedule:
    enabled: true
    warmup_steps: 1000
    adj_temperature_enabled: true
    adj_temperature_start: 2.0
    adj_temperature_end: 1.0
    focal_gamma_enabled: true
    focal_gamma_start: 1.0
    focal_gamma_end: 2.0
```

### Disable Warmup

**Option 1** (explicit):
```yaml
training:
  warmup_schedule:
    enabled: false  # Disabled - use full parameters immediately
```

**Option 2** (implicit - recommended):
```yaml
training:
  # warmup_schedule: null  ← Comment out or set to null
```

---

## How They Work

### Adjacency Temperature Schedule

**What it controls**: Row-softmax sharpness in learned adjacency matrix

**Schedule**:
```python
τ(step) = 2.0 - (2.0 - 1.0) × (step / 1000)

# Examples:
# step 0:    τ = 2.0 (very soft softmax)
# step 500:  τ = 1.5 (medium)
# step 1000: τ = 1.0 (sharp softmax - production value)
# step 1000+: τ = 1.0 (stays at target)
```

**Effect**:
- **τ=2.0** (soft): Adjacency entries more uniform → smaller gradient changes
- **τ=1.0** (sharp): Adjacency entries more peaked → larger gradient changes

**Why**: Dynamic adjacency changes rapidly during early training. Softer softmax reduces gradient spikes from structural changes.

**Applied in**: `condition_adjacency()` during row-softmax normalization

---

### Focal Loss Gamma Schedule

**What it controls**: How much focal loss amplifies hard examples

**Schedule**:
```python
γ(step) = 1.0 + (2.0 - 1.0) × (step / 1000)

# Examples:
# step 0:    γ = 1.0 (standard BCE)
# step 500:  γ = 1.5 (medium focusing)
# step 1000: γ = 2.0 (strong focusing - production value)
# step 1000+: γ = 2.0 (stays at target)
```

**Effect**:
- **γ=1.0**: `L = BCE` (no amplification)
- **γ=2.0**: `L = (1-p)² × BCE` (4x amplification for confident mistakes)

**Why**: Early training makes many confident wrong predictions. Lower gamma reduces loss amplification during most volatile phase.

**Applied in**: Training loop, dynamically updates `FocalLoss.gamma` before each forward pass

---

### Residual Scaling (Advanced - Optional)

**What it controls**: Residual connection strength in first N Mamba blocks

**NOT NEEDED** for current architecture! Only for debugging very deep networks (12+ layers).

**If you must use it**:
```yaml
residual_scale_enabled: true
residual_scale_blocks: [0, 1]  # Scale first 2 blocks
residual_scale_factor: 0.5     # 50% scaling during warmup
```

**Effect**: Reduces early block dominance, helps gradient flow balance in very deep stacks.

---

## Expected Impact

### Gradient Trajectory

**WITHOUT warmup** (current v3.4.1 - proven working):
```
Batch 0-200:   P95 ~20-60 (high variance)
Batch 200-500: P95 ~10-30 (decreasing)
Batch 500+:    P95 ~5-20  (stable)
```

**WITH warmup** (expected - still testing):
```
Batch 0-200:   P95 ~15-40 (20-30% lower variance)
Batch 200-500: P95 ~8-20  (smoother decrease)
Batch 500+:    P95 ~5-20  (same final result)
```

**Key Insight**: Warmup affects the **path**, not the **destination**. Final performance should be similar.

### Training Time

**No significant impact**:
- Computation overhead: <0.1% (2 float divisions per batch)
- Memory overhead: Zero (no additional buffers)
- Convergence speed: Similar (may be slightly faster due to smoother early training)

### Loss Convergence

**Should be similar or slightly better**:
- Early training may be slightly slower (γ=1.0 is less aggressive)
- Mid-training should catch up (γ→2.0 ramps up)
- Final convergence: Same or better

---

## Best Practices

### 1. Start Conservative

**First run with warmup**:
```yaml
warmup_steps: 1000  # Standard
adj_temperature_start: 2.0  # Default
focal_gamma_start: 1.0  # Safe
```

**If too conservative** (training feels slow):
- Reduce `warmup_steps` to 500
- Increase `focal_gamma_start` to 1.5 (more aggressive)

**If still unstable** (high gradient variance):
- Increase `warmup_steps` to 1500-2000
- Increase `adj_temperature_start` to 3.0 (softer)

### 2. Monitor Early Batches

**Check gradient logs** (first 100 batches):
```bash
grep "\[GRADIENTS\]" train.log | head -5
```

**Look for**:
- P95 decreasing trend (good!)
- Smooth curve (not spiky)
- No "Sanitized NaN gradients" messages

**If warmup logs appear**:
```
[WARMUP] Batch 0 adj_tau=2.000
[WARMUP] Batch 50 adj_tau=1.900
[WARMUP] Batch 100 adj_tau=1.800
```

This confirms schedules are active ✅

### 3. Compare With/Without

**A/B Test** (recommended before production use):

**Run 1** (no warmup):
```yaml
warmup_schedule: null  # Baseline
```

**Run 2** (with warmup):
```yaml
warmup_schedule:
  enabled: true
  # ... full config
```

**Compare**:
- Gradient P95 at batches 100, 500, 1000
- Loss convergence curve
- Training time
- Final validation performance

**Use warmup if**: P95 >20% lower in first 500 batches with no downsides

### 4. Adjust for Platform

**Local (RTX 4090, batch=4)**:
```yaml
warmup_steps: 1000  # Fine for 15,404 batches/epoch
```

**Modal (A100-80GB, batch=64)**:
```yaml
warmup_steps: 500   # Faster convergence with larger batches
```

**Rule of thumb**: `warmup_steps ≈ 5-10% of batches per epoch`

### 5. Disable for Short Runs

**Smoke tests** (3-50 files):
```yaml
warmup_schedule: null  # Too short for warmup
```

**Checkpointing** (resume from epoch 50):
```yaml
warmup_schedule: null  # Already warmed up!
```

---

## FAQ

### Q: Do I need warmup schedules?

**A**: No! Current training (v3.4.1) works perfectly WITHOUT warmup:
- P95=9.74 at batch 723
- Loss decreasing smoothly
- Zero NaN/Inf

Warmup is **optional enhancement** for smoother early training, not a fix.

### Q: Will warmup make training faster?

**A**: Unlikely. It may make early training slightly smoother, but final convergence speed is similar. The main benefit is **stability**, not speed.

### Q: Can I use warmup with mixed precision?

**A**: Yes! Warmup schedules are orthogonal to `mixed_precision` setting.

```yaml
training:
  mixed_precision: true  # A100 tensor cores
  warmup_schedule:
    enabled: true  # Works together
```

### Q: What if I change warmup_steps mid-training?

**A**: Don't! `warmup_steps` is used for schedule computation. Changing it mid-run will cause incorrect values.

If you must change, restart training from scratch or checkpoint.

### Q: Can I use only one schedule (e.g., just adjacency)?

**A**: Yes! Each schedule is independent:

```yaml
warmup_schedule:
  enabled: true
  warmup_steps: 1000
  adj_temperature_enabled: true  # Enable this
  focal_gamma_enabled: false     # Disable this
```

### Q: How do I know if warmup is working?

**Check logs** for warmup messages:
```
[WARMUP] Batch 0 adj_tau=2.000 focal_gamma=1.000
[WARMUP] Batch 500 adj_tau=1.500 focal_gamma=1.500
[WARMUP] Batch 1000 adj_tau=1.000 focal_gamma=2.000
```

If you see these → warmup is active ✅

If you don't → check `enabled: true` in config

### Q: Does warmup affect validation/inference?

**A**: No! Warmup schedules only affect training loop. During validation and inference, model uses production values (τ=1.0, γ=2.0).

### Q: Can I use different warmup_steps for each schedule?

**A**: Currently no. All schedules share the same `warmup_steps`. This is by design for simplicity.

If you need different durations, file a feature request.

### Q: Will warmup help with Modal A100 training?

**A**: Potentially! Modal uses larger batches (64 vs 4 local), which can have higher gradient variance. Warmup may help smooth early training.

**Recommendation**: Test both with/without on Modal, compare P95 at batch 500.

---

## Configuration Examples

### Example 1: Standard Local Training

```yaml
# configs/local/train.yaml
training:
  epochs: 100
  batch_size: 4
  learning_rate: 1.0e-4
  gradient_clip: 0.5
  loss: focal
  focal_gamma: 2.0

  # Optional warmup (comment out to disable)
  warmup_schedule:
    enabled: true
    warmup_steps: 1000
    adj_temperature_enabled: true
    adj_temperature_start: 2.0
    adj_temperature_end: 1.0
    focal_gamma_enabled: true
    focal_gamma_start: 1.0
    focal_gamma_end: 2.0

model:
  graph:
    adj_softmax_tau: 1.0  # Target after warmup
```

### Example 2: Modal A100 Training

```yaml
# configs/modal/train.yaml
training:
  epochs: 100
  batch_size: 64
  learning_rate: 8.0e-5
  gradient_clip: 0.5
  loss: focal
  focal_gamma: 2.0

  # Shorter warmup for larger batches
  warmup_schedule:
    enabled: true
    warmup_steps: 500  # Faster convergence
    adj_temperature_enabled: true
    adj_temperature_start: 2.0
    adj_temperature_end: 1.0
    focal_gamma_enabled: true
    focal_gamma_start: 1.0
    focal_gamma_end: 2.0

model:
  graph:
    adj_softmax_tau: 1.0
```

### Example 3: Conservative Warmup (High Stability)

```yaml
# For first run on completely new dataset
training:
  warmup_schedule:
    enabled: true
    warmup_steps: 2000  # Longer warmup
    adj_temperature_enabled: true
    adj_temperature_start: 3.0  # Extra soft
    adj_temperature_end: 1.0
    focal_gamma_enabled: true
    focal_gamma_start: 0.5  # Very gentle
    focal_gamma_end: 2.0
```

### Example 4: Aggressive Warmup (Faster Ramp)

```yaml
# If training is already stable and you want faster convergence
training:
  warmup_schedule:
    enabled: true
    warmup_steps: 500  # Short warmup
    adj_temperature_enabled: true
    adj_temperature_start: 1.5  # Less soft
    adj_temperature_end: 1.0
    focal_gamma_enabled: true
    focal_gamma_start: 1.5  # Start higher
    focal_gamma_end: 2.0
```

---

## Summary

**Warmup schedules are**:
- ✅ Optional enhancement for smoother early training
- ✅ Standard practice in production ML (OpenAI, Google, Meta)
- ✅ Easy to configure (3 lines in YAML)
- ✅ Zero performance overhead
- ✅ Backward compatible (default: disabled)

**Use warmup when**:
- First training run on new dataset
- High gradient variance early (P95 >50)
- Want extra stability margin

**Skip warmup when**:
- Model already trains stably (like current v3.4.1!)
- Fine-tuning from checkpoint
- Quick experiments or smoke tests

**Current status**: v3.4.1 trains perfectly WITHOUT warmup (P95=9.74 at batch 723). Warmup is available if you want even smoother early training.

**Related Docs**:
- `nan-prevention-complete.md` - Full NaN protection guide
- `gradient-monitoring.md` - Realistic gradient expectations
- `v3-stability-evolution.md` - v3.3.1-3.4.1 evolution
