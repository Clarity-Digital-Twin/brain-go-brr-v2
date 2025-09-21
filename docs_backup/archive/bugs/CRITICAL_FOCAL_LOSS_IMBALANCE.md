# CRITICAL: Focal Loss & Extreme Class Imbalance Configuration

## ⚠️ EXTREME IMBALANCE WARNING ⚠️

**TUSZ Dataset**: ~7.5-8% seizures, ~92% background (12:1 imbalance)

This extreme imbalance WILL cause model collapse (AUROC=0.5) without proper handling.

## The Problem: Silent Model Collapse

With extreme imbalance, models learn to predict all-zeros and still achieve 92% accuracy. This looks like:
- AUROC = 0.5 (random guessing)
- All predictions = 0 (never predicts seizure)
- Training loss decreases but model learns nothing useful
- Validation metrics look "good" if you only check accuracy

## The Solution: Three-Layer Defense

### 1. Focal Loss (NOT regular BCE)
```yaml
training:
  loss: focal                    # REQUIRED for imbalance
  focal_alpha: 0.5              # MUST BE 0.5 (neutral)
  focal_gamma: 2.0              # Focus on hard examples
```

**Critical**: `focal_alpha` must be 0.5 for neutral weighting. We rely on pos_weight for imbalance.

### 2. Balanced Sampling
```yaml
data:
  use_balanced_sampling: true    # REQUIRED for batches to see seizures
```

Without this, many batches will have ZERO seizures → model never learns.

### 3. Automatic pos_weight
The training loop automatically computes pos_weight from class ratios:
```python
pos_weight = num_negatives / num_positives  # ~12 for TUSZ
```

## Critical Bugs We Fixed

### Bug 1: Backwards Alpha (DOWN-WEIGHTED SEIZURES!)
```yaml
# WRONG - This DOWN-WEIGHTS seizures (the rare class we care about!)
focal_alpha: 0.25  # 25% weight to seizures, 75% to background

# CORRECT - Neutral weighting
focal_alpha: 0.5   # 50% each, let pos_weight handle imbalance
```

### Bug 2: Double-Counting Class Weights
```python
# WRONG - Using BOTH pos_weight AND alpha != 0.5
loss = FocalLoss(alpha=0.25)  # Class weighting
loss(logits, labels, pos_weight=12)  # ALSO class weighting → double!

# CORRECT - Our code now prevents this
if alpha != 0.5:
    pos_weight = None  # Disable to prevent double-counting
```

### Bug 3: Empty Positive Batches
```python
# WRONG - Random sampling with 8% positives
# Many batches have ZERO seizures → model never sees them

# CORRECT - Balanced sampling ensures every batch has seizures
use_balanced_sampling: true  # Forces minimum seizures per batch
```

## Verification Checklist

### Training Start (MUST SEE ALL):
```
[INIT] Using FOCAL loss (alpha=0.5, gamma=2.0)
[SAMPLER] Creating positive-aware balanced sampler...
[SAMPLER] Seizure ratio: X.XX%
[SAMPLER] Positive weight: X.XX
```

### During Training:
- Batch positive ratio > 10% (logged)
- Loss decreasing for BOTH classes
- AUROC > 0.5 immediately (not random)

### Red Flags (STOP IMMEDIATELY):
- AUROC = 0.5 exactly → model collapsed
- All predictions = 0 → predicting only background
- No "[SAMPLER] Creating positive-aware" message → not using balanced sampling
- focal_alpha != 0.5 in logs → wrong weighting

## Config Requirements

### Training Configs (MUST HAVE):
- `loss: focal`
- `focal_alpha: 0.5`
- `focal_gamma: 2.0`
- `use_balanced_sampling: true`

### Eval-Only Configs:
- Don't need loss settings (no training)
- Still need `use_balanced_sampling: true` for proper eval

## File Locations

- **Focal Loss Implementation**: `src/brain_brr/train/loop.py:178-210`
- **Anti-Double-Count Logic**: `src/brain_brr/train/loop.py:257-270`
- **Balanced Sampler**: `src/brain_brr/data/loader.py:create_balanced_sampler()`
- **Config Schema**: `src/brain_brr/config/schemas.py:focal_alpha`

## Lessons Learned

1. **Alpha is confusing**: In focal loss, alpha applies to POSITIVES. Lower alpha = less weight to positives.
2. **Double-counting is subtle**: Using both pos_weight and alpha multiplies the imbalance correction.
3. **Silent failures**: Model can look like it's training (loss decreasing) while learning nothing.
4. **Batch composition matters**: Even one batch without positives can destabilize training.

## The Fix That Saved Us

```python
# src/brain_brr/train/loop.py:257-270
# Prevent double-counting when alpha != 0.5
alpha_diff = abs(float(focal_alpha) - 0.5)
pass_pos_weight = alpha_diff < 1e-6  # Only use pos_weight if alpha=0.5

if not pass_pos_weight:
    print("[INIT] FOCAL: alpha != 0.5 → disabling pos_weight to avoid double-counting")

def compute_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    pw = pos_weight_t if pass_pos_weight else None
    return focal(x, y, pos_weight=pw)
```

## Modal vs Local

- **Local**: May see long "[SAMPLER]..." pause during indexing - normal
- **Modal**: Full dataset takes 10-20 min to index from S3
- **Both**: MUST see balanced sampler messages or training will fail

## TL;DR

**Without ALL three defenses (focal loss + alpha=0.5 + balanced sampling), the model WILL collapse to AUROC=0.5 and be useless.**

This is not optional. This is not a nice-to-have. This is REQUIRED for seizure detection to work at all.