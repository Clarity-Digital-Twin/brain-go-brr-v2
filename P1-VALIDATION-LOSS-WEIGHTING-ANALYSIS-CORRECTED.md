# P1: Training-Validation Loss Mismatch Analysis (CORRECTED)

**Status:** 🟡 OPEN (Non-blocking, safe to defer)
**Priority:** P1 (affects interpretability, not correctness)
**Date:** 2025-10-03 (Corrected after agent review)
**Author:** Deep investigation from first principles

---

## Executive Summary

**Issue:** Training uses **focal loss** (hard-example focusing), validation uses **plain BCE** (no focusing). This makes train/val loss curves not directly comparable.

**Impact:** Does NOT affect current training because we use `sensitivity_at_10fa` for early stopping and best model selection (not loss). However, it affects interpretability and future loss-based experiments.

**Recommendation:** Implement Option B (report both focal and plain BCE validation losses) after current Modal training completes.

**Training Status:** ✅ **SAFE TO CONTINUE** - Current 100-epoch Modal training (App: ap-BwyQN1PX1prmfzbWGlUDqS) is NOT affected.

---

## Critical Correction from Original Analysis

### ❌ WHAT I ORIGINALLY CLAIMED (WRONG!)
> "Training loss uses `pos_weight` for class imbalance weighting, validation loss does not."

### ✅ ACTUAL GROUND TRUTH
**Training uses FOCAL LOSS (not weighted BCE):**
```python
# train_step.py:179-187 - When loss_mode="focal" (both configs)
if loss_mode == "focal":  # ← TRUE in our configs!
    probs = torch.sigmoid(logits)
    pt = labels * probs + (1 - labels) * (1 - probs)
    at = labels * focal_alpha + (1 - labels) * (1 - focal_alpha)
    focal_weight = at * ((1 - pt) ** focal_gamma)
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    loss = (focal_weight * bce).mean()
```

**Validation uses PLAIN BCE:**
```python
# val_step.py:239
criterion = nn.BCEWithLogitsLoss()
```

**Why this matters:** The mismatch is NOT about class imbalance weighting (α=0.5 is neutral), but about **hard-example focusing** (γ=2.0 amplifies hard examples up to 100x!).

---

## Technical Details from First Principles

### Focal Loss Mathematics

**Standard BCE Loss:**
```
L_BCE = - [y × log(p) + (1-y) × log(1-p)]
```

**Focal Loss (Lin et al. 2017):**
```
L_focal = - α_t × (1 - p_t)^γ × [y × log(p) + (1-y) × log(1-p)]
          └─ α_t: class weight ─┘ └─ (1-p_t)^γ: hard-example focus ─┘
```

Where:
- `p_t = y × p + (1-y) × (1-p)` → probability of TRUE class
- `α_t = y × α + (1-y) × (1-α)` → class-dependent weight
- `(1 - p_t)^γ` → focusing parameter (higher γ = more focus on hard examples)

**Our Configuration:**
```yaml
focal_alpha: 0.5   # α_t = 0.5 always (NEUTRAL class weighting)
focal_gamma: 2.0   # γ = 2.0 (hard-example focusing)
```

**What this means:**
- `α_t = y × 0.5 + (1-y) × 0.5 = 0.5` (always!) → NO class imbalance weighting
- `(1 - p_t)^2.0` → Hard-example amplification:
  - Easy example (p_t=0.95): weight = (1-0.95)^2 = 0.0025 (400x DOWN-weighted!)
  - Hard example (p_t=0.6): weight = (1-0.6)^2 = 0.16 (6x DOWN-weighted)
  - Very hard example (p_t=0.1): weight = (1-0.1)^2 = 0.81 (1.2x DOWN-weighted)
  - Completely wrong (p_t=0.01): weight = (1-0.01)^2 = 0.98 (NO down-weighting)

**Key Insight:** Focal loss DOWN-weights easy examples (model already predicts correctly), letting optimizer focus on hard/wrong predictions!

### Current Implementation Deep-Dive

#### Training Loss (train_step.py:135-189)

```python
# Line 135-141: Setup (focal_alpha, focal_gamma from config)
if loss_mode == "focal":
    logger.info(f"[LOSS] Using focal loss (alpha={focal_alpha}, gamma={focal_gamma})")
else:
    logger.info(f"[LOSS] Using BCE loss (pos_weight={pos_weight_val:.2f})")

# NOTE: This criterion is ONLY used if loss_mode != "focal"
pos_weight_tensor = torch.tensor([pos_weight_val], device=device_obj)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

# Line 179-189: Actual loss computation
if loss_mode == "focal":  # ← TRUE in both modal and local configs
    # Custom focal loss (bypasses criterion entirely!)
    probs = torch.sigmoid(logits)
    pt = labels * probs + (1 - labels) * (1 - probs)
    at = labels * focal_alpha + (1 - labels) * (1 - focal_alpha)  # 0.5 always
    focal_weight = at * ((1 - pt) ** focal_gamma)  # Hard-example focus
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    loss = (focal_weight * bce).mean()
else:
    # Weighted BCE (NOT used in our configs!)
    loss = criterion(logits, labels)
```

**Configs verify focal loss is active:**
```yaml
# configs/modal/train.yaml:117-119
training:
  loss: focal              # ← Sets loss_mode="focal"
  focal_alpha: 0.5         # ← Neutral class weighting
  focal_gamma: 2.0         # ← Hard-example focusing

# configs/local/train.yaml:149-151
training:
  loss: focal
  focal_alpha: 0.5
  focal_gamma: 2.0
```

#### Validation Loss (val_step.py:239)

```python
# Plain BCE, no focal focusing
criterion = nn.BCEWithLogitsLoss()

# In loop (line 298):
loss = criterion(logits, labels)  # Plain average BCE
```

### The REAL Mismatch

| Aspect | Training (Focal) | Validation (Plain BCE) | Consequence |
|--------|------------------|------------------------|-------------|
| **Loss type** | Focal loss | Standard BCE | Different formulas |
| **Class weighting** | None (α=0.5) | None | Actually SAME! ✅ |
| **Hard-example focus** | YES (γ=2.0, up to 100x) | NO (equal weighting) | NOT comparable! ❌ |
| **Easy-example down-weight** | YES (0.0025x for p_t=0.95) | NO | NOT comparable! ❌ |
| **What it measures** | "Hard-example error" | "Average error" | Different semantics! |

**Example:** Suppose model has:
- 1000 easy examples (p_t > 0.9): Each contributes ~0.01 × BCE to focal loss
- 10 hard examples (p_t < 0.5): Each contributes ~0.50 × BCE to focal loss

**Result:**
- Focal loss: Dominated by hard examples (10 × 0.5 >> 1000 × 0.01)
- Plain BCE: Dominated by easy examples (1000 × 1.0 >> 10 × 1.0)
- **NOT COMPARABLE!**

---

## Why This Doesn't Affect Current Training

### Early Stopping & Model Selection (loop.py:255-280)

```python
# Line 255-258: Get metric from config
metric_name = config.training.early_stopping.metric
# ↑ "sensitivity_at_10fa" in both configs

current_metric = val_metrics[metric_name]
# ↑ val_metrics["sensitivity_at_10fa"], NOT val_metrics["val_loss"]!

# Line 258-263: Early stopping decision
if early_stopping(current_metric, epoch):
    # ↑ Compares SENSITIVITY, not loss!
    logger.info(f"Early stopping triggered")
    break

# Line 263-280: Save best model
if current_metric == early_stopping.best_score:
    # ↑ Compares SENSITIVITY, not loss!
    save_checkpoint(..., checkpoint_dir / CHECKPOINT_BEST)
    logger.info(f"New best {metric_name}: {current_metric:.4f}")
```

**Configs confirm sensitivity-based selection:**
```yaml
# configs/modal/train.yaml:143-145
early_stopping:
  patience: 5
  metric: sensitivity_at_10fa  # ← NOT val_loss!

# configs/local/train.yaml:183-185
early_stopping:
  patience: 5
  metric: sensitivity_at_10fa  # ← NOT val_loss!
```

**Conclusion:** Loss is logged to W&B but **NOT used** for training decisions!

---

## Impact Analysis

### ✅ UNAFFECTED (Training Correctness)

1. **Early stopping** - Uses `sensitivity_at_10fa`, not loss
2. **Best model selection** - Uses `sensitivity_at_10fa`, not loss
3. **Gradient updates** - Training focal loss is correct for hard-example mining
4. **Final model quality** - Selected by sensitivity (clinical metric!)

### 🟡 AFFECTED (Interpretability)

1. **Loss curves in W&B** - Train (focal) vs val (BCE) not comparable
2. **Overfitting diagnosis** - Can't use loss gap to detect overfitting
3. **Hard-example tracking** - Can't measure hard-example performance separately
4. **Future loss-based experiments** - Would need fix before switching to loss-based early stopping

### Example Misleading Scenario

**W&B shows:**
- Epoch 50: Train loss = 0.15, Val loss = 0.08
- **Wrong conclusion:** "Val loss lower, model generalizes well!"

**Reality:**
- Train focal loss = 0.15 (dominated by hard examples with high weight)
- Val plain BCE = 0.08 (average across all examples)
- If we compute val FOCAL loss, might be 0.12 (suggesting mild overfitting)
- But with α=0.5, no class weighting, so actually comparable on easy/hard dimension!

---

## Solution Options

### Option A: Mirror Focal Loss in Validation
```python
# val_step.py
def validate_epoch(...,
                   focal_alpha: float | None = None,
                   focal_gamma: float | None = None):
    if focal_alpha is not None and focal_gamma is not None:
        # Use focal loss (matches training)
        ...
    else:
        # Use plain BCE (original behavior)
        criterion = nn.BCEWithLogitsLoss()
```

**Pros:**
- ✅ Train/val losses directly comparable
- ✅ Same hard-example semantics

**Cons:**
- ❌ Val loss no longer represents "real-world" average error
- ❌ Loses ability to measure plain BCE performance

### Option B: Report Both Focal and Plain BCE Validation Losses ⭐ **RECOMMENDED**
```python
# val_step.py
def validate_epoch(...,
                   focal_alpha: float | None = None,
                   focal_gamma: float | None = None):
    # Plain BCE criterion (current behavior)
    criterion_bce = nn.BCEWithLogitsLoss()

    # Focal loss criterion (matches training if provided)
    use_focal = focal_alpha is not None and focal_gamma is not None

    total_loss_bce = 0.0
    total_loss_focal = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            logits = model(windows)
            labels = batch["label"]

            # Plain BCE loss (always compute)
            loss_bce = criterion_bce(logits, labels)
            total_loss_bce += loss_bce.item()

            # Focal loss (if requested)
            if use_focal:
                probs = torch.sigmoid(logits)
                pt = labels * probs + (1 - labels) * (1 - probs)
                at = labels * focal_alpha + (1 - labels) * (1 - focal_alpha)
                focal_weight = at * ((1 - pt) ** focal_gamma)
                bce = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, labels, reduction="none"
                )
                loss_focal = (focal_weight * bce).mean()
                total_loss_focal += loss_focal.item()

            num_batches += 1

    # Return both metrics
    metrics["val_loss"] = total_loss_bce / max(1, num_batches)
    if use_focal:
        metrics["val_loss_focal"] = total_loss_focal / max(1, num_batches)

    return metrics
```

**Pros:**
- ✅ Full visibility into both perspectives
- ✅ `val_loss` = real-world average performance (standard metric)
- ✅ `val_loss_focal` = directly comparable to train loss (hard-example focus)
- ✅ No information loss
- ✅ Can track both overfitting dimensions (average + hard-example)

**Cons:**
- ⚠️ Slightly more computation (2 loss calculations per batch)
- ⚠️ More logging/storage (negligible)

---

## Recommended Implementation Plan

### Phase 1: After Current Training (v3.7.0)

1. **Implement Option B** in `val_step.py`
2. **Update loop.py** to pass `focal_alpha` and `focal_gamma` from config
3. **Update W&B logging** to report both `val_loss` and `val_loss_focal`
4. **Add unit tests** for both loss modes
5. **Update CLAUDE.md** to document both metrics

### Phase 2: Next Training Run

1. **Monitor both curves** in W&B
2. **Verify focal loss tracks train loss** (should be parallel)
3. **Compare to plain BCE** (shows if hard examples overfitting vs average)

---

## Code Changes Required

### File 1: `src/brain_brr/train/val_step.py`

**Changes:**
1. Add `focal_alpha` and `focal_gamma` optional parameters (line ~218)
2. Implement dual-loss computation (see Option B pseudocode above)
3. Return both `val_loss` (BCE) and `val_loss_focal` (if focal params provided)

**Estimated lines changed:** ~30 lines

### File 2: `src/brain_brr/train/loop.py`

**Changes:**
1. Extract `focal_alpha` and `focal_gamma` from config (line ~195)
2. Pass to `validate_epoch()` call
3. Update W&B logging to handle optional `val_loss_focal`

**Example:**
```python
# Line ~195 (after training)
val_metrics = validate_epoch(
    model,
    val_loader,
    config.postprocessing,
    device=device,
    focal_alpha=getattr(config.training, "focal_alpha", None),
    focal_gamma=getattr(config.training, "focal_gamma", None),
)

# Line ~202 (W&B logging)
wandb_logger.log_metrics({
    "val_loss": val_metrics["val_loss"],
    "val_loss_focal": val_metrics.get("val_loss_focal", None),  # Optional
    ...
})
```

**Estimated lines changed:** ~5 lines

### File 3: `tests/unit/train/test_val_loss_focal.py` (NEW)

**New test file:**
```python
def test_validation_loss_focal_vs_bce():
    """Verify focal loss differs from plain BCE on hard examples."""
    # Setup model, create hard examples (p_t ~ 0.5)
    # Run validation with focal_alpha=0.5, focal_gamma=2.0
    # Assert: val_loss_focal ≈ val_loss (α=0.5 is neutral on class weight)
    # BUT verify focal loss down-weights easy examples

def test_validation_loss_backward_compat():
    """Verify focal_alpha=None preserves original behavior."""
    # Run validation without focal params
    # Assert: only val_loss returned, no val_loss_focal
```

---

## Testing Plan

### Unit Tests
1. ✅ Focal vs BCE loss comparison
2. ✅ Backward compatibility (None params)
3. ✅ Hard-example down-weighting verification

### Integration Tests
- ✅ No changes needed (early stopping uses sensitivity)

### Manual Validation (Smoke Test)
1. Run smoke test with Option B implementation
2. Verify both `val_loss` and `val_loss_focal` logged to W&B
3. Confirm focal loss ~= BCE loss (since α=0.5 is neutral)
4. Verify easy examples have lower focal weight

---

## Why Current Training is SAFE

### Decision Matrix

| Training Aspect | Uses Loss? | Safe? | Why? |
|----------------|------------|-------|------|
| **Early stopping** | ❌ NO (uses sensitivity) | ✅ YES | Metric unaffected by loss mismatch |
| **Best model selection** | ❌ NO (uses sensitivity) | ✅ YES | Saves best sensitivity model |
| **Gradient updates** | ✅ YES (focal loss) | ✅ YES | Focal loss is CORRECT for hard-example mining |
| **Overfitting diagnosis** | ✅ YES (loss gap) | 🟡 LIMITED | Can't use loss gap (misleading) |
| **W&B monitoring** | ✅ YES (plots) | 🟡 LIMITED | Curves not comparable |

**Conclusion:** Core training loop is unaffected. Only interpretability/debugging impacted.

---

## References

### Code Locations
- **Training focal loss:** `src/brain_brr/train/train_step.py:179-187`
- **Validation plain BCE:** `src/brain_brr/train/val_step.py:239`
- **Early stopping metric:** `src/brain_brr/train/loop.py:255-280`
- **Modal config:** `configs/modal/train.yaml:117-119,143-145`
- **Local config:** `configs/local/train.yaml:149-151,183-185`

### Related Papers
- **Focal Loss (Lin et al. 2017):** https://arxiv.org/abs/1708.02002
  - "Focal Loss for Dense Object Detection"
  - Used for extreme class imbalance in object detection
  - Key insight: Down-weight easy examples to focus on hard negatives

### PyTorch Documentation
- **BCEWithLogitsLoss:** https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
- **Focal loss implementations:** Many community implementations available

---

## Appendix: Focal Loss Weighting Examples

### Easy Example (Model Confident, Correct)
```
p_t = 0.95 (95% confidence, correct prediction)
focal_weight = (1 - 0.95)^2.0 = 0.0025
→ Loss contribution: 0.0025 × BCE = 400x DOWN-weighted!
```

### Hard Example (Model Unsure)
```
p_t = 0.6 (60% confidence, somewhat correct)
focal_weight = (1 - 0.6)^2.0 = 0.16
→ Loss contribution: 0.16 × BCE = 6x DOWN-weighted
```

### Wrong Prediction (Model Confident, Incorrect)
```
p_t = 0.05 (5% confidence on true class = wrong!)
focal_weight = (1 - 0.05)^2.0 = 0.9025
→ Loss contribution: 0.90 × BCE ≈ NO down-weighting
```

**Key Pattern:** Focal loss focuses optimizer on wrong predictions and hard examples, ignoring easy correct predictions.

---

## Decision Matrix (Corrected)

| Criterion | Option A (Mirror Focal) | Option B (Both) ⭐ | Status Quo |
|-----------|------------------------|-------------------|------------|
| Train/val loss comparable | ✅ Yes (focal) | ✅ Yes (focal) | ❌ No |
| Real-world average metric | ❌ No | ✅ Yes (BCE) | ✅ Yes |
| Hard-example tracking | ✅ Yes | ✅ Yes (focal) | ❌ No |
| Information completeness | ⚠️ Partial | ✅ Full | ⚠️ Partial |
| Code complexity | ⚠️ Medium | ⚠️ Medium+ | ✅ Low |
| Future flexibility | ⚠️ Limited | ✅ High | ❌ None |
| **Recommendation** | Not recommended | **BEST CHOICE** | Defer for now |

---

## Action Items

### Immediate (v3.6.0 - Current Training)
- [x] Correct analysis document (this doc!)
- [x] Archive incorrect analysis
- [x] Verify training uses focal loss (configs confirmed)
- [x] Verify early stopping uses sensitivity (code confirmed)
- [x] Continue training normally ✅

### Post-Training (v3.7.0)
- [ ] Implement Option B in `val_step.py`
- [ ] Update `loop.py` to pass focal params
- [ ] Add unit tests for dual-loss validation
- [ ] Update W&B logging
- [ ] Document in CLAUDE.md and AGENTS.md
- [ ] Run smoke test to validate

### Long-Term (v4.0+)
- [ ] Consider making focal params required for consistency
- [ ] Add validation to ensure train/val use same loss type
- [ ] Document focal loss best practices for future projects

---

**Last Updated:** 2025-10-03 (Corrected after agent review)
**Supersedes:** P1-VALIDATION-LOSS-WEIGHTING-ANALYSIS.md (incorrect, archived)
**Next Review:** After v3.6.0 Modal training completes
**Owner:** Deferred to post-training (P1, non-blocking)
