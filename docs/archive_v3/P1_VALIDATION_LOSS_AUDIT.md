# P1 Validation Loss Weighting – Complete First-Principles Audit

**Auditor:** Claude Code
**Date:** 2025-10-05
**Status:** ✅ AUDIT COMPLETE
**Verdict:** **NOT A BUG IN PRODUCTION** – Issue only affects unused BCE mode

---

## Executive Summary

**TL;DR:** The P1 document is **technically correct about the code** but **misleading about production impact**. Production uses focal loss in both training and validation, making the pos_weight discrepancy irrelevant. Early stopping uses `sensitivity_at_10fa`, not loss. **Recommendation: Downgrade to P3 or close.**

---

## 1. What the P1 Document Claims

From `P1_VALIDATION_LOSS_PLAN.md`:

> **Problem:** Validation loss is computed without class weighting, while training loss uses `pos_weight`. This makes train/val losses incomparable and can mislead model selection logic that relies on loss curves.

**Claimed Impact:**
- Training uses `BCEWithLogitsLoss(pos_weight=pos_weight_tensor)`
- Validation uses `torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)` (no `pos_weight`)
- Result: Train/val losses incomparable, model selection misleading

---

## 2. First-Principles Code Analysis

### 2.1 Production Configuration (ALL Configs)

```yaml
# configs/local/train.yaml:148
# configs/local/smoke.yaml:126
# configs/modal/train.yaml:127
# configs/modal/smoke.yaml:124
training:
  loss: focal  # ALL configs use focal loss
```

**Early Stopping Config:**
```yaml
early_stopping:
  patience: 5
  metric: sensitivity_at_10fa  # NOT loss!
```

### 2.2 Training Loss Computation (`train_step.py`)

**Lines 254-266: pos_weight computation**
```python
pos_weight_val = (1.0 - pos_ratio) / max(pos_ratio, EPSILON_NUMERICAL)
pos_weight_val = float(min(pos_weight_val, 20.0))
logger.info(f"[DATASET] Positive weight for loss: {pos_weight_val:.2f}")

pos_weight_tensor = torch.tensor([pos_weight_val], device=device_obj)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
```

**Lines 306-327: Actual loss used**
```python
if loss_mode == "focal":
    # PRODUCTION PATH (all configs use focal)
    probs = torch.sigmoid(logits)
    pt = labels * probs + (1 - labels) * (1 - probs)
    at = labels * focal_alpha + (1 - labels) * (1 - focal_alpha)
    current_gamma = get_focal_gamma(global_step, warmup_schedule, target_gamma=focal_gamma)
    focal_weight = at * ((1 - pt) ** current_gamma)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    loss = (focal_weight * bce).mean()  # Uses focal_alpha/gamma, NOT pos_weight
else:
    # UNUSED IN PRODUCTION
    loss = criterion(logits, labels)  # Would use pos_weight
```

**Finding:** Training computes `pos_weight` but **NEVER USES IT** in production (focal mode).

### 2.3 Validation Loss Computation (`val_step.py`)

**Lines 223: BCE criterion (no pos_weight)**
```python
criterion = nn.BCEWithLogitsLoss()  # No pos_weight parameter
```

**Lines 287-300: Actual losses computed**
```python
loss_bce = criterion(logits, labels)  # Unweighted BCE
total_loss += loss_bce.item()

if use_focal:  # True when focal_alpha/focal_gamma passed from loop.py
    assert focal_alpha is not None
    assert focal_gamma is not None
    pt = labels * probs + (1 - labels) * (1 - probs)
    at = labels * focal_alpha + (1 - labels) * (1 - focal_alpha)
    focal_weight = at * ((1 - pt) ** focal_gamma)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    loss_focal = (focal_weight * bce).mean()  # Same focal loss as training
    total_loss_focal += loss_focal.item()
```

**Lines 368-371: Metrics returned**
```python
metrics["val_loss"] = total_loss / max(1, num_batches)  # Unweighted BCE

if use_focal:
    metrics["val_loss_focal"] = total_loss_focal / max(1, num_batches)  # Focal loss
```

**Finding:** Validation computes TWO losses:
1. `val_loss` = unweighted BCE (always)
2. `val_loss_focal` = focal loss (when enabled, **which it is in production**)

### 2.4 How Validation is Called (`loop.py:215-224`)

```python
focal_alpha = config.training.focal_alpha if config.training.loss == "focal" else None
focal_gamma = config.training.focal_gamma if config.training.loss == "focal" else None
val_metrics = validate_epoch(
    model,
    val_loader,
    config.postprocessing,
    device=device,
    fa_rates=config.evaluation.fa_rates,
    focal_alpha=focal_alpha,  # Passed when loss=="focal"
    focal_gamma=focal_gamma,  # Passed when loss=="focal"
    ...
)
```

**Finding:** Production DOES pass `focal_alpha` and `focal_gamma` to validation.

### 2.5 Model Selection Logic (`loop.py:289-304`)

```python
metric_name = config.training.early_stopping.metric  # "sensitivity_at_10fa"

# Check if this is a NEW best
is_new_best = (
    current_metric > early_stopping.best_score
    if early_stopping.mode == "max"
    else current_metric < early_stopping.best_score
)

if early_stopping(current_metric, epoch):
    # Stop training
```

**Finding:** Model selection uses `sensitivity_at_10fa`, **NOT loss**.

---

## 3. What Actually Happens in Production

| Aspect | Training | Validation | Comparable? |
|--------|----------|------------|-------------|
| **Loss Mode** | `focal` | Both BCE + focal computed | ✅ |
| **Loss Logged** | `focal_loss` | `val_loss` (BCE) + `val_loss_focal` | ⚠️ |
| **Loss Parameters** | `focal_alpha=0.5`, `focal_gamma=2.0` | Same parameters when enabled | ✅ |
| **pos_weight** | Computed but unused | Not used at all | ✅ N/A |
| **Model Selection** | N/A | Uses `sensitivity_at_10fa` | ✅ N/A |

**Comparison Matrix:**
- `train_loss` (focal) vs `val_loss` (BCE) → ❌ **Incomparable** (different loss functions)
- `train_loss` (focal) vs `val_loss_focal` (focal) → ✅ **Comparable** (same loss function)
- Model selection → ✅ **Not affected** (uses sensitivity, not loss)

---

## 4. Is This a Bug?

### 4.1 For Production Use (focal loss)

**NO, NOT A BUG:**
- ✅ Training uses focal loss with alpha/gamma
- ✅ Validation computes focal loss with same alpha/gamma
- ✅ Both focal losses are comparable (`train_loss` vs `val_loss_focal`)
- ✅ Model selection uses `sensitivity_at_10fa`, not loss
- ✅ Early stopping unaffected

**Minor Monitoring Issue:**
- ⚠️ Primary logged metric is `val_loss` (BCE) instead of `val_loss_focal`
- ⚠️ W&B/TensorBoard might default to showing `val_loss`, which is inconsistent with `train_loss`
- ⚠️ Users need to manually compare `train_loss` vs `val_loss_focal`

### 4.2 For Hypothetical BCE Mode (loss != "focal")

**YES, WOULD BE A BUG:**
- ❌ Training would use `BCEWithLogitsLoss(pos_weight=~11.5)` for 12:1 imbalance
- ❌ Validation would use `BCEWithLogitsLoss()` (no pos_weight)
- ❌ Losses would be incomparable (weighted vs unweighted)
- ⚠️ Model selection STILL unaffected (uses sensitivity)

**But this scenario doesn't exist in production:**
- ALL configs use `loss: focal`
- No production code path uses BCE mode
- The `criterion` variable is created but never used

---

## 5. Verification Evidence

### 5.1 Code Paths Confirmed

```bash
# All configs use focal loss
$ rg "loss: focal" configs/
configs/local/smoke.yaml:126:  loss: focal
configs/local/train.yaml:148:  loss: focal
configs/modal/smoke.yaml:124:  loss: focal
configs/modal/train.yaml:127:  loss: focal

# Early stopping uses sensitivity, not loss
$ rg "metric: sensitivity" configs/
configs/local/smoke.yaml:136:    metric: sensitivity_at_10fa
configs/local/train.yaml:179:    metric: sensitivity_at_10fa
configs/modal/train.yaml:154:    metric: sensitivity_at_10fa
```

### 5.2 Validation Logging Output

From `val_step.py:372-377`:
```python
if use_focal:
    metrics["val_loss_focal"] = total_loss_focal / max(1, num_batches)
    logger.info(
        f"[VALIDATION] Done! Val Loss (BCE): {metrics['val_loss']:.4f} | "
        f"Val Loss (Focal): {metrics['val_loss_focal']:.4f}"
    )
```

**Both losses ARE computed and logged in production.**

### 5.3 W&B/TensorBoard Logging

From `loop.py:248-250`:
```python
writer.add_scalar("Loss/val", val_metrics["val_loss"], epoch)
if "val_loss_focal" in val_metrics:
    writer.add_scalar("Loss/val_focal", val_metrics["val_loss_focal"], epoch)
```

**Both metrics ARE sent to TensorBoard.**

From `loop.py:262-267`:
```python
wandb_metrics = {
    "val_loss": val_metrics["val_loss"],
    # ...
}
if "val_loss_focal" in val_metrics:
    wandb_metrics["val_loss_focal"] = val_metrics["val_loss_focal"]
```

**Both metrics ARE sent to W&B.**

---

## 6. Root Cause of Confusion

The P1 document was likely written based on:
1. **Incomplete code reading** – saw `pos_weight` computation in training, assumed it was used
2. **Not checking production configs** – didn't verify that `loss: focal` is universal
3. **Not checking validation focal path** – missed that `val_loss_focal` exists
4. **Not checking early stopping** – assumed model selection used loss

**The document is accurate for a hypothetical BCE mode, but irrelevant for production.**

---

## 7. Recommendations

### 7.1 Immediate Actions

**Option A: Close as Not a Bug (RECOMMENDED)**
- Status: ✅ Working as designed
- Rationale: Production uses focal loss consistently
- Action: Update P1 document with "CLOSED - Not applicable to production"

**Option B: Downgrade to P3 Polish**
- Issue: Primary metric name (`val_loss`) is misleading for focal mode
- Fix: Rename `val_loss` → `val_loss_bce` and `val_loss_focal` → `val_loss`
- Impact: Monitoring clarity only (no correctness change)

### 7.2 Documentation Updates

**Update `P1_VALIDATION_LOSS_PLAN.md`:**
```markdown
**Status:** CLOSED – Not applicable to production
**Reason:** Production uses focal loss in both train/val. Issue only affects unused BCE mode.
**Evidence:** All configs use `loss: focal`. Validation computes `val_loss_focal` with same parameters.
**Model Selection:** Uses `sensitivity_at_10fa`, not loss. Early stopping unaffected.
```

**Update `STATUS.md`:**
```markdown
| P1 (closed) | Validation loss weighting | Closed | Not applicable – focal loss used in production |
```

**Update `TODO.md`:**
```markdown
### ✅ RESOLVED: Validation Loss Weighting (was P1, now closed)
- Investigation: Validation DOES use focal loss in production
- Evidence: `val_loss_focal` computed with same alpha/gamma as training
- Model selection uses sensitivity, not loss
- Closed as not applicable
```

### 7.3 Optional Enhancement (P3)

If you want perfect monitoring clarity:

**File:** `src/brain_brr/train/val_step.py:368-371`
```python
# CURRENT (confusing naming)
metrics["val_loss"] = total_loss / max(1, num_batches)  # Unweighted BCE
if use_focal:
    metrics["val_loss_focal"] = total_loss_focal / max(1, num_batches)

# PROPOSED (clear naming)
metrics["val_loss_bce"] = total_loss / max(1, num_batches)  # Unweighted BCE (reference)
if use_focal:
    metrics["val_loss"] = total_loss_focal / max(1, num_batches)  # Primary metric
    metrics["val_loss_focal"] = metrics["val_loss"]  # Alias for compatibility
```

**Impact:** Makes `val_loss` match `train_loss` (both focal). Low priority.

---

## 8. Conclusion

**Verdict:** ✅ **NOT A BUG IN PRODUCTION**

**Summary:**
- P1 document is technically accurate about **code structure** (BCE mode would have mismatch)
- P1 document is **wrong about production impact** (focal mode is used, not BCE)
- Validation DOES compute focal loss with correct parameters
- Model selection uses sensitivity, not loss
- Issue is a **monitoring cosmetic** at worst (primary metric name)

**Recommended Action:**
1. Close P1 as "not applicable to production"
2. Update STATUS.md, TODO.md, P1_VALIDATION_LOSS_PLAN.md
3. Optional: Add P3 polish item to rename metrics for clarity

**No code fixes required for correctness.**

---

## 9. Appendix: Full Evidence Trail

### Code References

| File | Lines | Evidence |
|------|-------|----------|
| `configs/*.yaml` | ALL | `loss: focal` in all configs |
| `train_step.py` | 306-327 | Training uses focal loss, not pos_weight criterion |
| `val_step.py` | 287-300 | Validation computes both BCE and focal loss |
| `val_step.py` | 368-371 | Both losses returned in metrics dict |
| `loop.py` | 215-224 | focal_alpha/gamma passed to validation |
| `loop.py` | 248-250, 262-267 | Both losses logged to W&B/TensorBoard |
| `loop.py` | 289-304 | Early stopping uses `sensitivity_at_10fa` |
| `schemas.py` | 414-423 | EarlyStopping config uses metric, not loss |

### Grep Verification Commands

```bash
# All configs use focal
rg "loss: focal" configs/

# Early stopping uses sensitivity
rg "metric: sensitivity" configs/

# Validation computes focal loss
rg "val_loss_focal" src/brain_brr/train/

# pos_weight only in training, not used in focal mode
rg "pos_weight" src/brain_brr/train/train_step.py
```

**All evidence confirms: Production works correctly.**
