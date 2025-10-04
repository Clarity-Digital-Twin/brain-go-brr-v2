# P1: Validation Loss Weighting Analysis

**Status:** 🟡 OPEN (Non-blocking, safe to defer)
**Priority:** P1 (affects interpretability, not correctness)
**Date:** 2025-10-03
**Author:** Comprehensive codebase audit

---

## Executive Summary

**Issue:** Training loss uses `pos_weight` for class imbalance weighting, validation loss does not. This makes train/val loss curves not directly comparable.

**Impact:** Does NOT affect current training because we use `sensitivity_at_10fa` for early stopping and best model selection (not loss). However, it affects interpretability and future experiments.

**Recommendation:** Implement Option B (report both weighted + unweighted validation loss) after current Modal training completes.

**Training Status:** ✅ **SAFE TO CONTINUE** - Current 100-epoch Modal training (App: ap-BwyQN1PX1prmfzbWGlUDqS) is NOT affected.

---

## Technical Details

### Current Implementation

#### Training Loss (train_step.py:129-141)
```python
# Calculate pos_weight from dataset statistics
pos_weight_val = (1.0 - pos_ratio) / max(pos_ratio, EPSILON_NUMERICAL)
pos_weight_val = float(min(pos_weight_val, 20.0))  # Cap at 20.0
logger.info(f"[DATASET] Positive weight for loss: {pos_weight_val:.2f}")

# Create weighted BCE criterion
pos_weight_tensor = torch.tensor([pos_weight_val], device=device_obj)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)  # ✅ HAS pos_weight
```

**Expected pos_weight value:** ~11-12 (for 12:1 background:seizure ratio)

#### Validation Loss (val_step.py:239)
```python
# Create UNWEIGHTED BCE criterion
criterion = nn.BCEWithLogitsLoss()  # ❌ NO pos_weight!
```

### Why This Matters

**pos_weight** multiplies the loss for positive examples (seizures) by a factor:
- Training: `loss_seizure = 11.0 × BCE(logit, label)`
- Validation: `loss_seizure = 1.0 × BCE(logit, label)`

This means:
1. **Training loss** measures "imbalance-aware error" (seizures weighted 11x higher)
2. **Validation loss** measures "average error" (all samples equal)
3. **Not directly comparable** - different scales, different semantics

### Impact Analysis

#### ✅ SAFE (Not Affected)

1. **Early Stopping** - Uses `sensitivity_at_10fa` (configs/{modal,local}/train.yaml:145,185)
2. **Best Model Selection** - Uses `sensitivity_at_10fa` (loop.py:255-280)
3. **Final Metrics** - AUROC, TAES, PR-AUC all unaffected
4. **Model Training** - Gradient flow and optimization unaffected

#### 🟡 AFFECTED (Interpretability Issues)

1. **Loss Curves** - Train/val loss plots misleading in W&B/TensorBoard
2. **Convergence Analysis** - Can't use loss gap to diagnose overfitting
3. **Future Experiments** - If we want loss-based early stopping, need fix
4. **Debugging** - Harder to interpret loss behavior during training

### Example Scenario

Suppose at epoch 50:
- Train loss (weighted): 0.15
- Val loss (unweighted): 0.08

**Wrong conclusion:** "Val loss is lower, model generalizes well!"
**Reality:** Train loss is artificially inflated by 11x seizure weight. Actual unweighted train loss might be 0.05, suggesting overfitting.

---

## Solution Options

### Option A: Mirror pos_weight in Validation
```python
# val_step.py
def validate_epoch(..., pos_weight: float | None = None):
    if pos_weight is not None:
        pos_weight_tensor = torch.tensor([pos_weight], device=device_obj)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        criterion = nn.BCEWithLogitsLoss()
```

**Pros:**
- ✅ Train/val losses directly comparable
- ✅ Same metric semantics (imbalance-aware)
- ✅ Minimal code change

**Cons:**
- ❌ Val loss no longer represents "real-world" average error
- ❌ Loses ability to measure unweighted generalization

### Option B: Report Both Weighted + Unweighted Validation Loss ⭐ **RECOMMENDED**
```python
# val_step.py
def validate_epoch(..., pos_weight: float | None = None):
    # Unweighted criterion (current behavior)
    criterion_unweighted = nn.BCEWithLogitsLoss()

    # Weighted criterion (matches training)
    if pos_weight is not None:
        pos_weight_tensor = torch.tensor([pos_weight], device=device_obj)
        criterion_weighted = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # ... in validation loop ...
    loss_unweighted = criterion_unweighted(logits, labels)
    loss_weighted = criterion_weighted(logits, labels) if pos_weight else loss_unweighted

    # Return both
    metrics["val_loss"] = loss_unweighted.item()         # Real-world performance
    metrics["val_loss_weighted"] = loss_weighted.item()  # Comparable to train
```

**Pros:**
- ✅ Full visibility into both perspectives
- ✅ Unweighted loss = real-world performance metric
- ✅ Weighted loss = directly comparable to train loss
- ✅ No information loss
- ✅ Flexible for future experiments

**Cons:**
- ⚠️ Slightly more complex (but well worth it)
- ⚠️ More logging/storage (negligible impact)

---

## Recommended Implementation Plan

### Phase 1: After Current Training Completes (Post-v3.6.0)

1. **Implement Option B** in `val_step.py`
2. **Update W&B logging** to report both `val_loss` and `val_loss_weighted`
3. **Update early stopping config** to clarify which metric is used
4. **Add unit tests** for both loss modes
5. **Document in CLAUDE.md** why we report both

### Phase 2: During Next Training Run (v3.7.0+)

1. **Monitor both loss curves** in W&B
2. **Compare interpretability** - verify weighted loss tracks train loss
3. **Validate real-world metric** - verify unweighted loss represents actual generalization

### Code Changes Required

**File:** `src/brain_brr/train/val_step.py`

```python
# Add pos_weight parameter (line 218)
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    post_config: PostprocessingConfig,
    device: str = "cpu",
    fa_rates: list[float] | None = None,
    pos_weight: float | None = None,  # NEW: Optional class weighting
) -> dict[str, Any]:
    """Validate model with true streaming per-recording processing (low memory).

    Args:
        ...
        pos_weight: Optional positive class weight (if None, uses unweighted loss only)
    """
    ...

    # Create both criteria (line 239)
    criterion_unweighted = nn.BCEWithLogitsLoss()
    criterion_weighted = None
    if pos_weight is not None:
        pos_weight_tensor = torch.tensor([pos_weight], device=device_obj)
        criterion_weighted = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    total_loss_unweighted = 0.0
    total_loss_weighted = 0.0
    ...

    # In validation loop (line 298)
    loss_unweighted = criterion_unweighted(logits, labels)
    total_loss_unweighted += loss_unweighted.item()

    if criterion_weighted is not None:
        loss_weighted = criterion_weighted(logits, labels)
        total_loss_weighted += loss_weighted.item()

    # Return both metrics (line 369)
    metrics["val_loss"] = total_loss_unweighted / max(1, num_batches)
    if criterion_weighted is not None:
        metrics["val_loss_weighted"] = total_loss_weighted / max(1, num_batches)
```

**File:** `src/brain_brr/train/loop.py`

```python
# Pass pos_weight from train to validation (line 195)
# Need to extract pos_weight from train_epoch and pass to validate_epoch
# (Requires refactoring train_epoch to return pos_weight value)

val_metrics = validate_epoch(
    model,
    val_loader,
    config.postprocessing,
    device=device,
    pos_weight=train_pos_weight,  # NEW: Pass from training
)
```

**Estimated Work:** ~2 hours (implementation + tests)

---

## Why NOT a Blocker

### Current Training Configuration
```yaml
# configs/modal/train.yaml:143-145
early_stopping:
  patience: 5
  metric: sensitivity_at_10fa  # ← Uses SENSITIVITY, not loss!
```

**Early stopping logic (loop.py:255-280):**
```python
metric_name = config.training.early_stopping.metric  # "sensitivity_at_10fa"
current_metric = val_metrics[metric_name]            # Uses sensitivity
if early_stopping(current_metric, epoch):            # NOT loss!
    ...
if current_metric == early_stopping.best_score:      # Best by sensitivity
    best_metrics = {...}                             # Save best model
```

**Conclusion:** Validation loss is logged but **NOT used** for training decisions. Only affects interpretability, not correctness.

---

## Testing Plan

### Unit Tests (New)
```python
# tests/unit/train/test_val_loss_weighting.py

def test_validation_loss_weighted_vs_unweighted():
    """Verify weighted loss is higher than unweighted for imbalanced data."""
    # Setup: Create model, dataloader with 12:1 imbalance
    # Run validation with pos_weight=11.0
    # Assert: val_loss_weighted > val_loss (seizure errors amplified)

def test_validation_loss_backward_compat():
    """Verify pos_weight=None preserves original behavior."""
    # Run validation without pos_weight
    # Assert: only val_loss returned, no val_loss_weighted
```

### Integration Tests (Existing)
- ✅ No changes needed - early stopping uses sensitivity, unaffected
- ✅ Checkpoint tests use sensitivity, unaffected

---

## References

### Code Locations
- **Training loss with pos_weight:** `src/brain_brr/train/train_step.py:129-141`
- **Validation loss without pos_weight:** `src/brain_brr/train/val_step.py:239`
- **Early stopping metric:** `src/brain_brr/train/loop.py:255-280`
- **Config early stopping:** `configs/{modal,local}/train.yaml:143-145,183-185`

### Related Documentation
- **Bug tracker:** `docs/09-development/bug-tracker.md:33-38` (updated with code refs)
- **TODO.md:** Lines 30-48 (active P1 section)
- **TUSZ class imbalance:** ~12:1 background:seizure ratio
- **PyTorch BCEWithLogitsLoss:** https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

---

## Decision Matrix

| Criterion | Option A (Mirror) | Option B (Both) ⭐ | Status Quo |
|-----------|-------------------|-------------------|------------|
| Train/val loss comparable | ✅ Yes | ✅ Yes (weighted) | ❌ No |
| Real-world metric | ❌ No | ✅ Yes (unweighted) | ✅ Yes |
| Information completeness | ⚠️ Partial | ✅ Full | ⚠️ Partial |
| Code complexity | ✅ Low | ⚠️ Medium | ✅ Low |
| Future flexibility | ⚠️ Limited | ✅ High | ❌ None |
| **Recommendation** | Not recommended | **BEST CHOICE** | Acceptable for now |

---

## Action Items

### Immediate (v3.6.0 - Current Training)
- [x] Document issue in this analysis
- [x] Update bug tracker with severity and impact
- [x] Update TODO.md with active P1
- [x] Verify current training is NOT affected ✅
- [x] Monitor current training normally (sensitivity-based early stopping works)

### Post-Training (v3.7.0)
- [ ] Implement Option B in `val_step.py`
- [ ] Add unit tests for weighted/unweighted validation loss
- [ ] Update W&B logging to track both metrics
- [ ] Document in CLAUDE.md and AGENTS.md
- [ ] Validate on smoke test before next full training

### Long-Term (v4.0+)
- [ ] Consider making pos_weight required (not optional) for all training
- [ ] Add validation to ensure train/val use same weighting scheme
- [ ] Document best practices for imbalanced datasets in new projects

---

## Appendix: Why pos_weight Matters

### Mathematical Explanation

**Standard BCE Loss:**
```
L = - (y × log(p) + (1-y) × log(1-p))
```

**BCE with pos_weight:**
```
L = - (y × pos_weight × log(p) + (1-y) × log(1-p))
      └────── 11x amplification ─────┘
```

**Impact on gradient:**
```python
# For seizure sample (y=1):
grad_standard = -(1/p)              # Standard
grad_weighted = -(11/p)             # With pos_weight=11
# → 11x stronger gradient signal for seizure misclassifications!
```

**Why this helps with imbalance:**
- 12 background samples → 12 × 1.0 = 12.0 loss contribution
- 1 seizure sample → 1 × 11.0 = 11.0 loss contribution
- **Result:** Balanced influence despite 12:1 ratio!

---

**Last Updated:** 2025-10-03
**Next Review:** After v3.6.0 Modal training completes
**Owner:** Deferred to post-training (P1, non-blocking)
