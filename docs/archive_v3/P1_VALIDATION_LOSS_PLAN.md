# P1 Remediation Plan – Validation Loss Weighting Parity

**Owner:** N/A (issue closed)
**Last Updated:** 2025-10-05
**Status:** ✅ **CLOSED – Not applicable to production**
**Priority:** ~~P1~~ → **CLOSED**

---

## 🚫 CLOSURE NOTICE

**Audit Date:** 2025-10-05
**Audit Report:** `P1_VALIDATION_LOSS_AUDIT.md`
**Verdict:** ✅ **NOT A BUG IN PRODUCTION**

**Reason:** Production uses focal loss in both training and validation with identical parameters (`focal_alpha`, `focal_gamma`). The pos_weight discrepancy described below only affects the unused BCE mode.

**Evidence:**
- ✅ All configs use `loss: focal` (verified: configs/local/train.yaml:148, configs/modal/train.yaml:127)
- ✅ Validation computes `val_loss_focal` using same focal parameters as training (val_step.py:290-300)
- ✅ Model selection uses `sensitivity_at_10fa`, not loss (early_stopping.metric in all configs)
- ✅ Both `val_loss` (unweighted BCE reference) and `val_loss_focal` are logged to W&B/TensorBoard
- ✅ Training loop passes `focal_alpha` and `focal_gamma` to validation when `loss=="focal"` (loop.py:215-224)

**Impact:** None. Issue only applies to hypothetical BCE mode, which is not used in any production config.

**See:** `P1_VALIDATION_LOSS_AUDIT.md` for complete first-principles code analysis.

---

## Original Issue Description (Archived – Not Applicable)

### 1. Summary
- **Problem (INCORRECT FOR PRODUCTION):** Validation loss is computed without class weighting, while training loss uses `pos_weight`. This makes train/val losses incomparable and can mislead model selection logic that relies on loss curves.
- **Reality:** Production uses focal loss, not BCE+pos_weight. Validation computes focal loss. Model selection uses sensitivity.
- **Original Goal:** Ensure validation reports both a weighted and unweighted loss so we can compare apples-to-apples while retaining the ability to inspect raw metrics.

---

### 2. Current Behaviour (ARCHIVED – INACCURATE FOR PRODUCTION)

**⚠️ NOTE:** This section describes the BCE mode path, which is NOT used in production configs.

| Stage | File | Loss Definition | Actually Used? |
|-------|------|-----------------|----------------|
| Training | `src/brain_brr/train/train_step.py` line 266 | `BCEWithLogitsLoss(pos_weight=pos_weight_tensor)` | ❌ NO (focal mode used) |
| Training (actual) | `src/brain_brr/train/train_step.py` line 306-317 | Focal loss with alpha/gamma | ✅ YES |
| Validation | `src/brain_brr/train/val_step.py` line 287 | `criterion(logits, labels)` (no pos_weight) | ✅ YES (as reference) |
| Validation (actual) | `src/brain_brr/train/val_step.py` line 290-300 | Focal loss with alpha/gamma | ✅ YES |

**Original claim:** "During training we compute `pos_weight` from the dataset sample ratio each epoch. Validation ignores this weight, so reported losses look much lower than the weighted training loss even if behaviour is unchanged."

**Actual behavior:** Training computes `pos_weight` but uses focal loss instead. Validation computes both unweighted BCE (as reference) and focal loss (for comparison with training). Model selection uses `sensitivity_at_10fa`, not loss.

---

## 3. Proposed Solution
1. **Compute/Reuse `pos_weight` for validation:**
   - Reuse the same ratio computed in `train/train_step.py::_compute_dataset_stats` (stored as `pos_weight_val`).
   - Pass this value into `val_step.validate_epoch` so validation can produce a weighted loss alongside the existing unweighted loss.
2. **Report both metrics:**
   - Weighted loss (comparable with training loss) for dashboards/early stopping.
   - Unweighted loss (current behaviour) for raw interpretability.

---

## 4. Implementation Plan
| Step | Description | Files |
|------|-------------|-------|
| 1 | Expose `pos_weight` from `train_epoch` result (store in `TrainingEpochResult`) | `src/brain_brr/train/train_step.py`, `src/brain_brr/train/loop.py` |
| 2 | Thread `pos_weight` into `validate_epoch` call | `src/brain_brr/train/loop.py`, `src/brain_brr/train/val_step.py` |
| 3 | Compute weighted validation loss (`BCEWithLogitsLoss(pos_weight=...)`) alongside existing unweighted loss; log both to metrics dict | `src/brain_brr/train/val_step.py` |
| 4 | Update logging/W&B hooks to record new metric keys (e.g. `val_loss_weighted`, `val_loss_unweighted`) | `src/brain_brr/train/loop.py`, `src/brain_brr/train/wandb_integration.py` |
| 5 | Extend tests to assert both metrics are emitted | `tests/unit/train/test_loop.py`, `tests/integration/test_training_smoke.py` |
| 6 | Documentation: note metric changes in `README.md` and release notes | `README.md`, `RELEASE_NOTES.md` |

---

## 5. Definition of Done
- Validation returns two loss values (`loss_weighted`, `loss_unweighted`).
- `train/loop.py` chooses weighted loss for early stopping / best model selection to match training behaviour.
- Tests cover the new metrics (unit + integration).
- W&B / console logging displays both metrics clearly.
- Documentation updated.

---

## 6. Test/Verification Checklist
```bash
make q
pytest tests/unit/train/test_loop.py -k val
pytest tests/integration/test_training_smoke.py -k validation
rg 'val_loss_weighted' -n src/ tests/  # ensure metric names wired consistently
```

Manual sanity check: run a short training/validation cycle (`make s`) and confirm the console/W&B output now includes both validation loss values with sensible magnitudes.

---

## 7. Rollback Plan
If issues occur:
1. Revert the changes in `train/loop.py`, `train/val_step.py`, and tests.
2. Remove the new metric keys from logging/W&B integration.
3. Re-run `make q` to ensure clean state.

---

## 8. Open Questions
- Should we expose weighted/unweighted losses via CLI flags? (Default to both for now.)
- Do downstream scripts (e.g. modal monitoring, dashboards) expect the old metric names? Coordinate with DevOps if necessary.

---

## 9. References (ARCHIVED)
- Earlier TODO entry (validation loss weighting parity)
- Forensic audit script output: `/tmp/complete_audit.py`

---

## 10. Closure Rationale (Added 2025-10-05)

**Complete First-Principles Audit:** See `P1_VALIDATION_LOSS_AUDIT.md`

**Key Findings:**
1. ✅ Production uses `loss: focal` in ALL configs (local/smoke.yaml:126, local/train.yaml:148, modal/smoke.yaml:124, modal/train.yaml:127)
2. ✅ Training uses focal loss (train_step.py:306-317), NOT BCEWithLogitsLoss with pos_weight
3. ✅ Validation computes focal loss (val_step.py:290-300) with same alpha/gamma parameters
4. ✅ Both `val_loss` (BCE reference) and `val_loss_focal` (primary) are logged
5. ✅ Early stopping uses `metric: sensitivity_at_10fa` in ALL configs, NOT loss
6. ✅ Model selection unaffected by loss metric choice

**Why This Was Flagged:**
- The code DOES have a pos_weight path in training (line 266)
- But it's only used when `loss_mode != "focal"`
- Production configs ALL use `loss: focal`, so the pos_weight path is dead code
- The issue would only affect hypothetical BCE mode, which doesn't exist

**Recommendation:** Close as "not applicable". No code changes needed.

**Optional P3 Enhancement:** Rename `val_loss` → `val_loss_bce` and `val_loss_focal` → `val_loss` for monitoring clarity. Low priority cosmetic change.

