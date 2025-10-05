# P1 Remediation Plan – Validation Loss Weighting Parity

**Owner:** Open slot (assign when picked up)  
**Last Updated:** 2025-10-05  
**Status:** Not started  
**Priority:** P1 (non-blocking, but recommended for interpretability)

---

## 1. Summary
- **Problem:** Validation loss is computed without class weighting, while training loss uses `pos_weight`. This makes train/val losses incomparable and can mislead model selection logic that relies on loss curves.
- **Goal:** Ensure validation reports both a weighted and unweighted loss so we can compare apples-to-apples while retaining the ability to inspect raw metrics.

---

## 2. Current Behaviour
| Stage | File | Loss Definition |
|-------|------|-----------------|
| Training | `src/brain_brr/train/train_step.py` (around line 185) | `BCEWithLogitsLoss(pos_weight=pos_weight_tensor)` |
| Validation | `src/brain_brr/train/val_step.py` (around line 235) | `torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)` (no `pos_weight`) |

During training we compute `pos_weight` from the dataset sample ratio each epoch. Validation ignores this weight, so reported losses look much lower than the weighted training loss even if behaviour is unchanged.

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

## 9. References
- Earlier TODO entry (validation loss weighting parity)  
- Forensic audit script output: `/tmp/complete_audit.py`

