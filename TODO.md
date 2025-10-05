# Technical Debt Priority List

**Last Updated:** 2025-10-05  
**Status:** 🔴 1 active P1 (non-blocking) · 🟡 Optional polish backlog (unused constants/docs)

Key reminder from earlier investigation: `BGB_SANITIZE_GRADS=1` remains informational; gradient sanitisation relies on clipping + logging.

---

## Priority Legend

- **P0 (BLOCKER):** must fix before production deployment
- **P1 (HIGH):** significant impact on maintainability/velocity
- **P2 (MEDIUM):** should fix but not urgent
- **P3 (LOW):** nice to have / future polish
- **✅ RESOLVED:** completed items (kept for reference)

---

## P0: BLOCKERS

- None – production ready

---

## P1: HIGH PRIORITY

### 🔴 Validation Loss Weighting Under Imbalance (OPEN, non-blocking)
- Training loss uses `pos_weight`; validation loss does not (`train/val_step.py`)
- Impact: affects interpretability of train vs. val loss; no correctness impact
- Options: mirror `pos_weight` in validation or report both metrics
- Status: plan documented in `P1_VALIDATION_LOSS_PLAN.md` (defer execution until after current training run completes)

---

## P2: MEDIUM PRIORITY

- None – all previously tracked items (config validation, FA sweep refactor, metric helper, schema epsilons) completed in v3.7.0

---

## P3: LOW PRIORITY (Optional Backlog)

| Item | Notes |
|------|-------|
| Remove unused constants (labels, metric strings, legacy clamps) | 32 constants identified by `/tmp/complete_audit.py`; schedule for v3.8.0 |
| Refresh architecture/metrics documentation | Update diagrams + docs to highlight `format_sensitivity_key()` and SSOT patterns |

---

## Verification Commands

```bash
python /tmp/complete_audit.py   # constant usage + schema stats
make q                          # lint + format + mypy + config validation
```

---

## Historical Reference

Resolved items (kept for context): test-suite batch size fixture, redundant cleanup fixtures, GPU memory fraction parametrisation,
inline import removal, FA sweep refactor, metric key helper adoption, schema epsilons, ECE bin constant, constant wiring.
