# Technical Debt Priority List

**Last Updated:** 2025-10-05
**Status:** ✅ **ZERO ACTIVE P1** · 🟡 Optional polish backlog (unused constants/docs)

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

### ✅ RESOLVED: Validation Loss Weighting Under Imbalance (CLOSED 2025-10-05)

**Original Concern:** Training loss uses `pos_weight`; validation loss does not.

**Audit Verdict:** ✅ **NOT A BUG** – Production uses focal loss, not BCE+pos_weight.

**Evidence:**
- ✅ All configs use `loss: focal` (configs/local/train.yaml:148, configs/modal/train.yaml:127)
- ✅ Training uses focal loss (train_step.py:306-317), NOT BCEWithLogitsLoss with pos_weight
- ✅ Validation computes focal loss (val_step.py:290-300) with same alpha/gamma parameters
- ✅ Model selection uses `sensitivity_at_10fa`, not loss (all configs)
- ✅ Both `val_loss` (unweighted BCE reference) and `val_loss_focal` (primary) are logged

**Conclusion:** Issue only affects hypothetical BCE mode (not used). Production is correct.

**See:** `P1_VALIDATION_LOSS_AUDIT.md` for complete first-principles analysis.

---

## P2: MEDIUM PRIORITY

- None – all previously tracked items (config validation, FA sweep refactor, metric helper, schema epsilons) completed in v3.7.0

---

## P3: LOW PRIORITY (Optional Backlog)

| Item | Notes | Origin |
|------|-------|--------|
| Rename `val_loss` → `val_loss_bce` for clarity | Cosmetic monitoring improvement (optional) | P1 audit 2025-10-05 |
| Remove unused constants (labels, metric strings, legacy clamps) | 32 constants identified by `/tmp/complete_audit.py`; schedule for v3.8.0 | Constant audit |
| Refresh architecture/metrics documentation | Update diagrams + docs to highlight `format_sensitivity_key()` and SSOT patterns | Documentation debt |

---

## Verification Commands

```bash
python /tmp/complete_audit.py   # constant usage + schema stats
make q                          # lint + format + mypy + config validation
```

---

## Historical Reference

**v3.7.0 Completed Items:**
- ✅ P1 validation loss weighting audit (closed as "not a bug" – production uses focal loss correctly)
- ✅ SSOT constant wiring (88→90 constants, 56→58 used, zero hardcoded literals)
- ✅ Test-suite batch size fixture, redundant cleanup fixtures, GPU memory fraction parametrisation
- ✅ Inline import removal, FA sweep refactor, metric key helper adoption, schema epsilons, ECE bin constant
