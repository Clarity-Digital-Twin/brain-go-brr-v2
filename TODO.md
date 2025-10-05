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

### ✅ RESOLVED: BCE Mode Removed (CLOSED 2025-10-05)

**Original Concern:** Validation loss weighting inconsistency between BCE and focal modes.

**Final Action:** **BCE MODE DELETED** – Codebase now focal-only.

**Changes:**
- ✅ Removed BCE criterion from train_step.py and val_step.py
- ✅ Removed pos_weight computation (unused in focal mode)
- ✅ Locked schema to `loss: Literal["focal"]` only
- ✅ Updated tests to use focal mode
- ✅ Validation always uses focal loss (matches training)

**Rationale:** Focal loss is the proven standard for rare event detection (RetinaNet, medical AI). BCE mode was never tested on this dataset and added code complexity with no benefit.

**Result:** Clean, single-path production code. Ready for training.

---

## P2: MEDIUM PRIORITY

- None – all previously tracked items (config validation, FA sweep refactor, metric helper, schema epsilons) completed in v3.7.0

---

## P3: LOW PRIORITY (Optional Backlog)

| Item | Notes | Origin |
|------|-------|--------|
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
- ✅ **BCE mode removed** - Focal-only production, deleted dead code, locked schema
- ✅ SSOT constant wiring (88→90 constants, 58 used, zero hardcoded literals)
- ✅ Test-suite batch size fixture, redundant cleanup fixtures, GPU memory fraction parametrisation
- ✅ Inline import removal, FA sweep refactor, metric key helper adoption, schema epsilons, ECE bin constant
