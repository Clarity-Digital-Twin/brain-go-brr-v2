# Technical Debt Priority List

**Last Updated:** 2025-10-05 (Comprehensive Audit Complete)
**Status:** ✅ **ZERO P0/P1** · 🟡 3 P2 issues · 4 P3 polish items
**Comprehensive Audit:** See `COMPREHENSIVE_DEBT_AUDIT.md` for full first-principles analysis

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

**COUNT: 3** (see `COMPREHENSIVE_DEBT_AUDIT.md` for details)

1. **P2.1:** Deprecated env vars (`BGB_MID_EPOCH_*`) → Remove in v3.8.0
2. **P2.2:** Flaky performance test → ✅ **FIXED** (WSL2 degradation threshold)
3. **P2.3:** 32 unused constants → Audit in v3.8.0 polish sprint

---

## P3: LOW PRIORITY (Optional Backlog)

**COUNT: 4** (see `COMPREHENSIVE_DEBT_AUDIT.md` for details)

1. **P3.1:** 21 `# type: ignore` comments → Audit and remove where possible
2. **P3.2:** Debug assertions in production → Convert to exceptions (2-3 hours)
3. **P3.3:** 11 `pass` statements → Review for dead code
4. **P3.4:** Documentation refresh → Update diagrams for focal-only, SSOT patterns

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
