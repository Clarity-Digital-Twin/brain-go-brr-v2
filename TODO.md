# Technical Debt Priority List

**Last Updated:** 2025-10-05 (Zero-Debt Policy Enforcement)
**Status:** 🔴 **TRAINING BLOCKED** · 2 P2 + 4 P3 items REQUIRED before Modal A100
**Comprehensive Audit:** See `COMPREHENSIVE_DEBT_AUDIT.md` for full first-principles analysis
**Total Effort:** ~6.75 hours

**🚨 ZERO-DEBT POLICY: NO TRAINING UNTIL ALL P2/P3 PAID DOWN 🚨**

Key reminder from earlier investigation: `BGB_SANITIZE_GRADS=1` remains informational; gradient sanitisation relies on clipping + logging.

---

## Priority Legend

- **P0 (BLOCKER):** must fix before production deployment
- **P1 (HIGH):** significant impact on maintainability/velocity
- **P2 (MEDIUM):** **MUST FIX BEFORE TRAINING** (zero-debt policy)
- **P3 (LOW):** **MUST FIX BEFORE TRAINING** (zero-debt policy)
- **P4-P5 (OPTIMIZATION):** DEFER until after training (no premature optimization)
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

## P2: MEDIUM PRIORITY (MUST FIX BEFORE TRAINING)

**COUNT: 2** (total: 1.5 hours)

### P2.1: Deprecated Environment Variables (30 min)
**Location:** `src/brain_brr/utils/env.py:104-128`
- Delete `mid_epoch_minutes()` and `mid_epoch_keep()` functions
- Verify no usage: `rg "BGB_MID_EPOCH_MINUTES|BGB_MID_EPOCH_KEEP" configs/ deploy/`
- **Rationale:** Clean API surface, avoid runtime warnings

### P2.2: Unused Constants Cleanup (1 hour)
**Location:** `src/brain_brr/constants.py` (32/90 unused = 35.6%)
- Audit each of 32 unused constants
- DELETE if truly dead code (e.g., CSV_VERSION_HEADER)
- KEEP with `# OPTIONAL:` if intentional (e.g., LABEL_* for future multi-class)
- WIRE if should be used but isn't (e.g., FOCAL_GAMMA_PRODUCTION)
- **Rationale:** 35.6% dead code violates clean code principles

---

## P3: LOW PRIORITY (MUST FIX BEFORE TRAINING)

**COUNT: 4** (total: 5.25 hours)

### P3.1: Type Ignore Comments Audit (2 hours)
**Location:** Scattered (21 occurrences)
- Review each `# type: ignore`, try to remove
- Add proper type annotations or use `typing.cast()`
- Document remaining ignores with inline explanation
- **Rationale:** Type safety non-negotiable for production ML

### P3.2: Convert Assertions to Exceptions (1.5 hours)
**Location:** `src/brain_brr/models/detector.py` (11 assert statements)
- Convert all assertions to proper exceptions (RuntimeError, ValueError)
- **CRITICAL:** Assertions disabled with `python -O` (Modal may use this)
- **Rationale:** Production code MUST NOT use assertions for data validation

### P3.3: Review Pass Statements (45 min)
**Location:** Various (9 occurrences)
- Audit each `pass` statement
- Abstract classes → OK, add docstring
- Empty except blocks → **CRITICAL:** May hide bugs, add logging or handle properly
- Empty functions → Document intent or remove dead code
- **Rationale:** Silent error handling violates best practices

### P3.4: Documentation Accuracy Verification (1 hour)
**Location:** `docs/`, root-level docs
- Check for BCE references (should be removed)
- Verify focal-only messaging is clear
- Ensure all training commands are accurate (file counts, memory, batch sizes)
- **Rationale:** Inaccurate docs waste time and money during training

---

## P4-P5: DEFER (Post-Training Optimization Only)

**COUNT: 2** (DO NOT IMPLEMENT BEFORE TRAINING)

**🚨 WARNING: Performance optimizations ONLY after training proves baseline 🚨**

1. **P4.1:** Tensor `.item()` calls (7 in train/) → Profile first, optimize ONLY if bottleneck proven
2. **P4.2:** Refactor detector.py (480 lines) → DEFER, current structure is clear

**Philosophy:** Training has been tough. Premature optimization risks bugs for negligible gain. Training stability >>> theoretical performance.

---

## Verification Commands

```bash
# After completing each P2/P3 item:
make q                          # lint + format + mypy + config validation
make test                       # full test suite

# Final verification before training:
python /tmp/complete_audit.py   # verify constant usage
make test-performance           # verify stability
```

---

## Implementation Order (RECOMMENDED)

1. P2.1: Remove deprecated env vars (30 min)
2. P2.2: Audit unused constants (1 hour)
3. P3.2: Convert assertions to exceptions (1.5 hours) - **HIGHEST RISK**
4. P3.1: Audit type ignores (2 hours)
5. P3.3: Review pass statements (45 min)
6. P3.4: Verify documentation (1 hour)
7. **Final verification** (make q + make test)

**Total: ~6.75 hours + verification**

---

## Historical Reference

**v3.7.0 Completed Items:**
- ✅ **BCE mode removed** - Focal-only production, deleted dead code, locked schema
- ✅ **Comprehensive debt audit** - First-principles analysis, zero-debt policy enforcement
- ✅ **Performance test fix** - WSL2 degradation threshold (2.5× tolerance)
- ✅ SSOT constant wiring (90 constants total, 58 used = 64.4%, zero hardcoded literals)
- ✅ Test-suite batch size fixture, redundant cleanup fixtures, GPU memory fraction parametrisation
- ✅ Inline import removal, FA sweep refactor, metric key helper adoption, schema epsilons, ECE bin constant
