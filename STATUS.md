# Brain-Go-Brr v3.7.0 – Current Status

**Last Updated:** 2025-10-05 (Zero-Debt Policy Enforcement)
**Branch:** `main`
**Version:** v3.7.0 (Focal-only production, ZERO-DEBT POLICY IN EFFECT)
**Audit:** See `COMPREHENSIVE_DEBT_AUDIT.md` for complete first-principles analysis

---

## Production Readiness

**🚨 TRAINING BLOCKED - ZERO-DEBT POLICY IN EFFECT 🚨**

- ✅ **P0/P1 (Blockers):** 0 issues - All resolved
- 🔴 **P2 (Medium Priority):** 2 issues - **MUST FIX BEFORE TRAINING**
- 🔴 **P3 (Low Priority):** 4 issues - **MUST FIX BEFORE TRAINING**
- 🟡 **P4-P5 (Optimizations):** 2 issues - DEFER until after training

**Total Effort Required:** ~6.75 hours

**Policy:** NO Modal A100 training until ALL P2/P3 technical debt paid down. Training costs ~$319 for 100 epochs - we MUST have pristine codebase first.

**Recent Fixes:**
- ✅ Focal-only production (BCE mode removed)
- ✅ Performance test flakiness (WSL2 degradation threshold)
- ✅ `make q` green (lint, format, mypy, config validation)

---

## Latest Improvements (v3.7.0)

- **Comprehensive debt audit** - First-principles analysis of entire codebase (63 modules, 516 tests)
- **Performance test fix** - WSL2-aware degradation threshold (2.5× tolerance for thermal noise)
- Centralized constant usage across the training loop, GNN builders, and morphology pipeline
- Rebuilt `CONSTANTS_CONFIGS_REFERENCE.md` to reflect real counts (88 constants, 152 schema fields)
- Updated root documentation (`COMPREHENSIVE_DEBT_AUDIT.md`, `TODO.md`, `STATUS.md`) to match reality

---

## Outstanding Items (MUST FIX BEFORE TRAINING)

| Priority | Item | Effort | Status |
|----------|------|--------|--------|
| **P2** | **Deprecated env vars** (`BGB_MID_EPOCH_*`) | 30 min | 🔴 **REQUIRED** |
| **P2** | **Unused constants** (32/90 = 35.6% dead code) | 1 hour | 🔴 **REQUIRED** |
| **P3** | **Type ignore comments** (21 occurrences) | 2 hours | 🔴 **REQUIRED** |
| **P3** | **Debug assertions** (11 in detector.py) | 1.5 hours | 🔴 **REQUIRED** |
| **P3** | **Pass statements** (9 occurrences) | 45 min | 🔴 **REQUIRED** |
| **P3** | **Documentation accuracy** | 1 hour | 🔴 **REQUIRED** |

**Total:** 6 items, ~6.75 hours

**Completed:**
- ✅ Flaky performance test (WSL2 degradation threshold)
- ✅ BCE mode removed (focal-only production)
- ✅ SSOT constants wired

**Full details:** See `COMPREHENSIVE_DEBT_AUDIT.md` → Implementation Roadmap

---

## Validation Checklist

```bash
python /tmp/complete_audit.py   # verifies constant usage + schema stats
make q                          # lint + format + mypy + config validation
rg 'sample_size: int = 500' src/   # should return nothing
rg 'alpha: float = 0\.05' src/    # should return nothing
```

---

## Quick Facts

- **Training:** BLOCKED until all P2/P3 items fixed (~6.75 hours of work)
- **Target config:** Modal A100, 100 epochs, batch 32, grad accum 2 (~$319 cost)
- **Constant utilization:** 58/90 used (64.4%), 32 cleanup candidates
- **Code metrics:** 63 modules, 21 type ignores, 11 assertions (need conversion), 9 pass statements
- **Schema defaults:** 23 constants imported, 129 literal defaults (expected)
