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

## Outstanding Items

| Priority | Item | Status | Notes |
|----------|------|--------|-------|
| ~~P1~~ CLOSED | ~~Validation loss / BCE mode~~ | ✅ Removed | Focal-only production (v3.7.0) - BCE mode deleted |
| P2 | Deprecated env vars (`BGB_MID_EPOCH_*`) | Open | Remove in v3.8.0 (30 min) |
| P2 | Flaky performance test | ✅ **FIXED** | WSL2 degradation threshold (2.5× multiplier) |
| P2 | 32 unused constants | Open | Audit in v3.8.0 polish sprint (1 hour) |
| P3 | 21 type ignore comments | Open | Audit and remove where possible (2 hours) |
| P3 | Debug assertions in production | Open | Convert to exceptions (2-3 hours) |
| P3 | Documentation refresh | Open | Post-training sprint (4-6 hours) |

**Full details:** See `COMPREHENSIVE_DEBT_AUDIT.md`

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

- Training configuration unchanged: Modal run still targeting 100 epochs (batch 32, grad accum 2)
- Constant utilization: 56/88 used (63.6%), 32 optional constants remain
- Schema defaults: 23 constants imported, 129 literal defaults (expected)
