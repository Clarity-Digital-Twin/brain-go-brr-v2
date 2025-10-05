# Brain-Go-Brr v3.7.0 – Current Status

**Last Updated:** 2025-10-05 (Comprehensive Audit Complete)
**Branch:** `main`
**Version:** v3.7.0 (Focal-only production, zero P0/P1 debt)
**Audit:** See `COMPREHENSIVE_DEBT_AUDIT.md` for complete first-principles analysis

---

## Production Readiness

- ✅ **ZERO BLOCKING DEBT** - All P0/P1 items closed (comprehensive audit complete)
- ✅ **Focal-only production** - BCE mode removed, focal loss proven standard for rare event detection
- ✅ `make q` green (lint, format, mypy, config validation)
- ✅ Training stable on RTX 4090 + Modal A100 with current configs (no runs active right now)
- ✅ All critical constants wired (SSOT compliance)
- ✅ Performance test flakiness fixed (WSL2 degradation threshold)
- 🟡 3 P2 medium-priority items (deprecated env vars, unused constants) - defer to v3.8.0
- 🟡 4 P3 low-priority polish items (type ignores, docs refresh) - post-training

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
