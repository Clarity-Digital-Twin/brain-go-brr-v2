# Brain-Go-Brr v3.7.0 – Current Status

**Last Updated:** 2025-10-05
**Branch:** `main`
**Version:** v3.7.0 (Focal-only production, zero debt)

---

## Production Readiness

- ✅ **ZERO TECHNICAL DEBT** - All P0/P1 items closed
- ✅ **Focal-only production** - BCE mode removed, focal loss proven standard for rare event detection
- ✅ `make q` green (lint, format, mypy, config validation)
- ✅ Training stable on RTX 4090 + Modal A100 with current configs (no runs active right now)
- ✅ All critical constants wired (SSOT compliance)
- 🟡 Optional cleanup backlog limited to unused constants / docs (tracked in `POLISH_ITEMS.md`)

---

## Latest Improvements (v3.7.0)

- Centralized constant usage across the training loop, GNN builders, and morphology pipeline
- Rebuilt `CONSTANTS_CONFIGS_REFERENCE.md` to reflect real counts (88 constants, 152 schema fields)
- Updated root documentation (`DEBT_STATUS_TRUE.md`, `POLISH_ITEMS.md`, `REMAINING_DEBT_IMPLEMENTATION_GUIDE.md`, `TODO.md`) to match reality

---

## Outstanding Items

| Priority | Item | Status | Notes |
|----------|------|--------|-------|
| ~~P1~~ CLOSED | ~~Validation loss / BCE mode~~ | ✅ Removed | Focal-only production (v3.7.0) - BCE mode deleted |
| P3 (optional) | Remove unused constants (labels, metric strings) | Open | Optional v3.8.0 cleanup |
| P3 (optional) | Refresh diagrams / docs to highlight SSOT patterns | Open | Optional polish |

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
