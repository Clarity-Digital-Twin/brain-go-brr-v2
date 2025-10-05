# Brain-Go-Brr v3.7.0 – Current Status

**Last Updated:** 2025-10-05  
**Branch:** `main`  
**Version:** v3.7.0 (SSOT constant wiring)

---

## Production Readiness

- ✅ All blocking debt resolved (P0/P1 closed) and all critical constants wired
- ✅ `make q` green (lint, format, mypy, config validation)
- ✅ Training stable on RTX 4090 + Modal A100 with current configs (no runs active right now)
- ✅ P1 validation loss audit complete – no bugs found (production uses focal loss correctly)
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
| ~~P1~~ CLOSED | ~~Validation loss weighting parity~~ | ✅ Not a bug | Production uses focal loss in train+val. See `P1_VALIDATION_LOSS_AUDIT.md` |
| P3 (optional) | Rename val_loss → val_loss_bce for clarity | Open | Cosmetic monitoring improvement (not required) |
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
