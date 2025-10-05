# Technical Debt Priority List

**Last Updated:** 2025-10-05 (13:45 UTC)
**Status:** 🟢 **ZERO TECHNICAL DEBT** · Training approved
**Comprehensive Audit:** `COMPREHENSIVE_DEBT_AUDIT.md` (SSOT)

Key reminder: keep `BGB_SANITIZE_GRADS=1` informational; gradient sanitisation relies on clipping + logging.

---

## Priority Legend

- **P0 (BLOCKER):** must fix before production
- **P1 (HIGH):** significant maintainability impact
- **P2 (MEDIUM):** none outstanding (policy satisfied)
- **P3 (LOW):** none outstanding (policy satisfied)
- **P4-P5 (OPTIMISATIONS):** optional, post-training only
- **✅ RESOLVED:** historical reference of recent fixes

---

## Open Items

🟢 None. Codebase is clean. Maintain the zero-debt bar for future work.

Optional ideas (post-training):
1. Profile `.item()` calls (train/) after a successful run—optimise only if >1% wall-clock in GPU sync.
2. Consider modularising `models/detector.py` if future features make it unwieldy.

---

## Recently Completed

| Item | Details | Status |
|------|---------|--------|
| Deprecated env vars | Removed `mid_epoch_minutes()` / `mid_epoch_keep()` | ✅ DONE |
| Unused constants | Deleted 6 dead constants, documented 26 reserves | ✅ DONE |
| Type ignore audit | 21 → 17 (all remaining documented) | ✅ DONE |
| Assertions → exceptions | All 11 in `models/detector.py` converted | ✅ DONE |
| Pass statements | All 9 occurrences reviewed and annotated | ✅ DONE |
| Documentation | CLAUDE.md + guides synced (focal-only, batch sizes, memory) | ✅ DONE |
| Full verification | `make q`, `make test`, `make test-performance`, `/tmp/complete_audit.py` | ✅ DONE |

---

## Historical Reference

- ✅ BCE mode removed – focal-only production
- ✅ Comprehensive debt audit (v3.7.0) – zero-debt policy adopted
- ✅ WSL2 performance guard – latency degradation multiplier
- ✅ SSOT constant wiring – 84 constants tracked, 69% utilisation
- ✅ Test infrastructure hygiene – batch size fixture, GPU memory parametrisation, FA sweep refactor, metric key helper

---

## Next Step

```bash
modal run deploy/modal/app.py --action train --config configs/modal/smoke.yaml
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml
```

Keep this file in sync with every audit or training cycle.
