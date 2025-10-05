# Remaining Debt Implementation Guide (v3.7.0)

**Date:** October 5, 2025  
**Status:** ✅ P0/P1 resolved · 🟡 Optional polish only

---

## Executive Summary

The v3.7.0 SSOT/constant wiring effort closed all high- and medium-priority debt. What remains is optional cleanup (unused constants,
document polish). This guide now tracks those low-priority items for a future maintenance sprint.

**Recommended Timing:** v3.8.0 or later  
**Estimated Effort:** ~3 hours total  
**Risk:** Very low (no production impact)

---

## Optional Cleanup Checklist

| Item | Description | Effort | Suggested Approach |
|------|-------------|--------|--------------------|
| Unused constants | 32 constants (labels, metric strings, legacy clamps) remain defined but unused | 1.5h | Remove or migrate to appropriate modules; ensure tests pass |
| Metrics documentation | Update docs to mention `format_sensitivity_key()` and TAES defaults | 45m | Refresh developer docs / README sections |
| Architecture diagrams | Incorporate constant wiring + SSOT explanation into diagrams | 30m | Update `docs/04-model/` assets |

---

## Completed Debt (for reference)

The following were addressed in v3.7.0 and require no further action:

- Config validation script + Makefile target  
- FA sweep refactor (`train/val_step.py`)  
- Inline import cleanup  
- Metric key helper adoption  
- Schema epsilon + ECE bin constants  
- Six hardcoded literals replaced with constants  
- Root documentation synced with reality

---

## How to Use This Document

1. Treat the table above as a backlog of optional polish items.  
2. When you schedule a cleanup sprint, copy the relevant rows into a tracking issue / PR checklist.  
3. After completing an item, update this guide (and `POLISH_ITEMS.md`) with the new status.

---

## Verification Tools

- `/tmp/complete_audit.py` – constant usage + schema stats  
- `make q` – lint, format, mypy, config validation  
- `rg 'sample_size: int = 500' src/` – ensure literals stay gone  
- `rg 'alpha: float = 0\.05' src/` – ensure GNN default stays wired

