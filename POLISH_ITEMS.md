# Brain-Go-Brr v3.7.0 – Post-Debt Polish Tracker

**Last Audit:** 2025-10-05  
**Status:** ✅ High/medium polish items complete · 🟡 Optional cleanup remaining

---

## Completed Polish (v3.7.0)

| Item | Summary | Evidence |
|------|---------|----------|
| P2.1 Config validation | `scripts/validate_configs.py` wired into `make q` | `make q` → “Validating configs…” |
| P2.2 Config docs version | `configs/CONFIG_CONSISTENCY_CHECK.md` bumped to v3.7.0 | `head -1 configs/CONFIG_CONSISTENCY_CHECK.md` |
| P2.3 FA sweep refactor | `train/val_step.py` now delegates to `false_alarm.py` helper | `rg 'find_threshold_for_fa_target' src/brain_brr/train/val_step.py` |
| P2.4 Inline imports | All imports at module scope in `train/val_step.py` | `rg '^\s+from .*eval.metrics' src/brain_brr/train/val_step.py` → none |
| P2.5 Metric key helper | `format_sensitivity_key()` used in train & val loops | `rg 'format_sensitivity_key' src/brain_brr/train` |
| P3.1 Schema epsilons | Validation bounds now use `EPSILON_NUMERICAL` | `rg 'EPSILON_NUMERICAL' src/brain_brr/config/schemas.py` |
| P3.2 ECE bins constant | `ECE_NUM_BINS` used in metrics/val step | audit script output |
| P3.3 Constant wiring | Six production literals replaced with constants | `/tmp/complete_audit.py` summary |
| P3.4 Function signatures | No lingering literal defaults in code | `rg '\b=\s*\d+\b'` (spot checks) |
| P3.5 Schema literals | Validation assertions use SSOT constants | `rg '!=\s*256' src/brain_brr/config/schemas.py` → none |

---

## Optional Polish Backlog (Deferred)

These are nice-to-have cleanups that do **not** affect production training.

| Priority | Item | Notes | Suggested Target |
|----------|------|-------|------------------|
| P3 | Unused constants (labels, metric strings, legacy clamps) | 32 constants still defined but unused – create focused cleanup when convenient | v3.8.0 |
| P3 | Documentation polish | Refresh architecture diagrams + metrics guide to reference the new SSOT helper | v3.8.0 |
| P3 | Metric naming guide | Document `format_sensitivity_key()` usage in developer docs | v3.8.0 |

---

## Verification Commands

```bash
python /tmp/complete_audit.py   # confirms constant usage + schema stats
make q                          # lint + format + mypy + config validation
rg 'sample_size: int = 500' src/   # should return nothing
rg 'alpha: float = 0\.05' src/    # should return nothing
```

---

## Notes

- No additional action required unless we choose to tackle the optional backlog.  
- Re-run the audit script after any future constant or config changes.

