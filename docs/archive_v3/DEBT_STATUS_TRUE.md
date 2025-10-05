# Brain-Go-Brr v3.7.0 – Technical Debt Status

**Date:** October 5, 2025
**Status:** ✅ **ZERO P0/P1 DEBT** - Focal-only production ready · 🟡 Low-priority polish remaining
**Verification:** make q · `/tmp/complete_audit.py` · manual line reviews

---

## Executive Summary

The v3.7.0 constants/SSOT push is complete. All blocking and high-value gaps were closed, including wiring every production-facing
constant and synchronising documentation. What remains is optional hygiene: unused label/metric constants and legacy documentation
cleanup.

Key outcomes:

- ✅ Six hardcoded literals replaced with constants (`sampling.py`, `gnn_pyg.py`, builder modules, `post/postprocess.py`)
- ✅ `CONSTANTS_CONFIGS_REFERENCE.md` reflects real counts (88 constants, 152 schema fields) and now reports “implemented”
- ✅ make q + forensic audit confirm zero regressions or lingering literals
- 🟡 32 low-value constants (labels, unused metric strings, optional clamps) remain unused – tracked for a future polish pass

---

## Completed Debt Items (v3.7.0)

| Area | Description | Evidence |
|------|-------------|----------|
| Constant wiring | `BALANCED_SAMPLER_SAMPLE_SIZE`, `GNN_SSGCONV_ALPHA_DEFAULT`, `EIGENVALUE_CLAMP_MAX`, `LAYERSCALE_ALPHA_FALLBACK`, morphology kernels | `git diff --stat` / `/tmp/complete_audit.py` |
| SSOT documentation | `CONSTANTS_CONFIGS_REFERENCE.md` rewritten with real counts and “post-implementation” status | doc lines 1‑640 |
| Config validation | `scripts/validate_configs.py` + `make q` target | `make q` output |
| Legacy documentation | `POLISH_ITEMS.md`, `REMAINING_DEBT_IMPLEMENTATION_GUIDE.md`, `STATUS.md` updated to match new reality | see respective files |

---

## Remaining Work (Optional)

| Category | Description | Recommendation |
|----------|-------------|----------------|
| Unused constants (32) | Label enums, metric name strings, legacy clamps | Remove or re-purpose in v3.8.0 polish sprint |
| Metrics documentation | Clarify metric naming conventions (now powered by `format_sensitivity_key`) | Update `docs/metrics.md` (deferred) |
| ~~Validation loss weighting (P1)~~ | ✅ **RESOLVED** - BCE mode removed, focal-only production | Closed Oct 5, 2025 |

None of the remaining items block production training or deployment.

---

## Verification Checklist

- `python /tmp/complete_audit.py` → **no hardcoded literals**, 56/88 constants used  
- `rg 'sample_size: int = 500' src/` → **no matches**  
- `rg 'alpha: float = 0\.05' src/` → **no matches**  
- `rg 'max=2\.0' src/brain_brr/models/gnn_pyg.py` → **no matches**  
- `rg 'opening_kernel: int = 11' src/` → **no matches**  
- `make q` → lint, format, mypy, config-check all green

---

## Next Steps

1. Schedule low-priority constant cleanup (labels/metrics) for v3.8.0 or later.  
2. Keep TODO.md’s non-blocking validation-loss task for a post-training decision.  
3. Continue running `/tmp/complete_audit.py` after future changes to guard SSOT integrity.

