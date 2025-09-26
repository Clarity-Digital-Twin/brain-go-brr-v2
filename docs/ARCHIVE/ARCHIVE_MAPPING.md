# Archive → Canonical Mapping (SSOT)

This mapping shows where content from the legacy `docs/ARCHIVE/` was consolidated into canonical docs. With these mappings in place, the archived files can be safely removed.

Key migrations
- CRITICAL_FIX_PATIENT_LEAKAGE.md → `docs/tusz/tusz-splits.md`, `docs/05-training/modal.md`
- TUSZ_PIPELINE_AUDIT.md → `docs/tusz/tusz-splits.md`, `docs/02-data/overview.md`
- MODAL_CONFIG_STATUS.md → `docs/03-configuration/modal-configs.md`, `docs/05-training/modal.md`
- OOM_ROOT_CAUSE_ANALYSIS.md → `docs/08-operations/troubleshooting.md`
- NAN_LOGITS_ROOT_CAUSE_ANALYSIS.md → `docs/08-operations/v3-nan-explosion-resolution.md`, `docs/08-operations/troubleshooting.md`
- DYNAMIC_PE_MEMORY_SOLUTION.md → `docs/04-model/laplacian-pe.md`, `docs/08-operations/performance-optimization.md`
- DYNAMIC_LPE_IMPLEMENTATION.md → `docs/04-model/laplacian-pe.md`
- OPTIMAL_RTX4090_CONFIG.md → `docs/03-configuration/local-configs.md`, `docs/08-operations/performance-optimization.md`
- streaming.md → `docs/08-operations/streaming.md`
- BUGS.md / BUGS_FIXED.md / TEST_FIX_SUMMARY.md → `docs/09-development/bug-tracker.md`
- ARCHITECTURE_STATUS_REPORT.md / V3_ARCHITECTURE_AS_IMPLEMENTED.md / V3_FINAL_STATUS.md → `docs/04-model/v3-architecture.md`
- CONFIG_STATUS_FINAL.md → `docs/03-configuration/{local-configs,modal-configs}.md`

TUSZ references
- Keep dataset‑specific procedures under `docs/tusz/` and link from `docs/02-data/overview.md`.

Status
- All essential topics above now have canonical homes. It is safe to remove the corresponding archived files.
