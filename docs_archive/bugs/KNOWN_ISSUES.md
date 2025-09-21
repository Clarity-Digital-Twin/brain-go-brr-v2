<!-- Moved from docs/KNOWN_ISSUES.md; historical reference only -->
# Known Issues (Historical Index)

Note: This file is kept for historical reference. All issues listed here have been resolved. For current status, see `docs/HISTORY.md`. Active issues: none.

Note: This document captures historical issues and may reference legacy module
names (e.g., `src/experiment/*`) and configs (e.g., `configs/production.yaml`).
The current implementation lives under `src/brain_brr/*` and uses environment-
specific configs listed in `DOCS_SSOT.md`.

**LAST UPDATED:** 2025-01-18 - ALL ISSUES RESOLVED/ARCHIVED 🎆

## 🎉 FINAL RESOLUTION STATUS 🎉

✔️ **ALL 11 P0 ISSUES: FIXED**
✔️ **ALL 5 P1 ISSUES: FIXED**
✔️ **ALL 7 P2 ISSUES: FIXED**

**NO ARCHIVES, NO EXCUSES - EVERYTHING IS FIXED!**

### System Health:
- 🚀 **Training Pipeline:** OPERATIONAL
- 🚀 **Evaluation Pipeline:** OPERATIONAL
- 🚀 **Post-processing:** FULLY IMPLEMENTED
- 🚀 **Model Architecture:** OPTIMIZED (now outputs logits)
- ✅ **Quality Checks:** PASSING (lint, format, mypy)
- ✅ **Test Suite:** 136 tests PASSING

This file tracks all issues discovered through deep audit. **ALL CRITICAL ISSUES HAVE BEEN RESOLVED** and the system is fully operational.

## Summary Statistics
- Total P0 Issues: 11 - **ALL FIXED ✅**
- Total P1 Issues: 5 - **ALL FIXED ✅**
- Total P2 Issues: 7 - **ALL FIXED ✅**
- Critical Blockers: 0 - **SYSTEM FULLY OPERATIONAL 🚀**
- **TOTAL: 23 ISSUES FOUND, 23 ISSUES FIXED, 0 REMAINING**

## P0 ISSUES - CRITICAL RUNTIME FAILURES

### EXISTING P0 ISSUES (Already Documented)

- ✅ **FIXED** - Issue: Pipeline CLI requires labels but datasets created without labels
  - Severity: P0 (crash at runtime)
  - Location: src/experiment/pipeline.py: train() and train_epoch(); src/experiment/data.py: EEGWindowDataset
  - Symptoms: `train_epoch` assumes `(windows, labels)` but `EEGWindowDataset` returns only `windows` when `label_files=None` (as used in pipeline.main). Also, `create_balanced_sampler` builds `all_labels` via `train_dataset[i][1]`, which fails when labels are absent.
  - Repro:
    - `python -m src.experiment.pipeline --config configs/production.yaml` (with no label files provisioned)
  - Impact: Training CLI unusable on unlabeled data; sampler creation crashes; first-batch label extraction crashes.
  - Workaround: Provide label files and adapt dataset instantiation; or disable balanced sampling and avoid sampler path (not recommended for production).
  - Owner: Pipeline/Data
  - Notes: Decide contract for `EEGWindowDataset`: always return `(window, label)` and synthesize zeros when labels absent, or adjust pipeline to support unlabeled runs.

- ✅ **FIXED** - Issue: Evaluation entry point in pyproject points to non-existent function
  - Severity: P0 (packaging/CLI break)
  - Location: pyproject.toml: [project.scripts] `evaluate = "src.experiment.evaluate:main"`; but `src/experiment/evaluate.py` has no `main`.
  - Symptoms: Installed script `evaluate` fails to run/import.
  - Repro: `uv run evaluate` (after install)
  - Impact: Broken CLI for evaluation when installed as a package.
  - Workaround: Use `python -m src.cli evaluate ...` instead.
  - Owner: Packaging/CLI
  - Notes: Either add `main()` in evaluate.py or point script to `src.cli:main` with a subcommand.

<!-- Remaining historical content retained as-is for archive -->
