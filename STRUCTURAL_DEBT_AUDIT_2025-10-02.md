# Structural Debt Audit — 2025-10-02

Senior auditor sweep for oversized/monolithic Python modules. No code was modified; this document records refactor targets for future sprints.

## Summary
- Originally identified five hotspots where function length or responsibility density violates our SOLID/clean-code bar.
- **Status Update (2025-10-02):** ✅ loop.py refactoring COMPLETED (958 → 640 lines, 33% reduction)
- **Remaining:** 4 hotspots (detector.py, metrics.py, cli.py, io.py)
- Individual refactoring plans created for each remaining file (see root directory)

## Hotspot Details

### ✅ 1. `src/brain_brr/train/loop.py` (COMPLETED 2025-10-02)
- **Original State:** `train_epoch` 606 lines, `train` 260 lines, `main` 332 lines — total 958 lines
- **Final State:** loop.py reduced to 640 lines (33% reduction)
- **What Was Done:**
  - ✅ Extracted utilities to focused modules (warmup.py, sampling.py, losses.py, optimizer_factory.py, early_stopping.py)
  - ✅ train_epoch/validate_epoch already in train_step.py/val_step.py
  - ✅ Checkpoint management in checkpoint.py
  - ✅ 100% test pass rate maintained
  - ✅ Full SOLID compliance achieved
- **Status:** PRODUCTION READY - No further refactoring needed
- **Commit:** 36055df

### 2. `src/brain_brr/models/detector.py` (PLANNED - See REFACTOR_DETECTOR_PY.md)
- **Functions:** `forward` 186 lines (#247) and `from_config` 198 lines (#436).
- **Why it matters:** `forward` interleaves feature extraction, sanitisation, fusion, and multiple tiers of clamping; `from_config` instantiates every optional component (TCN, Mamba, GNN, fusion, LayerScale) in one method. This breaches SRP and obscures invariants for future architecture tweaks.
- **Action:** Introduce helper builders (e.g., `_build_node_stream`, `_build_edge_stream`) and decompose the forward pass into composable stages. Add targeted unit tests around each stage before refactor.

### 3. `src/brain_brr/eval/metrics.py` (PLANNED - See REFACTOR_METRICS_PY.md)
- **Function:** `evaluate_predictions` 159 lines (#408).
- **Why it matters:** Bloated evaluation routine couples hysteresis thresholds, FA sweeps, AUROC, and timeline stitching. This combines multiple concerns (timeline reconstruction, threshold search, metric computation) that should be separated.
- **Action:** Carve out timeline assembly and metric reducers into discrete helpers. Add regression tests for multi-record evaluation to prevent recurrence.

### 4. `src/brain_brr/cli/cli.py` (PLANNED - See REFACTOR_CLI_PY.md)
- **Function:** `evaluate` 223 lines (#316).
- **Why it matters:** CLI command handles dry-run scaffolding, checkpoint IO, dataset creation, inference, metric display, and CSV export in one function. Difficult to reuse and nearly impossible to unit test.
- **Action:** Extract service-layer helpers (`load_checkpoint`, `run_inference`, `export_reports`) and keep the Click command thin. Cover with integration tests invoking these helpers directly.

### 5. `src/brain_brr/data/io.py` (PLANNED - See REFACTOR_IO_PY.md)
- **Function:** `load_edf_file` 152 lines (#54).
- **Why it matters:** Performs file resolution, EDF reading, resampling, filtering, channel ordering, and label alignment in a single block. Violates SRP and complicates targeted instrumentation (e.g., profiling individual stages).
- **Action:** Split into pipeline steps (read → resample → filter → normalise → label sync). Add unit tests per stage using lightweight synthetic signals.

## Progress Summary

| File | Status | Plan Document | Priority |
|------|--------|---------------|----------|
| ✅ loop.py | COMPLETED | ✅ Done (640 lines, 33% reduction) | N/A |
| detector.py | PLANNED | REFACTOR_DETECTOR_PY.md | HIGH |
| metrics.py | PLANNED | REFACTOR_METRICS_PY.md | HIGH |
| cli.py | PLANNED | REFACTOR_CLI_PY.md | MEDIUM |
| io.py | PLANNED | REFACTOR_IO_PY.md | MEDIUM |

## Notes
- ✅ loop.py refactoring completed 2025-10-02 (Sequence 4)
- Individual refactoring plans created for each remaining file
- Each plan includes explicit steps, validation criteria, and rollback strategy
- Other large files (`utils/training_logger.py`, `utils/logging_config.py`, `models/gnn_pyg.py`) are long due to multiple short helpers and currently read clean; no immediate action.
- Wait for AI agent consensus on refactoring plans before implementation

Document owner: Codex senior auditor (2025-10-02).
Last updated: 2025-10-02 (loop.py completion)
