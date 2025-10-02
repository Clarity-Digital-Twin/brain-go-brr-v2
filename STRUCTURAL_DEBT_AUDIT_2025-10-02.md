# Structural Debt Audit — 2025-10-02

Senior auditor sweep for oversized/monolithic Python modules. No code was modified; this document records refactor targets for future sprints.

## Summary
- Identified five hotspots where function length or responsibility density violates our SOLID/clean-code bar.
- All findings align with prior refactoring backlog; nothing contradicts the active plan.
- Recommend tackling in the order shown once the evaluation-timeline P0 fix is complete.

## Hotspot Details

### 1. `src/brain_brr/train/loop.py`
- **Functions:** `train_epoch` 606 lines (#341), `train` 260 lines (#1137), `main` 332 lines (#1405).
- **Why it matters:** Mixing data statistics, AMP, logging, checkpointing, anomaly handling, and warmup plumbing in one routine breaks Single Responsibility and makes testing brittle. Any change requires editing a 600-line block.
- **Action:** Follow `REFACTORING_PLAN_V4.md` Phase 2—split into dedicated modules (`train_step`, `val_step`, checkpoint helpers). Maintain behaviour parity with regression tests + smoke run.

### 2. `src/brain_brr/models/detector.py`
- **Functions:** `forward` 186 lines (#247) and `from_config` 198 lines (#436).
- **Why it matters:** `forward` interleaves feature extraction, sanitisation, fusion, and multiple tiers of clamping; `from_config` instantiates every optional component (TCN, Mamba, GNN, fusion, LayerScale) in one method. This breaches SRP and obscures invariants for future architecture tweaks.
- **Action:** Introduce helper builders (e.g., `_build_node_stream`, `_build_edge_stream`) and decompose the forward pass into composable stages. Add targeted unit tests around each stage before refactor.

### 3. `src/brain_brr/eval/metrics.py`
- **Function:** `evaluate_predictions` 159 lines (#408).
- **Why it matters:** Bloated evaluation routine couples hysteresis thresholds, FA sweeps, AUROC, and (currently broken) timeline stitching. The P0 bug is rooted here because time alignment logic is buried among unrelated metrics code.
- **Action:** Once dataset metadata is emitted, carve out timeline assembly and metric reducers into discrete helpers. Add regression tests for multi-record evaluation to prevent recurrence.

### 4. `src/brain_brr/cli/cli.py`
- **Function:** `evaluate` 223 lines (#316).
- **Why it matters:** CLI command handles dry-run scaffolding, checkpoint IO, dataset creation, inference, metric display, and CSV export in one function. Difficult to reuse and nearly impossible to unit test.
- **Action:** Extract service-layer helpers (`load_checkpoint`, `run_inference`, `export_reports`) and keep the Click command thin. Cover with integration tests invoking these helpers directly.

### 5. `src/brain_brr/data/io.py`
- **Function:** `load_edf_file` 152 lines (#54).
- **Why it matters:** Performs file resolution, EDF reading, resampling, filtering, channel ordering, and label alignment in a single block. Violates SRP and complicates targeted instrumentation (e.g., profiling individual stages).
- **Action:** Split into pipeline steps (read → resample → filter → normalise → label sync). Add unit tests per stage using lightweight synthetic signals.

## Notes
- Other large files (`utils/training_logger.py`, `utils/logging_config.py`, `models/gnn_pyg.py`) are long due to multiple short helpers and currently read clean; no immediate action.
- Defer all structural refactors until the timeline P0 fix lands and fresh training runs start.

Document owner: Codex senior auditor (2025-10-02).
