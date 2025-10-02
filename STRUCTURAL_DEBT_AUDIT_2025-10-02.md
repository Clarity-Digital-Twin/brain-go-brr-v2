# Structural Debt Audit — 2025-10-02

Senior auditor sweep for oversized/monolithic Python modules. No code was modified; this document records refactor targets for future sprints and links to detailed action plans.

## Summary
- Originally identified five hotspots where function length or responsibility density violated our SOLID/clean-code bar.
- **Status Update (2025-10-02):** ✅ loop.py refactoring completed (958 → 640 lines, 33% reduction) with Sequence 4 documentation updated.
- **Remaining:** 4 hotspots (detector.py, metrics.py, cli.py, io.py) now have dedicated refactor playbooks drafted for consensus: `REFACTOR_DETECTOR_PY.md`, `REFACTOR_METRICS_PY.md`, `REFACTOR_CLI_PY.md`, `REFACTOR_IO_PY.md`.
- Next step: secure agreement on each plan, then execute one refactor at a time with regression gates defined in the playbooks.

## Hotspot Details

### ✅ 1. `src/brain_brr/train/loop.py` — COMPLETED 2025-10-02
- **Original:** `train_epoch` 606 lines, `train` 260 lines, `main` 332 lines — total 958 lines.
- **Final:** loop.py reduced to 640 lines (33% reduction) after extracting warmup, sampling, losses, optimizer, early-stopping utilities, and keeping train/validate logic in dedicated modules.
- **Verification:** Full suite (unit, integration, clinical) plus type and lint checks pass; streaming validation retains 77% memory reduction.
- **Reference:** Commit `36055df`, EXECUTION_PLAN_2025-10-02.md Sequence 4 marked complete.

### 2. `src/brain_brr/models/detector.py` — PLAN READY (see `REFACTOR_DETECTOR_PY.md`)
- **Pain Points:** `forward` (≈186 lines) blends preprocessing, dual-stream fusion, monitoring, and clamping; `from_config` (≈198 lines) instantiates TCN, BiMamba, GNN, fusion heads, and PR toggles in one block.
- **Refactor Strategy:**
  - Phase 1 extracts builder helpers (`_build_node_stream`, `_build_edge_stream`, `_build_fusion_head`, `_build_regularizers`).
  - Phase 2 decomposes `forward` into pipeline helpers (`_prepare_inputs`, `_run_node_stream`, `_run_edge_stream`, `_apply_fusion`, `_apply_postprocess`).
  - Includes baseline state_dict snapshot, regression tests for helper outputs, and rollback plan.
- **Status:** Awaiting consensus prior to implementation.

### 3. `src/brain_brr/eval/metrics.py` — PLAN READY (see `REFACTOR_METRICS_PY.md`)
- **Pain Point:** `evaluate_predictions` (≈159 lines) couples timeline assembly, FA sweeps, scalar metrics, and output formatting, hindering testability.
- **Refactor Strategy:**
  - Timeline helpers isolate hysteresis/morphology/merge logic.
  - False-alarm sweep helper preserves current conservative counting while documenting TODO for unique FA logic.
  - Scalar reducers and output formatter provide composable stages with dedicated unit tests.
  - Regression uses golden JSON fixtures from integration tests.
- **Status:** Awaiting consensus before code changes.

### 4. `src/brain_brr/cli/cli.py` — PLAN READY (see `REFACTOR_CLI_PY.md`)
- **Pain Point:** `evaluate` command (≈223 lines) intermixes CLI parsing, checkpoint IO, dataloader creation, inference, metrics, and export logic.
- **Refactor Strategy:**
  - Introduce `src/brain_brr/cli/services/` with evaluation/training helpers.
  - Thin Click commands to parse-and-delegate while preserving UX.
  - New service-layer unit tests plus existing CLI tests ensure parity.
- **Status:** Plan drafted; waiting on alignment before extracting helpers.

### 5. `src/brain_brr/data/io.py` — PLAN READY (see `REFACTOR_IO_PY.md`)
- **Pain Point:** `load_edf_file` (≈152 lines) performs path resolution, EDF read, resample, filtering, channel reordering, interpolation, and label alignment inline.
- **Refactor Strategy:**
  - Break pipeline into helpers for path resolution, EDF read, resample, filters, channel ordering/interpolation, label alignment, and output packaging.
  - Add synthetic-signal unit tests per stage; regression compares cached outputs with `numpy.allclose`.
- **Status:** Plan drafted; execution scheduled post-consensus.

## Progress Summary

| File | Status | Plan Document | Priority |
|------|--------|---------------|----------|
| `train/loop.py` | ✅ Completed | EXECUTION_PLAN_2025-10-02.md (Sequence 4) | N/A |
| `models/detector.py` | 📝 Planned | `REFACTOR_DETECTOR_PY.md` | High |
| `eval/metrics.py` | 📝 Planned | `REFACTOR_METRICS_PY.md` | High |
| `cli/cli.py` | 📝 Planned | `REFACTOR_CLI_PY.md` | Medium |
| `data/io.py` | 📝 Planned | `REFACTOR_IO_PY.md` | Medium |

## Next Actions
- Review each refactor plan with engineering leads; capture sign-off in STATUS.md.
- Once approved, schedule refactors sequentially (detector → metrics → CLI → IO) with regression checkpoints outlined in each document.
- Update TODO.md as phases begin/complete to keep debt tracker current.

Document owner: Codex senior auditor (updated 2025-10-02 after drafting refactor playbooks).
