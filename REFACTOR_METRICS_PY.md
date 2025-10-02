# Refactor Plan — src/brain_brr/eval/metrics.py
Status: Draft for consensus (2025-10-02)
Priority: HIGH
Owner: Senior ML auditor (Codex)
Scope: `evaluate_predictions` and supporting helpers

## 1. Context & Problem Statement
`evaluate_predictions` (≈185 lines, current span 448–632) currently couples:
- Timeline reconstruction (binarisation, hysteresis, morphology, merging)
- False-alarm sweeps at target FA rates
- Scalar metric reductions (sensitivity, duration stats, AUROC, ECE)
- Output assembly for CLI (dicts, CSV/JSON export metadata)

Mixing these concerns makes it difficult to reason about, extend (e.g., new FA buckets), or unit test intermediate steps. Developers are forced to read the entire function to understand any single responsibility.

## 2. Goals
1. Reduce `evaluate_predictions` to a thin orchestrator (≤120 lines) that delegates to focused helpers.
2. Isolate timeline assembly, FA sweep computation, and scalar reducers into composable functions with clear contracts.
3. Maintain public API compatibility: return structure and keys must remain identical.
4. Improve test coverage for each stage with deterministic fixtures.
5. Lay groundwork for future improvements (unique FA counting) without changing behaviour now.

## 3. Proposed Architecture
```
metrics.py
├── evaluate_predictions(...)                 # orchestrator
├── _build_record_timeline(...)
├── _merge_timelines(...)
├── _compute_false_alarm_sweep(...)
├── _select_threshold_for_target(...)
├── _compute_scalar_metrics(...)
└── _format_metrics_output(...)
```
- If file remains large, move helpers into `metrics_timeline.py`, `metrics_false_alarm.py`, `metrics_reducers.py` under `src/brain_brr/eval/` while keeping public API unchanged.

## 4. Step-by-Step Plan

### Phase 0 – Baseline Snapshot
- Run `make t` + `make test` to confirm green state.
- Capture golden metrics output (JSON) from `tests/integration/test_evaluation.py` fixture for regression comparison.

### Phase 1 – Timeline Helpers
1. Extract `_build_record_timeline(pred, ref, config)` handling hysteresis, morphology, merging.
2. Ensure helper returns structured data (namedtuple/dataclass) containing binary mask, event list, and durations.
3. Replace inline logic in `evaluate_predictions` with helper calls.
4. Add unit tests under `tests/unit/eval/test_metrics_timeline.py` covering perfect overlap, partial overlap, empty predictions, and multi-record merges.

### Phase 2 – False-Alarm Sweep Extraction
1. Extract `_compute_false_alarm_sweep(timelines, fa_targets, sample_rate)`.
2. Encapsulate current conservative FA counting; document TODO for unique FA logic.
3. Add unit tests verifying deterministic thresholds, unreachable targets, and monotonic behaviour.

### Phase 3 – Scalar Reducers
1. Extract `_compute_scalar_metrics(timelines, sweep_results)` for sensitivity, duration stats, seizure counts.
2. Extract `_compute_probability_metrics(probabilities, references)` for AUROC/ECE (may already exist; reuse or wrap).
3. Ensure helpers operate on NumPy/Torch arrays without side effects.
4. Add unit tests using synthetic probability distributions.

### Phase 4 – Output Assembly
- Create `_format_metrics_output(...)` to merge scalar metrics, sweep summaries, and probability metrics into final dict.
- Keep CLI expectations intact (keys, ordering).
- Update docstrings and inline comments to describe each stage.

### Phase 5 – Regression Validation
- Re-run targeted unit tests, integration suites, `make test`.
- Compare golden JSON output to baseline (use `pytest.approx` for floats).
- Update documentation (`STRUCTURAL_DEBT_AUDIT_2025-10-02.md`, TODO) to mark phase complete.

## 5. Testing Strategy
- Existing integration tests remain primary guard.
- New unit modules:
  - `tests/unit/eval/test_metrics_timeline.py`
  - `tests/unit/eval/test_metrics_false_alarm.py`
  - `tests/unit/eval/test_metrics_reducers.py`
- Add regression fixture verifying full evaluation result matches baseline on sample dataset.

## 6. Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| Behaviour change in FA counting | Preserve current algorithm exactly; add regression test capturing counts. |
| Floating-point drift | Use deterministic operations; compare with tolerances in tests. |
| Performance regression | Avoid repeated conversions or Python loops; profile after refactor if needed. |
| Module import churn | Keep helper functions internal (`_` prefix) unless promoting to new module. |

## 7. Rollback Plan
- Commit each phase separately.
- If regression occurs, revert to previous commit while retaining new tests for future attempts.
- Restore baseline metrics JSON to confirm rollback.

## 8. Open Questions
1. Should FA targets be configurable via config files in future? (Out of scope for initial refactor.)
2. Do we want to expose helper outputs (timelines/events) for downstream analytics?
3. When we implement unique FA counting, should it live in the same helper or a new strategy class?
