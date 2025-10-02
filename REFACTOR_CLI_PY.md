# Refactor Plan — src/brain_brr/cli/cli.py
Status: Draft for consensus (2025-10-02)
Priority: MEDIUM
Owner: Senior ML auditor (Codex)
Scope: Click command orchestration (`train`, `evaluate`, `validate`, etc.)

## 1. Context & Problem Statement
- `evaluate` command (≈223 lines, current span 316–539) currently orchestrates option parsing, checkpoint resolution, dataset construction, model loading, inference, metric computation, and export in a single function.
- `train` command shares similar issues (~160 lines) though less extreme.
- Lack of separation makes unit testing difficult (requires Click runner + heavy mocking) and blocks reuse of evaluation flow in other contexts (e.g., automated pipelines).

## 2. Goals
1. Reduce Click command bodies to thin wrappers delegating to service-layer helpers.
2. Keep CLI behaviour, options, and output formatting identical for users.
3. Enable unit tests to exercise evaluation logic without invoking Click.
4. Provide clear extension points for additional commands (e.g., batch evaluation, profiling).

## 3. Proposed Architecture
```
src/brain_brr/cli/
├── cli.py                       # Click command definitions only
└── services/
    ├── evaluation.py            # run_evaluation(...)
    ├── training.py              # run_training(...)
    └── io_utils.py              # shared path/context helpers
```
- Service helpers encapsulate business logic and can be imported by both CLI and external scripts.

## 4. Step-by-Step Plan

### Phase 0 – Baseline Snapshot
- Run existing CLI unit tests (`tests/unit/cli/test_cli_commands.py`, `test_cli_simple.py`) to confirm green state.
- Capture representative CLI outputs (stdout logs) for regression comparison.

### Phase 1 – Scaffold Service Layer
1. Create `src/brain_brr/cli/services/__init__.py` and `evaluation.py` skeleton with high-level functions:
   - `run_evaluation(config_path, overrides, output_opts, logger)`
   - `resolve_checkpoint_path(checkpoint_opt, config)`
   - `build_eval_dataloaders(config, env)`
2. Move corresponding chunks from `cli.py` into helpers, ensuring signatures allow dependency injection (logger, device provider).
3. Keep CLI command calling new helpers while preserving original flow.

### Phase 2 – Thin CLI Commands
1. Update `evaluate` Click command to:
   - Parse options / config overrides.
   - Invoke `run_evaluation(...)`.
   - Handle exceptions with Click-friendly messages.
2. Optionally repeat pattern for `train` if time permits (document if deferred).
3. Retain global options (e.g., `--device`, `--batch-size`) by passing them into service layer.

### Phase 3 – Testing Enhancements
- Add unit tests for service helpers (`tests/unit/cli/test_evaluation_service.py`) using mocks/fixtures for model loading and metrics.
- Update CLI tests to patch service functions, asserting they are invoked with expected parameters.
- Add integration test (non-Click) that calls service layer directly on smoke config to verify metrics dictionary.

### Phase 4 – Documentation & Cleanup
- Update developer docs (e.g., `docs/05-training/local.md`, CLI README sections) to mention new service modules.
- Ensure `STRUCTURAL_DEBT_AUDIT_2025-10-02.md` and TODO reflect refactor completion.

### Phase 5 – Regression Validation
- Run `make q`, `make test`, targeted CLI tests.
- Manually invoke `python -m src.cli evaluate --config configs/local/smoke.yaml` to confirm user-facing behaviour unchanged.

## 5. Testing Strategy
- Continue using Click tests for command-line coverage.
- Introduce service-layer unit tests with dependency injection (fake checkpoint, stub metrics) for fast validation.
- Optionally add golden log snapshot tests if CLI output formatting critical.

## 6. Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| CLI output formatting changes inadvertently | Capture baseline stdout and compare post-refactor using pytest snapshots. |
| Service layer introduces implicit dependencies | Accept logger, device resolver, and environment objects as parameters; document defaults. |
| Duplicate logic between training and evaluation helpers | Factor shared components into `io_utils.py` to avoid duplication. |
| Potential import cycles | Keep service modules free of Click imports; only CLI imports services. |

## 7. Rollback Plan
- Keep each phase in separate commit; easiest rollback is to restore original `cli.py` while leaving helper files (unused) for future.
- If services cause regressions, revert to baseline commit and retain new tests for next attempt.

## 8. Open Questions
1. Should we expose service helpers as part of public API (e.g., for Modal scripts)?
2. Do we refactor `train` command in same sprint or track separately?
3. Should W&B export logic move into dedicated helper or remain within evaluation service?
