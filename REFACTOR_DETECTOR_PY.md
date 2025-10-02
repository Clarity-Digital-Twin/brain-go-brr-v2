# Refactor Plan — src/brain_brr/models/detector.py
Status: Draft for consensus (2025-10-02)
Priority: HIGH
Owner: Senior ML auditor (Codex)
Scope: SeizureDetector construction + forward orchestration

## 1. Context & Problem Statement
- `forward` (~186 lines) interleaves feature extraction, dual-stream fusion, residual safety rails, and multiple clamping paths. Mixing tensor sanitisation with fusion logic obscures invariants and complicates future architecture changes.
- `from_config` (~198 lines) constructs every optional component (TCN, BiMamba, GNN, fusion heads, LayerScale, PR toggles) within a single method, with nested branching and duplicated defaults.
- Cross-cutting responsibilities make the module risky to modify, hard to unit test in isolation, and difficult to extend for V4 features.

## 2. Goals
1. Preserve the SeizureDetector public surface (constructor, attributes, behaviour) and checkpoints.
2. Restructure so that:
   - `forward()` ≤ 120 lines and reads as a sequence of clearly named stages.
   - Configuration/build logic lives in helpers, each ≤ 80 lines and single-responsibility.
3. Increase unit-test granularity to exercise node-stream, edge-stream, fusion, and clamping paths independently.
4. Maintain deterministic outputs for all existing tests and sample checkpoints.

## 3. Proposed Architecture
```
src/brain_brr/models/
├── detector.py                  # Thin orchestrator + public API
├── builders/
│   ├── node_stream.py           # build_node_stream(config)
│   ├── edge_stream.py           # build_edge_stream(config)
│   ├── fusion.py                # build_fusion_head(config)
│   └── regularization.py        # build_layerscale(), clamp policy helpers
└── pipelines/
    ├── node_pipeline.py         # run_node_pipeline(inputs)
    ├── edge_pipeline.py         # run_edge_pipeline(inputs, node_feats)
    └── fusion_pipeline.py       # fuse_streams(node, edge) → logits
```
- If directory split feels heavy, keep helpers in detector.py under clearly marked sections while retaining the responsibility boundaries.

## 4. Step-by-Step Plan

### Phase 0 – Baseline Snapshot
- Capture baseline by running `make t`, `make test`, and storing a minimal model `state_dict` for regression comparison.
- Record existing coverage for detector tests (`tests/unit/models/test_detector_v3.py`, integration suites).

### Phase 1 – Builder Extraction (no functional change)
1. Introduce `_build_node_stream(config: ModelConfig) -> nn.Module`.
2. Introduce `_build_edge_stream(config) -> nn.Module | None`.
3. Introduce `_build_fusion_head(config) -> nn.Module`.
4. Introduce `_build_regularizers(config) -> RegularizationBundle` (LayerScale, clamping thresholds, monitors).
5. Update `from_config` to delegate to helpers without altering attribute names or default values.
6. Add unit tests validating builder outputs (layer counts, dropout flags, parameter shapes).
7. Run full test suite.

### Phase 2 – Forward Pipeline Decomposition
1. Extract `_prepare_inputs(self, x)` for input sanitisation and residual gating.
2. Extract `_run_node_stream(self, node_in)` returning node features plus auxiliary artifacts.
3. Extract `_run_edge_stream(self, edge_in, node_feats)` handling disabled edge stream gracefully.
4. Extract `_apply_fusion(self, node_feats, edge_feats)` producing logits.
5. Extract `_apply_postprocess(self, logits, aux)` for clamping/monitoring.
6. Rewrite `forward` to compose these helpers sequentially with clear docstrings on tensor shapes.
7. Add targeted unit tests for each helper (mock modules as needed) asserting shape and value constraints.

### Phase 3 – Documentation & Cleanup
- Refresh type hints and section headers; ensure helpers are `_`-prefixed unless intentionally public.
- Update architecture docs (`docs/04-model/v3-architecture.md`) if the structure diagram or description changes.
- Note new helper locations in developer documentation and TODO trackers.

### Phase 4 – Regression Validation
- Re-run `make q`, `make t`, `make test`, and smoke training (`make s`).
- Compare saved `state_dict` with baseline to confirm key alignment (allowing for reordered submodule registration if names are unchanged).
- Update release notes / audit docs to reflect completion.

## 5. Testing Strategy
- Existing suites remain primary safety net.
- New tests under `tests/unit/models/test_detector_builders.py` and `test_detector_pipelines.py` to validate:
  - Builder helpers respect config variations (graph disabled, PR toggles, warmup schedules).
  - Pipeline helpers return tensors within expected bounds and detect NaNs via monitors.
  - Regression check ensuring logits match baseline within tolerance on synthetic input.

## 6. Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| `state_dict` key drift after extraction | Keep attribute names identical; add regression asserting expected keys. |
| Hidden coupling with PR feature flags | Expand unit tests to cover each flag combo before refactor; document assumptions per helper. |
| Performance regression due to extra calls | Helpers remain lightweight; profile forward pass pre/post (PyTorch profiler). |
| Merge conflicts with concurrent feature work | Coordinate scheduling post-training; keep commits small and focused. |

## 7. Rollback Plan
- Execute phases on dedicated branch with incremental commits.
- If regression detected, revert to baseline commit while retaining newly added tests for future attempt.
- Restore baseline `state_dict` to validate rollback success.

## 8. Open Questions
1. Should builder helpers be exposed publicly for reuse (e.g., alternate detectors), or remain internal?
2. Are upcoming V4 model experiments depending on current monolithic structure? Coordinate before landing refactor.
3. Do we want to introduce dataclass/typed containers for helper outputs to make contracts explicit?
