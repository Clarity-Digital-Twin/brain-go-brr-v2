# Refactor Plan — src/brain_brr/data/io.py
⚠️ **AUDIT FAILED - DO NOT IMPLEMENT** ⚠️
Status: BLOCKED - Requires complete rewrite (2025-10-02)
Priority: MEDIUM (after rewrite)
Owner: Senior ML auditor (Codex)
Scope: EDF loading pipeline (`load_edf_file` ≈153 lines)

**CRITICAL ISSUE:** This plan incorrectly describes operations that do NOT exist in `load_edf_file`.
- Claims function does resampling → FALSE (not in function)
- Claims function does filtering → FALSE (not in function)
- Claims function does label alignment → FALSE (not in function)

See `REFACTOR_AUDIT_REPORT_2025-10-02.md` for details. Plan must be rewritten based on actual code.

## 1. Context & Problem Statement
`load_edf_file` (≈153 lines, current span 54–206) performs multiple stages in a single function:
1. Path resolution / file validation
2. EDF reading via pyEDFlib
3. Resampling to 256 Hz
4. Bandpass + notch filtering
5. Channel reordering / interpolation
6. Label alignment and metadata packaging

Coupling these responsibilities reduces readability, complicates profiling, and makes targeted testing difficult (e.g., unit testing interpolation alone).

## 2. Goals
1. Decompose pipeline into focused helpers with strong contracts while preserving behaviour.
2. Maintain backwards compatibility for dataset loaders and cached outputs (tensors, metadata).
3. Improve test granularity (synthetic signals to validate each stage independently).
4. Provide clear extension points for future optimisations (GPU filtering, caching, streaming).

## 3. Proposed Architecture
```
io.py
├── load_edf_file(...)            # orchestrator (≤120 lines)
├── _resolve_paths(...)
├── _read_edf(...)
├── _resample_signal(...)
├── _apply_filters(...)
├── _order_channels(...)
├── _interpolate_missing_channels(...)
├── _align_labels(...)
└── _package_output(...)
```
- Helpers may stay in same file (private) or move into `src/brain_brr/data/edf_pipeline/` if file length still high.

## 4. Step-by-Step Plan

### Phase 0 – Baseline Snapshot
- Run existing data unit/integration tests to ensure green state.
- Cache sample outputs (NumPy arrays) from `load_edf_file` for regression comparisons.

### Phase 1 – Path & Read Separation
1. Extract `_resolve_paths(edf_path, annotation_path)` for validation and metadata.
2. Extract `_read_edf(edf_path)` returning raw signal matrix, sample rate, channel labels.
3. Preserve existing exception handling; unit test with mocks to simulate missing files.

### Phase 2 – Signal Processing Helpers
1. Extract `_resample_signal(signal, original_rate, target_rate)` using existing resample logic.
2. Extract `_apply_filters(signal, sample_rate, filter_config)` encapsulating bandpass/notch filters.
3. Unit tests using synthetic sine waves verifying frequency preservation and notch removal.

### Phase 3 – Channel Management
1. Extract `_order_channels(signal, channel_names, target_order)`; keep fallback warnings.
2. Extract `_interpolate_missing_channels(signal, missing_indices, method)` (if logic currently inline).
3. Tests covering exact match, missing channels, synonyms, and interpolation behaviour.

### Phase 4 – Label Alignment & Output Packaging
1. Extract `_align_labels(events, sample_rate, n_samples)` for converting annotations to masks/events.
2. Extract `_package_output(signal, labels, metadata)` producing final dict.
3. Tests verifying durations, sample indices, metadata fidelity.

### Phase 5 – Regression Validation
- Re-run unit/integration tests (`tests/unit/data/test_datasets.py`, `tests/integration/data/test_io_edge_cases.py`).
- Compare cached sample outputs pre/post refactor with `numpy.allclose` tolerance 1e-6.
- Update docs/TODO to mark completion.

## 5. Testing Strategy
- Add new unit tests under `tests/unit/data/test_io_helpers.py` or similar.
- Use `pytest` fixtures to generate synthetic EDF-like data.
- Employ monkeypatch to mock pyEDFlib when focusing on downstream stages.

## 6. Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| Filter coefficients accidentally modified | Copy existing logic verbatim; add unit tests verifying filter response curves. |
| Increased runtime due to extra function calls | Ensure helpers operate in-place; profile using representative files. |
| Hidden dependencies on module-level globals | Pass required config explicitly to helpers; document defaults. |
| Event alignment edge cases change behaviour | Write regression tests capturing current clamping/overlap rules. |

## 7. Rollback Plan
- Keep each extraction in dedicated commit.
- If regression observed, revert to baseline commit while retaining new tests for future work.
- Restore cached baseline outputs for sanity check.

## 8. Open Questions
1. Should we pre-compute and cache filter coefficients globally? (Potential follow-up.)
2. Do we need to expose helpers for streaming inference pipelines? (Coordinate with streaming team.)
3. Any dependency on upcoming V4 data pipeline changes that should influence helper APIs?
