# Technical Debt & Cleanup Plan (Authoritative)

This document is the single source of truth for technical debt and cleanup work. It lists exact code locations, phased deprecations, acceptance criteria, and verification steps. Line numbers are avoided to prevent drift; we reference modules, classes, and methods explicitly.

## ✅ Completed (V3 landing)
- [x] V3 dual‑stream path implemented in `src/brain_brr/models/detector.py` (`SeizureDetector.forward` V3 branch, `SeizureDetector.from_config` V3 setup)
- [x] Edge features + learned adjacency assembled in `src/brain_brr/models/edge_features.py` (`edge_scalar_series`, `assemble_adjacency`)
- [x] PyG GNN with Laplacian PE implemented in `src/brain_brr/models/gnn_pyg.py` (`GraphChannelMixerPyG`, vectorized path)
- [x] Detector consumes learned adjacency in V3 path (proj_to_electrodes → node_mamba/edge_mamba → assemble_adjacency → GNN → proj_from_electrodes)
- [x] Config surface updated for V3 (`ModelConfig.graph.edge_*`, `use_dynamic_pe`, `k_eigenvectors` in `src/brain_brr/config/schemas.py`)

## 🔎 Blocking Debt Summary
- Dual path complexity: V2 (heuristic graph) and V3 (edge stream) coexist in `SeizureDetector` increasing branching and surface area.
- Legacy parameters and config objects still exposed or referenced despite being unused (e.g., `encoder`, `rescnn`, legacy kwargs in `SeizureDetector.__init__`).
- Environment variable sprawl: multiple BGB_* toggles spread across modules; partial documentation in multiple places.
- Messaging mismatches (e.g., W&B model descriptor, CLI summaries) still reflect pre‑TCN state.

## 🧭 Deprecation & Removal Plan (Phased)

### Phase 0 — Alignment (no breaking changes) ✅ COMPLETE (commit 25a9362)
- [x] Update messaging to reflect TCN+BiMamba(+V3) everywhere
  - [x] `src/brain_brr/train/wandb_integration.py`: change model label to "TCN + Bi-Mamba-2 (+V3 edge/GNN)" and drop UNet/ResCNN fields
  - [x] `src/brain_brr/cli/cli.py`: config summary should show `model.architecture`, V3 graph settings, and replace deprecated `postprocessing.min_duration` with `postprocessing.duration.min_duration_s`
  - [ ] Docs: ensure `docs/04-model/v3-architecture.md`, `docs/00-overview/architecture-summary.md` match current code

### Phase 1 — Soft deprecation (warnings, defaults)
- [ ] Emit `DeprecationWarning` when:
  - [ ] `ModelConfig.architecture == "tcn"` in `SeizureDetector.from_config` (file: `src/brain_brr/models/detector.py`)
  - [ ] `SeizureDetector.__init__` receives legacy kwargs: `base_channels`, `encoder_depth`, `rescnn_blocks`, `rescnn_kernels`, and `dropout` as a surrogate for `mamba_dropout`
  - [ ] `DynamicGraphBuilder` is constructed (file: `src/brain_brr/models/graph_builder.py`)
- [ ] Set default architecture to `"v3"` in `ModelConfig` (keeping `"tcn"` allowed but deprecated)
- [ ] Add warning when `graph.use_pyg=false` (V2 heuristic path) suggesting migration to V3

### Phase 2 — Test and config migration (breaking tests only)
- [ ] Update tests to stop relying on V2 and legacy params (see inventories below)
  - [ ] Replace `architecture="tcn"` with `"v3"` where feasible
  - [ ] Remove creation/use of `DynamicGraphBuilder` assertions
  - [ ] Stop passing legacy kwargs to `SeizureDetector` in tests/fixtures
  - [ ] Update fixtures to use `ModelConfig` with V3 graph config where graph behavior is required
- [ ] Update example configs and docs to V3‑only (local + modal)

### Phase 3 — Removal (breaking API)
- [ ] Remove V2 code paths from `SeizureDetector.forward` and `from_config`
- [ ] Delete `src/brain_brr/models/graph_builder.py`
- [ ] Remove V2‑only fields from `GraphConfig`: `similarity`, `top_k`, `threshold`, `temperature`
- [ ] Remove legacy kwargs from `SeizureDetector.__init__`
- [ ] Remove deprecated `encoder`/`rescnn`/`decoder` objects from `ModelConfig` (or gate behind a compatibility flag for a final release)

### Verification gates (each phase)
- [ ] `make q` passes (ruff/format/mypy)
- [ ] `make t` passes; mark/skip GPU when not available
- [ ] Smoke run works locally: `make s` (CPU OK, PyG optional path guarded)
- [ ] Modal smoke config continues to run (if applicable)

---

## 🧹 Priority 1: Merge to Single V3 Path

**Actions:**
- [ ] Introduce deprecation warnings as above (Phase 1)
- [ ] Change `ModelConfig.architecture` default to `"v3"` and update docstrings
- [ ] Remove V2 forward branch in `SeizureDetector.forward` and the `use_gnn + graph_builder` path
- [ ] Remove V2 setup logic in `SeizureDetector.from_config` (heuristic graph builder import and projections for v2)
- [ ] Delete `src/brain_brr/models/graph_builder.py`
- [ ] Prune V2‑only fields from `GraphConfig` and callers

**Acceptance criteria:**
- No conditional branches on `architecture` remain in `SeizureDetector`
- No imports or references to `graph_builder` remain in repo
- Tests no longer assert V2 behavior and pass on V3 (CPU path remains valid)

**Impacted files:**
- `src/brain_brr/models/detector.py`
- `src/brain_brr/models/graph_builder.py` (delete)
- `src/brain_brr/config/schemas.py` (`GraphConfig`, `ModelConfig`)
- `tests/**/*` (see inventory below)
- Docs under `docs/04-model/*`, `docs/03-configuration/*`

## 🧹 Priority 2: Legacy Parameter Cleanup

**Actions:**
- [ ] Add `DeprecationWarning` in `SeizureDetector.__init__` if any legacy kwargs are passed: `base_channels`, `encoder_depth`, `rescnn_blocks`, `rescnn_kernels`, and using `dropout` as proxy for `mamba_dropout`
- [ ] Migrate call sites to only use config‑driven construction (`SeizureDetector.from_config(cfg)`) in tests and examples
- [ ] Remove legacy kwargs entirely (Phase 3)

**Acceptance criteria:**
- No tests pass legacy kwargs directly to `SeizureDetector`
- Mypy signatures reflect the reduced surface area

**Impacted files:**
- `src/brain_brr/models/detector.py`
- `tests/unit/train/test_loop.py` (direct constructor usage)
- `tests/**/conftest.py` (fixtures)

## 🧪 Priority 3: Test Suite Hardening

**Actions:**
- [ ] Replace V2‑specific tests with V3 equivalents
- [ ] Add resource cleanup fixtures to avoid worker/process leaks (torch, dataloaders, CUDA) in `tests/conftest.py`
- [ ] Ensure timeouts on slow integration/performance tests are present and calibrated
- [ ] Add CPU‑only V3 smoke forward for environments without PyG (GNN disabled) to keep CI green
- [ ] Investigate worker crashes if they still occur (run `pytest tests/ -n 4` to test parallel execution)

**Acceptance criteria:**
- Tests pass deterministically on CPU (no intermittent worker crashes)
- GPU tests are either gated or skip cleanly without CUDA

**Known V2‑dependent tests to migrate/remove:**
- `tests/integration/test_gnn_integration.py` and `tests/integration/test_gnn_integration_pyg.py` (assertions about graph_builder usage)
- `tests/unit/models/test_tcn.py` sections asserting `architecture="tcn"`
- `tests/unit/models/test_detector_v3.py::test_v2_still_works` (remove)
- `tests/unit/train/test_loop.py` direct `SeizureDetector(...)` with legacy kwargs

## 🧰 Priority 4: Config & CLI/W&B Consistency

**Actions:**
- [ ] `ModelConfig`: default `architecture="v3"`; plan removal of `encoder/rescnn/decoder` blocks after one release with warnings
- [ ] CLI summary in `src/brain_brr/cli/cli.py`
  - [ ] Show `model.architecture`, V3 graph knobs when present
  - [ ] Replace `postprocessing.min_duration` with `postprocessing.duration.min_duration_s`
- [ ] W&B metadata in `src/brain_brr/train/wandb_integration.py`
  - [ ] Update model string; drop UNet/ResCNN keys; add `model.architecture`, `tcn.*`, `graph.edge_*`

**Acceptance criteria:**
- `cli validate` and summary output reflect V3 accurately
- W&B runs show correct model description and keys (no UNet/ResCNN)

## 🧯 Priority 5: Env Vars → Single Source + Optional Config

**Inventory (current usage):**
- Model/streaming toggles
  - `BGB_EDGE_CLAMP`, `BGB_EDGE_CLAMP_MIN`, `BGB_EDGE_CLAMP_MAX` in `src/brain_brr/models/detector.py`
  - `SEIZURE_MAMBA_FORCE_FALLBACK` in `src/brain_brr/models/mamba.py`
  - `BGB_FORCE_TCN_EXT` in `src/brain_brr/models/tcn.py`
- Training/runtime
  - `BGB_SMOKE_TEST`, `BGB_LIMIT_FILES`, `BGB_FORCE_MANIFEST_REBUILD`, `BGB_DISABLE_TQDM`, `BGB_DISABLE_TB`, `BGB_MID_EPOCH_MINUTES`, `BGB_MID_EPOCH_KEEP`, `BGB_NAN_DEBUG`, `BGB_NAN_DEBUG_MAX`, `BGB_SANITIZE_INPUTS`, `BGB_SANITIZE_GRADS`, `BGB_SKIP_OPT_STEP_ON_NAN`, `BGB_ANOMALY_DETECT` in `src/brain_brr/train/loop.py` and `src/brain_brr/data/cache_utils.py`
- Testing/perf
  - `BGB_PERF_ALLOW_GPU`, `BGB_PERF_THREADS` in `tests/performance/*`
- Tooling/WSL2
  - `UV_LINK_MODE` (docs/install), optional env during setup

**Actions:**
- [ ] Centralize documentation: ensure all variables above appear in `docs/03-configuration/env-vars.md` (missing today: `BGB_EDGE_CLAMP*`, `BGB_DEBUG_FINITE`, `BGB_SANITIZE_GRADS`, `BGB_SKIP_OPT_STEP_ON_NAN`, perf vars)
- [ ] Add optional `DebugConfig` or `Experiment.debug: bool` in schemas to replace common toggles (nan debug, sanitize inputs/grads); env vars remain as overrides for quick experiments
- [ ] Add a small helper `src/brain_brr/utils/env.py` to read env once and expose typed accessors (prevents scattering)

**Acceptance criteria:**
- One canonical doc page lists all env vars and their default behavior
- Training loop reads from a single helper or config path for debug toggles

## 🧮 Priority 6: Numerical Guards Lifecycle

**Actions:**
- [ ] Keep `assert_finite` behind `BGB_DEBUG_FINITE` (already implemented in `src/brain_brr/models/debug_utils.py`); add a short doc link from V3 doc
- [ ] Maintain edge clamp in V3 path (`BGB_EDGE_CLAMP*`) with a plan to disable by default after stability milestone

**Disable plan (post‑stability):**
- [ ] Prove stability on full train (no NaN explosions) → flip default `BGB_EDGE_CLAMP=0` and remove guards after one release

## 🧾 Priority 7: Documentation Sweep

**Actions:**
- [ ] Update architecture pages to V3‑only path, drop V2 heuristics except for historical context
- [ ] Ensure config READMEs show V3 knobs (edge_*), and remove mentions of heuristic similarity/top_k/threshold after removal
- [ ] Modal/local guides: verify batch size, AMP guidance, and PE notes are consistent with code

**Acceptance criteria:**
- No references to UNet/ResCNN or heuristic graph builder remain in living docs

---

## 📦 Inventories (for migration work)

**Tests referencing V2 or legacy kwargs (must be updated):**
- `tests/unit/models/test_detector_v3.py`: `test_v2_still_works` asserts V2 path via heuristic graph builder
- `tests/unit/models/test_tcn.py`: several tests assert `architecture="tcn"`; convert to V3 or limit scope to TCN encoder only
- `tests/unit/train/test_loop.py`: constructs `SeizureDetector(...)` with legacy kwargs (`base_channels`, `encoder_depth`, `rescnn_blocks`, `rescnn_kernels`) — switch to `from_config` and remove legacy kwargs
- `tests/integration/test_gnn_integration.py` and `tests/integration/test_gnn_integration_pyg.py`: remove any reliance on heuristic adjacency, focus on V3 edge stream + GNN

**Files referencing deprecated config objects:**
- `src/brain_brr/train/wandb_integration.py`: logs `encoder/rescnn`; update to TCN/V3 fields
- `src/brain_brr/cli/cli.py`: prints `postprocessing.min_duration`; migrate to `postprocessing.duration.min_duration_s`
- `src/brain_brr/config/schemas.py`: `ModelConfig` still includes `encoder`, `rescnn`, `decoder` for compatibility; plan removal

**Env vars missing in the canonical doc (to add in `docs/03-configuration/env-vars.md`):**
- Model/stream: `BGB_EDGE_CLAMP`, `BGB_EDGE_CLAMP_MIN`, `BGB_EDGE_CLAMP_MAX`, `BGB_DEBUG_FINITE`
- Training: `BGB_SANITIZE_GRADS`, `BGB_SKIP_OPT_STEP_ON_NAN`
- Perf tests: `BGB_PERF_ALLOW_GPU`, `BGB_PERF_THREADS`

---

## 🔐 Acceptance Checklist (ready to ship V3‑only)
- [ ] No V2 code paths or files remain
- [ ] No tests reference `architecture="tcn"` or heuristic graph builder
- [ ] No legacy kwargs accepted by `SeizureDetector.__init__`
- [ ] W&B label and fields reflect TCN+BiMamba(+V3) accurately
- [ ] `docs/03-configuration/env-vars.md` enumerates all BGB_* toggles used in code
- [ ] `make q` and `make t` pass; smoke `make s` completes

## 🧱 Simplification Target
- `src/brain_brr/models/detector.py` can drop ~100 lines by removing V2 branches and legacy handling; goal: ≤ ~250 lines with clear V3 flow and helpers

## 🧪 Optional Future Enhancements (from EvoBrain roadmap)
- [ ] Alternative edge sequence models (GRU/LSTM) behind a config flag
- [ ] True K‑hop SSGConv or Chebyshev polynomial filters
- [ ] Additional edge features (e.g., coherence) with pluggable metric interface

## ⚠️ Risk Mitigation Strategy

### Critical Safety Checks Before Each Phase:
1. **Backup branch**: Create `pre-cleanup-phase-X` branch before starting each phase
2. **Test baseline**: Run full test suite and save results before changes
3. **Smoke test**: After each change, run `make s` to ensure basic functionality
4. **Rollback plan**: If tests fail, `git reset --hard` to last known good state

### Order of Operations (safest sequence):
1. **Phase 0 first** (messaging only) - Zero risk to functionality
2. **Add all warnings** before removing anything
3. **Test migration** before code removal
4. **Keep V2 path** until V3 proven stable in production (at least 1 release cycle)
