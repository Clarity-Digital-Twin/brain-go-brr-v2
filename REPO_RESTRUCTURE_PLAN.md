# Repository Operationalization & Refactor Plan

Status: draft for team review

This plan turns the current research-leaning layout into a production‑ready, testable, and extensible Python package while preserving public APIs during migration. No code changes in this patch — this is a roadmap.

## Current Snapshot (high level)

- Root: README.md, AGENTS.md, phase docs (PHASE1…PHASE5), Makefile, pyproject.toml
- Code: `src/experiment/` contains most modules (data, models, pipeline, postprocess, evaluate, schemas, events, export, streaming)
- CLI: `src/cli.py` with entry‑points
- Tests: `tests/` mirror major features (encoder/decoder/rescnn/mamba/model/training/postprocess/evaluate)
- Configs: `configs/{local,production,seizure_local}.yaml`

Observation: functionality is sound and covered, but most logic lives under a single namespace (`experiment`) with large files (e.g., `models.py`, `pipeline.py`). This is maintainable for a research sprint, but harder to operate at scale.

## Goals

- Stable public API and clear module boundaries
- Operational CLI with subcommands (train, eval, export) and typed configs
- Strong test structure: unit, integration, CUDA‑optional tests
- Clean separation of model, training, post‑processing, evaluation, and infra concerns
- Backwards compatible migration (no breaking changes to users during refactor)

## Target Package Layout

```
src/brain_brr/
  __init__.py                 # version, top‑level re‑exports
  cli/
    __init__.py
    main.py                   # Typer/Click app, `brain-brr` entrypoint
    commands/
      train.py                # train/eval/export subcommands
      eval.py
      export.py
  config/
    __init__.py
    schemas.py                # Pydantic models (Data/Model/Post/Eval/Train/Experiment)
    defaults.py               # default configs, helpers
    migrate.py                # legacy config shims
  data/
    __init__.py
    io.py                     # EDF I/O with MNE
    preprocess.py             # filtering, montage, resample, z‑score
    windows.py                # windowing utilities
    datasets.py               # EEGWindowDataset (+ streaming variant)
  models/
    __init__.py
    unet.py                   # encoder/decoder
    rescnn.py                 # residual temporal conv stack
    mamba.py                  # Bi‑Mamba‑2 layer + stack, CUDA/CPU dispatch
    detector.py               # full model wiring
    layers/                   # small reusable layers/blocks
  train/
    __init__.py
    loop.py                   # train/validate epoch logic
    losses.py
    optim.py                  # optimizers, schedulers, clipping
    sampler.py                # balanced sampling
    early_stopping.py
    checkpoints.py            # save/load
  post/
    __init__.py
    hysteresis.py             # dual‑tau with stability windows
    morphology.py             # CPU/GPU paths
    duration.py               # min/max, segmentation
    stitch.py                 # overlap‑add variants
  events/
    __init__.py
    intervals.py              # mask↔intervals
    merge.py                  # merging with tau_merge
    confidence.py             # mean/peak/percentile
    export.py                 # CSV_BI writer
  eval/
    __init__.py
    metrics.py                # AUROC, TAES helpers
    taes.py                   # TAES implementation
    fa_threshold.py           # τ_on search, thresholds table
    sensitivity.py            # sensitivity@{10,5,1} FA/24h
  infra/
    __init__.py
    device.py                 # CUDA gating, seeds
    logging.py                # logging setup
    cache.py                  # caching keys, dirs
    paths.py                  # results dirs, manifests
  utils/
    __init__.py
    tensor.py                 # common tensor helpers
    timing.py
```

Notes
- Keep public defaults (e.g., `d_conv=5`) and enforce CUDA kernel coercion internally (already implemented).
- Post‑processing and evaluation are strictly separated; evaluation consumes post APIs.
- CLI becomes a thin layer wiring configs to modules.

## Migration Strategy (no user breakage)

1) Introduce new package alongside existing code
   - Add `src/brain_brr/` with skeleton + adapters that import current `src/experiment` modules.
   - Keep tests green by re‑exporting in `src/brain_brr/__init__.py` from the old modules initially.

2) Move modules incrementally
   - Split `src/experiment/models.py` into `models/{unet,rescnn,mamba,detector}.py` and update imports.
   - Move post‑processing (`postprocess.py`) to `post/` submodules.
   - Move events/export and evaluation pieces into their dedicated packages.
   - Keep thin shims in `src/experiment/` that import from `brain_brr.*` and emit deprecation warnings.

3) Update CLI and entrypoints
   - Introduce `brain-brr` CLI (Typer/Click) under `brain_brr.cli`.
   - Route pyproject `[project.scripts]` to the new CLI while keeping old commands as aliases for one release.

4) Configs & schemas
   - Keep Pydantic schemas; add `config.migrate` to normalize legacy YAML keys.
   - Add `defaults.py` and a `brain-brr config validate` subcommand.

5) Tests (TDD)
   - Mirror test tree: `tests/brain_brr/...` while keeping current tests until parity reached.
   - Add CUDA‑only tests for Mamba dispatch (skip if no CUDA).
   - Add stitching tests covering cross‑window merges and time accounting.

6) CI & quality
   - GitHub Actions: CPU matrix (3.11, Linux). Optional CUDA workflow (nightly) guarded by label.
   - Enforce `make q` (ruff + format + mypy) and `pytest -q` in CI.
   - In CPU‑only CI, set `SEIZURE_MAMBA_FORCE_FALLBACK=1` to silence CUDA probing.

7) Deprecate legacy namespace
   - After 1–2 releases with shims, remove `src/experiment/`.
   - Provide a concise migration guide (import path changes only).

## Ideal Tests Layout

```
tests/
  unit/
    data/   models/   post/   events/   eval/   infra/
  integration/
    training_loop_test.py
    eval_pipeline_test.py
    stitching_endtoend_test.py
  gpu/
    test_mamba_cuda.py   # skip if no CUDA
```

## Makefile & Scripts

- `make q` → ruff check+format, mypy
- `make test` → coverage HTML
- `make test-fast` → -q -n auto
- `make eval-dev` / `make export-dev` → convenience targets for evaluation/export

## Backwards Compatibility Contracts

- Keep model, training, and evaluation behavior identical during refactor
- Preserve public config defaults; add migration shim for renamed fields
- Maintain CPU/GPU dispatch behavior and env overrides

## Sequencing & Effort (rough)

- Week 1: introduce new package skeleton + adapters; move post‑processing and events; add CLI shell
- Week 2: split models; move evaluation; update tests; add CUDA tests
- Week 3: finalize CLI, docs, manifests; deprecation notices; CI polish

## Risks & Mitigation

- Import churn → solved by adapters and deprecation shims
- Test fragility → migrate with TDD mirroring; keep old tests until parity
- CUDA variance → isolate with explicit skip markers; keep CPU deterministic paths

## Why this is better

- Clear boundaries and small modules improve readability
- Easier onboarding for contributors; reduces “research dump” feel
- Extensible for future phases (streaming inference, new post‑processing)
- Production‑grade CLI and CI support reproducibility and deployment

---

Appendix: Quick mapping from current modules

- `src/experiment/data.py` → `brain_brr.data.{io,preprocess,windows,datasets}`
- `src/experiment/models.py` → `brain_brr.models.{unet,rescnn,mamba,detector}`
- `src/experiment/pipeline.py` → `brain_brr.train.{loop,early_stopping,checkpoints}`
- `src/experiment/postprocess.py` → `brain_brr.post.{hysteresis,morphology,duration,stitch}`
- `src/experiment/events.py` → `brain_brr.events.{intervals,merge,confidence}`
- `src/experiment/export.py` → `brain_brr.events.export`
- `src/experiment/evaluate.py` → `brain_brr.eval.{metrics,taes,fa_threshold,sensitivity}`
- `src/experiment/schemas.py` → `brain_brr.config.schemas`
- `src/cli.py` → `brain_brr.cli.main`

