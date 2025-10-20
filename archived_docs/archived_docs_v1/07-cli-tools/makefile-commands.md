# Makefile Commands

**Last Updated**: October 16, 2025 (v4.0.0)

## Core Development

- `make q` (quality) — Lint + format + mypy + config-check
- `make t` (test-fast) — Fast tests without coverage
- `make ts` (test-safe) — Training-safe tests (CPU only) — **USE DURING TRAINING**
- `make test` — Full test suite with coverage
- `make setup` — Initial project setup
- `make setup-gpu` — Install GPU stack (Mamba+PyG+TCN)
- `make setup-fla` — Install FLA library (Gated DeltaNet research stack)

## Local Training (RTX 4090)

**Dual Production Stacks** (BiMamba2 baseline + FLA research):

- `make smoke-bimamba` — BiMamba2 smoke test (1 epoch, 3 files)
- `make smoke-fla` — FLA smoke test (1 epoch, 3 files)
- `make train-bimamba` — BiMamba2 full training (100 epochs)
- `make train-fla` — FLA full training (100 epochs)

**Aliases** (default to BiMamba2):
- `make s` → `make smoke-bimamba`
- `make train-local` → `make train-bimamba`

## Modal Cloud Deployment (A100-80GB)

**Cache Management**:
- `make create-manifests` — Build train/dev manifests for balanced sampling
- `make upload-cache` — Upload cache to S3 (includes manifests)
- `make populate-modal` — Populate Modal cache from S3 (uses --detach)

**Dual-Stack Training**:
- `make smoke-modal-bimamba` — BiMamba2 smoke test (50 files, uses --detach)
- `make smoke-modal-fla` — FLA smoke test (50 files, uses --detach)
- `make train-modal-bimamba` — BiMamba2 full training (uses --detach)
- `make train-modal-fla` — FLA full training (uses --detach)

**Aliases** (default to BiMamba2):
- `make smoke-modal` → `make smoke-modal-bimamba`
- `make train-modal` → `make train-modal-bimamba`

**Complete Pipeline**:
- `make deploy-modal` — Full deployment (upload → populate → train BiMamba2)

## Testing

- `make test-integration` — Integration tests only
- `make test-performance` — Performance benchmarks (skip during training)
- `make test-gpu` — GPU-specific tests
- `make test-cpu` — CPU tests (safe during training)
- `make test-edge` — Edge case tests
- `make test-clinical` — Clinical validation suite
- `make test-all` — ALL tests including performance

## Utilities

- `make lint` — Run ruff linter
- `make lint-fix` — Fix lint issues and format
- `make format` — Format code with ruff
- `make type-check` — Run mypy type checking
- `make config-check` — Validate YAML configs
- `make clean` — Clean all artifacts
- `make hooks` — Run pre-commit hooks
- `make update` — Update all dependencies
- `make notebook` — Start Jupyter notebook
- `make tensorboard` — Start TensorBoard
