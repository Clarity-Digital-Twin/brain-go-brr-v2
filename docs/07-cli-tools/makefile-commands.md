# Makefile Commands

- `make q` — Quality (lint+format+mypy)
- `make t` — Fast tests
- `make test` — Full tests with coverage
- `make setup` — Base env
- `make setup-gpu` — GPU stack (Mamba+PyG)
- `make s` — Smoke test
- `make train-local` — Full local training

## Modal Deployment Targets

- `make create-manifests` — Create train and dev manifests for balanced sampling
- `make upload-cache` — Upload cache to S3 (includes manifests)
- `make populate-modal` — Populate Modal cache from S3 (uses --detach)
- `make smoke-modal` — Run Modal smoke test (uses --detach)
- `make train-modal` — Start Modal training (uses --detach)
- `make deploy-modal` — Complete Modal deployment pipeline (upload → populate → train)
