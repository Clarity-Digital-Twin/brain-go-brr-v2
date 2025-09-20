# Documentation SSOT (Single Source of Truth)

Date: 2025-09-20

This file lists the canonical, up-to-date entry points and guides. Older docs may remain for historical context; when in doubt, follow these.

Canonical files:
- Deploy (Modal): `deploy/modal/app.py`
- CLI: `src/brain_brr/cli/cli.py`
- Configs:
  - Local long-run (WSL2-safe): `configs/tusz_train_wsl2.yaml`
  - Cloud A100 training: `configs/tusz_train_a100.yaml`
  - Smoke: `configs/smoke_test.yaml`
  - Dev/eval (tuning/test only): `configs/tusz_dev_tuning.yaml`, `configs/tusz_eval_final.yaml`

Canonical commands:
- Local smoke: `python -m src train configs/smoke_test.yaml`
- Local full (WSL2-safe): `python -m src train configs/tusz_train_wsl2.yaml`
- Modal smoke: `modal run deploy/modal/app.py --action train --config configs/smoke_test.yaml`
- Modal full (A100): `modal run deploy/modal/app.py --action train --config configs/tusz_train_a100.yaml --detach`
- Evaluate (dev): `python -m src evaluate <checkpoint.pt> data_ext4/tusz/edf/dev --config configs/tusz_dev_tuning.yaml --output-json results/dev_metrics.json`
- Evaluate (final): `python -m src evaluate <checkpoint.pt> data_ext4/tusz/edf/eval --config configs/tusz_eval_final.yaml --output-json results/final_metrics.json`

Notes on deprecated references:
- `modal_train.py` — replaced by `deploy/modal/app.py`
- `configs/production.yaml` — superseded by environment-specific configs above
- `src/experiment/*` — refactored into `src/brain_brr/*`

If you find drift, update the target document and add a bullet here if it affects canonical usage.
