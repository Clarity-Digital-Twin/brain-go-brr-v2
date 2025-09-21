Modal Deployment — Single Source of Truth

Principles
- Do not start training until cache manifest shows seizures (partial>0 or full>0)
- Keep data and cache mounts explicit; avoid writing to ephemeral paths by accident
- Kill long cache builds quickly if preflight fails

Preflight (Modal)
1) Verify project image builds and `make q` passes in container
2) Mount EDF+CSV root and a persistent cache directory
3) Build or rescan manifest on the mounted cache dir
4) Confirm partial>0 or full>0; otherwise STOP

Core commands (examples)
- Build cache (in container):
  - `python -m src build-cache --data-dir /mount/edf/train --cache-dir /mount/cache/train`
- Scan manifest:
  - `python -m src scan-cache --cache-dir /mount/cache/train`
- Train (auto-uses balanced when manifest exists):
  - `python -m src train configs/tusz_train_a100.yaml`

Cost control
- Prefer building cache once and reusing
- Validate manifest before launching large GPU jobs
- If a run shows 0 seizures: kill immediately and fix CSV/paths

Code anchors
- Modal entrypoint: `deploy/modal/app.py` (or `deploy/modal/README.md` for local CLI)
- Data pipeline details: docs_new/TUSZ/*

Troubleshooting
- 0 windows in BalancedSeizureDataset: re-scan manifest; fix CSV_BI parser; rebuild cache
- Slow cache build: ensure EDF/CSV and cache mounts are on fast storage
- Memory errors: reduce batch size; verify A100/L40 profile in config

