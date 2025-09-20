Codebase Cleanup and Training Readiness Report

Summary
- All tests pass: 150 passed, 2 skipped
- Lint, format, type check clean (ruff, mypy)
- Train/val split enforced inside training loop; no cross‑split leakage
- Configs aligned to use train/dev/eval correctly

Fixes applied
- CLI evaluate export: fixed DataLoader batch handling and sigmoid
  - Changed tuple vs dict access and convert logits→probs before eventization
  - File: src/brain_brr/cli/cli.py:311–319 → now iterates `for windows, _ in dataloader` and applies `torch.sigmoid`
- Config batch_size placement: moved from data → training
  - File: configs/production.yaml:10 removed, training.batch_size added at 58
  - File: configs/local.yaml:11 removed, training.batch_size added at 44

Validated items
- Splits
  - Train configs point to `.../edf/train`
  - Dev tuning uses `.../edf/dev` with epochs=0 (no training)
  - Final eval uses `.../edf/eval` with epochs=0 (no training)
- Training behavior
  - Uses `training.batch_size` for loaders and respects `validation_split`
  - Mixed precision only when device is CUDA/MPS (schema-enforced)
- Post‑processing and evaluation
  - Hysteresis thresholds and morphology wired correctly
  - Event conversion paths consistent with tests

Notable caveats (non‑blocking)
- Evaluating without labels: metrics like TAES/ROC require labels; running evaluate on unlabeled data will only produce events/exports but not meaningful metrics.
- `configs/seizure_local.yaml` is explicitly marked deprecated; safe to keep for now or archive.
- `.gitignore` lists `uv.lock`, but the file is tracked. Intentional for reproducible env; no action taken.

Runbook (Updated 2025-09-19)
- Quality: `make q`
- Fast tests: `make t`
- Train (local, WSL2-safe): `python -m src train configs/tusz_train_wsl2.yaml`
- Train (Modal A100): `modal run deploy/modal/app.py --action train --config configs/tusz_train_a100.yaml --detach`
- Dev tuning (no training): `python -m src evaluate <best.pt> data_ext4/tusz/edf/dev --config configs/tusz_dev_tuning.yaml --output-json results/dev_metrics.json`
- Final eval (no training): `python -m src evaluate <best.pt> data_ext4/tusz/edf/eval --config configs/tusz_eval_final.yaml --output-json results/final_metrics.json --output-csv-bi results/final_events.csv`

Conclusion
- Codebase is ready to train. The two config corrections and the evaluate export fix remove the only training‑critical inconsistencies found.
