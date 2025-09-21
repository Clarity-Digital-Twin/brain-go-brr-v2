# Modal vs Local Training – Troubleshooting & Consistency Guide

This doc explains common divergences between local (WSL2) and Modal (A100) runs, how to detect them, and how to fix them quickly without losing time.

## Fast Checklist

- Local config uses `num_workers: 4`, `pin_memory: true`, `persistent_workers: true` for throughput.
- Modal uses `num_workers: 8`, `pin_memory: true` and disables tqdm (`BGB_DISABLE_TQDM=1`).
- Cache dirs point to the same root for data and experiment: local `cache/tusz`, Modal `/results/cache/tusz`.
- If `BalancedSeizureDataset` errors about missing seizures, rebuild the manifest.

## Symptoms & Root Causes

- Symptom: 15–25s/batch locally, GPU util ~10–20%.
  - Cause: DataLoader starvation (`num_workers: 0`, `pin_memory: false`).
  - Fix: Use `num_workers: 4`, `pin_memory: true`, `persistent_workers: true`, `prefetch_factor: 2`.

- Symptom: On Modal, log shows `BalancedSeizureDataset failed: No partial seizure windows found in manifest! ...; falling back to EEGWindowDataset`.
  - Cause: Stale/corrupt manifest in persistent volume, or path mismatch.
  - Fix: Auto-rebuild now happens on startup (see Changes below). You can also force rebuild via env or CLI.

- Symptom: Manifest has 0 windows even though cache has `.npz` files.
  - Cause: Early empty manifest cached; tqdm issues in subprocesses.
  - Fix: We disable tqdm on Modal and validate manifests before use.

## Changes Implemented (Keeps things tight)

- Added `validate_manifest(cache_dir, manifest)` to verify manifest integrity:
  - Non-empty window count.
  - References match files in `cache_dir` (allows up to 5% missing during partial updates).

- Training loop improvements (`src/brain_brr/train/loop.py`):
  - Deletes invalid/empty manifests automatically and rebuilds from cache.
  - Honors `BGB_FORCE_MANIFEST_REBUILD=1` to force a rebuild on startup.

- `BalancedSeizureDataset` now filters and warns on missing manifest references instead of crashing.

- Added unit tests for manifest validation logic.

All quality checks pass: `ruff`, formatting, `mypy`.

## Manual Controls

- Force manifest rebuild on next run:
  - Local: `BGB_FORCE_MANIFEST_REBUILD=1 .venv/bin/python -m src train configs/local/train.yaml`
  - Modal (already sets `BGB_DISABLE_TQDM=1`): set `BGB_FORCE_MANIFEST_REBUILD=1` in the Modal env before launching.

- Rebuild manifest explicitly from cache:
  - Local: `python -m src scan-cache --cache-dir cache/tusz/train`
  - Modal shell: `python -m src scan-cache --cache-dir /results/cache/tusz/train`

- Pre-build cache to avoid slow first epochs:
  - Local: `python -m src build-cache --data-dir data_ext4/tusz/edf/train --cache-dir cache/tusz/train`
  - Modal: `python -m src build-cache --data-dir /data/edf/train --cache-dir /results/cache/tusz/train`

## Watching Runs

- Local tmux:
  - Start: `tmux new -s train_full '.venv/bin/python -m src train configs/local/train.yaml'`
  - Attach: `tmux attach -t train_full` (Ctrl+B then D to detach)

- GPU utilization:
  - `watch -n 1 nvidia-smi` or `nvidia-smi --loop=1`

- Modal detached run:
  - `modal run --detach deploy/modal/app.py::train --config-path configs/modal/train_a100.yaml`

## Config Parity Notes

- Local (WSL2): `configs/local/train.yaml`
  - `num_workers: 4`, `pin_memory: true`, `persistent_workers: true`

- Modal (Linux A100): `configs/modal/train_a100.yaml`
  - `num_workers: 8`, `pin_memory: true`, persistent workers enabled, larger `batch_size`.
  - Cache and outputs under `/results` volume.

## When To Suspect Data Issues

- Balanced dataset size changes drastically across runs without code/config changes.
- Sudden drop to 0 seizures in manifest.

Use `scan-cache` to inspect manifest counts and verify cache contents quickly.

