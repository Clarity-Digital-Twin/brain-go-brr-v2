# Checkpointing and Resume

The training loop is designed to survive Modal’s 24 h timeout and to recover from local interruptions without losing progress.

## Quick Reference

- **Local (BiMamba2)**: `python -m src train configs/local/train_bimamba.yaml --resume`
- **Modal (BiMamba2)**: `modal run --detach deploy/modal/app.py --action train --config configs/modal/train_bimamba.yaml --resume true`
- Swap to `*_train_fla.yaml` when resuming the FLA stack.
- `training.resume: true` in a YAML config has the same effect as `--resume` on the CLI.

## Load Order (Highest Priority First)

When `resume` is enabled, checkpoints are loaded in this order:

1. Latest `mid_epoch_*.pt` (saved every `training.mid_checkpoint_interval_s`, default 1800 s).
2. `timeout_exit.pt` (written when the wall-clock guard triggers at ~23 h on Modal).
3. `last.pt` (end-of-epoch snapshot).
4. Fresh start (no checkpoints found).

Each file contains model weights, optimizer, scheduler, AMP scaler, and RNG state (Python, NumPy, torch CPU, torch CUDA) so the resumed run continues exactly where it stopped.

## Atomic Saves & Deterministic State (v3.9.1)

- `save_checkpoint()` now writes to `<name>.pt.tmp`, calls `os.fsync()`, and atomically renames to `<name>.pt`. Partial/corrupt checkpoints can no longer appear if a job is killed mid-write.
- Loading legacy checkpoints (pre-v3.9.0) still works; the loader logs a warning when scaler/RNG fields are missing and falls back to best-effort resume.
- Mid-epoch saves are rotational (`training.mid_epoch_keep`, default 3) to keep disk usage small (<2 GB total).

## Modal Timeout Behaviour

- `deploy/modal/app.py` sets `BGB_WALL_CLOCK_LIMIT_S=82800` (23 h). The guard writes `timeout_exit.pt`, logs a warning, and exits cleanly about an hour before Modal’s enforced 24 h limit.
- Relaunch training with `--resume true`; the loader prefers `timeout_exit.pt` and training continues with no repeated batches.
- After a resume the guard resets automatically—expect to relaunch ~4–5 times over a 100-epoch run.

## W&B Run Persistence

- `wandb_integration.py` now stores the run ID in `.wandb_run_id` inside the checkpoint directory. This file is rewritten on every launch so resumes pick up the same dashboard automatically.
- Logs show `[W&B] Run resumed: …` when continuity succeeds. If the file is missing, the code creates a new run ID and logs `[W&B] Run created: …`.

## Verifying a Resume

- Check the logs for `[CHECKPOINT] Loading … mid_epoch_...` or `timeout_exit.pt`.
- Ensure the next log line reports the same epoch/batch you expect (e.g., `Resumed from epoch 3`).
- In W&B, metrics continue on the prior run; no new sweep entry should appear.

For deeper details see `docs/05-training/checkpoint-strategy.md` and `src/brain_brr/train/checkpoint.py`.
