# Your First Training Run

**Last Updated**: October 17, 2025  
**Codebase Version**: v4.0.0 (FLA Production + WSL2 Fix)  
**Estimated Time**: 30 minutes to launch, ~960 h (40 days) for FLA on RTX 4090, ~700-1200 h for BiMamba2 on Modal A100 (both for 100 epochs)

---

## What You’ll Accomplish

- Verify caches and configs for the full pipeline
- Launch the 100‑epoch training job locally or on Modal
- Learn how to monitor, checkpoint, and resume safely (including the 23 h Modal timeout guard)
- Know where to look for metrics and troubleshooting info

---

## Prerequisites

- ✅ Quickstart smoke test succeeded (`docs/getting-started/quickstart.md`)
- ✅ Memory-mapped caches present:
  - Local: `cache/tusz_mmap/{train,dev}`
  - Modal: `/results/cache/tusz_mmap/{train,dev}` (verify with `modal run deploy/modal/app.py --action check-cache`)
- ✅ ~50 GB free disk for caches + checkpoints
- ✅ tmux installed (recommended for long local runs)

---

## Part 1 – Preparation (≈5 min)

### 1. Verify Cache Health

```bash
python -m src scan-cache --cache-dir cache/tusz_mmap/train
python -m src scan-cache --cache-dir cache/tusz_mmap/dev
```
You should see partial > 0 or full > 0. If either report is 0, rebuild or investigate before training.

### 2. Validate Configuration

```bash
python -m src validate configs/local/train_bimamba.yaml   # or configs/modal/train_bimamba.yaml
```
Look for the four ✅ lines indicating paths, model, training, and evaluation settings are valid.

### 3. Start a tmux Session (local runs)

```bash
tmux new -s train
```
Detach with `Ctrl+B, D` and reattach via `tmux attach -t train`.

---

## Part 2 – Launch Training

### Local (RTX 4090)

**BiMamba2 Stack** (`configs/local/train_bimamba.yaml`):
```bash
# Optional diagnostics
export BGB_NAN_DEBUG=1
# export BGB_SANITIZE_GRADS=1   # Use only while debugging NaNs

make train-bimamba
```

**FLA Stack** (`configs/local/train_fla.yaml`):
```bash
# Optional diagnostics
export BGB_NAN_DEBUG=1

make train-fla
```

**NOTE**: FLA training requires cache on native ext4 filesystem (not Windows `/mnt/` drives) to avoid WSL2 SIGBUS crashes. See `INSTALLATION.md` section 6 for details.

Key defaults for both stacks (see config for full list):
```yaml
training:
  batch_size: 8
  epochs: 100
  mixed_precision: false        # Safer on RTX 4090
  gradient_clip: 0.5
  mid_checkpoint_interval_s: 1800
  mid_epoch_keep: 3
```

### Modal (A100-80GB)

**🚨 CRITICAL**: Use `--action schedule-training` for 100-epoch production runs (auto-restart every 23h).

**BiMamba2 Stack** (`configs/modal/train_bimamba.yaml`):
```bash
# Deploy Modal functions first
modal deploy deploy/modal/app.py

# Launch with auto-restart scheduler (hands-free 100 epochs)
modal run --detach deploy/modal/app.py \
  --action schedule-training \
  --config configs/modal/train_bimamba.yaml
```

**FLA Stack** (`configs/modal/train_fla.yaml`):
```bash
# Deploy Modal functions first
modal deploy deploy/modal/app.py

# Launch with auto-restart scheduler (hands-free 100 epochs)
modal run --detach deploy/modal/app.py \
  --action schedule-training \
  --config configs/modal/train_fla.yaml
```

**Modal-specific behaviour**:
- Auto-restart every ~23h until 100 epochs complete (zero manual intervention)
- Batch size 48, mixed precision enabled
- `.wandb_run_id` persisted so resumes continue same dashboard
- Checkpoint priority: `mid_epoch_*.pt` → `timeout_exit.pt` → `last.pt`

**Manual Mode** (use ONLY for experiments, NOT production):
```bash
# Runs ONCE, requires manual restart after 23h timeout
modal run --detach deploy/modal/app.py \
  --action train \
  --config configs/modal/train_bimamba.yaml \
  --resume
```

---

## Part 3 – Monitoring

### Local
- `tmux attach -t train` – follow logs in real time
- `nvidia-smi` – watch GPU memory and utilization
- `tail -f results/<run>/train.log` (if logging to file)

### Modal
- `modal app list` – active runs
- `modal app logs <app-id>` – stream stdout/stderr
- W&B dashboard (if enabled) – metrics + checkpoints

#### Healthy Log Signals
```
[INFO] Epoch 1/100 | Batch 50/389 | loss=0.63 lr=1.0e-04
[DEBUG] Large grad norm: 5.7e+00 (clipped to 0.5)
[TAES] sensitivity_at_10fa: 0.18 (will improve across epochs)
```

#### Investigate If You See
- Repeated `Sanitized NaN gradients` (>5% batches)
- Sudden gradient P95 spikes that never recover
- Validation dataset reporting 0 windows (run `check-cache`, rebuild manifest)

---

## Part 4 – Checkpoints & Resume

Checkpoint priority when `training.resume` or `--resume` is enabled:
1. Most recent `mid_epoch_*.pt` (saved every 30 min, rotated via `mid_epoch_keep`)
2. `timeout_exit.pt` (written by the wall-clock guard on Modal)
3. `last.pt`

`save_checkpoint()` now writes via temp file + fsync + atomic rename, so corrupted checkpoints are no longer possible. The state dict includes model, optimizer, scheduler, AMP scaler, and RNG (Python/NumPy/torch CPU/torch CUDA) for deterministic resume.

- Local resume: `python -m src train configs/local/train_bimamba.yaml --resume`
- Modal resume after timeout: use the `--resume` flag shown above.

---

## Part 5 – Troubleshooting Quick Hits

| Symptom | Suggested Action |
|---------|------------------|
| Validation loads 0 windows | `modal run deploy/modal/app.py --action check-cache`; rebuild dev manifest if prompted |
| NaN/Inf losses | Ensure cache built with latest preprocessing, keep gradient clip 0.5, optionally enable `BGB_SANITIZE_GRADS=1` |
| Modal job killed at 24 h | Expected. Timeout guard wrote `timeout_exit.pt`; relaunch with `--resume` |
| Cache mismatch warnings | Confirm `cache_dir` points to `.../tusz_mmap`, rerun `scan-cache`, or repopulate/convert |

More detail: see `docs/08-operations/troubleshooting.md` and `docs/08-operations/nan-prevention-complete.md`.

---

## Part 6 – After the Run

1. **Review metrics** (W&B, TensorBoard under `results/<run>/tensorboard`)
2. **Evaluate checkpoint** – follow `docs/06-evaluation/metrics-and-taes.md`
3. **Clean old artefacts** – keep disk/Modal volume below 70% usage
4. **Document results** – update your experiment log or issue tracker

---

## Next Steps

- Fine-tune or iterate configs (`configs/local/*.yaml`, `configs/modal/*.yaml`)
- Run evaluation/TAES pipeline on dev and eval sets
- Explore future work ideas in `docs/future-work/`

With the v4.0.0 tooling (atomic checkpoints + timeout guard + StatefulDataLoader + W&B persistence), resuming long runs is routine:

- **Modal**: Auto-restart every ~23h via `--action schedule-training` (hands-free 100 epochs)
- **Local**: Manual resume with `--resume` flag, loses at most 30 minutes of progress (mid-epoch checkpoint interval)
