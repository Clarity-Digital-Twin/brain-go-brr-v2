# Your First Training Run

**Last Updated**: October 8, 2025  
**Codebase Version**: v3.9.1 (Validation OOM Fix)  
**Estimated Time**: 30 minutes to launch, ~200–300 h (RTX 4090) or ~100 h (Modal A100) to complete

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

### Local (RTX 4090, `configs/local/train_bimamba.yaml`)

```bash
# Optional diagnostics
export BGB_NAN_DEBUG=1
# export BGB_SANITIZE_GRADS=1   # Use only while debugging NaNs

make train-local
```

Key defaults (see config for full list):
```yaml
training:
  batch_size: 8
  epochs: 100
  mixed_precision: false        # Safer on RTX 4090
  gradient_clip: 0.5
  mid_checkpoint_interval_s: 1800
  mid_epoch_keep: 3
  resume: true                  # Allows --resume flag or config toggle
```

### Modal (A100-80GB, `configs/modal/train_bimamba.yaml`)

```bash
# Always detach for long runs
modal run --detach deploy/modal/app.py \
  --action train \
  --config configs/modal/train_bimamba.yaml
```

Modal-specific behaviour:
- Batch size 48, mixed precision enabled
- Timeout guard saves `timeout_exit.pt` around 23 h (one hour before Modal’s hard kill)
- `.wandb_run_id` is persisted alongside checkpoints so resumes continue the same dashboard

When the guard triggers, relaunch with:
```bash
modal run --detach deploy/modal/app.py \
  --action train \
  --config configs/modal/train_bimamba.yaml \
  --resume true
```
The loader will pick the newest `mid_epoch_*.pt`, then `timeout_exit.pt`, then `last.pt`.

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
- Modal resume after timeout: use the `--resume true` flag shown above.

---

## Part 5 – Troubleshooting Quick Hits

| Symptom | Suggested Action |
|---------|------------------|
| Validation loads 0 windows | `modal run deploy/modal/app.py --action check-cache`; rebuild dev manifest if prompted |
| NaN/Inf losses | Ensure cache built with latest preprocessing, keep gradient clip 0.5, optionally enable `BGB_SANITIZE_GRADS=1` |
| Modal job killed at 24 h | Expected. Timeout guard wrote `timeout_exit.pt`; relaunch with `--resume true` |
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

With the v3.9.1 tooling (atomic checkpoints + timeout guard + disk-backed validation + W&B persistence), resuming long Modal runs is routine—expect to relaunch every ~23 h and lose at most 10–30 minutes of progress.
