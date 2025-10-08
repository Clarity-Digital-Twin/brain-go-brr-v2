# Modal Training (A100-80GB)

**Last Updated**: October 8, 2025  
**Baseline**: v3.9.0 – Production Training (Bulletproof Resume)

Modal is the production environment for the 100‑epoch A100 run. This guide covers the recommended workflow, timeout guard behaviour, cache hygiene, and observability.

---

## 1. Quick Command Reference

```bash
# Populate mmap cache from S3 (use --detach)
modal run --detach deploy/modal/app.py --action populate-cache

# CUDA sanity check (verifies Mamba kernels)
modal run deploy/modal/app.py --action test-mamba

# Cache health / manifest validation (train + dev)
modal run deploy/modal/app.py --action check-cache

# Smoke test (50 files, 1 epoch, ~5 min)
modal run --detach deploy/modal/app.py --action train --config configs/modal/smoke.yaml

# Full training (exits ~23 h with timeout_exit.pt)
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml

# Resume after timeout/interruption
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml --resume true

# Clean stray NPZ files if check-cache reports them
modal run deploy/modal/clean_stray_npz.py --confirm

# Monitoring
modal app list
modal app logs <app-id>
modal app stop <app-id>
```

---

## 2. Resources & Configuration

### Hardware (deploy/modal/app.py)
- **GPU**: 1 × A100‑80GB (`modal.gpu.A100`)
- **CPU**: 24 cores
- **RAM**: 96 GB
- **Volume**: `/results` persistent SSD (~500 GB)

### Training Profile (configs/modal/train.yaml)
```yaml
data:
  data_dir: /data/edf
  cache_dir: /results/cache/tusz_mmap
  num_workers: 4
  persistent_workers: true
  prefetch_factor: 2
training:
  batch_size: 48
  gradient_accumulation_steps: 1
  mixed_precision: true
  gradient_clip: 0.5
  mid_checkpoint_interval_s: 1800
  mid_epoch_keep: 3
experiment:
  output_dir: /results/v3_full_training
```

The job exports:
- `BGB_WALL_CLOCK_LIMIT_S=82800` (timeout guard)
- `BGB_LIMIT_FILES=50` for smoke configs
- `BGB_NAN_DEBUG=1` for additional NaN logging
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512`
- Unique Triton/Inductor cache directories per run

---

## 3. Timeout Guard & Resume Workflow

- The timeout guard monitors wall-clock time and, after ~23 h, saves `timeout_exit.pt`, logs a warning, and exits cleanly. Modal never hard-kills mid-checkpoint.
- Relaunch training with `--resume true`. The loader prefers, in order: newest `mid_epoch_*.pt`, `timeout_exit.pt`, then `last.pt`.
- Expect 4–5 resume cycles for a 100‑epoch run (~5 days wall-clock, ~$350).
- Each checkpoint contains model, optimizer, scheduler, AMP scaler, and RNG state (Python/NumPy/torch CPU/torch CUDA). Resumes are deterministic—no repeated batches.
- `.wandb_run_id` is stored in the checkpoint directory; resumed runs continue the same W&B dashboard (`[W&B] Run resumed: …` in logs).

---

## 4. Cache Hygiene & Manifest Health

| Action | Command | Notes |
|--------|---------|-------|
| Verify cache + manifests | `modal run deploy/modal/app.py --action check-cache` | Lists train/dev counts, warns about stale manifests or NPZ files |
| Remove stray NPZ files | `modal run deploy/modal/clean_stray_npz.py --confirm` | Confirms matching `_data.npy/_labels.npy` before deletion |
| Re-populate from S3 | `modal run --detach deploy/modal/app.py --action populate-cache` | Clears `/results/cache/tusz_mmap/{train,dev}` then copies fresh mmap cache |
| Clean cache completely | `modal run deploy/modal/app.py --action clean-cache` | Use prior to re-populating if you want a full refresh |

Manifests live alongside the data (`/results/cache/tusz_mmap/{train,dev}/manifest.json`). If `check-cache` reports a stale dev manifest, delete it and re-run training; the loader rebuilds it from the mmap cache on startup.

---

## 5. Launch Checklist

1. **Cache present**: `modal run deploy/modal/app.py --action check-cache` shows 4,667 train + 1,832 dev stems.
2. **Secrets**: `wandb` secret configured if you need logging (`modal secret create wandb …`).
3. **Smoke test**: Run the smoke config once after any significant change.
4. **Monitoring**: `modal app logs <app-id>` in one terminal, W&B dashboard in another.
5. **Resume plan**: Add a reminder to relaunch every ~23 h with `--resume true`.

---

## 6. Observability

### Logs
- Initialization takes ~10–15 min (manifest load, worker spawn, preflight batch).
- Heartbeat logging every ~10 batches; gradient norms printed when they spike (expected with FP16).
- Timeout guard emits `[TIMEOUT] Wall-clock limit approaching …` followed by `[TIMEOUT] Saved timeout_exit.pt`.

### W&B
- Metrics continue seamlessly thanks to `.wandb_run_id`.
- Check that new runs include “Epoch” panel resetting to the resumed epoch rather than restarting at 1.

---

## 7. Common Issues

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `ValidationDataset` loads 0 windows | Stale dev manifest | `check-cache` → delete manifest → relaunch (auto rebuild) |
| Modal times out at 24 h and kills job | No timeout guard (old build) | Ensure you’re on v3.9.0+; guard saves `timeout_exit.pt` |
| New W&B run after resume | `.wandb_run_id` missing | Check checkpoint directory permissions; file must be writable |
| NPZ files reported | Old aborted run | `modal run deploy/modal/clean_stray_npz.py --confirm` |
| Cache mismatch warnings | Wrong cache path in configs | Ensure `data.cache_dir: /results/cache/tusz_mmap` |
| OOM | Batch size incorrectly changed | Keep `batch_size: 48`, `gradient_accumulation_steps: 1` |

---

## 8. FAQ

**How often should I resume?**  
Modal exits ~23 h after launch; relaunch with `--resume true` as soon as you see the timeout log (max progress loss <30 min).

**Can I delete `timeout_exit.pt`?**  
Yes—it’s treated like a mid-epoch checkpoint. Once you resume successfully you may delete older snapshots to save space.

**Do I need to rebuild manifests manually?**  
Only if `check-cache` reports they’re stale. Training rebuilds them automatically when missing.

**What happens if W&B fails?**  
Training continues. A warning is logged and the job runs without telemetry; checkpoints remain intact.

---

## 9. Related Docs

- `docs/05-training/checkpoint-strategy.md` – Mid-epoch cadence, atomic save details.
- `docs/05-training/resume.md` – Resume prioritisation, W&B persistence.
- `docs/02-data/cache-layout.md` – Manifest format + mmap structure.
- `docs/08-operations/troubleshooting.md` – Modal-specific triage tips.

Keep this workflow and you’ll never lose more than ~30 minutes of progress, even across week-long Modal training runs.
