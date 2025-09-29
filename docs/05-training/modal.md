# Modal Training (A100-80GB)

## CRITICAL: Always Use --detach
Modal functions will stop after ~8 minutes when terminal disconnects unless you use `--detach`.
**All long-running commands MUST use --detach or run in tmux!**

Commands

- Populate cache (one-time, from S3): `modal run --detach deploy/modal/app.py --action populate-cache`
  - **IMPORTANT**: This command removes existing `/results/cache/tusz/{train,dev}` before copying fresh data from S3
  - Training uses the SSD cache and will NOT clear it unless you re-run populate-cache or clean-cache
- Test Mamba CUDA: `modal run deploy/modal/app.py --action test-mamba`
- Smoke: `modal run --detach deploy/modal/app.py --action train --config configs/modal/smoke.yaml`
- Full (detached): `modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml`
- Clean old cache (if needed): `modal run deploy/modal/app.py --action clean-cache`

Monitoring

- List running apps: `modal app list`
- Stream logs: `modal app logs <app-id>`
- Stop training: `modal app stop <app-id>`

Resources

- `cpu: 24`, `memory: 98304`, `batch_size: 64`, `mixed_precision: true`
- Prefer FP16 on A100 (tensor cores); keep dynamic PE enabled (default) — safeguards handle ill‑conditioned graphs.

Cache and volumes

- Raw data mounted at `/data/edf/` (read‑only dataset mount)
- Cache on persistent SSD volume at `/results/cache/tusz` (patient‑disjoint subdirs: `{train,dev}`)
- Results saved to `/results/` (same persistent volume)
- Ensure `data.data_dir: /data/edf`, `data.split_policy: official_tusz`
- Ensure `data.cache_dir: /results/cache/tusz` in configs
- Do not use S3 for cache on Modal; prebuilt caches should be synced into the Modal volume

Patient disjointness

- On startup, the app verifies that patient sets in `/data/edf/train` and `/data/edf/dev` are disjoint and aborts if not.

Notes

- If migrating legacy runs that used an S3 cache mount, copy the cache to the Modal volume once, then point configs to `/results/cache/tusz`.

Resuming

- Use `--resume` flag or set `training.resume: true`.
- Training prioritizes `mid_epoch_*.pt` when resuming; falls back to `last.pt`.

## Verification Checklist

### After populate-cache
Check logs for:
```
[COPY] ✅ Copied 4667 train files
[COPY] ✅ Copied 1832 dev files
[COPY] ✅ Copied metadata file
✅ Cache population complete! 4667 train, 1832 dev files
```

### During training startup
Expect to see:
```
[CACHE] ✅ Cache built with official_tusz policy
[DATASET] BalancedSeizureDataset: XXXX windows from manifest
[DATASET] Seizure ratio: XX% (from manifest)
[DATASET] Using pos_weight: X.XX (sqrt scaling)
```

### Warning Signs
- "Seizure ratio: 0%" → Missing manifest, rebuild and re-upload
- "Falling back to EEGWindowDataset" → Manifest creation failed
- Training hangs at epoch boundaries → Increase CPU/RAM allocation

Troubleshooting

- PyG/Mamba import issues: the image pins CUDA 12.1 + torch 2.2.2+cu121; rebuild if diverged.
- Slow/stuck at epoch boundaries: allocate 24 CPU and 96GB RAM (see function in `deploy/modal/app.py`).
- Zero seizures in batches: Ensure manifest exists and `use_balanced_sampling: true` in config.
- Logs: `modal app logs <app-id>`; stop: `modal app stop <app-id>`.
