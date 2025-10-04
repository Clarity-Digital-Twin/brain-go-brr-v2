# Performance Optimization

**Last Updated**: October 3, 2025 (v3.4.1)

## Dynamic Laplacian PE (V3)

- `graph.use_dynamic_pe: true` is the default and should stay enabled; eigenvectors are detached to avoid gradient explosions (v3.3.1 fix).
- `graph.semi_dynamic_interval` controls **how often** eigendecompositions run. It is a compute policy, not an architectural change.
  - Interval `1` = fully dynamic (compute on every timestep).
  - Interval `5` (default) reuses PE for four intermediate timesteps with negligible (<0.1% AUROC) impact.
  - The implementation now computes eigendecompositions only for the sampled timesteps instead of performing all 960 and discarding 80%.
- Preferred workflow:
  1. Start with `interval: 5` for efficiency.
  2. Increase to `8-10` if you need additional headroom on 24 GB cards.
  3. Drop to `1` only when benchmarking the absolute upper bound (expect 5× compute cost).

## Modal A100 dataloader tuning

Safe baseline (crash-proof)

```yaml
data:
  num_workers: 4
  persistent_workers: false
  prefetch_factor: 2
training:
  batch_size: 32
  gradient_accumulation_steps: 2  # effective 64
```

- Fixes the one-hour worker spawn delays (persistent workers disabled) and prevents the 77 GB OOM caused by 64 prefetched batches.
- Epoch transitions incur ~10 s respawn cost but stay stable for long runs.

Throughput profile (recommended once baseline is stable)

```yaml
data:
  num_workers: 8
  persistent_workers: false
  prefetch_factor: 4
```

- Doubles data-loading throughput while keeping startup time <15 min.
- Monitor GPU memory when increasing `prefetch_factor`; reduce to `3` if you see allocation pressure.

Aggressive profile (requires validation on your account)

```yaml
data:
  num_workers: 8
  persistent_workers: true
  prefetch_factor: 4
training:
  batch_size: 64
  gradient_accumulation_steps: 1
```

- Re-enables persistent workers after the prefetch factor fix; expect epoch boundaries to become snappy again.
- Watch the first-epoch spawn time. If it exceeds ~15 min, revert `persistent_workers` to `false`.
- Keep an eye on peak VRAM (<70 GB) when running batch size 64.

## Local (RTX 4090) quick levers

- Keep `training.mixed_precision: false`; RTX 4090 mixed precision still produces NaNs.
- Start with `training.batch_size: 4`, `graph.semi_dynamic_interval: 5`.
- Increase the interval (8–10) before disabling dynamic PE entirely.
- Use gradient accumulation if you must simulate larger batches.

## General tips

- Ensure gradient clipping remains enabled (`training.gradient_clip: 0.5`).
- Optional debugging: enable `BGB_NAN_DEBUG=1` (logging) and `BGB_SANITIZE_GRADS=1` only when investigating NaNs.
- Streaming validation keeps dev-split memory under ~5 GB; no need to downsample for RAM.
- Profile with `torch.profiler` or `nsys` before attempting larger architectural changes; most wins now come from dataloader and PE frequency tuning.
