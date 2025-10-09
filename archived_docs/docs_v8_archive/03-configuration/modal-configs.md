# Modal (A100-80GB) Configs

**Last Updated**: October 6, 2025 (v3.7.0)
**Target Hardware**: Modal A100-80GB

## Overview

The production configuration uses the V3 dual-stream pipeline with the mmap cache stored on the Modal SSD. Key differences from local training are the larger batch size, mixed precision, and persistent DataLoader workers.

## Key Differences vs Local (RTX 4090)

| Setting | Local | Modal | Reason |
|---------|-------|-------|--------|
| `data.cache_dir` | `cache/tusz_mmap` | `/results/cache/tusz_mmap` | SSD-backed memory-mapped cache |
| `batch_size` | 8 | 48 | 80 GB vs 24 GB VRAM |
| `gradient_accumulation_steps` | 1 | 1 | Effective batch 8 vs 48 |
| `mixed_precision` | false | true | A100 tensor cores (3.8× speedup) |
| `learning_rate` | 1.0e-4 | 8.0e-5 | Batch-size scaling |
| `num_workers` | 0 (WSL2) / 4 (native) | 4 | Parallel I/O on Modal |
| `persistent_workers` | false | true | Mmap cache keeps RSS low |
| `prefetch_factor` | 2 | 2 | Stable loader footprint |
| CPU cores | N/A | 24 | Prevent dataloader bottlenecks |
| RAM | N/A | 96 GB | Headroom for mmap + workers |

## Resource declaration (deploy/modal/app.py)

```python
@app.function(
    gpu=modal.gpu.A100(count=1, size="80GB"),
    cpu=24,
    memory=98304,  # 96 GB
)
```

## Data configuration (excerpt from `configs/modal/train.yaml`)

```yaml
data:
  dataset: tuh_eeg
  data_dir: /data/edf
  cache_dir: /results/cache/tusz_mmap
  num_workers: 4
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2
  use_balanced_sampling: true
```

## Training configuration

```yaml
training:
  epochs: 100
  batch_size: 48
  gradient_accumulation_steps: 1
  mixed_precision: true
  learning_rate: 8.0e-5
  weight_decay: 0.05
  optimizer: adamw
  scheduler:
    type: cosine
    warmup_ratio: 0.03
  warmup_schedule:
    enabled: true
    warmup_steps: 1000
    adj_temperature_enabled: true
    focal_gamma_enabled: true
  gradient_clip: 0.5
  loss: focal
  focal_alpha: 0.5
  focal_gamma: 2.0
  early_stopping:
    patience: 5
    metric: sensitivity_at_10fa
  checkpoint_interval: 1
  mid_checkpoint_interval_s: 1800
  mid_epoch_keep: 3
```

## Model (unchanged V3 dual-stream)

See `configs/modal/train.yaml` for the full block. Highlights:
- Boundary LayerNorm + LayerScale across streams (PR‑1).
- Edge stream bounded and conditioned (PR‑2/PR‑3/PR‑5).
- Dynamic Laplacian PE with `semi_dynamic_interval: 5`.

## Operational fixes carried forward

### XID 31 mitigation
```python
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_cache_run_{run_id}"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/tii_cache_run_{run_id}"
```

### Faster logging + hang detection
```python
os.environ["BGB_LOG_EVERY_N_STEPS"] = "10"
os.environ["BGB_NAN_DEBUG"] = "1"
heartbeat_interval = 120  # seconds
```

## Commands recap

```bash
# Populate mmap cache (one-time)
modal run --detach deploy/modal/app.py --action populate-cache

# Smoke test (50 files)
modal run deploy/modal/app.py --action train --config configs/modal/smoke.yaml

# Full training (detached)
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml
```

Maintain the mmap cache on the Modal SSD and keep at least 600 GB free on the volume to avoid populate failures.
