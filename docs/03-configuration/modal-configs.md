# Modal (A100-80GB) Configs

Resources

- `resources.cpu: 24`, `resources.memory: 98304` (96GB)
- Run with an A100‑80GB GPU

Training

- `training.batch_size: 64` (A100‑80GB)
- `training.mixed_precision: true` (A100 tensor cores)
- `loss: focal` with `focal_alpha: 0.5`, `focal_gamma: 2.0`

Data

- `data.data_dir: /data/edf` (parent containing `train/`, `dev/`, `eval/`)
- `data.cache_dir: /results/cache/tusz` (Modal persistent SSD; contains `{train,dev}`)
- `data.split_policy: official_tusz` (enforce patient‑disjoint official splits)
- `data.num_workers: 8`, `pin_memory: true`, `persistent_workers: true`, `prefetch_factor: 4`

Graph and V3

- `model.architecture: v3`
- `graph.enabled: true` with edge stream and vectorized GNN defaults
- `graph.use_dynamic_pe: true` with `semi_dynamic_interval: 1` (full dynamic on A100)

Commands

- Smoke: `modal run deploy/modal/app.py --action train --config configs/modal/smoke.yaml`
- Full (detached): `modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml`
 - Cache cleanup (once, after the split fix): `modal run deploy/modal/app.py --action clean-cache`
