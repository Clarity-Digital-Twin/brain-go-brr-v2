# Modal (A100-80GB) Configs

Recommended resources

- `resources.cpu: 24`
- `resources.memory: 98304`

Training

- `training.batch_size: 64`
- `training.mixed_precision: true`

Data

- `data.cache_dir: /results/cache/tusz`

Commands

- Smoke: `modal run deploy/modal/app.py --action train --config configs/modal/smoke.yaml`
- Full (detached): `modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml`
