# Modal Training (A100-80GB)

Commands

- Test Mamba CUDA: `modal run deploy/modal/app.py --action test-mamba`
- Smoke: `modal run deploy/modal/app.py --action train --config configs/modal/smoke.yaml`
- Full (detached): `modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml`

Resources

- `cpu: 24`, `memory: 98304`, `batch_size: 64`, `mixed_precision: true`
