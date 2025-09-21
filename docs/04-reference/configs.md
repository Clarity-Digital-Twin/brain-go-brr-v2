Configs & Metrics (Reference)

Scope
- Config structure and safe defaults for WSL2/local vs A100/Modal.

Guidelines
- WSL2/local: `num_workers: 0`, `pin_memory: false`, small batch sizes; keep data on ext4.
- A100/Modal: larger batch sizes; ensure CUDA kernels compiled/available; monitor VRAM.
- Use `smoke_test.yaml` to validate pipeline and cache/manifest before longer runs.

Metrics/logging
- Enable unbuffered logs for long runs; consider TensorBoard/W&B under `results/`.
- Log dataset composition at train start to confirm balanced sampling is active.

Code anchors
- configs/*
- src/brain_brr/train/loop.py (train hyperparams, logging points)

Docs
- deployment/SENIOR_REVIEW_TRAINING_CONFIGS.md
- implementation/EVALUATION_CHECKLIST.md
