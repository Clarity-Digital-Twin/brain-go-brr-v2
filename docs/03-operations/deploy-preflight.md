Deployment Preflight (Must Pass Before Any Run)

1) Code quality and env
- `make q` (ruff, format, mypy) — must pass
- `make setup` once per environment to ensure hooks and uv

2) Data availability
- TUSZ EDF+CSV_BI present and accessible
- For local: verify path in config (`data.data_dir`)
- For Modal: verify mount paths point to EDF+CSV and writable cache

3) Cache + manifest sanity
- If cache does not exist: build it
  - `python -m src build-cache --data-dir <edf_root> --cache-dir <cache_dir>`
- Scan or (re)build manifest
  - `python -m src scan-cache --cache-dir <cache_dir>`
- Required: partial > 0 or full > 0; otherwise STOP and fix CSV parsing/paths

Quick strategy (small first)
- Build/scan a tiny cache subset to verify parser/labels before large builds.
- Run a 1-epoch smoke: `python -m src train configs/local/smoke.yaml` and confirm batches show seizures > 0%.

4) Balanced dataset check
- `from src.brain_brr.data import BalancedSeizureDataset as B; len(B(Path('<cache_dir>')))`
- Required: `len(dataset) > 0`

5) Hardware + runtime
- WSL2: use `num_workers=0` and avoid pin_memory in configs
- GPU (if used): ensure correct drivers/CUDA; Mamba kernels coerce `d_conv` (fallback available)

## Data Split Integrity

**CRITICAL**: TUSZ provides pre-defined splits that MUST be respected:
```
data_ext4/tusz/edf/
├── train/   (581 patients) - TRAINING ONLY
├── dev/     (55 patients)  - HYPERPARAMETER TUNING
└── eval/    (45 patients)  - FINAL TEST - ONE SHOT ONLY!
```

**Never**:
- Use eval set for threshold selection
- Combine train/dev/eval for "more data"
- Peek at eval metrics during training
- Mix same patient across splits
- Tune hyperparameters on train set performance

References
- TUSZ preflight/troubleshooting: ../01-data-pipeline/tusz-preflight.md
- Cache+sampling details: ../01-data-pipeline/tusz-cache-sampling.md

