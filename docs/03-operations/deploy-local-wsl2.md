Local (WSL2/Linux) — Stable Training Guide

Why special care
- WSL2 can hang with multiprocessing; IO across Windows <-> WSL is slower
- Keep data and cache on the WSL ext4 filesystem; avoid Windows-mounted paths for heavy IO

Environment setup
- `make setup`
- `uv sync` (CPU) or `uv sync -E gpu` (if training with CUDA/Mamba-SSM)
- WSL tip: `export UV_LINK_MODE=copy` (Makefile sets this by default)

Config tips
- `num_workers: 0` (avoid WSL2 multiprocessing hangs)
- `pin_memory: false` (CPU/WSL runs)
- Use `data.cache_dir` on ext4 (e.g., `cache/tusz/train` under repo)

Preflight
- See deploy-preflight.md and ../01-data-pipeline/tusz-preflight.md
- Ensure manifest shows seizures; do not proceed otherwise

Commands
- Build cache: `python -m src build-cache --data-dir data_ext4/tusz/edf/train --cache-dir cache/tusz/train`
- Scan manifest: `python -m src scan-cache --cache-dir cache/tusz/train`
- Smoke train: `python -m src train configs/local/smoke.yaml`
- Full train: `python -m src train configs/local/train.yaml`

Mamba CUDA note
- CUDA kernels coerce unsupported `d_conv` to 4; to force Conv1d fallback: `SEIZURE_MAMBA_FORCE_FALLBACK=1`

Troubleshooting (quick)
- Hangs at data loading: set `num_workers=0`; ensure cache/manifest are local (ext4)
- 0% seizures in batches: rescan manifest; fix CSV_BI parsing; rebuild cache
- Slow IO: avoid Windows drives; keep data/cache in WSL ext4

Deep dives (WSL2)
- GPU setup options and CUDA wheels: ./WSL2/WSL2_GPU_SETUP.md
- CUDA performance findings and pytest multiprocessing: ./WSL2/CUDA_PERFORMANCE_FINDINGS.md
- Full WSL2 troubleshooting (I/O, env vars, threads): ./WSL2/WSL2_TROUBLESHOOTING.md

