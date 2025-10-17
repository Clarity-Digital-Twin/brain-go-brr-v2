# Preflight Checks

GPU and CUDA

- `.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"` → True, 12.4
- `nvcc --version` → release 12.4
- `.venv/bin/python -c "from mamba_ssm import Mamba2; print('✅ Mamba')"`
- `.venv/bin/python -c "import torch_geometric as tg; print('✅ PyG', tg.__version__)"`

WSL2

- `echo $UV_LINK_MODE` → `copy`
- Prefer `num_workers: 0` in local configs if hitting dataloader hangs.

Dataset cache

- Ensure cache exists with expected counts: `python -m src scan-cache --cache-dir cache/tusz_mmap/train`
- Expect `partial>0` or `full>0`; otherwise inspect CSV parsing and data paths.

Smoke test

- Local BiMamba2: `make smoke-bimamba` or `make s` (1 epoch, 3 files)
- Local FLA: `make smoke-fla` (1 epoch, 3 files)
- Confirm training runs end-to-end without NaNs or SIGBUS crashes.

Troubleshooting

- See `docs/08-operations/troubleshooting.md` and `docs/01-installation/gpu-stack.md`.
