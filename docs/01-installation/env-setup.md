# Environment Setup

System requirements

- GPU: NVIDIA RTX 4090 (24GB) for local; A100‑80GB on Modal
- RAM: 32GB minimum (64GB recommended)
- Disk: 200GB+ free (datasets + caches)
- OS: Ubuntu 20.04+ or WSL2 on Windows
- Python: 3.11 (tested 3.11.13)

Exact versions (do not change)

- PyTorch 2.2.2+cu121
- CUDA Toolkit 12.1 (must match PyTorch CUDA)
- mamba‑ssm 2.2.2
- causal‑conv1d 1.4.0
- torch‑geometric 2.6.1
- numpy 1.26.4

Install steps

1) Base environment
- `make setup`

2) GPU stack and PyG (prebuilt wheels)
- `make setup-gpu`

3) Verify toolchain
- `.venv/bin/python -c "import torch; print(f'CUDA: {torch.version.cuda}')"`  → 12.1
- `nvcc --version` → release 12.1
- `.venv/bin/python -c "from mamba_ssm import Mamba2; print('✅ Mamba')"`
- `.venv/bin/python -c "import torch_geometric as tg; print('✅ PyG', tg.__version__)"`

PyG prebuilt wheels

- If needed, install explicitly with prebuilt wheels:
- `.venv/bin/pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html`
- `.venv/bin/pip install torch-geometric==2.6.1`

UV build isolation note

- CUDA extensions (mamba‑ssm, causal‑conv1d) require PyTorch at build time.
- Use `--no-build-isolation` (already handled in `make setup-gpu`).

WSL2 specifics

- Set `UV_LINK_MODE=copy` to avoid hard‑link issues.
- Prefer `data.num_workers: 0` to avoid dataloader hangs.
- Keep repo/venv on WSL ext4 (not `/mnt/c`).

Common pitfalls

- PyG fails to build: use prebuilt wheels (see link above).
- Mamba CUDA errors: ensure CUDA 12.1 toolkit and rebuild mamba‑ssm with `--no-build-isolation`.
- Modal stuck: allocate 24 CPU cores and 96GB RAM.

Quick smoke check

- Local: `make s` (1 epoch, 3 files)
- Modal: `modal run deploy/modal/app.py --action train --config configs/modal/smoke.yaml`
