# GPU Stack and Versions

Exact versions (locked)

- PyTorch: 2.5.0+cu124
- CUDA Toolkit: 12.4
- mamba‑ssm: 2.2.5 (includes A100 int64 indexing fix)
- causal‑conv1d: 1.5.2 (latest stable for PyTorch 2.5+)
- torch‑geometric: 2.6.1
- numpy: 1.26.4

Install order

1) `make setup` — base env + PyTorch 2.5.0+cu124
2) `make setup-gpu` — CUDA extensions (mamba‑ssm, causal‑conv1d) and PyG wheels

What `make setup-gpu` does

- Exports `CUDA_HOME=/usr/local/cuda-12.4` and installs CUDA extensions with `--no-build-isolation`:
  - `uv pip install --no-build-isolation causal-conv1d==1.5.2`
  - `uv pip install --no-build-isolation mamba-ssm==2.2.5`
- Installs PyG using prebuilt wheels for torch 2.5.0+cu124:
  - `.venv/bin/pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html`
  - `.venv/bin/pip install torch-geometric==2.6.1`
- Installs TCN: `uv pip install pytorch-tcn==1.2.3`
- Verifies Mamba‑SSM, PyG, and TCN installs.

Manual verification

- `.venv/bin/python -c "import torch; print(torch.version.cuda)"` → 12.4
- `.venv/bin/python -c "from mamba_ssm import Mamba2; print('OK')"`
- `.venv/bin/python -c "import torch_geometric as tg; print(tg.__version__)"` → 2.6.1

Troubleshooting

- PyG install error: ensure correct wheel index URL for torch 2.5.0+cu124; install scatter/sparse/cluster/spline, then `torch-geometric==2.6.1`.
- `RuntimeError: no kernel image` for mamba‑ssm: confirm CUDA 12.4 toolkit in PATH/LD_LIBRARY_PATH, rebuild with `--no-build-isolation`.
- WSL2: set `UV_LINK_MODE=copy`; keep project on ext4 (avoid `/mnt/c`).
