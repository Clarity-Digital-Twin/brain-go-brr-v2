# GPU Stack and Versions

Exact versions (locked)

- PyTorch: 2.5.0+cu124 (includes CUDA 12.4 **runtime**)
- **CUDA Toolkit: 12.4** (REQUIRED for building mamba-ssm)
- Triton: 3.1.0 (pairs with PyTorch 2.5.0; FLA will warn that 3.2.0 is “recommended”—this is cosmetic and expected, do **not** upgrade while we rely on mamba-ssm 2.2.5 and PyG 2.6.1)
- mamba‑ssm: 2.2.5 (includes A100 int64 indexing fix)
- causal‑conv1d: 1.5.2 (latest stable for PyTorch 2.5+)
- torch‑geometric: 2.6.1
- numpy: 1.26.4
- NVIDIA driver: **581.42** (Oct 2025) on RTX 4090; older 572.xx builds crash with SIGBUS around batch ~3000. Verify with `nvidia-smi` after install.

## CRITICAL: CUDA Toolkit Installation

**PyTorch 2.5.0+cu124 includes the CUDA 12.4 runtime, but NOT the toolkit.** The toolkit is required to compile CUDA extensions like mamba-ssm.

### Install CUDA 12.4 Toolkit (Ubuntu/WSL2)
```bash
# Check if already installed
/usr/local/cuda-12.4/bin/nvcc --version

# If not found, install it:
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-4

# Verify
/usr/local/cuda-12.4/bin/nvcc --version
# Should output: "Cuda compilation tools, release 12.4, V12.4.131"
```

Install order

1) Install CUDA 12.4 toolkit (see above)
2) `make setup` — base env + PyTorch 2.5.0+cu124
3) `make setup-gpu` — CUDA extensions (mamba‑ssm, causal‑conv1d) and PyG wheels

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
- `nvidia-smi | head -n 3` → driver version should read `581.42`

### Triton version warning (expected)

Flash Linear Attention emits a warning when Triton <3.2.0 is detected:

```
WARNING fla.utils: Current Triton version 3.1.0 is below the recommended 3.2.0 … please consider upgrading.
```

Stay on **Triton 3.1.0** while the stack is locked to PyTorch 2.5.0—upgrading Triton forces a PyTorch 2.6+ jump, which breaks the validated mamba-ssm build and PyG wheels. The warning is informational only; kernels operate at full speed on 3.1.0.

Troubleshooting

### Symbol Mismatch (Most Common)
**Error**: `undefined symbol: _ZN3c104cuda9SetDeviceEab`

**Root Cause**: mamba-ssm compiled against wrong CUDA version (cached wheel)

**Solution**: Force rebuild from source
```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"

# Purge caches and rebuild
rm -rf ~/.cache/uv ~/.cache/pip
uv pip uninstall mamba-ssm causal-conv1d
uv pip install --no-build-isolation --no-binary causal-conv1d causal-conv1d==1.5.2
uv pip install --no-build-isolation --no-binary mamba-ssm mamba-ssm==2.2.5

# Verify
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('✅')"
```

### Other Issues
- **PyG install error**: ensure correct wheel index URL for torch 2.5.0+cu124; install scatter/sparse/cluster/spline, then `torch-geometric==2.6.1`.
- **`RuntimeError: no kernel image`**: confirm CUDA 12.4 toolkit installed, rebuild with `--no-build-isolation` and `--no-binary`.
- **WSL2**: set `UV_LINK_MODE=copy`; keep project on ext4 (avoid `/mnt/c`).
