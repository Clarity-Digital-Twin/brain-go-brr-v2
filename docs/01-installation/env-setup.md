# Environment Setup

Required versions

- PyTorch `2.2.2+cu121`
- CUDA Toolkit `12.1`
- mamba-ssm `2.2.2`
- causal-conv1d `1.4.0`
- torch-geometric `2.6.1`
- numpy `1.26.4`

Commands

- Base: `make setup`
- GPU: `make setup-gpu`
- Verify: `.venv/bin/python -c "from mamba_ssm import Mamba2; print('✅')"`

Notes

- PyG: use prebuilt wheels matching torch/cu121
- WSL2: set `UV_LINK_MODE=copy`
