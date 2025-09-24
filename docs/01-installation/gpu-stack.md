# GPU Stack and Versions

Exact lock (do not change)

- PyTorch: `2.2.2+cu121`
- CUDA Toolkit: `12.1`
- mamba-ssm: `2.2.2`
- causal-conv1d: `1.4.0`
- torch-geometric: `2.6.1`
- numpy: `1.26.4`

Install order

1) `make setup`
2) `make setup-gpu`

Troubleshooting

- PyG install failures → prebuilt wheels
- Mamba CUDA errors → ensure CUDA 12.1 toolchain
