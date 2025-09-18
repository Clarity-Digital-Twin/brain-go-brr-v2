#!/bin/bash
# Run pipeline with full GPU power + mamba-ssm CUDA support

# CRITICAL: Set library path for mamba-ssm CUDA linking in WSL2
export LD_LIBRARY_PATH=/home/jj/proj/brain-go-brr-v2/.venv/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.6

# These ONLY limit CPU threads for scipy/numpy - GPU runs FULL POWER!
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

echo "🚀 Running with RTX 4090 + MAMBA-SSM!"
echo "✅ LD_LIBRARY_PATH set for CUDA"
echo "✅ O(N) sequence modeling with Bi-Mamba-2"
.venv/bin/python -m src.experiment.pipeline --config "${1:-configs/smoke_test.yaml}"