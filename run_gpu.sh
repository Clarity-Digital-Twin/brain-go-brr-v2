#!/bin/bash
# Run pipeline with full GPU power + mamba-ssm CUDA support

# CRITICAL: Set library path for mamba-ssm CUDA linking in WSL2 (dynamic detection)
export LD_LIBRARY_PATH="$(.venv/bin/python -c 'import site,os,torch; print(os.path.join(site.getsitepackages()[0],"torch","lib"))'):${LD_LIBRARY_PATH}"
export CUDA_HOME=/usr/local/cuda-12.6
export TORCH_CUDA_ARCH_LIST="8.9"  # RTX 4090 architecture
export LIBTORCH_USE_RTLD_GLOBAL=YES
export PYTHONFAULTHANDLER=1

# WSL2 stability fixes
export PYTHONUNBUFFERED=1              # Force unbuffered output to see debug prints
export PYTHONFAULTHANDLER=1            # Dump tracebacks if something deadlocks
export LIBTORCH_USE_RTLD_GLOBAL=YES    # Help .so symbol resolution

# These ONLY limit CPU threads for scipy/numpy - GPU runs FULL POWER!
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

echo "🚀 Running with RTX 4090 + MAMBA-SSM!"
echo "✅ LD_LIBRARY_PATH set for CUDA"
echo "✅ O(N) sequence modeling with Bi-Mamba-2"
.venv/bin/python -V
echo "PY=$(which .venv/bin/python)"
.venv/bin/python -m src.experiment.pipeline --config "${1:-configs/smoke_test.yaml}"
