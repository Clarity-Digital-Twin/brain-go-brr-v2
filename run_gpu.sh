#!/bin/bash
# Run pipeline with full GPU power + scipy workaround

# These ONLY limit CPU threads for scipy/numpy - GPU runs FULL POWER!
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Optional: More CPU threads for data loading (if scipy doesn't hang)
# export OMP_NUM_THREADS=4

echo "🚀 Running with RTX 4090 - ALL 16,384 CUDA cores active!"
echo "Thread limits only affect CPU ops (scipy), not GPU!"
python -m src.experiment.pipeline --config "${1:-configs/smoke_test.yaml}"