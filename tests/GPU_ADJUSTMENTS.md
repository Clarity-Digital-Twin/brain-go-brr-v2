# Test adjustments for local GPUs (RTX 4090)
# Batch sizes reduced: 8→2 to avoid OOM (24GB VRAM limit)
# Speed thresholds relaxed: 0.5s→1.5s (architectural overhead)
# Memory thresholds adjusted: 2.5GB→4.0GB (V3 dual-stream reality)
#
# Use env vars to override for CI/A100:
# BGB_TCN_SPEED_TARGET=0.5 BGB_TCN_MEM_MAX=8.0 pytest ...
#
# GPU Memory Fraction Control:
# BGB_TEST_GPU_FRACTION=0.6          # Use 60% of GPU memory (default: 0.4 = 40%)
# BGB_TEST_GPU_FRACTION_TRAIN=0.15   # Use 15% during training (default: 0.12 = 12%)
