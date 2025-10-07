# GPU test adjustments (RTX 4090 default profile)
#
# The values below mirror `tests/test_config.py` and `tests/gpu_memory_guard.py`.
# Override them via environment variables when validating on different hardware.
#
# Batch size guardrails:
#   • RTX 4090 (24 GB): TEST_MAX_BATCH_SIZE=4      (default)
#   • A100‑80 GB:        TEST_MAX_BATCH_SIZE=8
#   • Unknown/CPU:       TEST_MAX_BATCH_SIZE=1
#     → set explicitly with TEST_BATCH_SIZE or TEST_MAX_BATCH_SIZE if required.
#
# Performance thresholds:
#   • PERF_LATENCY_THRESHOLD['cuda'] = 150 ms (median)
#   • PERF_MEMORY_THRESHOLD['cuda'] = 4096 MB
# Adjust for slower cards with:
#   BGB_TCN_SPEED_TARGET=200  BGB_TCN_MEM_MAX=6000  pytest -m performance
#
# GPU memory caps enforced by gpu_memory_guard:
#   • Normal mode: 40% VRAM (BGB_TEST_GPU_FRACTION, default 0.4)
#   • When training detected / low free VRAM: 12% (BGB_TEST_GPU_FRACTION_TRAIN)
# Raise or lower the limits if you need to share the GPU with other workloads:
#   export BGB_TEST_GPU_FRACTION=0.6
#   export BGB_TEST_GPU_FRACTION_TRAIN=0.15
#
# Skip GPU workloads entirely:
#   export BGB_SKIP_GPU_TESTS=1
