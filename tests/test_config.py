"""Test configuration for memory-safe and fast test execution."""

import os

import torch

# Memory-safe test defaults
TEST_BATCH_SIZE = int(os.getenv("TEST_BATCH_SIZE", "1"))  # Conservative default
TEST_WINDOW_SIZE = 15360  # 60s at 256Hz
TEST_NUM_WORKERS = 0  # Prevent multiprocessing issues

# GPU configuration
TEST_USE_GPU = torch.cuda.is_available() and os.getenv("TEST_GPU", "auto") != "false"
TEST_DEVICE = "cuda" if TEST_USE_GPU else "cpu"

# Memory limits by device type
if TEST_USE_GPU:
    # Conservative GPU memory limits to prevent OOM
    MAX_BATCH_SIZE = {
        "RTX 4090": 4,  # 24GB VRAM
        "A100": 8,  # 80GB VRAM
        "default": 2,  # Unknown GPU, be conservative
    }

    # Get GPU name and set appropriate batch size
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if "4090" in gpu_name:
            TEST_MAX_BATCH_SIZE = MAX_BATCH_SIZE["RTX 4090"]
        elif "A100" in gpu_name:
            TEST_MAX_BATCH_SIZE = MAX_BATCH_SIZE["A100"]
        else:
            TEST_MAX_BATCH_SIZE = MAX_BATCH_SIZE["default"]
    else:
        TEST_MAX_BATCH_SIZE = 1
else:
    TEST_MAX_BATCH_SIZE = 1  # CPU: minimal memory usage

# Test data sizes
TEST_SMOKE_FILES = 3  # Quick smoke tests
TEST_INTEGRATION_FILES = 10  # Integration tests
TEST_FULL_FILES = 50  # Full test suite

# Timeouts (seconds)
TEST_TIMEOUT_UNIT = 30  # Unit test timeout
TEST_TIMEOUT_INTEGRATION = 120  # Integration test timeout
TEST_TIMEOUT_PERFORMANCE = 300  # Performance test timeout

# Model configuration for tests
TEST_MODEL_CONFIG = {
    "architecture": "v3",
    "d_model": 64,  # Reduced from 512 for faster tests
    "tcn": {
        "num_layers": 2,  # Reduced from 8
        "channels": [32, 64],  # Reduced from [64, 128, 256, 512]
    },
    "mamba": {
        "n_layers": 2,  # Reduced from 6
        "d_state": 8,  # Reduced from 16
    },
    "graph": {
        "use_gnn": True,
        "use_dynamic_pe": False,  # Disable for most tests (expensive)
        "k_eigenvectors": 8,  # Reduced from 16
        "semi_dynamic_interval": 10,  # Conservative for tests
    },
}

# Performance test thresholds
PERF_LATENCY_THRESHOLD = {
    "cpu": 500,  # ms - relaxed for CI
    "cuda": 150,  # ms - reasonable for GPU
}

PERF_MEMORY_THRESHOLD = {
    "cpu": 2048,  # MB
    "cuda": 4096,  # MB
}

# Fixtures to skip in low-memory environments
SKIP_MEMORY_INTENSIVE = os.getenv("TEST_LOW_MEMORY", "false").lower() == "true"
