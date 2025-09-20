"""Performance test configuration to prevent resource contention."""

import os

import pytest
import torch


@pytest.fixture(autouse=True, scope="session")
def perf_env_guard():
    """Configure environment for stable performance testing.

    Defaults: CPU-only, single-thread to reduce noise.
    Override with env vars for opt-in scenarios:
      - BGB_PERF_ALLOW_GPU=1 to allow GPU
      - BGB_PERF_THREADS=<N> to set CPU thread count
    """
    allow_gpu = os.getenv("BGB_PERF_ALLOW_GPU", "0") == "1"
    if not allow_gpu:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    # Pin CPU threads to prevent contention (can be overridden)
    threads = int(os.getenv("BGB_PERF_THREADS", "1"))
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)

    # Set PyTorch threads
    torch.set_num_threads(threads)

    return
