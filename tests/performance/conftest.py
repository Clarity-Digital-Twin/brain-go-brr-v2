"""Performance test configuration to prevent resource contention."""

import os

import pytest
import torch


@pytest.fixture(autouse=True, scope="session")
def perf_env_guard():
    """Configure environment for stable performance testing."""
    # Disable GPU for performance tests (we test logic, not raw GPU speed)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    # Pin CPU threads to prevent contention
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    # Set PyTorch threads
    torch.set_num_threads(1)

    yield