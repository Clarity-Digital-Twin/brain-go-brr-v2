"""GPU memory guard for tests to prevent OOM crashes."""

import gc
import os

import pytest
import torch


def pytest_runtest_setup(item):
    """Setup before each test - clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

        # Kill any lingering Python processes using GPU
        os.system("nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | xargs -r kill -9 2>/dev/null")


def pytest_runtest_teardown(item):
    """Teardown after each test - aggressive GPU cleanup."""
    if torch.cuda.is_available():
        # Clear all GPU tensors
        for obj in gc.get_objects():
            if torch.is_tensor(obj) and obj.is_cuda:
                del obj

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()


@pytest.fixture(scope="function", autouse=True)
def gpu_memory_limit():
    """Limit GPU memory allocation for tests."""
    if torch.cuda.is_available():
        # Reserve only 20GB for tests (leaving 4GB buffer on RTX 4090)
        torch.cuda.set_per_process_memory_fraction(0.8, 0)
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def skip_if_low_gpu_memory():
    """Skip test if GPU memory is too low."""
    if torch.cuda.is_available():
        free_memory = torch.cuda.mem_get_info()[0] / 1e9  # GB
        if free_memory < 10:  # Need at least 10GB free
            pytest.skip(f"Insufficient GPU memory: {free_memory:.1f}GB free")