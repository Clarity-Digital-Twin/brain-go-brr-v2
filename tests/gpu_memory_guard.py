"""GPU memory guard for tests to prevent OOM crashes."""

import gc
import os
import subprocess

import pytest
import torch


def _is_training_active() -> bool:
    """Check if training is currently running on GPU."""
    if not torch.cuda.is_available():
        return False
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        lines = result.stdout.split("\n")
        for line in lines:
            if ("python -m src train" in line or "make train-local" in line) and "grep" not in line:
                return True
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return False


def _get_available_gpu_memory() -> float:
    """Get available GPU memory in GB."""
    if not torch.cuda.is_available():
        return 0.0
    free_memory, _total_memory = torch.cuda.mem_get_info()
    return free_memory / 1e9


def pytest_runtest_setup(item):
    """Setup before each test - clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

        # Kill any lingering Python processes using GPU
        # NOTE: Disabled as it can hang test execution
        # os.system("nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | xargs -r kill -9 2>/dev/null")


def pytest_runtest_teardown(item):
    """
    Teardown after each test - aggressive GPU cleanup.

    This session hook performs the DEFINITIVE cleanup by iterating
    gc.get_objects() and deleting CUDA tensors. This is the ONLY place
    where this expensive operation should happen.

    The conftest.py cleanup_torch_resources() fixture performs lightweight
    cache clearing only, to avoid duplicate iterations.
    """
    if torch.cuda.is_available():
        # Clear all GPU tensors (DEFINITIVE CLEANUP - only here!)
        for obj in gc.get_objects():
            if torch.is_tensor(obj) and obj.is_cuda:
                del obj

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()


@pytest.fixture(autouse=True, scope="session")
def gpu_memory_limit():
    """Limit GPU memory allocation for tests with training-aware adjustment."""
    if torch.cuda.is_available():
        training_active = _is_training_active()
        available_memory = _get_available_gpu_memory()

        if training_active or available_memory < 10:
            # Training detected or low memory: use minimal allocation (2-3GB)
            torch.cuda.set_per_process_memory_fraction(0.12, 0)  # 12% of 24GB = ~3GB
            print(
                f"\n⚠️  Training detected or low GPU memory ({available_memory:.1f}GB free)"
                f"\n   Tests limited to 3GB VRAM. Use BGB_SKIP_GPU_TESTS=1 to skip GPU tests."
            )
        else:
            # Normal test mode: conservative limit (10GB)
            torch.cuda.set_per_process_memory_fraction(0.4, 0)  # 40% of 24GB = ~10GB

    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def skip_if_low_gpu_memory():
    """Skip test if GPU memory is too low or training is active."""
    if os.getenv("BGB_SKIP_GPU_TESTS", "0") == "1":
        pytest.skip("GPU tests disabled via BGB_SKIP_GPU_TESTS=1")

    if torch.cuda.is_available():
        training_active = _is_training_active()
        free_memory = _get_available_gpu_memory()

        if training_active:
            pytest.skip(
                f"Training active on GPU ({free_memory:.1f}GB free). "
                f"Run 'make test-cpu' or 'make t' for CPU-only tests."
            )

        if free_memory < 5:  # Need at least 5GB for minimal tests
            pytest.skip(
                f"Insufficient GPU memory: {free_memory:.1f}GB free (need 5GB). "
                f"Stop training or use 'make test-cpu'."
            )
