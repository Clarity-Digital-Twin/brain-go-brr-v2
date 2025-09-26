"""Performance test configuration to prevent resource contention."""

import gc
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


@pytest.fixture(autouse=True)
def cleanup_gpu():
    """Clean up GPU memory after each test to prevent OOM."""
    yield
    # Clean up after test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


@pytest.fixture
def minimal_model_no_leak():
    """Create minimal model with proper cleanup."""
    from src.brain_brr.config.schemas import MambaConfig, ModelConfig, TCNConfig
    from src.brain_brr.models import SeizureDetector

    config = ModelConfig(
        architecture="v3",
        tcn=TCNConfig(num_layers=4, channels=[32, 64, 128, 256], kernel_size=3),
        mamba=MambaConfig(n_layers=2, d_model=256, d_state=16),
    )
    model = SeizureDetector.from_config(config)
    model.eval()

    # Move to GPU if available
    if torch.cuda.is_available() and os.getenv("BGB_PERF_ALLOW_GPU", "0") == "1":
        model = model.cuda()

    yield model

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
