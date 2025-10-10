"""Tests for RNG state device handling in checkpoint save/load.

REGRESSION TESTS for RNG_STATE_DEVICE_BUG.md:
- torch.set_rng_state() requires CPU ByteTensor
- torch.cuda.set_rng_state_all() requires GPU tensors
- Checkpoints loaded with map_location move ALL tensors (including RNG states)
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.brain_brr.config.schemas import Config, ModelConfig
from src.brain_brr.models.detector import SeizureDetector
from src.brain_brr.train.checkpoint import load_checkpoint, save_checkpoint
from src.brain_brr.train.optimizer_factory import create_optimizer


@pytest.fixture
def model_config() -> ModelConfig:
    """Minimal model config for testing."""
    return ModelConfig(
        architecture="v3",
        tcn={
            "num_layers": 4,
            "kernel_size": 3,
            "stride_down": 16,
            "dropout": 0.1,
        },
        mamba={
            "n_layers": 1,
            "d_state": 16,
            "conv_kernel": 4,
            "dropout": 0.1,
        },
        graph={
            "enabled": True,
            "k_eigenvectors": 4,
            "use_dynamic_pe": True,
        },
    )


@pytest.fixture
def full_config(model_config: ModelConfig) -> Config:
    """Full config with training parameters."""
    return Config(
        model=model_config,
        experiment={
            "output_dir": "/tmp/test",
            "seed": 42,
        },
        training={
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 1e-3,
        },
        data={
            "data_dir": "/tmp/test",
            "cache_dir": "/tmp/test",
        },
        preprocessing={},
        postprocessing={},
        evaluation={},
        logging={},
    )


def test_rng_cpu_save_cpu_load(
    tmp_path: Path, model_config: ModelConfig, full_config: Config
) -> None:
    """Test RNG state save/load when both use CPU (baseline).

    This is the simple case that already worked.
    """
    checkpoint_path = tmp_path / "test_rng_cpu.pt"

    # Create model and save checkpoint on CPU
    model1 = SeizureDetector.from_config(model_config)
    optimizer1 = create_optimizer(model1, full_config.training)

    # Set a known RNG state
    torch.manual_seed(12345)
    initial_rng = torch.get_rng_state().clone()

    save_checkpoint(
        model1,
        optimizer1,
        epoch=0,
        best_metric=0.5,
        checkpoint_path=checkpoint_path,
        save_rng=True,
    )

    # Change RNG state
    torch.manual_seed(99999)
    assert not torch.equal(torch.get_rng_state(), initial_rng)

    # Load checkpoint on CPU (should restore RNG)
    model2 = SeizureDetector.from_config(model_config)
    load_checkpoint(checkpoint_path, model2, device="cpu", restore_rng=True)

    # RNG state should match initial
    assert torch.equal(torch.get_rng_state(), initial_rng)


@pytest.mark.gpu
def test_rng_cpu_save_cuda_load(
    tmp_path: Path, model_config: ModelConfig, full_config: Config
) -> None:
    """Test RNG state save/load: CPU save, CUDA load.

    REGRESSION TEST for RNG_STATE_DEVICE_BUG.md:
    - Checkpoint saved with device="cpu" (RNG on CPU)
    - Checkpoint loaded with device="cuda" (moves all tensors to CUDA)
    - torch.set_rng_state() must receive CPU tensor (requires .cpu() call)
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    checkpoint_path = tmp_path / "test_rng_cpu_cuda.pt"

    # Create model and save checkpoint on CPU
    model1 = SeizureDetector.from_config(model_config)
    optimizer1 = create_optimizer(model1, full_config.training)

    # Set a known RNG state
    torch.manual_seed(12345)
    initial_cpu_rng = torch.get_rng_state().clone()

    save_checkpoint(
        model1,
        optimizer1,
        epoch=0,
        best_metric=0.5,
        checkpoint_path=checkpoint_path,
        save_rng=True,
    )

    # Change RNG state
    torch.manual_seed(99999)
    assert not torch.equal(torch.get_rng_state(), initial_cpu_rng)

    # Load checkpoint with device="cuda" (THIS IS THE BUG SCENARIO)
    model2 = SeizureDetector.from_config(model_config)
    load_checkpoint(checkpoint_path, model2, device="cuda", restore_rng=True)

    # Should succeed without "RNG state must be a torch.ByteTensor" error
    # CPU RNG state should match initial (was forced back to CPU with .cpu())
    restored_rng = torch.get_rng_state()
    assert torch.equal(restored_rng, initial_cpu_rng)


@pytest.mark.gpu
def test_rng_cuda_save_cuda_load(
    tmp_path: Path, model_config: ModelConfig, full_config: Config
) -> None:
    """Test RNG state save/load: CUDA save, CUDA load.

    This tests the normal Modal resume workflow (A100 → A100).
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    checkpoint_path = tmp_path / "test_rng_cuda_cuda.pt"

    # Create model and save checkpoint on CUDA
    model1 = SeizureDetector.from_config(model_config).cuda()
    optimizer1 = create_optimizer(model1, full_config.training)

    # Set a known RNG state
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)
    initial_cpu_rng = torch.get_rng_state().clone()
    initial_cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]

    save_checkpoint(
        model1,
        optimizer1,
        epoch=0,
        best_metric=0.5,
        checkpoint_path=checkpoint_path,
        save_rng=True,
    )

    # Change RNG states
    torch.manual_seed(99999)
    torch.cuda.manual_seed_all(99999)
    assert not torch.equal(torch.get_rng_state(), initial_cpu_rng)

    # Load checkpoint with device="cuda" (Modal resume workflow)
    model2 = SeizureDetector.from_config(model_config)
    load_checkpoint(checkpoint_path, model2, device="cuda", restore_rng=True)

    # Both CPU and CUDA RNG states should be restored
    assert torch.equal(torch.get_rng_state(), initial_cpu_rng)
    restored_cuda_rng = torch.cuda.get_rng_state_all()
    for initial, restored in zip(initial_cuda_rng, restored_cuda_rng, strict=False):
        assert torch.equal(initial, restored)


@pytest.mark.gpu
def test_rng_cuda_save_cpu_load(
    tmp_path: Path, model_config: ModelConfig, full_config: Config
) -> None:
    """Test RNG state save/load: CUDA save, CPU load.

    This tests the edge case of loading a GPU checkpoint on CPU.
    CUDA RNG state should be safely ignored (no CUDA available).
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    checkpoint_path = tmp_path / "test_rng_cuda_cpu.pt"

    # Create model and save checkpoint on CUDA
    model1 = SeizureDetector.from_config(model_config).cuda()
    optimizer1 = create_optimizer(model1, full_config.training)

    # Set a known RNG state
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)
    initial_cpu_rng = torch.get_rng_state().clone()

    save_checkpoint(
        model1,
        optimizer1,
        epoch=0,
        best_metric=0.5,
        checkpoint_path=checkpoint_path,
        save_rng=True,
    )

    # Change RNG state
    torch.manual_seed(99999)
    assert not torch.equal(torch.get_rng_state(), initial_cpu_rng)

    # Load checkpoint with device="cpu" (CUDA RNG moved to CPU by map_location)
    model2 = SeizureDetector.from_config(model_config)
    load_checkpoint(checkpoint_path, model2, device="cpu", restore_rng=True)

    # CPU RNG state should be restored
    # CUDA RNG state should be moved to CUDA before calling set_rng_state_all
    assert torch.equal(torch.get_rng_state(), initial_cpu_rng)
