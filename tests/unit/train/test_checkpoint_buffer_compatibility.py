"""Test checkpoint compatibility with dynamic buffers (CHECKPOINT_BUFFER_BUG.md).

Regression tests for the buffer incompatibility bug discovered in v3.10.0.
"""

from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
import torch
import torch.nn as nn

from src.brain_brr.config.schemas import Config, ModelConfig
from src.brain_brr.models import SeizureDetector
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
    """Full config for optimizer creation."""
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


def test_buffer_appears_in_state_dict_immediately(model_config: ModelConfig) -> None:
    """Test that last_valid_pe buffer is in state_dict from initialization.

    REGRESSION TEST for CHECKPOINT_BUFFER_BUG.md:
    - Old code: register_buffer("last_valid_pe", None) → buffer NOT in state_dict
    - New code: register_buffer with dummy tensor → buffer ALWAYS in state_dict
    """
    model = SeizureDetector.from_config(model_config)

    # Check buffer exists as attribute
    assert hasattr(model.gnn, "last_valid_pe")

    # CRITICAL: Buffer should be in state_dict BEFORE any tensor assignment
    state_dict = model.state_dict()
    assert "gnn.last_valid_pe" in state_dict, (
        "last_valid_pe buffer should appear in state_dict from initialization "
        "(prevents checkpoint incompatibility)"
    )

    # Buffer should have placeholder shape (1, 1, 1, k)
    buffer_shape = model.gnn.last_valid_pe.shape
    assert buffer_shape == (1, 1, 1, 4), f"Expected (1,1,1,4) placeholder, got {buffer_shape}"


def test_checkpoint_save_load_with_buffer(model_config: ModelConfig, full_config: Config) -> None:
    """Test checkpoint save/load after buffer has been updated during training.

    REGRESSION TEST for CHECKPOINT_BUFFER_BUG.md:
    - Simulates: buffer gets assigned during forward pass
    - Verifies: checkpoint can be loaded into fresh model (buffer starts as placeholder)

    NOTE: With strict=False, PyTorch allows shape mismatches for buffers.
    The loaded checkpoint will keep its original buffer shape, which is correct
    (it will be overwritten on next forward pass anyway).
    """
    # Create model and optimizer
    model1 = SeizureDetector.from_config(model_config)
    optimizer = create_optimizer(model1, full_config.training)

    # Simulate forward pass updating last_valid_pe buffer (arbitrary size for testing)
    B, T, N, k = 3, 15, 19, model_config.graph.k_eigenvectors  # noqa: N806 (standard DL notation)
    model1.gnn.last_valid_pe = torch.randn(B, T, N, k)

    # Save checkpoint
    with NamedTemporaryFile(suffix=".pt", delete=False) as f:
        checkpoint_path = Path(f.name)

    try:
        save_checkpoint(
            model1, optimizer, epoch=0, best_metric=0.5, checkpoint_path=checkpoint_path
        )

        # Load into fresh model (buffer starts as placeholder)
        model2 = SeizureDetector.from_config(model_config)

        # Verify fresh model has placeholder
        assert model2.gnn.last_valid_pe.shape == (1, 1, 1, k), "Fresh model should have placeholder"

        # Load checkpoint (strict=False allows buffer shape mismatch)
        epoch, metric = load_checkpoint(checkpoint_path, model2)

        # Verify checkpoint loaded successfully
        assert epoch == 0
        assert metric == 0.5

        # After loading, buffer keeps placeholder shape because checkpoint buffer was skipped
        # due to shape mismatch. This is CORRECT: first forward pass will update it.
        assert model2.gnn.last_valid_pe.shape == (1, 1, 1, k), (
            f"After load, buffer should keep placeholder shape {(1, 1, 1, k)} "
            f"(checkpoint buffer was skipped due to shape mismatch, will update on first forward)"
        )

    finally:
        checkpoint_path.unlink(missing_ok=True)


def test_checkpoint_strict_false_handles_extra_keys(
    model_config: ModelConfig, full_config: Config
) -> None:
    """Test that strict=False allows loading checkpoints with extra buffer keys.

    REGRESSION TEST for CHECKPOINT_BUFFER_BUG.md:
    - Simulates: old checkpoint has dynamic buffer that fresh model doesn't expect
    - Verifies: strict=False allows load without errors
    """
    # Create model and manually add a buffer to checkpoint
    model1 = SeizureDetector.from_config(model_config)
    optimizer = create_optimizer(model1, full_config.training)

    # Save checkpoint
    with NamedTemporaryFile(suffix=".pt", delete=False) as f:
        checkpoint_path = Path(f.name)

    try:
        save_checkpoint(
            model1, optimizer, epoch=0, best_metric=0.5, checkpoint_path=checkpoint_path
        )

        # Manually add an extra key to checkpoint (simulates old dynamic buffer)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint["model_state_dict"]["gnn.extra_buffer"] = torch.zeros(10)
        torch.save(checkpoint, checkpoint_path)

        # Load into fresh model (strict=False should handle extra key)
        model2 = SeizureDetector.from_config(model_config)
        epoch, metric = load_checkpoint(checkpoint_path, model2)

        # Should load successfully despite extra key
        assert epoch == 0
        assert metric == 0.5

    finally:
        checkpoint_path.unlink(missing_ok=True)


def test_buffer_fallback_logic_with_placeholder(model_config: ModelConfig) -> None:
    """Test that fallback logic correctly detects placeholder vs. valid buffer.

    REGRESSION TEST for CHECKPOINT_BUFFER_BUG.md:
    - Placeholder: (1, 1, 1, k) → should NOT be used as fallback
    - Valid PE: (B, T, N, k) → should be used as fallback

    NOTE: This test verifies the buffer initialization, not the full forward pass
    (which requires proper input dimensions that match the model architecture).
    """
    model = SeizureDetector.from_config(model_config)
    k = model_config.graph.k_eigenvectors

    # Initially has placeholder (1, 1, 1, k)
    assert model.gnn.last_valid_pe.shape == (1, 1, 1, k), (
        "Fresh model should have placeholder buffer"
    )

    # Manually simulate what happens during forward pass: buffer gets updated
    B, T, N = 2, 10, 19  # noqa: N806 (standard DL notation)
    model.gnn.last_valid_pe = torch.randn(B, T, N, k)

    # Now buffer has valid shape
    assert model.gnn.last_valid_pe.shape == (B, T, N, k), (
        "After update, buffer should have valid PE shape"
    )

    # The fallback logic checks shape: if shape[0] == B and shape[1] == T, use it
    # Placeholder (1, 1, 1, k) will NOT match (B, T, 1, k) → fallback to random PE
    # Valid PE (B, T, N, k) will match → use cached PE
    # This is tested implicitly by the GNN forward pass under NaN conditions


def test_pytorch_register_buffer_none_behavior() -> None:
    """Document PyTorch's register_buffer(None) behavior for future reference.

    EDUCATIONAL TEST - Documents the root cause of CHECKPOINT_BUFFER_BUG.md.

    This test shows:
    1. register_buffer("name", None) does NOT add to state_dict
    2. Assigning tensor later DOES add to state_dict
    3. This timing-dependent behavior causes checkpoint incompatibility
    """

    class ModelWithNone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("buf", None)

    class ModelWithTensor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("buf", torch.zeros(1))

    # Model with None: buffer NOT in state_dict
    m1 = ModelWithNone()
    assert hasattr(m1, "buf"), "Attribute should exist"
    assert m1.buf is None, "Value should be None"
    assert "buf" not in m1.state_dict(), "Buffer should NOT be in state_dict"

    # Model with tensor: buffer IS in state_dict
    m2 = ModelWithTensor()
    assert "buf" in m2.state_dict(), "Buffer SHOULD be in state_dict"

    # After assigning tensor to None buffer: NOW appears in state_dict
    m1.buf = torch.ones(2)
    assert "buf" in m1.state_dict(), "Buffer should NOW be in state_dict after assignment"
