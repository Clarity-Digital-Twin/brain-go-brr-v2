"""Unit tests for gradient sanitization functionality."""

from __future__ import annotations

import logging

import pytest
import torch
import torch.nn as nn

from src.brain_brr.train.train_step import _sanitize_gradients


class SimpleModel(nn.Module):
    """Minimal model for testing gradient sanitization."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def model():
    """Create a simple model for testing."""
    return SimpleModel()


@pytest.fixture
def logger():
    """Create a logger for testing."""
    return logging.getLogger(__name__)


class TestGradientSanitization:
    """Test suite for _sanitize_gradients() function."""

    def test_sanitize_nan_gradients(self, model, logger):
        """Verify NaN gradients are replaced with zeros."""
        for param in model.parameters():
            param.grad = torch.ones_like(param)
            param.grad.flatten()[1] = float("nan")
            break

        count = _sanitize_gradients(model, logger, batch_idx=0)

        assert count == 1
        for param in model.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all()
                assert param.grad.flatten()[1] == 0.0
                break

    def test_sanitize_inf_gradients(self, model, logger):
        """Verify inf gradients are replaced with zeros."""
        for param in model.parameters():
            param.grad = torch.ones_like(param)
            param.grad.flatten()[1] = float("inf")
            param.grad.flatten()[2] = -float("inf")
            break

        count = _sanitize_gradients(model, logger, batch_idx=0)

        assert count == 1
        for param in model.parameters():
            if param.grad is not None:
                assert param.grad.flatten()[1] == 0.0
                assert param.grad.flatten()[2] == 0.0
                break

    def test_sanitize_mixed_nonfinite(self, model, logger):
        """Verify mixed NaN/Inf gradients are replaced."""
        for param in model.parameters():
            param.grad = torch.ones_like(param)
            flat = param.grad.flatten()
            flat[0] = float("nan")
            flat[1] = float("inf")
            flat[2] = 1.0
            flat[3] = -float("inf")
            flat[4] = 2.0
            break

        count = _sanitize_gradients(model, logger, batch_idx=0)

        assert count == 1
        for param in model.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all()
                flat = param.grad.flatten()
                assert flat[0] == 0.0
                assert flat[1] == 0.0
                assert flat[2] == 1.0
                assert flat[3] == 0.0
                assert flat[4] == 2.0
                break

    def test_noop_when_finite(self, model, logger):
        """Verify no changes when all gradients are finite."""
        for param in model.parameters():
            param.grad = torch.randn_like(param)

        original_grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]
        count = _sanitize_gradients(model, logger, batch_idx=0)

        assert count == 0
        for orig, param in zip(original_grads, model.parameters()):
            if param.grad is not None:
                assert torch.equal(orig, param.grad)

    def test_noop_when_no_gradients(self, model, logger):
        """Verify no crashes when model has no gradients."""
        for param in model.parameters():
            param.grad = None

        count = _sanitize_gradients(model, logger, batch_idx=0)

        assert count == 0

    def test_sanitize_multiple_parameters(self, logger):
        """Verify sanitization works across multiple parameters."""

        class MultiParamModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 5)
                self.linear2 = nn.Linear(5, 1)

        model = MultiParamModel()

        params = list(model.parameters())
        params[0].grad = torch.tensor([[float("nan")]] * 10)
        params[1].grad = torch.tensor([float("inf")] * 5)
        params[2].grad = torch.randn(5, 1)
        params[3].grad = torch.tensor([float("-inf")])

        count = _sanitize_gradients(model, logger, batch_idx=0)

        assert count == 3
        for param in model.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all()

    def test_preserves_finite_values(self, model, logger):
        """Verify finite values are preserved during sanitization."""
        for param in model.parameters():
            param.grad = torch.ones_like(param)
            flat = param.grad.flatten()
            flat[0] = 1.5
            flat[1] = float("nan")
            flat[2] = -3.7
            flat[3] = float("inf")
            flat[4] = 0.0
            break

        _sanitize_gradients(model, logger, batch_idx=0)

        for param in model.parameters():
            if param.grad is not None:
                flat = param.grad.flatten()
                assert flat[0] == 1.5
                assert flat[2] == -3.7
                assert flat[4] == 0.0
                break

    def test_inplace_modification(self, model, logger):
        """Verify sanitization modifies gradients in-place."""
        for param in model.parameters():
            param.grad = torch.ones_like(param)
            param.grad.flatten()[1] = float("nan")
            grad_id = id(param.grad)
            break

        _sanitize_gradients(model, logger, batch_idx=0)

        for param in model.parameters():
            if param.grad is not None:
                assert id(param.grad) == grad_id
                break

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_sanitize_cuda_gradients(self, logger):
        """Verify sanitization works on CUDA tensors."""
        model = SimpleModel().cuda()

        for param in model.parameters():
            param.grad = torch.ones_like(param).cuda()
            param.grad.flatten()[1] = float("nan")
            break

        count = _sanitize_gradients(model, logger, batch_idx=0)

        assert count == 1
        for param in model.parameters():
            if param.grad is not None:
                assert param.grad.device.type == "cuda"
                assert torch.isfinite(param.grad).all()
                assert param.grad.flatten()[1] == 0.0
                break
