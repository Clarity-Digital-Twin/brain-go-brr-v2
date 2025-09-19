"""Integration tests for the full SeizureDetector assembly (Phase 2.5)."""

import pytest
import torch

from src.brain_brr.models import SeizureDetector


class TestSeizureDetector:
    @pytest.fixture
    def model(self) -> SeizureDetector:
        return SeizureDetector(
            in_channels=19,
            base_channels=64,
            encoder_depth=4,
            mamba_layers=6,
            mamba_d_state=16,
            rescnn_blocks=3,
        )

    @pytest.fixture
    def sample_input(self) -> torch.Tensor:
        return torch.randn(4, 19, 15360)

    def test_forward_shape(self, model: SeizureDetector, sample_input: torch.Tensor) -> None:
        output = model(sample_input)
        assert output.shape == (4, 15360)

    def test_output_range(self, model: SeizureDetector, sample_input: torch.Tensor) -> None:
        """Test that model outputs raw logits (can be any real value)."""
        output = model(sample_input)
        # Logits can be any real value, just check they're finite
        assert torch.all(torch.isfinite(output))

    def test_gradient_flow(self, model: SeizureDetector, sample_input: torch.Tensor) -> None:
        sample_input.requires_grad = True
        output = model(sample_input)

        target = torch.rand_like(output)
        # Use BCEWithLogitsLoss since model outputs logits
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target)
        loss.backward()

        assert sample_input.grad is not None
        assert not torch.isnan(sample_input.grad).any()
        assert not torch.isinf(sample_input.grad).any()

    def test_no_nan_inf(self, model: SeizureDetector, sample_input: torch.Tensor) -> None:
        output = model(sample_input)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_parameter_count(self, model: SeizureDetector) -> None:
        info = model.get_layer_info()
        component_sum = (
            info["encoder_params"]
            + info["rescnn_params"]
            + info["mamba_params"]
            + info["decoder_params"]
            + info["head_params"]
        )
        assert component_sum == info["total_params"]
        assert 10_000_000 < info["total_params"] < 50_000_000

    def test_memory_usage(self, model: SeizureDetector) -> None:
        mem_info = model.get_memory_usage(batch_size=16)
        assert mem_info["model_size_mb"] < 200
        assert mem_info["total_size_mb"] < 4000

    def test_different_batch_sizes(self, model: SeizureDetector) -> None:
        for batch_size in [1, 8, 16, 32]:
            x = torch.randn(batch_size, 19, 15360)
            y = model(x)
            assert y.shape == (batch_size, 15360)

    def test_deterministic(self, model: SeizureDetector) -> None:
        torch.manual_seed(42)
        x = torch.randn(2, 19, 15360)
        out1 = model(x)
        torch.manual_seed(42)
        x2 = torch.randn(2, 19, 15360)
        out2 = model(x2)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_config_storage(self, model: SeizureDetector) -> None:
        info = model.get_layer_info()
        config = info["config"]
        assert config["in_channels"] == 19
        assert config["base_channels"] == 64
        assert config["encoder_depth"] == 4
        assert config["mamba_layers"] == 6
        assert config["mamba_d_state"] == 16
