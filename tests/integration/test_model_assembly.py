"""Integration tests for the full SeizureDetector assembly (Phase 2.5)."""

import pytest
import torch

from src.brain_brr.config.schemas import ModelConfig
from src.brain_brr.models import SeizureDetector


class TestSeizureDetector:
    @pytest.fixture
    def model(self) -> SeizureDetector:
        # Use minimum allowed values for small test model
        cfg = ModelConfig(
            architecture="v3",
            tcn={
                "num_layers": 4,
                "kernel_size": 3,
                "stride_down": 16,
                "dropout": 0.1,
            },  # min 4 layers
            mamba={
                "n_layers": 1,
                "d_state": 16,
                "conv_kernel": 4,
                "dropout": 0.1,
            },  # d_state must be 16
            graph={"enabled": False},
        )
        return SeizureDetector.from_config(cfg)

    @pytest.fixture
    def sample_input(self) -> torch.Tensor:
        return torch.randn(4, 19, 15360)

    @pytest.mark.serial
    def test_forward_shape(self, model: SeizureDetector, sample_input: torch.Tensor) -> None:
        output = model(sample_input)
        assert output.shape == (4, 15360)

    @pytest.mark.serial
    def test_output_range(self, model: SeizureDetector, sample_input: torch.Tensor) -> None:
        """Test that model outputs raw logits (can be any real value)."""
        output = model(sample_input)
        # Logits can be any real value, just check they're finite
        assert torch.all(torch.isfinite(output))

    @pytest.mark.serial
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
            info["tcn_params"] + info["mamba_params"] + info["proj_params"] + info["head_params"]
        )
        assert component_sum == info["total_params"]
        # Small TCN+Mamba model: ~1-10M params
        assert 500_000 < info["total_params"] < 10_000_000

    def test_memory_usage(self, model: SeizureDetector) -> None:
        mem_info = model.get_memory_usage(batch_size=16)
        assert mem_info["model_size_mb"] < 200
        assert mem_info["total_size_mb"] < 4000

    @pytest.mark.serial
    @pytest.mark.gpu  # Large batch sizes need GPU memory
    def test_different_batch_sizes(self, model: SeizureDetector) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Use smaller batch sizes for small model
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 19, 15360, device=device)
            y = model(x)
            assert y.shape == (batch_size, 15360)

            # Clear cache to prevent OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def test_deterministic(self, model: SeizureDetector) -> None:
        torch.manual_seed(42)
        x = torch.randn(2, 19, 15360)
        model.eval()  # Ensure eval mode for deterministic behavior
        with torch.no_grad():
            out1 = model(x)

        torch.manual_seed(42)
        x2 = torch.randn(2, 19, 15360)
        with torch.no_grad():
            out2 = model(x2)

        assert torch.allclose(out1, out2, atol=1e-6)

    def test_config_storage(self, model: SeizureDetector) -> None:
        info = model.get_layer_info()
        config = info["config"]
        assert config["in_channels"] == 19
        assert config["tcn_layers"] == 4  # minimum allowed
        assert config["tcn_kernel_size"] == 3
        assert config["mamba_layers"] == 1
        assert config["mamba_d_state"] == 16  # required value
