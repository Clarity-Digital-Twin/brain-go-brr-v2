"""Test TCN encoder module - WRITTEN BEFORE IMPLEMENTATION (TDD)."""


import pytest
import torch


class TestTCNEncoder:
    """Test TCN encoder shape contracts and integration."""

    @pytest.fixture
    def batch_shape(self) -> tuple[int, int, int]:
        """Standard batch shape for testing."""
        return (4, 19, 15360)  # (batch, channels, time @ 256Hz)

    @pytest.fixture
    def expected_output_shape(self) -> tuple[int, int, int]:
        """Expected TCN output shape for Mamba compatibility."""
        return (4, 512, 960)  # (batch, features, time/16)

    def test_tcn_encoder_shape_contract(self, batch_shape, expected_output_shape):
        """TCN must produce exact shape for Mamba input."""
        from src.brain_brr.models.tcn import TCNEncoder

        model = TCNEncoder(
            input_channels=19,
            output_channels=512,
            num_layers=8,
            kernel_size=7,
            dropout=0.15,
            causal=False  # Non-causal for offline training
        )

        x = torch.randn(*batch_shape)
        output = model(x)

        assert output.shape == expected_output_shape, (
            f"TCN output shape {output.shape} != expected {expected_output_shape}"
        )

    def test_tcn_encoder_gradient_flow(self, batch_shape):
        """Ensure gradients flow through TCN without vanishing."""
        from src.brain_brr.models.tcn import TCNEncoder

        model = TCNEncoder(
            input_channels=19,
            output_channels=512,
            num_layers=8,
            kernel_size=7
        )

        x = torch.randn(*batch_shape, requires_grad=True)
        output = model(x)
        loss = output.mean()
        loss.backward()

        assert x.grad is not None, "No gradients computed"
        assert not torch.isnan(x.grad).any(), "NaN gradients"
        assert x.grad.abs().mean() > 1e-8, "Vanishing gradients"

    def test_tcn_encoder_parameter_efficiency(self):
        """TCN must have fewer parameters than U-Net+ResCNN (~47M)."""
        from src.brain_brr.models.tcn import TCNEncoder

        model = TCNEncoder(
            input_channels=19,
            output_channels=512,
            num_layers=8,
            kernel_size=7
        )

        total_params = sum(p.numel() for p in model.parameters())

        # Should be much less than 47M (U-Net + ResCNN)
        assert total_params < 10_000_000, (
            f"TCN has {total_params/1e6:.1f}M params, should be <10M"
        )

    def test_tcn_handles_variable_batch_size(self):
        """TCN should handle different batch sizes."""
        from src.brain_brr.models.tcn import TCNEncoder

        model = TCNEncoder(
            input_channels=19,
            output_channels=512,
            num_layers=8
        )

        # Test different batch sizes
        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 19, 15360)
            output = model(x)
            assert output.shape == (batch_size, 512, 960)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_tcn_cuda_optimization(self):
        """TCN should use CUDA efficiently when available."""
        from src.brain_brr.models.tcn import TCNEncoder

        model = TCNEncoder(
            input_channels=19,
            output_channels=512,
            use_cuda_optimizations=True
        ).cuda()

        x = torch.randn(4, 19, 15360).cuda()

        # Should not crash with CUDA optimizations
        with torch.cuda.amp.autocast():
            output = model(x)
            assert output.is_cuda
            assert output.shape == (4, 512, 960)


class TestTCNProjectionHead:
    """Test the projection + upsampling head after Mamba."""

    def test_projection_head_shape_restoration(self):
        """Projection head must restore full temporal resolution."""
        from src.brain_brr.models.tcn import ProjectionHead

        head = ProjectionHead(
            input_channels=512,
            output_channels=19,
            upsample_factor=16
        )

        # Input from Mamba: (B, 512, 960)
        x = torch.randn(4, 512, 960)
        output = head(x)

        # Must restore to (B, 19, 15360) for detection head
        assert output.shape == (4, 19, 15360), (
            f"Head output {output.shape} != expected (4, 19, 15360)"
        )

    def test_projection_head_gradient_flow(self):
        """Gradients must flow through projection head."""
        from src.brain_brr.models.tcn import ProjectionHead

        head = ProjectionHead(512, 19, 16)
        x = torch.randn(2, 512, 960, requires_grad=True)
        output = head(x)
        loss = output.mean()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestTCNIntegration:
    """Integration tests for TCN in the full detector pipeline."""

    def test_detector_with_tcn_flag(self):
        """Detector should use TCN when architecture='tcn'."""
        from src.brain_brr.config.schemas import ModelConfig
        from src.brain_brr.models.detector import SeizureDetector

        # Create config with TCN architecture
        config = ModelConfig(
            architecture="tcn",
            tcn={
                "num_layers": 8,
                "kernel_size": 7,
                "dropout": 0.15,
                "channels": [64, 128, 256, 512]
            },
            mamba={
                "conv_kernel": 4  # Avoid coercion warning
            }
        )

        detector = SeizureDetector.from_config(config)

        # Should have TCN components
        assert hasattr(detector, 'tcn_encoder')
        assert hasattr(detector, 'proj_512_to_19')
        assert hasattr(detector, 'upsample')

        # Should NOT have U-Net components
        assert not hasattr(detector, 'encoder')
        assert not hasattr(detector, 'decoder')
        assert not hasattr(detector, 'rescnn')

    def test_detector_with_unet_flag(self):
        """Detector should use U-Net when architecture='unet' (backwards compat)."""
        from src.brain_brr.config.schemas import ModelConfig
        from src.brain_brr.models.detector import SeizureDetector

        config = ModelConfig(
            architecture="unet"  # Old path
        )

        detector = SeizureDetector.from_config(config)

        # Should have U-Net components
        assert hasattr(detector, 'encoder')
        assert hasattr(detector, 'decoder')
        assert hasattr(detector, 'rescnn')

        # Should NOT have TCN components
        assert not hasattr(detector, 'tcn_encoder')

    def test_detector_forward_with_tcn(self):
        """Full forward pass with TCN should produce correct output shape."""
        from src.brain_brr.config.schemas import ModelConfig
        from src.brain_brr.models.detector import SeizureDetector

        config = ModelConfig(
            architecture="tcn",
            tcn={
                "num_layers": 4,  # Smaller for testing
                "kernel_size": 3,
                "dropout": 0.1
            }
        )

        detector = SeizureDetector.from_config(config)
        x = torch.randn(2, 19, 15360)

        output = detector(x)

        # Must maintain per-sample output at 256Hz
        assert output.shape == (2, 15360), (
            f"Detector output {output.shape} != expected (2, 15360)"
        )

    def test_loss_compatibility_with_tcn(self):
        """TCN path must produce outputs compatible with existing loss."""
        import torch.nn.functional as functional

        from src.brain_brr.config.schemas import ModelConfig
        from src.brain_brr.models.detector import SeizureDetector

        config = ModelConfig(architecture="tcn")
        detector = SeizureDetector.from_config(config)

        x = torch.randn(2, 19, 15360)
        labels = torch.randint(0, 2, (2, 15360)).float()

        output = detector(x)

        # Should be able to compute loss without shape errors
        loss = functional.binary_cross_entropy_with_logits(output, labels)
        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss)

    def test_config_gating_works(self):
        """Switching architecture flag should change detector behavior."""
        from src.brain_brr.config.schemas import ModelConfig
        from src.brain_brr.models.detector import SeizureDetector

        # Test both paths work
        for arch in ["tcn", "unet"]:
            config = ModelConfig(architecture=arch)
            detector = SeizureDetector.from_config(config)
            x = torch.randn(1, 19, 15360)
            output = detector(x)
            assert output.shape == (1, 15360), f"Failed for architecture={arch}"
