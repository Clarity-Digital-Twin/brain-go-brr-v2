"""Integration tests for TCN replacement - WRITTEN BEFORE IMPLEMENTATION (TDD)."""

import tempfile
from pathlib import Path

import pytest
import torch


@pytest.mark.integration
class TestTCNFullPipeline:
    """End-to-end tests for TCN integration."""

    def test_full_pipeline_with_tcn(self):
        """TCN path should work end-to-end: EEG → TCN → Mamba → Output."""
        from src.brain_brr.config.schemas import ModelConfig
        from src.brain_brr.models.detector import SeizureDetector

        config = ModelConfig(
            architecture="tcn",
            tcn={
                "num_layers": 8,
                "kernel_size": 7,
                "dropout": 0.15,
                "channels": [64, 128, 256, 512],
                "causal": False
            },
            mamba={
                "conv_kernel": 4,  # Avoid CUDA coercion
                "d_model": 512,
                "n_layers": 6
            }
        )

        detector = SeizureDetector(config)
        detector.eval()

        # Simulate batch of EEG data
        batch_size = 4
        x = torch.randn(batch_size, 19, 15360)  # 60s @ 256Hz

        with torch.no_grad():
            output = detector(x)

        # Check output shape
        assert output.shape == (batch_size, 15360)

        # Check output is valid probabilities after sigmoid
        probs = torch.sigmoid(output)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_tcn_training_step(self):
        """TCN should work with training pipeline."""
        from src.brain_brr.train.loss import FocalLoss

        from src.brain_brr.config.schemas import ModelConfig
        from src.brain_brr.models.detector import SeizureDetector

        model_config = ModelConfig(
            architecture="tcn",
            tcn={"num_layers": 4, "kernel_size": 3}  # Small for testing
        )

        detector = SeizureDetector(model_config)
        optimizer = torch.optim.Adam(detector.parameters(), lr=1e-4)
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

        # Training step
        x = torch.randn(2, 19, 15360)
        labels = torch.randint(0, 2, (2, 15360)).float()

        # Forward
        logits = detector(x)
        loss = loss_fn(logits, labels)

        # Backward
        loss.backward()
        optimizer.step()

        # Should not crash and loss should be finite
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_tcn_checkpoint_save_load(self):
        """TCN model should save and load correctly."""
        from src.brain_brr.config.schemas import ModelConfig
        from src.brain_brr.models.detector import SeizureDetector

        config = ModelConfig(
            architecture="tcn",
            tcn={"num_layers": 4}
        )

        detector = SeizureDetector(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pth"

            # Save
            torch.save(detector.state_dict(), checkpoint_path)

            # Load into new model
            detector2 = SeizureDetector(config)
            detector2.load_state_dict(torch.load(checkpoint_path))

            # Test both produce same output
            x = torch.randn(1, 19, 15360)
            detector.eval()
            detector2.eval()

            with torch.no_grad():
                out1 = detector(x)
                out2 = detector2(x)

            assert torch.allclose(out1, out2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_tcn_mixed_precision(self):
        """TCN should work with mixed precision training."""
        from src.brain_brr.config.schemas import ModelConfig
        from src.brain_brr.models.detector import SeizureDetector

        config = ModelConfig(
            architecture="tcn",
            tcn={"num_layers": 4}
        )

        detector = SeizureDetector(config).cuda()
        x = torch.randn(2, 19, 15360).cuda()

        with torch.cuda.amp.autocast():
            output = detector(x)

        assert output.dtype == torch.float16  # Should be FP16 in autocast
        assert output.shape == (2, 15360)

    def test_config_backward_compatibility(self):
        """Old configs without 'architecture' field should default to U-Net."""
        from src.brain_brr.config.schemas import ModelConfig
        from src.brain_brr.models.detector import SeizureDetector

        # Old config without architecture field
        config_dict = {
            "mamba": {"d_model": 512, "n_layers": 6}
        }

        config = ModelConfig(**config_dict)

        # Should default to U-Net
        assert config.architecture == "unet"

        detector = SeizureDetector(config)
        assert hasattr(detector, 'encoder')  # U-Net encoder
        assert not hasattr(detector, 'tcn_encoder')


@pytest.mark.integration
class TestTCNPerformance:
    """Performance comparison tests (run locally, not in CI)."""

    @pytest.mark.slow
    def test_tcn_faster_than_unet(self):
        """TCN should be faster than U-Net+ResCNN per batch."""
        import time

        from src.brain_brr.config.schemas import ModelConfig
        from src.brain_brr.models.detector import SeizureDetector

        # Create both models
        unet_config = ModelConfig(architecture="unet")
        tcn_config = ModelConfig(architecture="tcn")

        unet_model = SeizureDetector(unet_config)
        tcn_model = SeizureDetector(tcn_config)

        x = torch.randn(4, 19, 15360)

        # Warmup
        for _ in range(3):
            _ = unet_model(x)
            _ = tcn_model(x)

        # Time U-Net
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        for _ in range(10):
            _ = unet_model(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        unet_time = time.perf_counter() - t0

        # Time TCN
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        for _ in range(10):
            _ = tcn_model(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        tcn_time = time.perf_counter() - t0

        print(f"U-Net: {unet_time:.3f}s, TCN: {tcn_time:.3f}s")
        # TCN should be at least 20% faster
        assert tcn_time < 0.8 * unet_time

    @pytest.mark.slow
    def test_tcn_memory_usage(self):
        """TCN should use less memory than U-Net+ResCNN."""
        from src.brain_brr.config.schemas import ModelConfig
        from src.brain_brr.models.detector import SeizureDetector

        if not torch.cuda.is_available():
            pytest.skip("CUDA required for memory profiling")

        # Create models
        unet_config = ModelConfig(architecture="unet")
        tcn_config = ModelConfig(architecture="tcn")

        # Measure U-Net memory
        torch.cuda.reset_peak_memory_stats()
        unet_model = SeizureDetector(unet_config).cuda()
        x = torch.randn(4, 19, 15360).cuda()
        _ = unet_model(x)
        unet_memory = torch.cuda.max_memory_allocated() / 1e9

        # Measure TCN memory
        torch.cuda.reset_peak_memory_stats()
        tcn_model = SeizureDetector(tcn_config).cuda()
        x = torch.randn(4, 19, 15360).cuda()
        _ = tcn_model(x)
        tcn_memory = torch.cuda.max_memory_allocated() / 1e9

        print(f"U-Net: {unet_memory:.2f}GB, TCN: {tcn_memory:.2f}GB")
        # TCN should use at least 30% less memory
        assert tcn_memory < 0.7 * unet_memory


@pytest.mark.integration
class TestCodeCleanliness:
    """Verify no U-Net/ResNet imports remain after migration."""

    @pytest.mark.skip(reason="Run after full migration")
    def test_no_unet_imports_remain(self):
        """After migration, no U-Net/ResNet imports should remain."""
        import subprocess

        # Search for U-Net/ResNet imports in src/
        result = subprocess.run(
            ["grep", "-r", "UNetEncoder\\|UNetDecoder\\|ResCNN", "src/"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 1, (
            f"Found U-Net/ResNet references:\n{result.stdout}"
        )

    @pytest.mark.skip(reason="Run after full migration")
    def test_files_deleted(self):
        """Old U-Net/ResNet files should be deleted."""
        from pathlib import Path

        files_should_not_exist = [
            "src/brain_brr/models/unet.py",
            "src/brain_brr/models/rescnn.py",
            "tests/unit/models/test_unet.py",
            "tests/unit/models/test_rescnn.py"
        ]

        for filepath in files_should_not_exist:
            assert not Path(filepath).exists(), f"{filepath} still exists!"