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
                "causal": False,
            },
            mamba={
                "conv_kernel": 4,  # Avoid CUDA coercion
                "d_model": 512,
                "n_layers": 6,
            },
        )

        detector = SeizureDetector.from_config(config)
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
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_tcn_training_step(self):
        """TCN should work with training pipeline."""
        import torch.nn.functional as functional

        from src.brain_brr.config.schemas import ModelConfig
        from src.brain_brr.models.detector import SeizureDetector

        model_config = ModelConfig(
            architecture="tcn",
            tcn={"num_layers": 4, "kernel_size": 3},  # Small for testing
        )

        detector = SeizureDetector.from_config(model_config)
        optimizer = torch.optim.Adam(detector.parameters(), lr=1e-4)

        # Training step
        x = torch.randn(2, 19, 15360)
        labels = torch.randint(0, 2, (2, 15360)).float()

        # Forward
        logits = detector(x)
        loss = functional.binary_cross_entropy_with_logits(logits, labels)

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

        config = ModelConfig(architecture="tcn", tcn={"num_layers": 4})

        detector = SeizureDetector.from_config(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pth"

            # Save
            torch.save(detector.state_dict(), checkpoint_path)

            # Load into new model
            detector2 = SeizureDetector.from_config(config)
            detector2.load_state_dict(torch.load(checkpoint_path))

            # Test both produce same output
            x = torch.randn(1, 19, 15360)
            detector.eval()
            detector2.eval()

            with torch.no_grad():
                out1 = detector(x)
                out2 = detector2(x)

            assert torch.allclose(out1, out2)

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_tcn_mixed_precision(self):
        """TCN should work with mixed precision training."""
        from src.brain_brr.config.schemas import ModelConfig
        from src.brain_brr.models.detector import SeizureDetector

        config = ModelConfig(architecture="tcn", tcn={"num_layers": 4})

        detector = SeizureDetector.from_config(config).cuda()
        x = torch.randn(2, 19, 15360).cuda()

        with torch.cuda.amp.autocast():
            output = detector(x)

        assert output.dtype == torch.float16  # Should be FP16 in autocast
        assert output.shape == (2, 15360)

    def test_config_defaults_to_tcn(self):
        """Config should default to TCN architecture."""
        from src.brain_brr.config.schemas import ModelConfig
        from src.brain_brr.models.detector import SeizureDetector

        # Config should default to TCN
        config = ModelConfig()
        assert config.architecture == "tcn"

        # Should create TCN model
        detector = SeizureDetector.from_config(config)
        assert hasattr(detector, "tcn_encoder")
        assert hasattr(detector, "proj_head")


@pytest.mark.integration
class TestTCNPerformance:
    """TCN architecture performance tests."""

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="GPU required for performance testing"
    )
    def test_tcn_inference_speed(self):
        """TCN should have fast inference on GPU."""
        import time

        from src.brain_brr.config.schemas import ModelConfig
        from src.brain_brr.models.detector import SeizureDetector

        # Create TCN model
        tcn_config = ModelConfig(architecture="tcn")
        tcn_model = SeizureDetector.from_config(tcn_config).cuda()

        x = torch.randn(4, 19, 15360).cuda()

        # Warmup
        for _ in range(3):
            _ = tcn_model(x)

        # Time TCN
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            _ = tcn_model(x)
        torch.cuda.synchronize()
        tcn_time = time.perf_counter() - t0

        print(f"TCN: {tcn_time:.3f}s for 10 batches")
        # TCN should be fast (< 0.5s for 10 batches on GPU)
        assert tcn_time < 0.5

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for memory profiling")
    def test_tcn_memory_efficient(self):
        """TCN should be memory efficient."""
        from src.brain_brr.config.schemas import ModelConfig
        from src.brain_brr.models.detector import SeizureDetector

        # Create TCN model
        tcn_config = ModelConfig(architecture="tcn")

        # Measure TCN memory
        torch.cuda.reset_peak_memory_stats()
        tcn_model = SeizureDetector.from_config(tcn_config).cuda()
        x = torch.randn(4, 19, 15360).cuda()
        _ = tcn_model(x)
        tcn_memory = torch.cuda.max_memory_allocated() / 1e9

        print(f"TCN: {tcn_memory:.2f}GB")
        # TCN should use less than 2GB for batch size 4
        assert tcn_memory < 2.0


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
            text=True,
        )

        assert result.returncode == 1, f"Found U-Net/ResNet references:\n{result.stdout}"

    @pytest.mark.skip(reason="Run after full migration")
    def test_files_deleted(self):
        """Old U-Net/ResNet files should be deleted."""
        from pathlib import Path

        files_should_not_exist = [
            "src/brain_brr/models/unet.py",
            "src/brain_brr/models/rescnn.py",
            "tests/unit/models/test_unet.py",
            "tests/unit/models/test_rescnn.py",
        ]

        for filepath in files_should_not_exist:
            assert not Path(filepath).exists(), f"{filepath} still exists!"
