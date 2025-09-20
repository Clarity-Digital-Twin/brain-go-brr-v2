"""Training pipeline smoke tests."""

import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.brain_brr.config.schemas import Config, EarlyStoppingConfig, TrainingConfig
from src.brain_brr.models import SeizureDetector
from src.brain_brr.train import (
    create_optimizer,
    create_scheduler,
    load_checkpoint,
    save_checkpoint,
    train_epoch,
    validate_epoch,
)


class TestTrainingSmoke:
    """Smoke tests for training pipeline."""

    @pytest.fixture
    def model(self) -> SeizureDetector:
        """Create small model for testing."""
        return SeizureDetector(
            base_channels=32,  # Smaller for speed
            encoder_depth=2,
            mamba_layers=1,
            rescnn_blocks=1,
        )

    @pytest.fixture
    def synthetic_data(self) -> tuple[DataLoader, DataLoader]:
        """Create synthetic balanced dataset."""
        # Create balanced dataset (10 windows)
        windows = torch.randn(10, 19, 15360)
        labels = torch.zeros(10, 15360)

        # Make 50% positive
        labels[::2, 5000:10000] = 1

        # Split into train/val
        train_dataset = TensorDataset(windows[:8], labels[:8])
        val_dataset = TensorDataset(windows[8:], labels[8:])

        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

        return train_loader, val_loader

    @pytest.mark.serial
    def test_single_epoch(
        self, model: SeizureDetector, synthetic_data: tuple[DataLoader, DataLoader]
    ) -> None:
        """Test single training epoch."""
        train_loader, _ = synthetic_data

        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Train one epoch
        train_loss = train_epoch(model, train_loader, optimizer, device="cpu", use_amp=False)

        assert train_loss > 0
        assert train_loss < 10  # Reasonable range
        assert not torch.isnan(torch.tensor(train_loss))

    @pytest.mark.serial
    def test_validation(
        self, model: SeizureDetector, synthetic_data: tuple[DataLoader, DataLoader]
    ) -> None:
        """Test validation epoch."""
        _, val_loader = synthetic_data

        # Validate
        from src.brain_brr.config.schemas import HysteresisConfig, PostprocessingConfig

        post_cfg = PostprocessingConfig(
            hysteresis=HysteresisConfig(tau_on=0.86, tau_off=0.78),
            morphology={"kernel_size": 5, "operation": "closing"},
            min_duration=1.0,
        )

        metrics = validate_epoch(model, val_loader, post_cfg, device="cpu")

        # Check metrics structure
        assert "val_loss" in metrics
        assert "taes" in metrics
        assert "auroc" in metrics
        assert "sensitivity_at_10fa" in metrics

        # Check ranges
        assert 0 <= metrics["taes"] <= 1
        assert 0 <= metrics["auroc"] <= 1
        assert metrics["val_loss"] > 0

    def test_optimizer_creation(self, model: SeizureDetector) -> None:
        """Test optimizer creation from config."""
        config = TrainingConfig(
            optimizer="adamw",
            learning_rate=1e-3,
            weight_decay=1e-4,
        )

        optimizer = create_optimizer(model, config)

        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]["lr"] == 1e-3
        assert optimizer.param_groups[0]["weight_decay"] == 1e-4

    def test_scheduler_creation(self, model: SeizureDetector) -> None:
        """Test scheduler creation."""
        from src.brain_brr.config.schemas import SchedulerConfig

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler_cfg = SchedulerConfig(
            type="cosine",
            warmup_ratio=0.1,
        )

        scheduler = create_scheduler(optimizer, scheduler_cfg, total_steps=100)

        # Check warmup works
        initial_lr = optimizer.param_groups[0]["lr"]
        # Need to call optimizer.step() before scheduler.step() in PyTorch 1.1+
        optimizer.step()
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] < initial_lr  # Starts low

    def test_checkpoint_save_load(self, model: SeizureDetector) -> None:
        """Test checkpoint saving and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"

            # Save checkpoint
            optimizer = torch.optim.AdamW(model.parameters())
            save_checkpoint(
                model, optimizer, epoch=5, best_metric=0.95, checkpoint_path=checkpoint_path
            )

            assert checkpoint_path.exists()

            # Load checkpoint
            new_model = SeizureDetector(
                base_channels=32,
                encoder_depth=2,
                mamba_layers=1,
                rescnn_blocks=1,
            )
            new_optimizer = torch.optim.AdamW(new_model.parameters())

            epoch, best_metric = load_checkpoint(checkpoint_path, new_model, new_optimizer)

            assert epoch == 5
            assert best_metric == 0.95

    def test_gradient_clipping(
        self, model: SeizureDetector, synthetic_data: tuple[DataLoader, DataLoader]
    ) -> None:
        """Test gradient clipping during training."""
        train_loader, _ = synthetic_data

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Train with gradient clipping
        _ = train_epoch(
            model, train_loader, optimizer, device="cpu", use_amp=False, gradient_clip=1.0
        )

        # Check gradients were clipped (no explosion)
        for param in model.parameters():
            if param.grad is not None:
                assert torch.norm(param.grad).item() <= 1.1  # Small tolerance

    def test_mixed_precision(
        self, model: SeizureDetector, synthetic_data: tuple[DataLoader, DataLoader]
    ) -> None:
        """Test mixed precision training (CPU fallback)."""
        train_loader, _ = synthetic_data

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Should work even on CPU (no-op)
        train_loss = train_epoch(model, train_loader, optimizer, device="cpu", use_amp=True)

        assert train_loss > 0
        assert not torch.isnan(torch.tensor(train_loss))

    def test_early_stopping(self) -> None:
        """Test early stopping logic."""
        from src.brain_brr.train import EarlyStopping

        config = EarlyStoppingConfig(
            patience=3,
            metric="sensitivity_at_10fa",
            mode="max",
        )

        early_stopping = EarlyStopping(config)

        # Simulate improving metrics
        assert not early_stopping(0.5)
        assert not early_stopping(0.6)
        assert not early_stopping(0.7)

        # Simulate plateauing
        assert not early_stopping(0.7)
        assert not early_stopping(0.7)
        assert not early_stopping(0.7)
        assert early_stopping(0.7)  # Should stop after patience

    def test_balanced_sampling(self) -> None:
        """Test balanced sampler creation."""
        from unittest.mock import MagicMock
        from src.brain_brr.train import create_balanced_sampler

        # Create mock dataset with imbalanced labels (90% negative, 10% positive)
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)

        # Mock dataset items - first 10 have seizures, rest don't
        def mock_getitem(idx):
            labels = torch.zeros(15360)
            if idx < 10:
                labels[5000:10000] = 1  # Seizure
            return torch.zeros(19, 15360), labels

        mock_dataset.__getitem__ = MagicMock(side_effect=mock_getitem)

        sampler = create_balanced_sampler(mock_dataset, sample_size=50)

        # Should return a sampler (not None since we have seizures)
        assert sampler is not None

        # Sample indices should oversample minorities
        indices = list(sampler)
        assert len(indices) == 100

        # Note: Can't easily test exact distribution due to probabilistic assignment
        # but sampler should be created successfully

    def test_full_training_loop(
        self, model: SeizureDetector, synthetic_data: tuple[DataLoader, DataLoader]
    ) -> None:
        """Test complete training loop for 2 epochs."""
        from src.brain_brr.train import train

        train_loader, val_loader = synthetic_data

        # Mini config
        config = Config()
        config.training.epochs = 2
        config.training.learning_rate = 1e-3

        with tempfile.TemporaryDirectory() as tmpdir:
            config.experiment.output_dir = tmpdir

            # Train for 2 epochs
            best_metrics = train(model, train_loader, val_loader, config)

            # Check that we got metrics
            assert "best_epoch" in best_metrics
            assert "best_taes" in best_metrics
            assert best_metrics["best_epoch"] <= 2

            # Check checkpoint exists
            checkpoint_path = Path(tmpdir) / "checkpoints" / "best.pt"
            assert checkpoint_path.exists()
