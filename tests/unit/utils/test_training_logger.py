"""Unit tests for specialized training logger.

Tests training session management, metric aggregation, NaN detection,
and performance optimizations.
"""

import logging
from unittest.mock import patch

import pytest
import torch

from src.brain_brr.utils.training_logger import (
    MetricBuffer,
    TrainingLogger,
    log_data_stats,
    log_model_info,
)


class TestMetricBuffer:
    """Test the MetricBuffer for zero-allocation aggregation."""

    def test_add_and_stats(self):
        """Test adding values and computing statistics."""
        buffer = MetricBuffer(capacity=10)

        # Add some values
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            buffer.add(v)

        stats = buffer.get_stats()
        assert stats["mean"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["last"] == 5.0
        assert stats["count"] == 5

    def test_capacity_limit(self):
        """Test that buffer respects capacity."""
        buffer = MetricBuffer(capacity=3)

        # Add more than capacity
        for i in range(10):
            buffer.add(float(i))

        # Should only keep last 3
        stats = buffer.get_stats()
        assert stats["count"] == 3
        assert stats["min"] == 7.0
        assert stats["max"] == 9.0

    def test_clear(self):
        """Test buffer clearing."""
        buffer = MetricBuffer()
        buffer.add(1.0)
        buffer.add(2.0)

        assert buffer.get_stats()["count"] == 2

        buffer.clear()
        assert buffer.get_stats() == {}

    def test_empty_stats(self):
        """Test stats on empty buffer."""
        buffer = MetricBuffer()
        assert buffer.get_stats() == {}


class TestTrainingLogger:
    """Test the TrainingLogger for ML training loops."""

    @pytest.fixture
    def logger(self):
        """Create a training logger for testing."""
        logger = logging.getLogger("test.training")
        return TrainingLogger(
            name="test",
            logger=logger,
            use_rich=False,  # Disable rich for tests
            log_every_n_steps=2,
            aggregate_window=5,
        )

    def test_training_lifecycle(self, logger, test_batch_size):
        """Test complete training lifecycle."""
        # Start training
        logger.start_training(total_epochs=2, total_steps=10)
        assert logger.total_epochs == 2

        # Logger tests don't need large batches
        batch_size = min(test_batch_size, 32)

        # Run epochs
        for epoch in range(1, 3):
            logger.start_epoch(epoch, total_batches=5)
            assert logger.epoch == epoch

            for step in range(5):
                logger.log_batch(
                    step=step,
                    loss=1.0 / (step + 1),
                    metrics={"accuracy": 0.5 + step * 0.1},
                    lr=1e-3,
                    batch_size=batch_size,
                )

            logger.end_epoch(metrics={"val_loss": 0.5})

        # End training
        logger.end_training(final_metrics={"final_acc": 0.95})

    def test_log_batch_gating(self, logger, caplog):
        """Test that batch logging is gated by step frequency."""
        caplog.set_level(logging.INFO)

        # Log every 2 steps
        for step in range(10):
            logger.log_batch(step=step, loss=1.0)

        # Should log steps: 0, 2, 4, 6, 8
        logged_steps = [r for r in caplog.records if "Step" in r.message]
        assert len(logged_steps) == 5

    def test_metric_aggregation(self, logger):
        """Test metric buffer aggregation."""
        # Add metrics
        for i in range(10):
            logger.log_batch(
                step=i,
                loss=float(i),
                metrics={"acc": float(i * 2)},
            )

        # Check buffers were created
        assert "loss" in logger.metric_buffers
        assert "acc" in logger.metric_buffers

        # Check aggregation (last 5 values due to window size)
        loss_stats = logger.metric_buffers["loss"].get_stats()
        assert loss_stats["min"] == 5.0
        assert loss_stats["max"] == 9.0

    def test_nan_detection(self, logger, caplog):
        """Test NaN detection logging."""
        caplog.set_level(logging.WARNING)

        # Create tensor with NaN
        tensor = torch.tensor([1.0, float("nan"), 3.0])

        logger.log_nan_detection(
            location="test",
            tensor_name="test_tensor",
            tensor=tensor,
        )

        # Check NaN was logged
        assert logger.nan_count == 1
        assert "test" in logger.nan_locations
        assert any("NaN detected" in r.message for r in caplog.records)

    def test_nan_detection_context(self, logger):
        """Test NaN detection with context information."""
        tensor = torch.tensor([[1.0, 2.0], [float("nan"), float("inf")]])

        logger.log_nan_detection(
            location="gradients",
            tensor_name="layer.weight",
            tensor=tensor,
            extra_context={"epoch": 5, "step": 100},
        )

        assert logger.nan_count == 1
        assert logger.last_nan_step == 0  # Global step is 0

    def test_gradient_stats(self, logger, caplog):
        """Test gradient statistics logging."""
        caplog.set_level(logging.INFO)

        # Create a simple model
        model = torch.nn.Linear(10, 5)

        # Simulate gradients
        for param in model.parameters():
            param.grad = torch.randn_like(param) * 0.01

        stats = logger.log_gradient_stats(model)

        assert stats is not None
        assert "grad_norm" in stats
        assert "grad_min" in stats
        assert "grad_max" in stats
        assert "num_params_with_grad" in stats
        assert stats["num_params_with_grad"] == 2  # weight and bias

    def test_gradient_stats_with_nan(self, logger, caplog):
        """Test gradient stats when NaN is present."""
        caplog.set_level(logging.WARNING)

        model = torch.nn.Linear(5, 3)

        # Add NaN to gradients
        for param in model.parameters():
            param.grad = torch.full_like(param, float("nan"))

        stats = logger.log_gradient_stats(model)

        assert stats["has_nan_grads"] is True
        assert any("Gradient anomaly" in r.message for r in caplog.records)

    def test_context_manager(self, logger):
        """Test context manager usage."""
        with logger as ctx_logger:
            assert ctx_logger is logger
            ctx_logger.start_training(total_epochs=1)

        # Progress should be stopped on exit
        if logger.progress:
            assert not logger.progress.live.is_started


class TestUtilityFunctions:
    """Test standalone utility functions."""

    def test_log_model_info(self, caplog):
        """Test model info logging."""
        caplog.set_level(logging.INFO)

        model = torch.nn.Sequential(
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10),
        )

        info = log_model_info(model)

        assert info["total_parameters"] == 100 * 50 + 50 + 50 * 10 + 10
        assert info["trainable_parameters"] == info["total_parameters"]
        assert info["model_size_mb"] > 0
        assert info["num_layers"] > 0

        # Check that info was logged
        assert any("Model initialized" in r.message for r in caplog.records)

    def test_log_data_stats(self, caplog):
        """Test dataset statistics logging."""
        caplog.set_level(logging.INFO)

        log_data_stats(
            dataset_name="test_dataset",
            num_samples=1000,
            num_positive=100,
            extra_stats={"num_files": 50},
        )

        # Check that stats were logged
        records = [r for r in caplog.records if "Dataset test_dataset" in r.message]
        assert len(records) == 1

        # Check message content
        message = records[0].message
        assert "1,000 samples" in message
        assert "100 positive" in message
        assert "10.0%" in message  # Positive percentage
        assert "imbalance 1:10.0" in message

    def test_log_data_stats_edge_cases(self):
        """Test data stats with edge cases."""
        # Zero positive samples
        log_data_stats(
            dataset_name="empty",
            num_samples=100,
            num_positive=0,
        )

        # All positive samples
        log_data_stats(
            dataset_name="all_positive",
            num_samples=100,
            num_positive=100,
        )


class TestPerformanceOptimizations:
    """Test performance-related features."""

    def test_metric_buffer_reuse(self):
        """Test that metric buffers are reused across epochs."""
        logger = TrainingLogger(name="perf", use_rich=False)

        # First epoch
        logger.start_epoch(1)
        logger.log_batch(0, loss=1.0)
        buffer1 = logger.metric_buffers["loss"]

        # Second epoch - should reuse buffer
        logger.start_epoch(2)
        logger.log_batch(1, loss=2.0)
        buffer2 = logger.metric_buffers["loss"]

        # Should be the same buffer instance (cleared and reused)
        assert buffer1 is buffer2

    @patch("time.time")
    def test_throughput_calculation(self, mock_time, caplog):
        """Test throughput calculation in batch logging."""
        caplog.set_level(logging.INFO)

        logger = TrainingLogger(name="throughput", use_rich=False, log_every_n_steps=1)

        # Mock time for consistent throughput
        mock_time.side_effect = [
            0.0,
            1.0,
            1.0,
            2.0,
        ]  # start, end of batch 1, start batch 2, end batch 2

        logger.batch_start_time = 0.0
        logger.log_batch(0, loss=1.0, batch_size=32)

        # Check throughput was calculated
        records = [r for r in caplog.records if "Speed:" in r.message]
        assert len(records) == 1
        assert "32.0 samples/s" in records[0].message


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_epoch(self):
        """Test handling of empty epoch."""
        logger = TrainingLogger(name="empty", use_rich=False)

        logger.start_training(total_epochs=1)
        logger.start_epoch(1)
        # No batches logged
        logger.end_epoch()
        logger.end_training()

        # Should not crash

    def test_nan_in_metrics(self):
        """Test handling of NaN in metrics."""
        logger = TrainingLogger(name="nan", use_rich=False)

        # Should handle NaN in metrics gracefully
        logger.log_batch(
            step=0,
            loss=float("nan"),
            metrics={"acc": float("inf")},
        )

        # Should not crash

    def test_very_large_tensor_stats(self):
        """Test NaN detection with very large tensors."""
        logger = TrainingLogger(name="large", use_rich=False)

        # Create large tensor (should skip detailed stats)
        large_tensor = torch.randn(10000, 10000)
        large_tensor[0, 0] = float("nan")

        logger.log_nan_detection(
            location="large",
            tensor_name="huge_tensor",
            tensor=large_tensor,
        )

        # Should handle without computing all stats
        assert logger.nan_count == 1
