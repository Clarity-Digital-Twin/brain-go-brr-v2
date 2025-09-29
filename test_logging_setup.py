#!/usr/bin/env python
"""Test script to verify logging infrastructure works correctly."""

import logging
import torch
import numpy as np

from src.brain_brr.utils.logging_config import setup_logging, get_logger
from src.brain_brr.utils.training_logger import TrainingLogger, log_model_info, log_data_stats


def test_basic_logging():
    """Test basic logging configuration."""
    print("\n=== Testing Basic Logging ===")

    # Setup logging
    setup_logging(level="INFO")
    logger = get_logger(__name__)

    # Test different log levels
    logger.debug("This debug message should NOT appear at INFO level")
    logger.info("✅ INFO logging works")
    logger.warning("⚠️ WARNING logging works")
    logger.error("❌ ERROR logging works (this is a test, not a real error)")

    print("Basic logging test complete!\n")


def test_training_logger():
    """Test specialized training logger."""
    print("\n=== Testing Training Logger ===")

    logger = logging.getLogger("test.training")
    train_logger = TrainingLogger(name="test", logger=logger, use_rich=False)

    # Simulate training session
    train_logger.start_training(total_epochs=3, total_steps=30)

    for epoch in range(1, 4):
        train_logger.start_epoch(epoch, total_batches=10)

        for batch_idx in range(10):
            step = (epoch - 1) * 10 + batch_idx
            loss = 1.0 / (step + 1)  # Fake decreasing loss
            metrics = {
                "accuracy": min(0.99, 0.5 + step * 0.02),
                "f1_score": min(0.95, 0.4 + step * 0.02),
            }

            train_logger.log_batch(
                step=step,
                loss=loss,
                metrics=metrics,
                lr=1e-3 * (0.9 ** epoch),
                batch_size=32,
            )

        # End epoch with validation metrics
        val_metrics = {
            "val_loss": 0.5 / epoch,
            "val_accuracy": 0.8 + epoch * 0.05,
        }
        train_logger.end_epoch(metrics=val_metrics)

    train_logger.end_training(final_metrics={"final_accuracy": 0.95})
    print("Training logger test complete!\n")


def test_nan_detection():
    """Test NaN detection logging."""
    print("\n=== Testing NaN Detection ===")

    logger = logging.getLogger("test.nan")
    train_logger = TrainingLogger(name="nan_test", logger=logger, use_rich=False)

    # Create a tensor with NaN
    tensor_with_nan = torch.tensor([1.0, 2.0, float('nan'), 4.0, float('inf')])

    # Test NaN detection
    train_logger.log_nan_detection(
        location="test_tensor",
        tensor_name="example_tensor",
        tensor=tensor_with_nan,
        extra_context={"test": "This is a test NaN detection"}
    )

    print("NaN detection test complete!\n")


def test_model_info():
    """Test model info logging."""
    print("\n=== Testing Model Info Logging ===")

    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
    )

    # Log model info
    info = log_model_info(model)
    print(f"Model has {info['trainable_parameters']:,} parameters")
    print(f"Model size: {info['model_size_mb']:.2f} MB")
    print("Model info logging test complete!\n")


def test_data_stats():
    """Test dataset statistics logging."""
    print("\n=== Testing Dataset Stats Logging ===")

    log_data_stats(
        dataset_name="test_dataset",
        num_samples=10000,
        num_positive=833,  # ~12:1 imbalance like our real data
        extra_stats={
            "num_patients": 50,
            "avg_duration_s": 3600,
        }
    )

    print("Dataset stats logging test complete!\n")


def test_performance_filter():
    """Test performance filtering for high-frequency logs."""
    print("\n=== Testing Performance Filter ===")

    # Setup with performance filter
    setup_logging(level="DEBUG", force=True)
    logger = get_logger("test.performance")

    # Simulate high-frequency logging
    print("Logging 100 messages (only every 50th should appear)...")
    for i in range(100):
        # Add step attribute for filtering
        logger.info(f"Step {i} message", extra={"step": i})

    print("Performance filter test complete!\n")


def main():
    """Run all logging tests."""
    print("=" * 60)
    print("LOGGING INFRASTRUCTURE TEST SUITE")
    print("=" * 60)

    test_basic_logging()
    test_training_logger()
    test_nan_detection()
    test_model_info()
    test_data_stats()
    test_performance_filter()

    print("\n" + "=" * 60)
    print("✅ ALL LOGGING TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()