"""Performance-optimized logging patterns for inner loops.

PHASE 2 REFERENCE: Example patterns for migrating high-frequency print statements.
"""

import logging
import os
from typing import Any

# Performance constants from environment
LOG_EVERY_N_STEPS = int(os.getenv("BGB_LOG_EVERY_N_STEPS", "50"))


def log_batch_metrics(
    logger: logging.Logger,
    step: int,
    loss: float,
    accuracy: float | None = None,
    **extra_metrics: Any,
) -> None:
    """Example of efficient inner-loop logging with gating.

    Pattern for Phase 2 migration of train/loop.py batch logging.
    """
    # CRITICAL: Gate before doing ANY string formatting
    if not logger.isEnabledFor(logging.DEBUG):
        return

    # Secondary gate for step frequency
    if step % LOG_EVERY_N_STEPS != 0 and step != 0:
        return

    # Use parameterized logging to avoid formatting cost when disabled
    # DO NOT use f-strings here - they always evaluate
    logger.debug(
        "Step %d | Loss: %.4f | Acc: %.4f | %s",
        step,
        loss,
        accuracy or 0.0,
        " | ".join(f"{k}={v:.4f}" for k, v in extra_metrics.items()),
        extra={"step": step, "metrics": {"loss": loss, "accuracy": accuracy, **extra_metrics}},
    )


def log_epoch_progress(
    logger: logging.Logger,
    epoch: int,
    total_epochs: int,
    phase: str,
    **metrics: Any,
) -> None:
    """Example of epoch-level logging (less frequent).

    This replaces print statements at epoch boundaries.
    """
    # Epoch logs are INFO level - always shown
    logger.info(
        "[%s] Epoch %d/%d completed | %s",
        phase.upper(),
        epoch,
        total_epochs,
        " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
        extra={"epoch": epoch, "phase": phase, "metrics": metrics},
    )


def log_nan_detection_efficient(
    logger: logging.Logger,
    location: str,
    step: int,
    should_debug: bool = False,
) -> None:
    """Example of conditional NaN logging.

    Only logs details when BGB_NAN_DEBUG is set.
    """
    if should_debug or os.getenv("BGB_NAN_DEBUG") == "1":
        # Detailed debug info
        logger.debug(
            "NaN detected at %s (step %d) - entering debug mode",
            location,
            step,
            extra={"nan_location": location, "step": step},
        )
    else:
        # Brief warning
        logger.warning("NaN at %s", location)


def log_data_loading_progress(
    logger: logging.Logger,
    files_processed: int,
    total_files: int,
    current_file: str,
) -> None:
    """Example for data pipeline logging.

    Pattern for Phase 3 migration of data loading prints.
    """
    # Gate by percentage completion
    percent = (files_processed * 100) // total_files
    if percent % 10 == 0 and files_processed > 0:  # Log every 10%
        logger.info(
            "Processing: %d/%d files (%d%%) - Current: %s",
            files_processed,
            total_files,
            percent,
            current_file,
        )


# Example usage patterns for Phase 2
"""
MIGRATION EXAMPLES:

Before (train/loop.py):
    print(f"[TRAIN] Batch {i}: loss={loss:.4f}", flush=True)

After:
    from src.brain_brr.utils.logging_patterns import log_batch_metrics
    log_batch_metrics(logger, step=i, loss=loss)

Before (with multiple metrics):
    print(f"loss={loss:.4f} acc={acc:.4f} lr={lr:.6f}", flush=True)

After:
    if logger.isEnabledFor(logging.DEBUG) and step % LOG_EVERY_N_STEPS == 0:
        logger.debug("loss=%.4f acc=%.4f lr=%.6f", loss, acc, lr)

Before (NaN detection):
    if math.isnan(loss):
        print(f"WARNING: NaN loss at step {step}!", flush=True)

After:
    if math.isnan(loss):
        log_nan_detection_efficient(logger, "loss_computation", step)
"""


__all__ = [
    "LOG_EVERY_N_STEPS",
    "log_batch_metrics",
    "log_data_loading_progress",
    "log_epoch_progress",
    "log_nan_detection_efficient",
]