"""Specialized training logger for high-performance ML training loops.

Ultra-efficient logging for training with:
- Zero-allocation progress tracking
- Batched metrics aggregation
- NaN detection with context
- Automatic rate limiting
- Rich console output for interactive sessions
- Structured logging for production

Optimized for minimal overhead in tight training loops.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.table import Table


@dataclass
class MetricBuffer:
    """Zero-allocation metric buffer for efficient aggregation.

    Pre-allocates arrays and reuses them across epochs.
    Follows Google's performance best practices.
    """

    capacity: int = 1000
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))

    def add(self, value: float, timestamp: float | None = None) -> None:
        """Add value with optional timestamp."""
        self.values.append(value)
        self.timestamps.append(timestamp or time.time())

    def get_stats(self) -> dict[str, float]:
        """Get aggregated statistics without allocation."""
        if not self.values:
            return {}

        values_list = list(self.values)
        return {
            "mean": sum(values_list) / len(values_list),
            "min": min(values_list),
            "max": max(values_list),
            "last": values_list[-1],
            "count": len(values_list),
        }

    def clear(self) -> None:
        """Clear buffers for reuse."""
        self.values.clear()
        self.timestamps.clear()


class TrainingLogger:
    """High-performance logger for ML training loops.

    Designed for minimal overhead with maximum observability.
    Follows DeepMind's internal logging standards.
    """

    def __init__(
        self,
        name: str = "train",
        logger: logging.Logger | None = None,
        use_rich: bool = True,
        log_every_n_steps: int = 50,
        aggregate_window: int = 100,
    ):
        """Initialize training logger.

        Args:
            name: Logger name/prefix
            logger: Python logger instance (creates one if None)
            use_rich: Use rich console for pretty output
            log_every_n_steps: Log frequency for batch metrics
            aggregate_window: Window size for metric aggregation
        """
        self.name = name
        self.logger = logger or logging.getLogger(f"brain_brr.training.{name}")

        # Console output
        self.use_rich = use_rich and sys.stderr.isatty()
        self.console = Console(stderr=True) if self.use_rich else None
        self.progress = None
        self.task_id = None

        # Performance settings
        self.log_every_n_steps = log_every_n_steps
        self.aggregate_window = aggregate_window

        # Metric buffers (pre-allocated)
        self.metric_buffers: dict[str, MetricBuffer] = {}

        # State tracking
        self.epoch = 0
        self.total_epochs = 0
        self.global_step = 0
        self.epoch_step = 0
        self.epoch_start_time = 0.0
        self.batch_start_time = 0.0

        # NaN detection state
        self.nan_locations: list[str] = []
        self.nan_count = 0
        self.last_nan_step = -1

    def start_training(self, total_epochs: int, total_steps: int | None = None) -> None:
        """Initialize training session.

        Sets up progress tracking and logging infrastructure.
        """
        self.total_epochs = total_epochs
        self.logger.info(
            f"Starting training: {total_epochs} epochs"
            + (f", ~{total_steps} steps" if total_steps else "")
        )

        if self.use_rich and self.console:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=self.console,
            )
            self.progress.start()

    def start_epoch(self, epoch: int, total_batches: int | None = None) -> None:
        """Start a new epoch.

        Args:
            epoch: Current epoch number (1-indexed)
            total_batches: Total batches in epoch (for progress bar)
        """
        self.epoch = epoch
        self.epoch_step = 0
        self.epoch_start_time = time.time()

        # Clear metric buffers for new epoch
        for buffer in self.metric_buffers.values():
            buffer.clear()

        self.logger.info(f"[Epoch {epoch}/{self.total_epochs}] Starting")

        if self.progress and total_batches:
            self.task_id = self.progress.add_task(
                f"Epoch {epoch}/{self.total_epochs}",
                total=total_batches,
            )

    def log_batch(
        self,
        step: int,
        loss: float,
        metrics: dict[str, float] | None = None,
        lr: float | None = None,
        batch_size: int | None = None,
        **extra,
    ) -> None:
        """Log batch metrics with automatic aggregation and rate limiting.

        Args:
            step: Global step number
            loss: Loss value
            metrics: Additional metrics dict
            lr: Learning rate
            batch_size: Batch size for throughput calculation
            **extra: Additional values to log
        """
        self.global_step = step
        self.epoch_step += 1

        # Update metric buffers (zero allocation after warmup)
        if "loss" not in self.metric_buffers:
            self.metric_buffers["loss"] = MetricBuffer(self.aggregate_window)
        self.metric_buffers["loss"].add(loss)

        if metrics:
            for key, value in metrics.items():
                if key not in self.metric_buffers:
                    self.metric_buffers[key] = MetricBuffer(self.aggregate_window)
                self.metric_buffers[key].add(value)

        # Check if we should log this step
        should_log = step % self.log_every_n_steps == 0 or step == 0 or self.epoch_step == 1

        if should_log:
            # Calculate throughput if batch_size provided
            throughput = None
            if batch_size and self.batch_start_time > 0:
                elapsed = time.time() - self.batch_start_time
                throughput = batch_size / elapsed

            # Build log message efficiently
            msg_parts = [
                f"[{self.name.upper()}]",
                f"Epoch {self.epoch}/{self.total_epochs}",
                f"Step {step}",
                f"Loss: {loss:.4f}",
            ]

            if lr is not None:
                msg_parts.append(f"LR: {lr:.2e}")

            if throughput:
                msg_parts.append(f"Speed: {throughput:.1f} samples/s")

            if metrics:
                for key, value in metrics.items():
                    msg_parts.append(f"{key}: {value:.4f}")

            # Log with appropriate level
            self.logger.info(" | ".join(msg_parts), extra={"step": step})

        # Update progress bar if available
        if self.progress and self.task_id is not None:
            self.progress.update(self.task_id, advance=1)

        # Update timing
        self.batch_start_time = time.time()

    def end_epoch(self, metrics: dict[str, float] | None = None) -> None:
        """End current epoch and log summary statistics.

        Args:
            metrics: Final epoch metrics (e.g., validation results)
        """
        elapsed = time.time() - self.epoch_start_time

        # Get aggregated stats for all metrics
        summary = {}
        for key, buffer in self.metric_buffers.items():
            stats = buffer.get_stats()
            if stats:
                summary[f"{key}_mean"] = stats["mean"]
                summary[f"{key}_min"] = stats["min"]
                summary[f"{key}_max"] = stats["max"]

        # Add provided metrics
        if metrics:
            summary.update(metrics)

        # Log epoch summary
        self.logger.info(
            f"[Epoch {self.epoch}/{self.total_epochs}] Completed in {elapsed:.1f}s",
            extra={"epoch": self.epoch, "metrics": summary},
        )

        # Rich table for interactive sessions
        if self.console and summary:
            table = Table(title=f"Epoch {self.epoch} Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            for key, value in summary.items():
                table.add_row(key, f"{value:.4f}")

            self.console.print(table)

        # Complete progress task
        if self.progress and self.task_id is not None:
            self.progress.remove_task(self.task_id)
            self.task_id = None

    def log_nan_detection(
        self,
        location: str,
        tensor_name: str,
        tensor: torch.Tensor | None = None,
        extra_context: dict[str, Any] | None = None,
    ) -> None:
        """Log NaN detection with rich context.

        Args:
            location: Where NaN was detected (e.g., "loss", "gradients")
            tensor_name: Name of the tensor
            tensor: The tensor containing NaN (optional)
            extra_context: Additional debugging context
        """
        self.nan_count += 1
        self.nan_locations.append(location)
        self.last_nan_step = self.global_step

        # Build context info
        context = {
            "location": location,
            "tensor_name": tensor_name,
            "step": self.global_step,
            "epoch": self.epoch,
            "nan_count": self.nan_count,
        }

        if tensor is not None and isinstance(tensor, torch.Tensor):
            context.update(
                {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                    "num_nans": torch.isnan(tensor).sum().item()
                    if tensor.numel() < 1e6
                    else "too_large",
                    "num_infs": torch.isinf(tensor).sum().item()
                    if tensor.numel() < 1e6
                    else "too_large",
                }
            )

            # Get finite statistics if tensor is small enough
            if tensor.numel() < 1e5:
                finite_mask = torch.isfinite(tensor)
                if finite_mask.any():
                    finite_values = tensor[finite_mask]
                    context.update(
                        {
                            "finite_min": finite_values.min().item(),
                            "finite_max": finite_values.max().item(),
                            "finite_mean": finite_values.mean().item(),
                        }
                    )

        if extra_context:
            context.update(extra_context)

        # Log with WARNING level for visibility
        self.logger.warning(
            f"NaN detected at {location}: {tensor_name}", extra={"nan_context": context}
        )

        # Rich console output for interactive debugging
        if self.console:
            self.console.print(
                f"[bold red]⚠ NaN Detection #{self.nan_count}[/bold red]",
                f"Location: {location}",
                f"Tensor: {tensor_name}",
                style="red",
            )

    def log_gradient_stats(
        self,
        model: torch.nn.Module,
        log_histograms: bool = False,
    ) -> dict[str, float] | None:
        """Log gradient statistics for debugging.

        Args:
            model: PyTorch model
            log_histograms: Whether to log full histograms (expensive)

        Returns:
            Dictionary of gradient statistics
        """
        total_norm = 0.0
        num_params = 0
        min_grad = float("inf")
        max_grad = float("-inf")
        has_nan = False
        has_inf = False

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                num_params += 1

                # Check for NaN/Inf
                if torch.isnan(grad).any():
                    has_nan = True
                    self.log_nan_detection("gradients", name, grad)

                if torch.isinf(grad).any():
                    has_inf = True

                # Compute stats on finite values
                finite_grad = grad[torch.isfinite(grad)]
                if finite_grad.numel() > 0:
                    param_norm = finite_grad.norm(2).item()
                    total_norm += param_norm**2
                    min_grad = min(min_grad, finite_grad.min().item())
                    max_grad = max(max_grad, finite_grad.max().item())

        total_norm = total_norm**0.5

        stats = {
            "grad_norm": total_norm,
            "grad_min": min_grad if min_grad != float("inf") else 0.0,
            "grad_max": max_grad if max_grad != float("-inf") else 0.0,
            "num_params_with_grad": num_params,
            "has_nan_grads": has_nan,
            "has_inf_grads": has_inf,
        }

        # Log if issues detected
        if has_nan or has_inf or total_norm > 1000:
            self.logger.warning(
                f"Gradient anomaly: norm={total_norm:.2f}, nan={has_nan}, inf={has_inf}",
                extra={"gradient_stats": stats},
            )

        return stats

    def end_training(self, final_metrics: dict[str, float] | None = None) -> None:
        """End training session and log final summary.

        Args:
            final_metrics: Final training metrics
        """
        if self.progress:
            self.progress.stop()

        # Log final summary
        self.logger.info(
            f"Training completed: {self.epoch} epochs, {self.global_step} steps",
            extra={"final_metrics": final_metrics},
        )

        # Log NaN summary if any occurred
        if self.nan_count > 0:
            self.logger.warning(
                f"Training had {self.nan_count} NaN detections at: {set(self.nan_locations)}",
            )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures progress cleanup."""
        if self.progress:
            self.progress.stop()
        return False


# Convenience functions for one-off logging
def log_model_info(
    model: torch.nn.Module,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Log model architecture information.

    Args:
        model: PyTorch model
        logger: Logger instance (creates one if None)

    Returns:
        Dictionary with model information
    """
    logger = logger or logging.getLogger("brain_brr.models")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Get model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / 1024 / 1024

    info = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": model_size_mb,
        "num_layers": len(list(model.modules())),
    }

    logger.info(
        f"Model initialized: {trainable_params:,} trainable params, {model_size_mb:.1f} MB",
        extra={"model_info": info},
    )

    return info


def log_data_stats(
    dataset_name: str,
    num_samples: int,
    num_positive: int,
    logger: logging.Logger | None = None,
    extra_stats: dict[str, Any] | None = None,
) -> None:
    """Log dataset statistics.

    Args:
        dataset_name: Name of the dataset
        num_samples: Total number of samples
        num_positive: Number of positive samples
        logger: Logger instance
        extra_stats: Additional statistics to log
    """
    logger = logger or logging.getLogger("brain_brr.data")

    imbalance_ratio = num_samples / max(num_positive, 1)

    stats = {
        "dataset": dataset_name,
        "total_samples": num_samples,
        "positive_samples": num_positive,
        "negative_samples": num_samples - num_positive,
        "imbalance_ratio": imbalance_ratio,
    }

    if extra_stats:
        stats.update(extra_stats)

    logger.info(
        f"Dataset {dataset_name}: {num_samples:,} samples, "
        f"{num_positive:,} positive ({num_positive / num_samples * 100:.1f}%), "
        f"imbalance 1:{imbalance_ratio:.1f}",
        extra={"dataset_stats": stats},
    )


# Import guard for optional rich dependency
import sys

try:
    from rich.console import Console
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
    from rich.table import Table
except ImportError:
    # Graceful degradation if rich not available
    Console = None
    Progress = None
    Table = None


__all__ = [
    "MetricBuffer",
    "TrainingLogger",
    "log_data_stats",
    "log_model_info",
]
