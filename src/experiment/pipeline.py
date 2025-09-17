"""Training and evaluation pipeline orchestration (Phase 3).

SOLID principles applied:
- Single Responsibility: Each function does one thing
- Open/Closed: Extensible via configs, not modification
- Liskov Substitution: Interfaces respect contracts
- Interface Segregation: Small, focused interfaces
- Dependency Inversion: Depend on abstractions (configs)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LRScheduler
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # type: ignore[import-untyped]

from src.experiment.evaluate import evaluate_predictions
from src.experiment.models import SeizureDetector
from src.experiment.schemas import (
    Config,
    EarlyStoppingConfig,
    PostprocessingConfig,
    SchedulerConfig,
    TrainingConfig,
)

# ============================================================================
# Reproducibility utilities (Single Responsibility)
# ============================================================================


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """Initialize worker seeds for DataLoader determinism."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================================
# Sampling utilities (Interface Segregation)
# ============================================================================


def create_balanced_sampler(labels: torch.Tensor) -> WeightedRandomSampler:
    """Create balanced sampler for imbalanced datasets.

    Args:
        labels: (N, T) binary labels tensor

    Returns:
        WeightedRandomSampler for balanced mini-batches
    """
    # Aggregate to window-level: has seizure or not
    window_has_seizure = (labels.max(dim=1).values > 0).float()
    pos_ratio = float(window_has_seizure.mean().item())

    # Avoid division by zero
    if pos_ratio < 1e-8:
        pos_ratio = 1e-8

    # Weight calculation: oversample minority class
    pos_weight = (1 - pos_ratio) / pos_ratio
    sample_weights = torch.where(window_has_seizure > 0, pos_weight, 1.0)

    return WeightedRandomSampler(
        weights=sample_weights.tolist(),  # Convert to list for type checking
        num_samples=len(sample_weights),
        replacement=True,
        generator=torch.Generator().manual_seed(42),  # Deterministic
    )


# ============================================================================
# Optimizer & Scheduler factories (Open/Closed Principle)
# ============================================================================


def create_optimizer(model: nn.Module, config: TrainingConfig) -> Optimizer:
    """Create optimizer from config.

    Factory pattern for optimizer creation.
    """
    if config.optimizer == "adamw":
        return AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")


def create_scheduler(
    optimizer: Optimizer,
    config: SchedulerConfig,
    total_steps: int,
) -> LRScheduler:
    """Create learning rate scheduler.

    Supports cosine with warmup.
    """
    warmup_steps = int(config.warmup_ratio * total_steps)

    if config.type == "cosine":
        # Use CosineAnnealingWarmRestarts for per-iteration scheduling
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=total_steps - warmup_steps,
            T_mult=1,
            eta_min=1e-7,
        )

        # Manual warmup wrapper
        class WarmupScheduler(LRScheduler):
            """Wrapper for warmup + base scheduler."""

            def __init__(self) -> None:
                self.warmup_steps = warmup_steps
                self.base_scheduler = scheduler
                self.current_step = 0
                super().__init__(optimizer, last_epoch=-1)

            def get_lr(self) -> list[float]:
                if self.current_step < self.warmup_steps:
                    # Linear warmup
                    warmup_factor = self.current_step / max(1, self.warmup_steps)
                    return [base_lr * warmup_factor for base_lr in self.base_lrs]
                return self.base_scheduler.get_last_lr()

            def step(self, epoch: int | None = None) -> None:
                self.current_step += 1
                if self.current_step > self.warmup_steps:
                    self.base_scheduler.step()
                else:
                    super().step()

        return WarmupScheduler()
    else:
        raise ValueError(f"Unknown scheduler: {config.type}")


# ============================================================================
# Training epoch (Single Responsibility)
# ============================================================================


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    device: str = "cpu",
    use_amp: bool = False,
    gradient_clip: float = 1.0,
    scheduler: LRScheduler | None = None,
) -> float:
    """Train for one epoch.

    Args:
        model: SeizureDetector model
        dataloader: Training DataLoader
        optimizer: Optimizer instance
        device: Device to train on
        use_amp: Use automatic mixed precision
        gradient_clip: Max gradient norm
        scheduler: Optional LR scheduler (per-iteration)

    Returns:
        Average training loss
    """
    model.train()
    device_obj = torch.device(device)
    scaler = GradScaler(enabled=use_amp and device != "cpu")

    # BCE loss (model already applies sigmoid)
    criterion = nn.BCELoss(reduction="none")

    # Calculate class weights from first batch (approximation)
    first_batch = next(iter(dataloader))
    _, first_labels = first_batch
    pos_ratio = float((first_labels > 0).float().mean())
    pos_weight = (1 - pos_ratio) / max(pos_ratio, 1e-8) if pos_ratio > 0 else 1.0

    total_loss = 0.0
    num_batches = 0

    progress = tqdm(dataloader, desc="Training", leave=False)
    for windows, labels in progress:
        windows = windows.to(device_obj)
        labels = labels.to(device_obj)

        # Handle multi-channel labels: aggregate to any-seizure
        if labels.dim() == 3:  # (B, C, T)
            labels = labels.max(dim=1)[0]  # (B, T)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass with AMP
        with autocast(enabled=use_amp and device != "cpu"):
            outputs = model(windows)  # (B, T) probabilities

            # Element-wise weighted loss (normalize by total weight)
            per_element_loss = criterion(outputs, labels)
            weights = torch.where(labels > 0, pos_weight, 1.0)
            loss = (per_element_loss * weights).sum() / (weights.sum() + 1e-8)

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        # Scheduler step (per-iteration)
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        progress.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(1, num_batches)


# ============================================================================
# Validation epoch (Single Responsibility)
# ============================================================================


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    post_config: PostprocessingConfig,
    device: str = "cpu",
    fa_rates: list[float] | None = None,
) -> dict[str, float]:
    """Validate model and compute metrics.

    Args:
        model: SeizureDetector model
        dataloader: Validation DataLoader
        post_config: Post-processing configuration
        device: Device to evaluate on
        fa_rates: FA/24h targets for sensitivity

    Returns:
        Dictionary of metrics
    """
    if fa_rates is None:
        fa_rates = [10, 5, 1]

    model.eval()
    device_obj = torch.device(device)
    criterion = nn.BCELoss()

    all_probs = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for windows, labels in tqdm(dataloader, desc="Validating", leave=False):
            windows = windows.to(device_obj)
            labels = labels.to(device_obj)

            # Handle multi-channel labels
            if labels.dim() == 3:
                labels = labels.max(dim=1)[0]

            outputs = model(windows)
            loss = criterion(outputs, labels)

            all_probs.append(outputs.cpu())
            all_labels.append(labels.cpu())

            total_loss += loss.item()
            num_batches += 1

    # Concatenate all batches
    all_probs_tensor = torch.cat(all_probs, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)

    # Compute metrics
    metrics = evaluate_predictions(
        all_probs_tensor,
        all_labels_tensor,
        fa_rates,
        post_config,
        sampling_rate=256,
    )

    # Add validation loss
    metrics["val_loss"] = total_loss / max(1, num_batches)

    return metrics


# ============================================================================
# Checkpointing (Single Responsibility)
# ============================================================================


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    best_metric: float,
    checkpoint_path: Path,
    scheduler: LRScheduler | None = None,
    config: Config | None = None,
) -> None:
    """Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        best_metric: Best metric value
        checkpoint_path: Where to save
        scheduler: Optional scheduler state
        config: Optional config to save
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if config is not None:
        checkpoint["config"] = config.model_dump()

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
) -> tuple[int, float]:
    """Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load into
        optimizer: Optional optimizer to restore
        scheduler: Optional scheduler to restore

    Returns:
        (epoch, best_metric)
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint["epoch"], checkpoint.get("best_metric", 0.0)


# ============================================================================
# Early stopping (Single Responsibility)
# ============================================================================


class EarlyStopping:
    """Early stopping handler.

    Encapsulates early stopping logic.
    """

    def __init__(self, config: EarlyStoppingConfig) -> None:
        self.patience = config.patience
        self.metric = config.metric
        self.mode = config.mode
        self.best_score = float("-inf") if self.mode == "max" else float("inf")
        self.counter = 0
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int = 0) -> bool:
        """Check if should stop.

        Args:
            score: Current metric value
            epoch: Current epoch

        Returns:
            True if should stop
        """
        improved = score > self.best_score if self.mode == "max" else score < self.best_score

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


# ============================================================================
# Main training orchestrator (Dependency Inversion)
# ============================================================================


def train(
    model: SeizureDetector,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config,
) -> dict[str, Any]:
    """Main training loop orchestrator.

    Args:
        model: SeizureDetector model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        config: Full configuration

    Returns:
        Dictionary of best metrics
    """
    # Setup
    set_seed(config.experiment.seed)
    device = config.experiment.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config.training)

    total_steps = config.training.epochs * len(train_loader)
    scheduler = (
        create_scheduler(optimizer, config.training.scheduler, total_steps)
        if config.training.scheduler
        else None
    )

    # Setup logging
    output_dir = Path(config.experiment.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(output_dir / "tensorboard")

    # Early stopping
    early_stopping = EarlyStopping(config.training.early_stopping)

    # Resume if checkpoint exists
    start_epoch = 0
    best_metric = 0.0
    last_checkpoint = checkpoint_dir / "last.pt"
    if last_checkpoint.exists() and config.training.resume:
        start_epoch, best_metric = load_checkpoint(last_checkpoint, model, optimizer, scheduler)
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    best_metrics: dict[str, Any] = {"best_epoch": 0}

    for epoch in range(start_epoch, config.training.epochs):
        print(f"\nEpoch {epoch + 1}/{config.training.epochs}")

        # Train
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            use_amp=config.training.mixed_precision,
            gradient_clip=config.training.gradient_clip,
            scheduler=scheduler,
        )

        # Validate
        val_metrics = validate_epoch(
            model,
            val_loader,
            config.postprocessing,
            device=device,
            fa_rates=config.evaluation.fa_rates,
        )

        # Log metrics
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_metrics["val_loss"], epoch)
        writer.add_scalar("Metrics/TAES", val_metrics["taes"], epoch)
        writer.add_scalar("Metrics/AUROC", val_metrics["auroc"], epoch)

        for fa_rate in config.evaluation.fa_rates:
            key = f"sensitivity_at_{fa_rate}fa"
            if key in val_metrics:
                writer.add_scalar(f"Metrics/{key}", val_metrics[key], epoch)

        # Print metrics
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"  TAES: {val_metrics['taes']:.4f}")
        print(f"  AUROC: {val_metrics['auroc']:.4f}")

        # Track best model
        metric_name = config.training.early_stopping.metric
        current_metric = val_metrics.get(metric_name, 0.0)

        if early_stopping(current_metric, epoch):
            print(f"Early stopping at epoch {epoch + 1}")
            break

        # Save best model
        if current_metric == early_stopping.best_score:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                current_metric,
                checkpoint_dir / "best.pt",
                scheduler,
                config,
            )
            best_metrics = {
                "best_epoch": epoch + 1,
                "best_taes": val_metrics["taes"],
                "best_auroc": val_metrics["auroc"],
                f"best_{metric_name}": current_metric,
            }
            print(f"  New best {metric_name}: {current_metric:.4f}")

        # Save last checkpoint
        save_checkpoint(
            model,
            optimizer,
            epoch,
            best_metric,
            checkpoint_dir / "last.pt",
            scheduler,
            config,
        )

    writer.close()
    print(f"\nTraining complete. Best epoch: {best_metrics['best_epoch']}")

    return best_metrics


# ============================================================================
# CLI entry point
# ============================================================================


def main() -> None:
    """CLI entry point for training."""
    import argparse

    from src.experiment.data import EEGWindowDataset

    parser = argparse.ArgumentParser(description="Train seizure detection model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )

    args = parser.parse_args()

    # Load config
    config = Config.from_yaml(Path(args.config))
    config.training.resume = args.resume

    # Create datasets
    edf_files = list(Path(config.data.data_dir).glob("**/*.edf"))

    # Split train/val
    val_split = int(len(edf_files) * config.data.validation_split)
    train_files = edf_files[val_split:]
    val_files = edf_files[:val_split]

    print(f"Loading {len(train_files)} train, {len(val_files)} val files")

    train_dataset = EEGWindowDataset(
        train_files,
        cache_dir=Path(config.data.cache_dir) / "train",
    )

    val_dataset = EEGWindowDataset(
        val_files,
        cache_dir=Path(config.data.cache_dir) / "val",
    )

    # Create data loaders
    train_sampler = None
    if config.data.use_balanced_sampling:
        # Get all labels for sampler
        all_labels = torch.stack([train_dataset[i][1] for i in range(len(train_dataset))])
        train_sampler = create_balanced_sampler(all_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Create model
    model = SeizureDetector.from_config(config.model)
    print(f"Model parameters: {model.count_parameters():,}")

    # Train
    best_metrics = train(model, train_loader, val_loader, config)

    print("\nFinal metrics:")
    for key, value in best_metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
