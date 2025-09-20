"""Training and evaluation pipeline orchestration (Phase 3).

SOLID principles applied:
- Single Responsibility: Each function does one thing
- Open/Closed: Extensible via configs, not modification
- Liskov Substitution: Interfaces respect contracts
- Interface Segregation: Small, focused interfaces
- Dependency Inversion: Depend on abstractions (configs)
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # type: ignore[import-untyped]

from src.brain_brr.config.schemas import (
    Config,
    EarlyStoppingConfig,
    PostprocessingConfig,
    SchedulerConfig,
    TrainingConfig,
)
from src.brain_brr.eval.metrics import evaluate_predictions
from src.brain_brr.models import SeizureDetector

# WSL2-safe multiprocessing defaults (must be before any DataLoader creation)
if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)

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

    Uses a LambdaLR for linear warmup followed by cosine decay.
    Designed to step once per optimization update.
    """
    warmup_steps = max(1, int(config.warmup_ratio * total_steps))

    if config.type == "cosine":
        import math

        # Preserve initial learning rates so creating the scheduler does not
        # mutate optimizer.param_groups (some schedulers may do this).
        initial_lrs = [g["lr"] for g in optimizer.param_groups]

        def lr_lambda(step: int) -> float:
            # Linear warmup
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            # Cosine decay to 0
            progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        sched = LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=-1)
        # Reset any change at construction time.
        for g, lr in zip(optimizer.param_groups, initial_lrs, strict=False):
            g["lr"] = lr
        return sched
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
    global_step: int = 0,
    *,
    return_step: bool = False,
) -> float | tuple[float, int]:
    """Train for one epoch.

    Args:
        model: SeizureDetector model
        dataloader: Training DataLoader
        optimizer: Optimizer instance
        device: Device to train on
        use_amp: Use automatic mixed precision
        gradient_clip: Max gradient norm
        scheduler: Optional LR scheduler (per-iteration)
        global_step: Global step counter for scheduler
        return_step: If True, return (loss, global_step). If False, return just loss.

    Returns:
        Average training loss (default) or tuple of (loss, global_step) if return_step=True
    """
    import time

    model.train()
    device_obj = torch.device(device)
    # Only construct GradScaler when actually using CUDA AMP
    scaler = GradScaler(enabled=(use_amp and device == "cuda"))

    # Heartbeat timer for Modal visibility
    last_heartbeat = time.time()
    heartbeat_interval = 300  # 5 minutes

    # Calculate class weights from first batch (approximation)
    first_batch = next(iter(dataloader))
    _, first_labels = first_batch
    pos_ratio = float((first_labels > 0).float().mean())
    pos_weight_val = (1 - pos_ratio) / max(pos_ratio, 1e-8) if pos_ratio > 0 else 1.0

    # AMP-safe, numerically stable loss on logits
    # (We keep model outputs as probabilities elsewhere for tests/inference)
    pos_weight_t = torch.as_tensor(pos_weight_val, device=device_obj, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight_t)

    total_loss = 0.0
    num_batches = 0

    # Robust tqdm handling for Modal/non-TTY environments
    use_tqdm = not os.getenv("BGB_DISABLE_TQDM")
    if use_tqdm:
        try:
            # Try to create tqdm, but catch if it returns None or fails
            progress_bar = tqdm(dataloader, desc="Training", leave=False)
            if progress_bar is None:
                print("[WARNING] tqdm returned None, falling back to plain iteration", flush=True)
                progress = dataloader
            else:
                progress = progress_bar
        except Exception as e:
            print(f"[WARNING] tqdm failed ({e}), using plain iteration", flush=True)
            progress = dataloader
    else:
        progress = dataloader

    # Use enumerate for batch indexing (satisfies ruff SIM113)
    # But track global_step separately for proper scheduler behavior
    for batch_idx, (windows, labels) in enumerate(progress):
        windows = windows.to(device_obj)
        labels = labels.to(device_obj)

        # Handle multi-channel labels: aggregate to any-seizure
        if labels.dim() == 3:  # (B, C, T)
            labels = labels.max(dim=1)[0]  # (B, T)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass with AMP (model returns raw logits)
        with autocast(enabled=(use_amp and device == "cuda")):
            logits = model(windows)  # (B, T) raw logits
            per_element_loss = criterion(logits, labels)
            # Mean reduction since pos_weight is already in criterion
            loss = per_element_loss.mean()

        # Backward pass with proper scaler handling
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

        # Increment global step counter
        global_step += 1

        # Scheduler step AFTER optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        if use_tqdm and hasattr(progress, "set_postfix"):
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        # Modal progress logging - print every 100 batches for visibility
        if batch_idx > 0 and batch_idx % 100 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"[PROGRESS] Batch {batch_idx}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f} | LR: {current_lr:.2e}",
                flush=True,
            )

        # Heartbeat for Modal (every 5 minutes)
        if time.time() - last_heartbeat > heartbeat_interval:
            print(
                f"[HEARTBEAT] Still training... Batch {batch_idx}/{len(dataloader)} | "
                f"Avg Loss: {total_loss / max(1, num_batches):.4f}",
                flush=True,
            )
            last_heartbeat = time.time()

    avg_loss = total_loss / max(1, num_batches)
    return (avg_loss, global_step) if return_step else avg_loss


# ============================================================================
# Validation epoch (Single Responsibility)
# ============================================================================


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    post_config: PostprocessingConfig,
    device: str = "cpu",
    fa_rates: list[float] | None = None,
) -> dict[str, Any]:
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
    criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for raw logits

    all_probs = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    # Robust tqdm handling for Modal/non-TTY environments
    use_tqdm = not os.getenv("BGB_DISABLE_TQDM")
    with torch.no_grad():
        if use_tqdm:
            try:
                progress_bar = tqdm(dataloader, desc="Validating", leave=False)
                if progress_bar is None:
                    print("[WARNING] tqdm returned None in validation, using plain iteration", flush=True)
                    iterator = dataloader
                else:
                    iterator = progress_bar
            except Exception as e:
                print(f"[WARNING] tqdm failed in validation ({e}), using plain iteration", flush=True)
                iterator = dataloader
        else:
            iterator = dataloader
        for windows, labels in iterator:
            windows = windows.to(device_obj)
            labels = labels.to(device_obj)

            # Handle multi-channel labels
            if labels.dim() == 3:
                labels = labels.max(dim=1)[0]

            logits = model(windows)  # Model now outputs raw logits
            loss = criterion(logits, labels)

            # Convert logits to probabilities for evaluation
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())
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
        # Allow exactly `patience` non-improving epochs; stop on the next one.
        return self.counter > self.patience


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

    writer: SummaryWriter | None = None
    if not os.getenv("BGB_DISABLE_TB"):
        writer = SummaryWriter(output_dir / "tensorboard")

    # Early stopping
    early_stopping = EarlyStopping(config.training.early_stopping)

    # Resume if checkpoint exists
    start_epoch = 0
    best_metric = 0.0
    last_checkpoint = checkpoint_dir / "last.pt"
    if last_checkpoint.exists() and config.training.resume:
        start_epoch, best_metric = load_checkpoint(last_checkpoint, model, optimizer, scheduler)
        print(f"Resumed from epoch {start_epoch}", flush=True)

    # Training loop
    best_metrics: dict[str, Any] = {"best_epoch": 0}
    global_step = 0  # Track global step across epochs for scheduler

    for epoch in range(start_epoch, config.training.epochs):
        print(f"\nEpoch {epoch + 1}/{config.training.epochs}", flush=True)

        # Train
        result = train_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            use_amp=config.training.mixed_precision,
            gradient_clip=config.training.gradient_clip,
            scheduler=scheduler,
            global_step=global_step,
            return_step=True,
        )
        # Type narrowing for mypy
        assert isinstance(result, tuple), "return_step=True should return tuple"
        train_loss, global_step = result

        # Validate
        val_metrics = validate_epoch(
            model,
            val_loader,
            config.postprocessing,
            device=device,
            fa_rates=config.evaluation.fa_rates,
        )

        # Log metrics
        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_metrics["val_loss"], epoch)
            writer.add_scalar("Metrics/TAES", val_metrics["taes"], epoch)
            writer.add_scalar("Metrics/AUROC", val_metrics["auroc"], epoch)

        for fa_rate in config.evaluation.fa_rates:
            key = f"sensitivity_at_{fa_rate}fa"
            if key in val_metrics and writer is not None:
                writer.add_scalar(f"Metrics/{key}", val_metrics[key], epoch)

        # Print metrics with flush for Modal visibility
        print(f"  Train Loss: {train_loss:.4f}", flush=True)
        print(f"  Val Loss: {val_metrics['val_loss']:.4f}", flush=True)
        print(f"  TAES: {val_metrics['taes']:.4f}", flush=True)
        print(f"  AUROC: {val_metrics['auroc']:.4f}", flush=True)

        # Print sensitivity at FA rates
        for fa_rate in config.evaluation.fa_rates:
            key = f"sensitivity_at_{fa_rate}fa"
            if key in val_metrics:
                print(f"  Sensitivity@{fa_rate}FA/24h: {val_metrics[key]:.4f}", flush=True)

        # Track best model
        metric_name = config.training.early_stopping.metric
        current_metric = val_metrics.get(metric_name, 0.0)

        if early_stopping(current_metric, epoch):
            print(f"Early stopping at epoch {epoch + 1}", flush=True)
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
            print(f"  New best {metric_name}: {current_metric:.4f}", flush=True)

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

    if writer is not None:
        writer.close()
    print(f"\nTraining complete. Best epoch: {best_metrics['best_epoch']}", flush=True)

    return best_metrics


# ============================================================================
# CLI entry point
# ============================================================================


def main() -> None:
    """CLI entry point for training."""
    import argparse

    from src.brain_brr.data import EEGWindowDataset

    parser = argparse.ArgumentParser(description="Train seizure detection model")
    parser.add_argument(
        "config",  # Make positional argument for easier CLI usage
        type=str,
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

    # Create datasets (discover EDF files and paired CSV_BI annotations if present)
    data_root = Path(config.data.data_dir)
    edf_files = sorted(data_root.glob("**/*.edf"))

    # Split train/val
    val_split = int(len(edf_files) * config.data.validation_split)
    train_files = edf_files[val_split:]
    val_files = edf_files[:val_split]

    print(f"Loading {len(train_files)} train, {len(val_files)} val files")

    # Optional file limit for fast bring-up via env var (does not change config)
    limit_env = os.getenv("BGB_LIMIT_FILES")
    if limit_env:
        try:
            limit = max(1, int(limit_env))
            train_files = train_files[:limit]
            val_files = val_files[: max(1, min(len(val_files), max(1, limit // 5)))]
            print(
                f"[DEBUG] BGB_LIMIT_FILES={limit}: using {len(train_files)} train, {len(val_files)} val files"
            )
        except Exception:
            pass

    # Pair label files (CSV next to EDF with same stem); pass even if missing
    train_label_files = [p.with_suffix(".csv") for p in train_files]
    val_label_files = [p.with_suffix(".csv") for p in val_files]

    train_dataset = EEGWindowDataset(
        train_files,
        label_files=train_label_files,
        cache_dir=Path(config.data.cache_dir) / "train",
    )

    val_dataset = EEGWindowDataset(
        val_files,
        label_files=val_label_files,
        cache_dir=Path(config.data.cache_dir) / "val",
    )

    # Create data loaders
    train_sampler = None
    if config.data.use_balanced_sampling and len(train_dataset) > 0:
        # Memory-optimized: sample subset to estimate class balance
        sample_size = min(500, len(train_dataset))
        sample_indices = torch.randperm(len(train_dataset))[:sample_size]

        seizure_count = 0
        for idx in sample_indices:
            _, label = train_dataset[idx.item()]
            if label.max().item() > 0.5:
                seizure_count += 1

        if seizure_count > 0:
            # Create approximate balanced sampler
            seizure_ratio = seizure_count / sample_size
            weights = torch.ones(len(train_dataset))

            # Randomly assign higher weights based on estimated ratio
            pos_weight = (1 - seizure_ratio) / max(seizure_ratio, 1e-8)
            n_seizure_est = int(len(train_dataset) * seizure_ratio)
            seizure_indices = torch.randperm(len(train_dataset))[:n_seizure_est]
            weights[seizure_indices] = pos_weight

            train_sampler = WeightedRandomSampler(weights.tolist(), len(weights), replacement=True)

    train_loader_kwargs: dict[str, Any] = {
        "batch_size": config.training.batch_size,
        "sampler": train_sampler,
        "shuffle": (train_sampler is None),
        "num_workers": config.data.num_workers,
        "pin_memory": bool(config.data.pin_memory),
        "worker_init_fn": worker_init_fn,
    }
    if config.data.num_workers > 0:
        train_loader_kwargs["persistent_workers"] = bool(config.data.persistent_workers)
        train_loader_kwargs["prefetch_factor"] = int(config.data.prefetch_factor)
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)

    val_loader_kwargs: dict[str, Any] = {
        "batch_size": config.training.batch_size,
        "shuffle": False,
        "num_workers": config.data.num_workers,
        "pin_memory": bool(config.data.pin_memory),
    }
    if config.data.num_workers > 0:
        val_loader_kwargs["persistent_workers"] = bool(config.data.persistent_workers)
        val_loader_kwargs["prefetch_factor"] = int(config.data.prefetch_factor)
    val_loader = DataLoader(val_dataset, **val_loader_kwargs)

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
