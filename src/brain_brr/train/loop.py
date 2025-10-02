"""Training and evaluation pipeline orchestration (Phase 3).

SOLID principles applied:
- Single Responsibility: Each function does one thing
- Open/Closed: Extensible via configs, not modification
- Liskov Substitution: Interfaces respect contracts
- Interface Segregation: Small, focused interfaces
- Dependency Inversion: Depend on abstractions (configs)
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as tnf
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.utils.data import DataLoader, WeightedRandomSampler

# Make TensorBoard optional
try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None  # type: ignore[assignment, misc]

from src.brain_brr.config.schemas import (
    Config,
    EarlyStoppingConfig,
    SchedulerConfig,
    TrainingConfig,
    WarmupScheduleConfig,
)
from src.brain_brr.models import SeizureDetector
from src.brain_brr.train.checkpoint import load_checkpoint, save_checkpoint
from src.brain_brr.train.train_step import train_epoch
from src.brain_brr.train.train_utils import set_seed, worker_init_fn
from src.brain_brr.train.val_step import validate_epoch
from src.brain_brr.train.wandb_integration import WandBLogger
from src.brain_brr.utils.env import env

# Module logger
logger = logging.getLogger(__name__)

# Training-specific logger with gating
LOG_EVERY_N_STEPS = int(os.getenv("BGB_LOG_EVERY_N_STEPS", "50"))

# WSL2-safe multiprocessing defaults (must be before any DataLoader creation)
if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)

# ============================================================================
# Warmup Schedule Utilities (Production ML Best Practice)
# ============================================================================


def get_focal_gamma(
    global_step: int,
    warmup_config: WarmupScheduleConfig | None,
    target_gamma: float = 2.0,
) -> float:
    """Compute focal loss gamma for current step.

    Linear interpolation from start_gamma → end_gamma over warmup_steps.
    Reduces loss amplification during early training for gradient stabilization.

    Standard practice in production ML (OpenAI, Google, Meta).

    Args:
        global_step: Current training step
        warmup_config: Warmup schedule configuration
        target_gamma: Target gamma (from config focal_gamma)

    Returns:
        Effective gamma for current step
    """
    if warmup_config is None or not warmup_config.enabled or not warmup_config.focal_gamma_enabled:
        return target_gamma

    if global_step >= warmup_config.warmup_steps:
        return target_gamma

    # Linear interpolation: start → end over warmup_steps
    progress = global_step / warmup_config.warmup_steps
    start_gamma = warmup_config.focal_gamma_start
    end_gamma = warmup_config.focal_gamma_end
    current_gamma = start_gamma + progress * (end_gamma - start_gamma)

    return current_gamma


# ============================================================================
# Reproducibility utilities (Single Responsibility)
# ============================================================================


# ============================================================================
# Sampling utilities (Interface Segregation)
# ============================================================================


def create_balanced_sampler(dataset: Any, sample_size: int = 500) -> WeightedRandomSampler | None:
    """Create positive-aware balanced sampler for imbalanced datasets.

    Args:
        dataset: EEGWindowDataset instance
        sample_size: Number of windows to sample for statistics

    Returns:
        WeightedRandomSampler for balanced mini-batches or None if no seizures found
    """
    logger.info("[SAMPLER] Creating positive-aware balanced sampler...")

    # Skip expensive sampling in smoke test mode
    if env.smoke_test():
        logger.info(
            "[SMOKE TEST MODE] Skipping sampler window checking - returning None for uniform sampling"
        )
        return None

    # Sample dataset to find which windows have seizures
    sample_size = min(sample_size, len(dataset))
    sample_indices = torch.randperm(len(dataset))[:sample_size]

    # Track which windows actually have seizures
    window_has_seizure = torch.zeros(len(dataset), dtype=torch.float32)
    sampled_seizure_count = 0

    logger.info(f"[SAMPLER] Checking {sample_size} windows for seizures...")
    for i, idx in enumerate(sample_indices):
        batch = dataset[idx.item()]
        label = batch["label"]
        if (label > 0).any():
            window_has_seizure[idx] = 1.0
            sampled_seizure_count += 1

        # Progress update every 1000 windows
        if (i + 1) % 1000 == 0:
            logger.debug(
                f"[SAMPLER] Checked {i + 1}/{sample_size} windows, found {sampled_seizure_count} with seizures"
            )

    # Estimate seizure ratio
    seizure_ratio = sampled_seizure_count / sample_size
    logger.info(
        f"[SAMPLER] Final: {sampled_seizure_count}/{sample_size} windows with seizures ({seizure_ratio:.2%})"
    )

    if seizure_ratio < 1e-8:
        logger.info("[SAMPLER] WARNING: No seizures found in sample! Using uniform sampling.")
        return None

    # Calculate weight for positive samples (sqrt to prevent explosion)
    pos_weight = math.sqrt((1 - seizure_ratio) / seizure_ratio)

    # Extrapolate to full dataset
    # For unsampled indices, assign weight probabilistically
    weights = torch.ones(len(dataset), dtype=torch.float32)

    # Known seizure windows get high weight
    weights[window_has_seizure > 0] = pos_weight

    # Estimate weights for unsampled windows
    unsampled_mask = torch.ones(len(dataset), dtype=torch.bool)
    unsampled_mask[sample_indices] = False
    n_unsampled_seizures = int(unsampled_mask.sum() * seizure_ratio)

    if n_unsampled_seizures > 0:
        unsampled_indices = torch.where(unsampled_mask)[0]
        random_seizure_indices = unsampled_indices[
            torch.randperm(len(unsampled_indices))[:n_unsampled_seizures]
        ]
        weights[random_seizure_indices] = pos_weight

    logger.info(f"[SAMPLER] Seizure ratio: {seizure_ratio:.2%}")
    logger.info(f"[SAMPLER] Positive weight: {pos_weight:.2f}")
    logger.info(f"[SAMPLER] Estimated seizure windows: {(weights > 1).sum().item()}/{len(dataset)}")

    return WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=len(weights),
        replacement=True,
        generator=torch.Generator().manual_seed(42),
    )


# ============================================================================
# Loss functions (Open/Closed Principle)
# ============================================================================


class FocalLoss(nn.Module):
    """Binary focal loss on logits with optional pos_weight.

    This wraps BCE-with-logits and applies focal modulation:
        loss = alpha_t * (1 - p_t)^gamma * BCEWithLogitsLoss(logits, targets)

    - logits: (B, T)
    - targets: (B, T) in {0,1}
    - pos_weight: optional scalar tensor to up-weight positives (same semantics as BCEWithLogitsLoss)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        pos_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Per-element BCE on logits for numerical stability
        # Clamp logits to prevent overflow in BCE computation
        logits_clamped = logits.clamp(min=-100, max=100)
        bce = tnf.binary_cross_entropy_with_logits(
            logits_clamped, targets, reduction="none", pos_weight=pos_weight
        )
        # Probabilities (use clamped logits for numerical stability)
        p = torch.sigmoid(logits_clamped)
        # Critical: Clamp probabilities to avoid log(0) or log(1) issues
        p = p.clamp(min=1e-6, max=1 - 1e-6)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        # Class-balanced alpha
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        # Focal modulation with numerical stability
        # Clamp p_t away from 1 to prevent (1-p_t)^gamma from underflowing to 0
        p_t_stable = p_t.clamp(min=1e-7, max=1 - 1e-7)
        mod = (1.0 - p_t_stable).pow(self.gamma)
        focal_loss = alpha_t * mod * bce

        # Additional safety: clamp output to prevent extreme values
        focal_loss = focal_loss.clamp(max=100.0)  # Prevent loss explosion
        return cast(torch.Tensor, focal_loss)


# ============================================================================
# Optimizer & Scheduler factories (Open/Closed Principle)
# ============================================================================


def create_optimizer(model: nn.Module, config: TrainingConfig) -> Optimizer:
    """Create optimizer from config.

    Factory pattern for optimizer creation.
    Applies weight decay only to weights, not biases or normalization parameters.
    """
    if config.optimizer == "adamw":
        # Separate parameters into decay and no_decay groups
        # This prevents weight decay from corrupting normalization layers
        no_decay = ["bias", "bn", "ln", "layernorm", "norm", "rmsnorm"]
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Check if parameter name contains any no_decay keyword
            if any(nd in name.lower() for nd in no_decay):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # Create parameter groups with different weight decay
        param_groups = [
            {
                "params": decay_params,
                "weight_decay": config.weight_decay,
                "lr": config.learning_rate,
            },
            {"params": no_decay_params, "weight_decay": 0.0, "lr": config.learning_rate},
        ]

        logger.info("[OPTIMIZER] Created parameter groups:")
        logger.info(f"  - Decay group: {len(decay_params)} parameters")
        logger.info(f"  - No-decay group: {len(no_decay_params)} parameters")

        return AdamW(param_groups, lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-8)
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
    if env.anomaly_detect():
        try:
            torch.autograd.set_detect_anomaly(True)
            logger.info("[DEBUG] Enabled torch.autograd anomaly detection")
        except Exception:
            pass
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
    if HAS_TENSORBOARD and not env.disable_tensorboard():
        writer = SummaryWriter(output_dir / "tensorboard")
    elif not HAS_TENSORBOARD and not env.disable_tensorboard():
        logger.info("TensorBoard not installed. Install with: pip install tensorboard")

    # Initialize W&B logging
    wandb_logger = WandBLogger(config)

    # Early stopping
    early_stopping = EarlyStopping(config.training.early_stopping)

    # Resume from checkpoint (prioritize mid-epoch > last > best)
    start_epoch = 0
    best_metric = 0.0

    # Check for mid-epoch checkpoints first (for crash recovery)
    mid_epoch_checkpoints = sorted(checkpoint_dir.glob("mid_epoch_*.pt"))
    if mid_epoch_checkpoints and config.training.resume:
        latest_mid = mid_epoch_checkpoints[-1]
        logger.info(f"[RESUME] Found mid-epoch checkpoint: {latest_mid.name}")
        ckpt = torch.load(latest_mid, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"]
        best_metric = ckpt.get("best_metric", 0.0)
        if best_metric == 0.0 and (checkpoint_dir / "last.pt").exists():
            try:
                _last = torch.load(
                    checkpoint_dir / "last.pt", map_location="cpu", weights_only=False
                )
                best_metric = _last.get("best_metric", 0.0)
            except Exception:
                pass
        logger.info(f"Resumed from epoch {start_epoch + 1}, batch {ckpt.get('batch_idx', '?')}")
        # Note: This resumes from start of epoch, not exact batch
    elif (checkpoint_dir / "last.pt").exists() and config.training.resume:
        start_epoch, best_metric = load_checkpoint(
            checkpoint_dir / "last.pt", model, optimizer, scheduler
        )
        logger.info(f"Resumed from epoch {start_epoch + 1}")

    # Training loop
    best_metrics: dict[str, Any] = {"best_epoch": 0}
    global_step = 0  # Track global step across epochs for scheduler

    for epoch in range(start_epoch, config.training.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{config.training.epochs}")

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
            loss_mode=getattr(config.training, "loss", "bce"),
            focal_alpha=getattr(config.training, "focal_alpha", 0.25),
            focal_gamma=getattr(config.training, "focal_gamma", 2.0),
            return_step=True,
            checkpoint_dir=checkpoint_dir,
            epoch_index=epoch,
            mid_epoch_minutes=(
                float(env.mid_epoch_minutes() or 0)
                if config.training.resume and env.mid_epoch_minutes() is not None
                else getattr(
                    config.experiment,
                    "mid_epoch_checkpoint_minutes",
                    10.0 if config.training.resume else None,
                )
            ),
            mid_epoch_keep=int(env.mid_epoch_keep()),
            warmup_schedule=config.training.warmup_schedule,
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

        # COLLAPSE DETECTION: Stop if model outputs all-negative
        if val_metrics["auroc"] < 0.55 and epoch > 2:
            logger.info(f"\n⚠️ MODEL COLLAPSE DETECTED! AUROC={val_metrics['auroc']:.3f}")
            logger.info("Model is predicting all-negative. Stopping training.")
            logger.info("Potential causes:")
            logger.info("  1. Dataset has too few seizures (<1%)")
            logger.info("  2. Class weighting is insufficient")
            logger.info("  3. Learning rate too high/low")
            logger.info("\nRecommendations:")
            logger.info("  - Increase BGB_LIMIT_FILES to include more seizure files")
            logger.info("  - Use focal loss or stronger class weighting")
            logger.info("  - Check dataset statistics logged at start")
            break

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

        # Log to W&B
        wandb_metrics = {
            "train_loss": train_loss,
            "val_loss": val_metrics["val_loss"],
            "taes": val_metrics["taes"],
            "auroc": val_metrics["auroc"],
        }
        for fa_rate in config.evaluation.fa_rates:
            key = f"sensitivity_at_{fa_rate}fa"
            if key in val_metrics:
                wandb_metrics[key] = val_metrics[key]
        wandb_logger.log(wandb_metrics, step=epoch)

        # Print metrics with flush for Modal visibility
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        logger.info(f"  TAES: {val_metrics['taes']:.4f}")
        logger.info(f"  AUROC: {val_metrics['auroc']:.4f}")

        # Print sensitivity at FA rates
        for fa_rate in config.evaluation.fa_rates:
            key = f"sensitivity_at_{fa_rate}fa"
            if key in val_metrics:
                logger.info(f"  Sensitivity@{fa_rate}FA/24h: {val_metrics[key]:.4f}")

        # Track best model
        metric_name = config.training.early_stopping.metric
        current_metric = val_metrics.get(metric_name, 0.0)

        if early_stopping(current_metric, epoch):
            logger.info(f"Early stopping at epoch {epoch + 1}")
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
            best_metric = current_metric  # FIX: Update best_metric when we find a new best
            best_metrics = {
                "best_epoch": epoch + 1,
                "best_taes": val_metrics["taes"],
                "best_auroc": val_metrics["auroc"],
                f"best_{metric_name}": current_metric,
            }
            logger.info(f"  New best {metric_name}: {current_metric:.4f}")

            # Log best model to W&B
            wandb_logger.log_model(checkpoint_dir / "best.pt", name=f"best-{metric_name}")

        # Save periodic checkpoint based on checkpoint_interval
        checkpoint_interval = getattr(
            config.experiment,
            "checkpoint_interval",
            getattr(config.training, "checkpoint_interval", 0),
        )
        if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1:03d}.pt"
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_metric,
                checkpoint_path,
                scheduler,
                config,
            )
            logger.info(f"  Saved periodic checkpoint: {checkpoint_path.name}")

        # Always save last checkpoint for resume capability
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

    # Finish W&B run
    wandb_logger.finish()

    logger.info(f"\nTraining complete. Best epoch: {best_metrics['best_epoch']}")

    return best_metrics


# ============================================================================
# CLI entry point
# ============================================================================


def main() -> None:
    """CLI entry point for training."""
    import argparse
    import logging

    from src.brain_brr.data import BalancedSeizureDataset, EEGWindowDataset
    from src.brain_brr.utils.logging_config import setup_logging

    # Initialize elite logging infrastructure for training
    setup_logging()
    logging.captureWarnings(True)  # Capture Python warnings into logging system

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

    # Check if we're in smoke test mode
    is_smoke_test = env.smoke_test()
    if is_smoke_test:
        logger.warning("\n" + "=" * 60)
        logger.info("SMOKE TEST MODE ACTIVE")
        logger.info("Pipeline validation only - model will NOT learn meaningful patterns")
        logger.info("DO NOT use this for real training!")
        logger.info("=" * 60 + "\n")

    # Load TUSZ official splits (PATIENT-DISJOINT!)
    # For TUSZ: train on train/, validate on dev/, never touch eval/
    from src.brain_brr.data.tusz_splits import load_tusz_for_training

    data_root = Path(config.data.data_dir)

    # Get the parent directory that contains train/, dev/, eval/
    if data_root.name in ["train", "dev", "eval"]:
        # If pointing to a specific split, go up to parent
        data_root = data_root.parent

    # Load official TUSZ splits with patient disjointness validation
    splits = load_tusz_for_training(data_root, use_eval=False, verbose=True)
    train_files, train_label_files = splits["train"]
    val_files, val_label_files = splits["dev"]  # Use dev for validation

    # Extract and validate patient IDs for transparency
    from src.brain_brr.data.tusz_splits import extract_patient_id

    train_patients = {extract_patient_id(f) for f in train_files}
    val_patients = {extract_patient_id(f) for f in val_files}

    # Final paranoid check - should never trigger if tusz_splits.py works
    overlap = train_patients & val_patients
    if overlap:
        raise ValueError(
            f"CRITICAL: Patient leakage detected! {len(overlap)} patients in both splits:\n"
            f"  {sorted(overlap)[:10]}"
        )

    logger.info("\n[SPLIT STATS] OFFICIAL TUSZ SPLITS:")
    logger.info(f"  Train: {len(train_patients)} patients, {len(train_files)} files")
    logger.info(f"  Val:   {len(val_patients)} patients, {len(val_files)} files")
    logger.info("  ✅ PATIENT DISJOINTNESS VERIFIED - No leakage!")

    # Optional file limit for fast bring-up via env var (does not change config)
    limit_env_val = env.limit_files()
    if limit_env_val is None and is_smoke_test:
        limit_env_val = 3  # Safe default for smoke mode
    if limit_env_val is not None:
        try:
            limit = max(1, int(limit_env_val))
            train_files = train_files[:limit]
            train_label_files = train_label_files[:limit]
            val_limit = max(1, min(len(val_files), max(1, limit // 5)))
            val_files = val_files[:val_limit]
            val_label_files = val_label_files[:val_limit]
            logger.debug(
                f"BGB_LIMIT_FILES={limit}: using {len(train_files)} train, {len(val_files)} val files"
            )
        except Exception:
            pass

    # Cache directory sanity and preflight
    data_cache_root = Path(config.data.cache_dir)
    exp_cache_root = Path(config.experiment.cache_dir)
    if data_cache_root.resolve() != exp_cache_root.resolve():
        logger.warning(
            f"config.data.cache_dir ({data_cache_root}) != config.experiment.cache_dir ({exp_cache_root})"
        )

    try:
        from src.brain_brr.data.cache_utils import check_cache_completeness

        train_cache = data_cache_root / "train"
        val_cache = data_cache_root / "dev"
        st_train = check_cache_completeness(train_files, train_cache)
        st_val = check_cache_completeness(val_files, val_cache)
        if st_train.missing_files > 0 or st_val.missing_files > 0:
            logger.info(
                "[DATA] Cache incomplete: "
                f"train {st_train.cached_files}/{st_train.total_files}, "
                f"val {st_val.cached_files}/{st_val.total_files}"
            )
            logger.info(
                "[HINT] Pre-build cache to avoid slow training:\n"
                f"  python -m src build-cache --data-dir {config.data.data_dir} --cache-dir {data_cache_root / 'train'}\n"
                f"  python -m src build-cache --data-dir {config.data.data_dir} --cache-dir {data_cache_root / 'dev'}"
            )
    except Exception:
        pass

    train_cache_dir = data_cache_root / "train"
    use_balanced = bool(config.data.use_balanced_sampling)
    manifest_path = train_cache_dir / "manifest.json"

    # Force manifest rebuild if requested or if it exists but is invalid
    if use_balanced and manifest_path.exists():
        import json

        force_rebuild = env.force_manifest_rebuild()
        try:
            with open(manifest_path) as f:
                manifest_data = json.load(f)
            if force_rebuild:
                logger.info("[DATA] BGB_FORCE_MANIFEST_REBUILD=1 → deleting manifest for rebuild")
                manifest_path.unlink()
            else:
                from src.brain_brr.data.cache_utils import validate_manifest

                if not validate_manifest(train_cache_dir, manifest_data):
                    logger.warning("Invalid/stale manifest detected → deleting for rebuild")
                    manifest_path.unlink()
        except Exception as e:
            logger.info(f"[WARNING] Failed to read/validate manifest: {e}, deleting...")
            manifest_path.unlink()

    if use_balanced and not manifest_path.exists():
        # CRITICAL: Only build manifest if cache already has files!
        # Bug fix: Don't build manifest from empty directory
        train_cache_dir.mkdir(parents=True, exist_ok=True)
        existing_cache_files = list(train_cache_dir.glob("*.npz"))
        if existing_cache_files:
            try:
                from src.brain_brr.data.cache_utils import scan_existing_cache

                _ = scan_existing_cache(train_cache_dir)
                logger.info(f"[DATA] Built manifest from {len(existing_cache_files)} cached files")
            except Exception as e:
                logger.info(f"[WARNING] Manifest build failed: {e}")
        else:
            logger.info("[DATA] Skipping manifest build - cache not yet populated")

    # Create training dataset - either balanced (from manifest) or standard
    train_dataset: BalancedSeizureDataset | EEGWindowDataset
    if use_balanced and manifest_path.exists():
        try:
            train_dataset = BalancedSeizureDataset(train_cache_dir)
            logger.info(
                f"[DATASET] BalancedSeizureDataset: {len(train_dataset)} windows from manifest"
            )
            if len(train_dataset) == 0:
                is_smoke_test = env.smoke_test()
                if is_smoke_test:
                    logger.info(
                        "[SMOKE TEST MODE] Balanced manifest empty - will fallback to EEGWindowDataset"
                    )
                    raise Exception("Empty manifest in smoke test - triggering fallback")
                else:
                    logger.info("[FATAL] Balanced manifest produced 0 windows")
                    import sys

                    sys.exit(1)
        except Exception as e:
            logger.info(
                f"[WARNING] BalancedSeizureDataset failed: {e}; falling back to EEGWindowDataset"
            )
            train_dataset = EEGWindowDataset(
                train_files,
                label_files=train_label_files,
                cache_dir=train_cache_dir,
                allow_on_demand=True,
            )
    else:
        train_dataset = EEGWindowDataset(
            train_files,
            label_files=train_label_files,
            cache_dir=train_cache_dir,
            allow_on_demand=True,
        )

    # Validation cache uses "dev" subdir (TUSZ official naming)
    val_split_name = "dev"
    val_dataset = EEGWindowDataset(
        val_files,
        label_files=val_label_files,
        cache_dir=data_cache_root / val_split_name,
        allow_on_demand=True,
    )

    # CRITICAL FIX: If we just built cache via EEGWindowDataset and manifest doesn't exist,
    # build it now and switch to BalancedSeizureDataset!
    if (
        use_balanced
        and not isinstance(train_dataset, BalancedSeizureDataset)
        and not manifest_path.exists()
    ):
        logger.info("[DATA] Cache built, now creating manifest for balanced sampling...")
        try:
            from src.brain_brr.data.cache_utils import scan_existing_cache

            _ = scan_existing_cache(train_cache_dir)
            if manifest_path.exists():
                # Switch to BalancedSeizureDataset now that manifest exists
                train_dataset = BalancedSeizureDataset(train_cache_dir)
                logger.info(
                    f"[DATA] Switched to BalancedSeizureDataset: {len(train_dataset)} windows"
                )
        except Exception as e:
            logger.info(f"[WARNING] Post-cache manifest build failed: {e}")

    # Create positive-aware balanced sampler (fallback if BalancedSeizureDataset not used)
    train_sampler = None
    if (
        config.data.use_balanced_sampling
        and len(train_dataset) > 0
        and not isinstance(train_dataset, BalancedSeizureDataset)
    ):
        # CRITICAL: TUSZ has extreme imbalance (0.1-1% seizures at window level)
        # We MUST sample enough windows to guarantee finding seizures
        # Math: P(0 seizures) = (1-p)^n, for p=0.001, n=20000 → P≈0.00000002
        sample_size = min(20000, len(train_dataset))  # Sample 20k windows for safety
        logger.info(f"[SAMPLER] Sampling {sample_size} windows to detect seizures...")
        train_sampler = create_balanced_sampler(train_dataset, sample_size=sample_size)

        if train_sampler is None:
            # Check if we're in smoke test mode
            is_smoke_test = env.smoke_test()

            if is_smoke_test:
                logger.info("=" * 60)
                logger.info("[SMOKE TEST MODE] No seizures found - continuing anyway")
                logger.info("[SMOKE TEST MODE] Using uniform sampling for pipeline validation")
                logger.info("[SMOKE TEST MODE] This model will NOT learn - testing only!")
                logger.info("=" * 60)
                # Continue with default sampler for smoke testing
            else:
                logger.info("=" * 60)
                logger.info(f"[FATAL] No seizures found in {sample_size} windows!")
                logger.info("[FATAL] Training will produce a USELESS model!")
                logger.info("[FATAL] Check your data or increase sample size!")
                logger.info("=" * 60)
                # Fail fast - don't waste GPU hours on doomed training
                import sys

                sys.exit(1)

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

    # Create model (v3.4.1: pass warmup_schedule for gradient stabilization)
    model = SeizureDetector.from_config(
        config.model, warmup_schedule=config.training.warmup_schedule
    )
    logger.info(f"Model parameters: {model.count_parameters():,}")

    # Train
    best_metrics = train(model, train_loader, val_loader, config)

    logger.info("\nFinal metrics:")
    for key, value in best_metrics.items():
        logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    main()
