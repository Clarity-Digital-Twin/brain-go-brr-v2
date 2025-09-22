"""Training and evaluation pipeline orchestration (Phase 3).

SOLID principles applied:
- Single Responsibility: Each function does one thing
- Open/Closed: Extensible via configs, not modification
- Liskov Substitution: Interfaces respect contracts
- Interface Segregation: Small, focused interfaces
- Dependency Inversion: Depend on abstractions (configs)
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as tnf
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
from src.brain_brr.train.wandb_integration import WandBLogger

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


def create_balanced_sampler(dataset: Any, sample_size: int = 500) -> WeightedRandomSampler | None:
    """Create positive-aware balanced sampler for imbalanced datasets.

    Args:
        dataset: EEGWindowDataset instance
        sample_size: Number of windows to sample for statistics

    Returns:
        WeightedRandomSampler for balanced mini-batches or None if no seizures found
    """
    print("[SAMPLER] Creating positive-aware balanced sampler...", flush=True)

    # Skip expensive sampling in smoke test mode
    if os.environ.get("BGB_SMOKE_TEST", "0") == "1":
        print(
            "[SMOKE TEST MODE] Skipping sampler window checking - returning None for uniform sampling",
            flush=True,
        )
        return None

    # Sample dataset to find which windows have seizures
    sample_size = min(sample_size, len(dataset))
    sample_indices = torch.randperm(len(dataset))[:sample_size]

    # Track which windows actually have seizures
    window_has_seizure = torch.zeros(len(dataset), dtype=torch.float32)
    sampled_seizure_count = 0

    print(f"[SAMPLER] Checking {sample_size} windows for seizures...", flush=True)
    for i, idx in enumerate(sample_indices):
        _, label = dataset[idx.item()]
        if (label > 0).any():
            window_has_seizure[idx] = 1.0
            sampled_seizure_count += 1

        # Progress update every 1000 windows
        if (i + 1) % 1000 == 0:
            print(
                f"[SAMPLER] Checked {i + 1}/{sample_size} windows, found {sampled_seizure_count} with seizures",
                flush=True,
            )

    # Estimate seizure ratio
    seizure_ratio = sampled_seizure_count / sample_size
    print(
        f"[SAMPLER] Final: {sampled_seizure_count}/{sample_size} windows with seizures ({seizure_ratio:.2%})",
        flush=True,
    )

    if seizure_ratio < 1e-8:
        print("[SAMPLER] WARNING: No seizures found in sample! Using uniform sampling.", flush=True)
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

    print(f"[SAMPLER] Seizure ratio: {seizure_ratio:.2%}", flush=True)
    print(f"[SAMPLER] Positive weight: {pos_weight:.2f}", flush=True)
    print(
        f"[SAMPLER] Estimated seizure windows: {(weights > 1).sum().item()}/{len(dataset)}",
        flush=True,
    )

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
        bce = tnf.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=pos_weight
        )
        # Probabilities
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        # Class-balanced alpha
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        # Focal modulation
        mod = (1.0 - p_t).clamp(min=0.0, max=1.0).pow(self.gamma)
        return cast(torch.Tensor, alpha_t * mod * bce)


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
    loss_mode: str = "bce",
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    return_step: bool = False,
    checkpoint_dir: Path | None = None,
    epoch_index: int | None = None,
    mid_epoch_minutes: float | None = None,
    mid_epoch_keep: int = 3,
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
    last_mid_save = time.time()
    mid_interval_s = (
        None if mid_epoch_minutes is None else float(max(0.0, mid_epoch_minutes)) * 60.0
    )

    # Calculate class weights from dataset sample (not just first batch!)
    # Sample a significant portion to get accurate statistics
    print("\n" + "=" * 60, flush=True)
    print("[INIT] DATASET STATISTICS", flush=True)
    print("=" * 60, flush=True)

    dataset = dataloader.dataset
    dataset_len = len(dataset)  # type: ignore[arg-type]

    # Skip expensive sampling in smoke test mode
    is_smoke_test = os.environ.get("BGB_SMOKE_TEST", "0") == "1"
    if is_smoke_test:
        print(
            "[SMOKE TEST MODE] Skipping dataset sampling - using default pos_weight=1.0", flush=True
        )
        pos_weight_val = 1.0
        pos_ratio = 0.5  # Assume balanced for smoke test

    # CRITICAL FIX: BalancedSeizureDataset already knows its exact seizure ratio!
    # No need to sample 1000 windows (which takes 2+ hours on Modal)
    else:
        from src.brain_brr.data.datasets import BalancedSeizureDataset

        if isinstance(dataset, BalancedSeizureDataset):
            # Use the pre-computed ratio from manifest statistics
            pos_ratio = dataset.seizure_ratio
            print("[DATASET] Using BalancedSeizureDataset known distribution", flush=True)
            print(f"[DATASET] Seizure ratio: {100 * pos_ratio:.1f}% (from manifest)", flush=True)
        else:
            # Fallback: sample windows for regular datasets
            sample_size = min(100, dataset_len)  # Reduced from 1000 for speed
            sample_indices = torch.randperm(dataset_len)[:sample_size]

            pos_count = 0
            total_samples = 0

            print(
                f"[DATASET] Sampling {sample_size} windows to estimate distribution...", flush=True
            )
            for idx in sample_indices:
                _, label = dataset[idx.item()]
                if (label > 0).any():
                    pos_count += 1
                total_samples += 1

            pos_ratio = pos_count / total_samples if total_samples > 0 else 1e-8
            print(f"[DATASET] Sampled {sample_size} windows", flush=True)
            print(
                f"[DATASET] Windows with seizures: {pos_count}/{sample_size} ({100 * pos_ratio:.2f}%)",
                flush=True,
            )

        # Use sqrt scaling for extreme imbalance (prevents explosion)
        if pos_ratio > 0 and pos_ratio < 0.5:
            pos_weight_val = math.sqrt((1 - pos_ratio) / pos_ratio)
        else:
            pos_weight_val = 1.0

        print(f"[DATASET] Using pos_weight: {pos_weight_val:.2f} (sqrt scaling)", flush=True)
    print("=" * 60 + "\n", flush=True)

    # Validate dataset has seizures
    if pos_ratio < 0.001:  # Less than 0.1% seizures
        print("\n" + "!" * 60, flush=True)
        print(f"[CRITICAL] Dataset has only {100 * pos_ratio:.4f}% seizures!", flush=True)
        print("[CRITICAL] Model will likely collapse to all-negative predictions.", flush=True)
        print("[CRITICAL] Increase BGB_LIMIT_FILES or use different data split.", flush=True)
        print("!" * 60 + "\n", flush=True)

    # Get first batch for preflight check
    first_batch = next(iter(dataloader))
    _, _first_labels = first_batch

    # AMP-safe, numerically stable loss on logits (build per-element loss fn)
    # (We keep model outputs as probabilities elsewhere for tests/inference)
    pos_weight_t = torch.as_tensor(pos_weight_val, device=device_obj, dtype=torch.float32)
    use_focal = (loss_mode or "bce").lower() == "focal"

    if use_focal:
        focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        # Avoid double-counting class priors: if alpha != 0.5 (class-balanced),
        # do NOT pass pos_weight to BCE term. Alpha<0.5 down-weights positives.
        alpha_diff = abs(float(focal_alpha) - 0.5)
        pass_pos_weight = alpha_diff < 1e-6
        if not pass_pos_weight:
            if focal_alpha < 0.5:
                print(
                    "[WARNING] focal_alpha < 0.5 down-weights positives; ensure this is intended",
                    flush=True,
                )
            print(
                "[INIT] FOCAL: alpha != 0.5 → disabling pos_weight to avoid double-counting",
                flush=True,
            )

        def compute_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            pw = pos_weight_t if pass_pos_weight else None
            return cast(torch.Tensor, focal(x, y, pos_weight=pw))

        print(
            f"[INIT] Using FOCAL loss (alpha={focal_alpha}, gamma={focal_gamma}, "
            f"pos_weight={'on' if pass_pos_weight else 'off'})",
            flush=True,
        )
    else:
        bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight_t)

        def compute_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return cast(torch.Tensor, bce(x, y))

        print("[INIT] Using BCEWithLogits loss", flush=True)

    # GUARDRAILS: Validate critical components before training
    if compute_loss is None or not callable(compute_loss):
        raise TypeError("compute_loss must be a callable")
    if scheduler is not None and not hasattr(scheduler, "step"):
        raise TypeError(f"scheduler must have 'step' method, got {type(scheduler)}")
    if use_amp and not hasattr(scaler, "scale"):
        raise TypeError(f"GradScaler expected for AMP, got {type(scaler)}")

    # PREFLIGHT CHECK: Test one batch to catch errors early
    print("[PREFLIGHT] Testing one batch before training...", flush=True)
    test_windows, test_labels = first_batch
    test_windows = test_windows.to(device_obj)
    test_labels = test_labels.to(device_obj)

    # Handle multi-channel labels for test
    if test_labels.dim() == 3:  # (B, C, T)
        test_labels = test_labels.max(dim=1)[0]  # (B, T)

    model.eval()
    try:
        with torch.no_grad(), autocast(enabled=(use_amp and device == "cuda")):
            test_logits = model(test_windows)  # (B, T) raw logits
            test_loss = compute_loss(test_logits, test_labels)
            if test_loss is None:
                raise ValueError("Loss computation returned None")
            print(
                f"[PREFLIGHT] ✓ Model forward pass OK, loss shape: {test_loss.shape}",
                flush=True,
            )
    except Exception as e:
        print(f"[PREFLIGHT] ✗ Failed on test batch: {e}", flush=True)
        print("[PREFLIGHT] Debug info:", flush=True)
        print(f"  - Model type: {type(model)}", flush=True)
        print(f"  - Input shape: {test_windows.shape}", flush=True)
        print(f"  - Labels shape: {test_labels.shape}", flush=True)
        print(f"  - Loss mode: {loss_mode}", flush=True)
        print(f"  - Device: {device_obj}", flush=True)
        raise
    finally:
        model.train()

    print(f"[TRAIN] Starting epoch with {len(dataloader)} batches", flush=True)
    total_loss = 0.0
    num_batches = 0

    # Robust tqdm handling for Modal/non-TTY environments
    use_tqdm = not os.getenv("BGB_DISABLE_TQDM")
    progress_bar = None  # Initialize to None for cleanup

    if use_tqdm:
        try:
            # Force tqdm to use simple ASCII output for subprocess compatibility
            # Disable dynamic ncols to prevent terminal detection issues
            progress_bar = tqdm(
                dataloader,
                desc="Training",
                leave=False,
                file=sys.stderr,  # Explicitly use stderr
                ascii=True,  # Use ASCII characters only
                ncols=80,  # Fixed width to avoid terminal detection
                disable=None,  # Let tqdm auto-detect if it should disable
            )
            if progress_bar is None or not hasattr(progress_bar, "__iter__"):
                print("[WARNING] tqdm initialization failed, using plain iteration", flush=True)
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
    try:
        for batch_idx, (windows, labels) in enumerate(progress):
            windows = windows.to(device_obj)
            labels = labels.to(device_obj)

            # Handle multi-channel labels: aggregate to any-seizure
            if labels.dim() == 3:  # (B, C, T)
                labels = labels.max(dim=1)[0]  # (B, T)

            optimizer.zero_grad(set_to_none=True)

            # Forward pass with AMP (model returns raw logits)
            with autocast(enabled=(use_amp and device == "cuda")):
                try:
                    logits = model(windows)  # (B, T) raw logits
                    if logits is None:
                        raise ValueError(f"Model returned None for input shape {windows.shape}")
                    per_element_loss = compute_loss(logits, labels)
                    if per_element_loss is None:
                        raise ValueError("Loss computation returned None")
                except Exception as e:
                    print(
                        f"[ERROR] Forward/loss computation failed at batch {batch_idx}:", flush=True
                    )
                    print(f"  - Error: {e}", flush=True)
                    print(f"  - Model: {type(model)}", flush=True)
                    print(f"  - Windows shape: {windows.shape}", flush=True)
                    print(f"  - Labels shape: {labels.shape}", flush=True)
                    print(f"  - Device: {windows.device}", flush=True)
                    raise
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

            # Scheduler step ONLY after a real optimizer step
            if scheduler is not None:
                step_count = getattr(optimizer, "_step_count", 0)
                if isinstance(step_count, int) and step_count > 0:
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

            if (
                checkpoint_dir is not None
                and epoch_index is not None
                and mid_interval_s is not None
                and (time.time() - last_mid_save) >= mid_interval_s
            ):
                mid_path = checkpoint_dir / f"mid_epoch_{epoch_index + 1:03d}_{batch_idx:06d}.pt"
                try:
                    save_checkpoint(
                        model,
                        optimizer,
                        epoch_index,
                        0.0,
                        mid_path,
                        scheduler,
                        None,
                        extra={"batch_idx": batch_idx, "kind": "mid_epoch"},
                    )
                    print(f"[CHECKPOINT] Saved mid-epoch snapshot: {mid_path.name}", flush=True)
                    last_mid_save = time.time()
                    mids = sorted(
                        checkpoint_dir.glob("mid_epoch_*.pt"), key=lambda p: p.stat().st_mtime
                    )
                    if len(mids) > int(max(0, mid_epoch_keep)):
                        for old in mids[: len(mids) - int(mid_epoch_keep)]:
                            with suppress(Exception):
                                old.unlink()
                except Exception as e:
                    print(f"[WARNING] Failed to save mid-epoch checkpoint: {e}", flush=True)

    except Exception as e:
        # Clean up tqdm if it exists
        if progress_bar is not None and hasattr(progress_bar, "close"):
            with suppress(Exception):
                progress_bar.close()
        # Re-raise the actual error with context
        print(f"[ERROR] Training loop failed at batch {num_batches}: {e}", flush=True)
        raise
    finally:
        # Always clean up tqdm progress bar
        if progress_bar is not None and hasattr(progress_bar, "close"):
            with suppress(Exception):
                progress_bar.close()

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
    progress_bar = None  # Initialize for cleanup

    with torch.no_grad():
        if use_tqdm:
            try:
                # Use same safe tqdm settings as training
                progress_bar = tqdm(
                    dataloader,
                    desc="Validating",
                    leave=False,
                    file=sys.stderr,
                    ascii=True,
                    ncols=80,
                    disable=None,
                )
                if progress_bar is None or not hasattr(progress_bar, "__iter__"):
                    print(
                        "[WARNING] tqdm initialization failed in validation, using plain iteration",
                        flush=True,
                    )
                    iterator = dataloader
                else:
                    iterator = progress_bar
            except Exception as e:
                print(
                    f"[WARNING] tqdm failed in validation ({e}), using plain iteration", flush=True
                )
                iterator = dataloader
        else:
            iterator = dataloader

        try:
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
        finally:
            # Clean up tqdm progress bar
            if progress_bar is not None and hasattr(progress_bar, "close"):
                with suppress(Exception):
                    progress_bar.close()

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
    extra: dict[str, Any] | None = None,
) -> None:
    """Save training checkpoint with verification.

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
        "timestamp": time.time(),  # Add timestamp for tracking
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if config is not None:
        checkpoint["config"] = config.model_dump()
    if extra:
        checkpoint.update(extra)

    # Save to temp file first, then rename (atomic operation)
    temp_path = checkpoint_path.with_suffix(".tmp")
    torch.save(checkpoint, temp_path)

    # Verify the checkpoint can be loaded
    try:
        test_ckpt = torch.load(temp_path, map_location="cpu")
        if "model_state_dict" not in test_ckpt:
            raise ValueError("Checkpoint missing model_state_dict")
        temp_path.rename(checkpoint_path)
    except Exception as e:
        print(f"[ERROR] Checkpoint verification failed: {e}", flush=True)
        temp_path.unlink(missing_ok=True)
        raise


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
        print(f"[RESUME] Found mid-epoch checkpoint: {latest_mid.name}", flush=True)
        ckpt = torch.load(latest_mid, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"]
        best_metric = ckpt.get("best_metric", 0.0)
        print(
            f"Resumed from epoch {start_epoch + 1}, batch {ckpt.get('batch_idx', '?')}", flush=True
        )
        # Note: This resumes from start of epoch, not exact batch
    elif (checkpoint_dir / "last.pt").exists() and config.training.resume:
        start_epoch, best_metric = load_checkpoint(
            checkpoint_dir / "last.pt", model, optimizer, scheduler
        )
        print(f"Resumed from epoch {start_epoch + 1}", flush=True)

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
            loss_mode=getattr(config.training, "loss", "bce"),
            focal_alpha=getattr(config.training, "focal_alpha", 0.25),
            focal_gamma=getattr(config.training, "focal_gamma", 2.0),
            return_step=True,
            checkpoint_dir=checkpoint_dir,
            epoch_index=epoch,
            mid_epoch_minutes=(
                float(os.getenv("BGB_MID_EPOCH_MINUTES", "0"))
                if config.training.resume and os.getenv("BGB_MID_EPOCH_MINUTES")
                else (
                    getattr(
                        config.experiment,
                        "mid_epoch_checkpoint_minutes",
                        10.0 if config.training.resume else None,
                    )
                )
            ),
            mid_epoch_keep=int(
                os.getenv(
                    "BGB_MID_EPOCH_KEEP", str(getattr(config.experiment, "mid_epoch_keep", 3))
                )
            ),
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
            print(f"\n⚠️ MODEL COLLAPSE DETECTED! AUROC={val_metrics['auroc']:.3f}", flush=True)
            print("Model is predicting all-negative. Stopping training.", flush=True)
            print("Potential causes:", flush=True)
            print("  1. Dataset has too few seizures (<1%)", flush=True)
            print("  2. Class weighting is insufficient", flush=True)
            print("  3. Learning rate too high/low", flush=True)
            print("\nRecommendations:", flush=True)
            print("  - Increase BGB_LIMIT_FILES to include more seizure files", flush=True)
            print("  - Use focal loss or stronger class weighting", flush=True)
            print("  - Check dataset statistics logged at start", flush=True)
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
            best_metric = current_metric  # FIX: Update best_metric when we find a new best
            best_metrics = {
                "best_epoch": epoch + 1,
                "best_taes": val_metrics["taes"],
                "best_auroc": val_metrics["auroc"],
                f"best_{metric_name}": current_metric,
            }
            print(f"  New best {metric_name}: {current_metric:.4f}", flush=True)

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
            print(f"  Saved periodic checkpoint: {checkpoint_path.name}", flush=True)

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

    print(f"\nTraining complete. Best epoch: {best_metrics['best_epoch']}", flush=True)

    return best_metrics


# ============================================================================
# CLI entry point
# ============================================================================


def main() -> None:
    """CLI entry point for training."""
    import argparse

    from src.brain_brr.data import BalancedSeizureDataset, EEGWindowDataset

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
    is_smoke_test = os.environ.get("BGB_SMOKE_TEST", "0") == "1"
    if is_smoke_test:
        print("\n" + "=" * 60)
        print("SMOKE TEST MODE ACTIVE")
        print("Pipeline validation only - model will NOT learn meaningful patterns")
        print("DO NOT use this for real training!")
        print("=" * 60 + "\n", flush=True)

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

    # Cache directory sanity and preflight
    data_cache_root = Path(config.data.cache_dir)
    exp_cache_root = Path(config.experiment.cache_dir)
    if data_cache_root.resolve() != exp_cache_root.resolve():
        print(
            f"[WARNING] config.data.cache_dir ({data_cache_root}) != config.experiment.cache_dir ({exp_cache_root})",
            flush=True,
        )

    try:
        from src.brain_brr.data.cache_utils import check_cache_completeness

        train_cache = data_cache_root / "train"
        val_cache = data_cache_root / "val"
        st_train = check_cache_completeness(train_files, train_cache)
        st_val = check_cache_completeness(val_files, val_cache)
        if st_train.missing_files > 0 or st_val.missing_files > 0:
            print(
                "[DATA] Cache incomplete: "
                f"train {st_train.cached_files}/{st_train.total_files}, "
                f"val {st_val.cached_files}/{st_val.total_files}",
                flush=True,
            )
            print(
                "[HINT] Pre-build cache to avoid slow training:\n"
                f"  python -m src build-cache --data-dir {config.data.data_dir} --cache-dir {data_cache_root / 'train'}\n"
                f"  python -m src build-cache --data-dir {config.data.data_dir} --cache-dir {data_cache_root / 'val'}",
                flush=True,
            )
    except Exception:
        pass

    train_cache_dir = data_cache_root / "train"
    use_balanced = bool(config.data.use_balanced_sampling)
    manifest_path = train_cache_dir / "manifest.json"

    # Force manifest rebuild if requested or if it exists but is invalid
    if use_balanced and manifest_path.exists():
        import json

        force_rebuild = os.getenv("BGB_FORCE_MANIFEST_REBUILD", "").strip() == "1"
        try:
            with open(manifest_path) as f:
                manifest_data = json.load(f)
            if force_rebuild:
                print(
                    "[DATA] BGB_FORCE_MANIFEST_REBUILD=1 → deleting manifest for rebuild",
                    flush=True,
                )
                manifest_path.unlink()
            else:
                from src.brain_brr.data.cache_utils import validate_manifest

                if not validate_manifest(train_cache_dir, manifest_data):
                    print(
                        "[WARNING] Invalid/stale manifest detected → deleting for rebuild",
                        flush=True,
                    )
                    manifest_path.unlink()
        except Exception as e:
            print(f"[WARNING] Failed to read/validate manifest: {e}, deleting...", flush=True)
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
                print(
                    f"[DATA] Built manifest from {len(existing_cache_files)} cached files",
                    flush=True,
                )
            except Exception as e:
                print(f"[WARNING] Manifest build failed: {e}", flush=True)
        else:
            print("[DATA] Skipping manifest build - cache not yet populated", flush=True)

    # Create training dataset - either balanced (from manifest) or standard
    train_dataset: BalancedSeizureDataset | EEGWindowDataset
    if use_balanced and manifest_path.exists():
        try:
            train_dataset = BalancedSeizureDataset(train_cache_dir)
            print(
                f"[DATASET] BalancedSeizureDataset: {len(train_dataset)} windows from manifest",
                flush=True,
            )
            if len(train_dataset) == 0:
                is_smoke_test = os.environ.get("BGB_SMOKE_TEST", "0") == "1"
                if is_smoke_test:
                    print(
                        "[SMOKE TEST MODE] Balanced manifest empty - will fallback to EEGWindowDataset",
                        flush=True,
                    )
                    raise Exception("Empty manifest in smoke test - triggering fallback")
                else:
                    print("[FATAL] Balanced manifest produced 0 windows", flush=True)
                    import sys

                    sys.exit(1)
        except Exception as e:
            print(f"[WARNING] BalancedSeizureDataset failed: {e}; falling back to EEGWindowDataset")
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

    val_dataset = EEGWindowDataset(
        val_files,
        label_files=val_label_files,
        cache_dir=data_cache_root / "val",
        allow_on_demand=True,
    )

    # CRITICAL FIX: If we just built cache via EEGWindowDataset and manifest doesn't exist,
    # build it now and switch to BalancedSeizureDataset!
    if (
        use_balanced
        and not isinstance(train_dataset, BalancedSeizureDataset)
        and not manifest_path.exists()
    ):
        print("[DATA] Cache built, now creating manifest for balanced sampling...", flush=True)
        try:
            from src.brain_brr.data.cache_utils import scan_existing_cache

            _ = scan_existing_cache(train_cache_dir)
            if manifest_path.exists():
                # Switch to BalancedSeizureDataset now that manifest exists
                train_dataset = BalancedSeizureDataset(train_cache_dir)
                print(
                    f"[DATA] Switched to BalancedSeizureDataset: {len(train_dataset)} windows",
                    flush=True,
                )
        except Exception as e:
            print(f"[WARNING] Post-cache manifest build failed: {e}", flush=True)

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
        print(f"[SAMPLER] Sampling {sample_size} windows to detect seizures...", flush=True)
        train_sampler = create_balanced_sampler(train_dataset, sample_size=sample_size)

        if train_sampler is None:
            # Check if we're in smoke test mode
            is_smoke_test = os.environ.get("BGB_SMOKE_TEST", "0") == "1"

            if is_smoke_test:
                print("=" * 60, flush=True)
                print("[SMOKE TEST MODE] No seizures found - continuing anyway", flush=True)
                print(
                    "[SMOKE TEST MODE] Using uniform sampling for pipeline validation", flush=True
                )
                print("[SMOKE TEST MODE] This model will NOT learn - testing only!", flush=True)
                print("=" * 60, flush=True)
                # Continue with default sampler for smoke testing
            else:
                print("=" * 60, flush=True)
                print(f"[FATAL] No seizures found in {sample_size} windows!", flush=True)
                print("[FATAL] Training will produce a USELESS model!", flush=True)
                print("[FATAL] Check your data or increase sample size!", flush=True)
                print("=" * 60, flush=True)
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
