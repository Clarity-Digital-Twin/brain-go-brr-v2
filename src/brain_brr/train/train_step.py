"""Training epoch implementation extracted from loop.py.

Single Responsibility: Execute one training epoch with proper logging and checkpointing.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Sized
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

# tqdm has no type stubs (third-party library)
from tqdm import tqdm  # type: ignore[import-untyped]

from src.brain_brr.config.schemas import WarmupScheduleConfig
from src.brain_brr.constants import (
    DATASET_DISTRIBUTION_SAMPLE_SIZE,
    FOCAL_ALPHA_DEFAULT,
    FOCAL_GAMMA_DEFAULT,
    HEARTBEAT_INTERVAL_SEC,
    LOG_EVERY_N_STEPS,
    PERCENTILE_P25,
    PERCENTILE_P50,
    PERCENTILE_P75,
    PERCENTILE_P95,
)
from src.brain_brr.train.checkpoint import save_checkpoint
from src.brain_brr.train.train_utils import get_memory_stats
from src.brain_brr.train.warmup import get_focal_gamma
from src.brain_brr.utils.env import env

logger = logging.getLogger(__name__)


def _sanitize_gradients(
    model: nn.Module,
    logger: logging.Logger,
    batch_idx: int,
) -> int:
    """Sanitize non-finite gradients to zero.

    This is a DEBUGGING TOOL, not core protection. Gradient clipping
    is the primary protection mechanism and is always applied.

    Args:
        model: Model with gradients to sanitize
        logger: Logger for warnings
        batch_idx: Current batch number

    Returns:
        Number of parameters with sanitized gradients

    Note:
        Only called when BGB_SANITIZE_GRADS=1 (debugging mode).
        Not required for normal training.
    """
    sanitized_count = 0

    for param in model.parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            n_nonfinite = (~torch.isfinite(param.grad)).sum().item()
            sanitized_count += 1

            param.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

            if sanitized_count == 1:
                logger.debug(
                    f"[GRAD_SANITIZE] First occurrence at batch {batch_idx}: "
                    f"param_shape={param.shape}, "
                    f"non_finite_count={n_nonfinite}"
                )

    return sanitized_count


def _compute_gradient_stats(model: nn.Module) -> dict[str, float]:
    """Compute gradient statistics for monitoring.

    Cheap summary stats (median, IQR, P95, max) matching current format.

    Args:
        model: Model with gradients

    Returns:
        Dict with gradient statistics
    """
    grad_norms = []
    for param in model.parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.detach().norm().item())

    if not grad_norms:
        return {"median": 0.0, "iqr": 0.0, "p95": 0.0, "max": 0.0}

    import numpy as np

    grad_array = np.array(grad_norms)
    p25, median, p75, p95 = np.percentile(
        grad_array, [PERCENTILE_P25, PERCENTILE_P50, PERCENTILE_P75, PERCENTILE_P95]
    )
    iqr = p75 - p25
    max_norm = float(grad_array.max())

    return {
        "median": float(median),
        "iqr": float(iqr),
        "p95": float(p95),
        "max": max_norm,
    }


def _compute_weight_stats(model: nn.Module) -> dict[str, float]:
    """Compute weight statistics for monitoring.

    Cheap summary stats (median, IQR, P95, max) for weights.

    Args:
        model: Model with parameters

    Returns:
        Dict with weight statistics
    """
    weight_norms = []
    for param in model.parameters():
        if param.requires_grad:
            weight_norms.append(param.detach().norm().item())

    if not weight_norms:
        return {"median": 0.0, "iqr": 0.0, "p95": 0.0, "max": 0.0}

    import numpy as np

    weight_array = np.array(weight_norms)
    p25, median, p75, p95 = np.percentile(
        weight_array, [PERCENTILE_P25, PERCENTILE_P50, PERCENTILE_P75, PERCENTILE_P95]
    )
    iqr = p75 - p25
    max_norm = float(weight_array.max())

    return {
        "median": float(median),
        "iqr": float(iqr),
        "p95": float(p95),
        "max": max_norm,
    }


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
    loss_mode: str = "focal",
    focal_alpha: float = FOCAL_ALPHA_DEFAULT,
    focal_gamma: float = FOCAL_GAMMA_DEFAULT,
    return_step: bool = False,
    checkpoint_dir: Path | None = None,
    epoch_index: int | None = None,
    mid_epoch_minutes: float | None = None,
    mid_epoch_keep: int = 3,
    warmup_schedule: WarmupScheduleConfig | None = None,
    gradient_accumulation_steps: int = 1,
    log_every_n_steps: int = LOG_EVERY_N_STEPS,
    log_gradients: bool = False,
    log_weights: bool = False,
    wandb_logger: Any | None = None,
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
        warmup_schedule: Optional warmup schedule configuration for gradient stabilization

    Returns:
        Average training loss (default) or tuple of (loss, global_step) if return_step=True
    """
    model.train()
    device_obj = torch.device(device)
    scaler = GradScaler(enabled=(use_amp and device == "cuda"))

    supports_training_state = hasattr(model, "set_training_state")
    if supports_training_state:
        try:
            model.set_training_state(global_step, warmup_schedule)
        except TypeError:
            supports_training_state = False

    last_heartbeat = time.time()
    heartbeat_interval = HEARTBEAT_INTERVAL_SEC
    last_mid_save = time.time()
    mid_interval_s = (
        None if mid_epoch_minutes is None else float(max(0.0, mid_epoch_minutes)) * 60.0
    )

    logger.info("\n" + "=" * 60)
    logger.info("[INIT] DATASET STATISTICS")
    logger.info("=" * 60)

    dataset = dataloader.dataset
    dataset_len = len(cast(Sized, dataset))

    is_smoke_test = env.smoke_test()
    if is_smoke_test:
        logger.info("[SMOKE TEST MODE] Skipping dataset sampling")
        pos_ratio = 0.5
    else:
        from src.brain_brr.data.datasets import BalancedSeizureDataset

        if isinstance(dataset, BalancedSeizureDataset):
            pos_ratio = dataset.seizure_ratio
            logger.info("[DATASET] Using BalancedSeizureDataset known distribution")
            logger.info(f"[DATASET] Seizure ratio: {100 * pos_ratio:.1f}% (from manifest)")
        else:
            sample_size = min(DATASET_DISTRIBUTION_SAMPLE_SIZE, dataset_len)
            sample_indices = torch.randperm(dataset_len)[:sample_size]

            pos_count = 0
            total_samples = 0

            logger.info(f"[DATASET] Sampling {sample_size} windows to estimate distribution...")
            for idx in sample_indices:
                batch = dataset[idx.item()]
                label = batch["label"]
                if (label > 0).any():
                    pos_count += 1
                total_samples += 1

            pos_ratio = pos_count / max(total_samples, 1)
            logger.info(f"[DATASET] Estimated seizure ratio: {100 * pos_ratio:.1f}%")

    logger.info("=" * 60 + "\n")
    logger.info(f"[LOSS] Using focal loss (alpha={focal_alpha}, gamma={focal_gamma})")

    gradient_norms = []
    total_loss = 0.0
    num_batches = 0
    accumulation_counter = 0

    use_tqdm = not env.disable_tqdm()
    progress_bar = None

    try:
        if use_tqdm:
            try:
                progress_bar = tqdm(
                    dataloader, desc="Training", leave=False, file=sys.stderr, ascii=True, ncols=80
                )
                if progress_bar is None or not hasattr(progress_bar, "__iter__"):
                    logger.warning("tqdm initialization failed, using plain iteration")
                    progress = dataloader
                else:
                    progress = progress_bar
            except Exception as e:
                logger.warning(f"tqdm failed ({e}), using plain iteration")
                progress = dataloader
        else:
            progress = dataloader

        for batch_idx, batch in enumerate(progress):
            windows = batch["window"].to(device_obj)
            labels = batch["label"].to(device_obj)

            if labels.dim() == 3:
                labels = labels.max(dim=1)[0]

            if accumulation_counter == 0:
                optimizer.zero_grad(set_to_none=True)

            # torch.amp.autocast not in PyTorch type stubs (known issue)
            with torch.amp.autocast(device_type=device, enabled=(use_amp and device == "cuda")):  # type: ignore[attr-defined]
                logits = model(windows)

                probs = torch.sigmoid(logits)
                pt = labels * probs + (1 - labels) * (1 - probs)
                at = labels * focal_alpha + (1 - labels) * (1 - focal_alpha)
                current_gamma = get_focal_gamma(
                    global_step, warmup_schedule, target_gamma=focal_gamma
                )
                focal_weight = at * ((1 - pt) ** current_gamma)
                bce = nn.functional.binary_cross_entropy_with_logits(
                    logits, labels, reduction="none"
                )
                loss = (focal_weight * bce).mean()

                if (
                    warmup_schedule
                    and warmup_schedule.enabled
                    and warmup_schedule.focal_gamma_enabled
                    and batch_idx % 100 == 0
                ):
                    logger.info(f"[WARMUP] Batch {batch_idx} focal_gamma={current_gamma:.3f}")

            raw_loss = loss.detach()
            loss = loss / gradient_accumulation_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulation_counter += 1

            if accumulation_counter >= gradient_accumulation_steps:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)

                if env.sanitize_grads():
                    sanitized_count = _sanitize_gradients(model, logger, batch_idx)
                    if sanitized_count > 0:
                        logger.warning(
                            f"[GRAD_SANITIZE] Replaced {sanitized_count} non-finite gradients "
                            f"at batch {batch_idx} (investigate root cause)"
                        )

                pre_clip_norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

                if (
                    log_every_n_steps
                    and batch_idx % log_every_n_steps == 0
                    and pre_clip_norm > gradient_clip * 2
                ):
                    post_clip_norm = min(float(pre_clip_norm), gradient_clip)
                    logger.debug(
                        f"[GRAD_CLIP] Batch {batch_idx}: "
                        f"pre={pre_clip_norm:.2f} → post={post_clip_norm:.2f} "
                        f"(clipped {(pre_clip_norm - post_clip_norm) / pre_clip_norm * 100:.1f}%)"
                    )

                # Log gradient statistics if enabled
                if log_gradients and log_every_n_steps and batch_idx % log_every_n_steps == 0:
                    grad_stats = _compute_gradient_stats(model)
                    logger.info(
                        f"[GRAD_STATS] Batch {batch_idx}: "
                        f"P50={grad_stats['median']:.2e} | "
                        f"IQR={grad_stats['iqr']:.2e} | "
                        f"P95={grad_stats['p95']:.2e} | "
                        f"Max={grad_stats['max']:.2e}"
                    )
                    if wandb_logger and hasattr(wandb_logger, "enabled") and wandb_logger.enabled:
                        wandb_logger.log(
                            {
                                "gradients/median": grad_stats["median"],
                                "gradients/iqr": grad_stats["iqr"],
                                "gradients/p95": grad_stats["p95"],
                                "gradients/max": grad_stats["max"],
                            },
                            step=global_step,
                        )

                # Log weight statistics if enabled
                if log_weights and log_every_n_steps and batch_idx % log_every_n_steps == 0:
                    weight_stats = _compute_weight_stats(model)
                    logger.info(
                        f"[WEIGHT_STATS] Batch {batch_idx}: "
                        f"P50={weight_stats['median']:.2e} | "
                        f"IQR={weight_stats['iqr']:.2e} | "
                        f"P95={weight_stats['p95']:.2e} | "
                        f"Max={weight_stats['max']:.2e}"
                    )
                    if wandb_logger and hasattr(wandb_logger, "enabled") and wandb_logger.enabled:
                        wandb_logger.log(
                            {
                                "weights/median": weight_stats["median"],
                                "weights/iqr": weight_stats["iqr"],
                                "weights/p95": weight_stats["p95"],
                                "weights/max": weight_stats["max"],
                            },
                            step=global_step,
                        )

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                accumulation_counter = 0
                gradient_norms.append(float(pre_clip_norm))

                if scheduler is not None:
                    scheduler.step()
                    global_step += 1

            if supports_training_state:
                with suppress(Exception):
                    model.set_training_state(global_step, warmup_schedule)

            loss_val = float(raw_loss)

            if torch.isfinite(torch.tensor(loss_val)):
                total_loss += loss_val
                num_batches += 1
            else:
                if log_every_n_steps and batch_idx % log_every_n_steps == 0:
                    logger.warning(
                        f"Non-finite loss detected at batch {batch_idx}, skipping in average"
                    )

            if use_tqdm and hasattr(progress, "set_postfix"):
                if not torch.isfinite(torch.tensor(loss_val)):
                    progress.set_postfix({"loss": "NaN"})
                else:
                    progress.set_postfix({"loss": f"{loss_val:.4f}"})

            if log_every_n_steps and batch_idx > 0 and batch_idx % log_every_n_steps == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                if not torch.isfinite(torch.tensor(loss_val)):
                    logger.info(
                        f"[PROGRESS] Batch {batch_idx}/{len(dataloader)} | "
                        f"Loss: nan | LR: {current_lr:.2e}"
                    )
                else:
                    logger.info(
                        f"[PROGRESS] Batch {batch_idx}/{len(dataloader)} | "
                        f"Loss: {loss_val:.4f} | LR: {current_lr:.2e}"
                    )

            if time.time() - last_heartbeat > heartbeat_interval:
                if num_batches > 0:
                    avg_loss = total_loss / num_batches
                    logger.info(
                        f"[HEARTBEAT] Still training... Batch {batch_idx}/{len(dataloader)} | "
                        f"Avg Loss: {avg_loss:.4f}"
                    )
                else:
                    logger.info(
                        f"[HEARTBEAT] Still training... Batch {batch_idx}/{len(dataloader)} | "
                        f"Avg Loss: N/A (all NaN)"
                    )
                last_heartbeat = time.time()

                mem_stats = get_memory_stats()
                if mem_stats:
                    mem_log = "[MEMORY]"
                    if "gpu_allocated_gb" in mem_stats:
                        mem_log += f" GPU: {mem_stats['gpu_allocated_gb']:.2f}GB alloc"
                        mem_log += f" / {mem_stats['gpu_reserved_gb']:.2f}GB res"
                    if "ram_used_gb" in mem_stats:
                        mem_log += f" | RAM: {mem_stats['ram_used_gb']:.2f}GB used"
                        mem_log += f" / {mem_stats['ram_available_gb']:.2f}GB avail"
                    if "swap_used_gb" in mem_stats and mem_stats["swap_used_gb"] > 0.1:
                        mem_log += f" | SWAP: {mem_stats['swap_used_gb']:.2f}GB"
                    logger.info(mem_log)

                if len(gradient_norms) > 10:
                    sorted_norms = sorted(gradient_norms)
                    n = len(sorted_norms)

                    finite_norms = [x for x in sorted_norms if torch.isfinite(torch.tensor(x))]

                    if len(finite_norms) > 0:
                        grad_p50 = finite_norms[len(finite_norms) // 2]
                        grad_p25 = finite_norms[int(len(finite_norms) * 0.25)]
                        grad_p75 = finite_norms[int(len(finite_norms) * 0.75)]
                        grad_p95 = finite_norms[int(len(finite_norms) * 0.95)]
                        grad_max = finite_norms[-1]
                        grad_iqr = grad_p75 - grad_p25

                        logger.info(
                            f"[GRADIENTS] Last {n} batches: "
                            f"P50={grad_p50:.2f} | IQR={grad_iqr:.2f} | "
                            f"P95={grad_p95:.2f} | Max={grad_max:.2f}"
                        )

                        n_inf = n - len(finite_norms)
                        if n_inf > 0:
                            logger.info(
                                f"[GRADIENTS] {n_inf}/{n} batches had inf pre-clip norm "
                                f"(normal with FP16, clipping handles it)"
                            )
                    else:
                        logger.warning(
                            f"[GRADIENTS] All {n} batches had inf pre-clip norm "
                            f"(verify gradient clipping is working)"
                        )

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
                    logger.info(f"[CHECKPOINT] Saved mid-epoch snapshot: {mid_path.name}")
                    last_mid_save = time.time()
                    mids = sorted(
                        checkpoint_dir.glob("mid_epoch_*.pt"), key=lambda p: p.stat().st_mtime
                    )
                    if len(mids) > int(max(0, mid_epoch_keep)):
                        for old in mids[: len(mids) - int(mid_epoch_keep)]:
                            with suppress(Exception):
                                old.unlink()
                except Exception as e:
                    logger.info(f"[WARNING] Failed to save mid-epoch checkpoint: {e}")

        if accumulation_counter > 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)

            if env.sanitize_grads():
                sanitized_count = _sanitize_gradients(model, logger, batch_idx)
                if sanitized_count > 0:
                    logger.warning(
                        f"[GRAD_SANITIZE] Replaced {sanitized_count} non-finite gradients "
                        f"at batch {batch_idx} (investigate root cause)"
                    )

            pre_clip_norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            gradient_norms.append(float(pre_clip_norm))

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if scheduler is not None:
                scheduler.step()
                global_step += 1

            logger.info(
                f"[GRAD_ACCUM] Flushed {accumulation_counter} leftover microbatch(es) at epoch end"
            )

            optimizer.zero_grad(set_to_none=True)
            accumulation_counter = 0

    except Exception as e:
        if progress_bar is not None and hasattr(progress_bar, "close"):
            with suppress(Exception):
                progress_bar.close()
        logger.info(f"[ERROR] Training loop failed at batch {num_batches}: {e}")
        raise
    finally:
        if progress_bar is not None and hasattr(progress_bar, "close"):
            with suppress(Exception):
                progress_bar.close()

    avg_loss = total_loss / max(1, num_batches)
    return (avg_loss, global_step) if return_step else avg_loss
