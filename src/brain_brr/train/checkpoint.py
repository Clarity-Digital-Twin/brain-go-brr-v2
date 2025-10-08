"""Checkpoint saving and loading utilities.

Enhanced with atomic saves and full state capture for bulletproof resume.
"""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.brain_brr.config.schemas import Config

logger = logging.getLogger(__name__)

CHECKPOINT_VERSION = "3.9.0"  # Schema version for compatibility tracking


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    best_metric: float,
    checkpoint_path: Path,
    scheduler: LRScheduler | None = None,
    config: Config | None = None,
    scaler: GradScaler | None = None,
    save_rng: bool = True,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save training checkpoint with atomic writes and full state capture.

    Enhanced for bulletproof resume:
    - Atomic writes (temp + fsync + rename) to prevent corruption
    - Full state capture (scaler + RNG) for deterministic resume
    - Backward compatible with old checkpoints

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        best_metric: Best metric value
        checkpoint_path: Where to save
        scheduler: Optional scheduler state
        config: Optional config to save
        scaler: Optional AMP grad scaler (CRITICAL for FP16 training)
        save_rng: Whether to save RNG states for reproducibility
        extra: Extra metadata to include
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
        "timestamp": time.time(),
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if config is not None:
        checkpoint["config"] = config.model_dump()

    # AMP scaler (CRITICAL for FP16 training resume)
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    # RNG states for deterministic resume
    if save_rng:
        checkpoint["rng_state"] = {
            "torch": torch.get_rng_state(),
            "torch_cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }

    if extra:
        checkpoint.update(extra)

    # Atomic save: temp file + fsync + rename
    temp_path = checkpoint_path.with_suffix(".tmp")

    try:
        # Save to temp file
        with open(temp_path, "wb") as f:
            torch.save(checkpoint, f)
            f.flush()
            os.fsync(f.fileno())  # Force write to disk

        # Verify checkpoint integrity
        test_ckpt = torch.load(temp_path, map_location="cpu", weights_only=False)
        if "model_state_dict" not in test_ckpt:
            raise ValueError("Checkpoint missing model_state_dict")

        # Atomic rename (POSIX guarantees atomicity)
        os.replace(str(temp_path), str(checkpoint_path))

        logger.debug(
            f"[CHECKPOINT] Saved atomically: {checkpoint_path.name} "
            f"(scaler={'yes' if scaler else 'no'}, rng={'yes' if save_rng else 'no'})"
        )

    except Exception as e:
        logger.error(f"[CHECKPOINT] Save failed: {e}")
        temp_path.unlink(missing_ok=True)
        raise


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    scaler: GradScaler | None = None,
    restore_rng: bool = True,
    device: str = "cpu",
) -> tuple[int, float]:
    """Load training checkpoint with full state restoration.

    Enhanced for bulletproof resume:
    - Restores scaler state for FP16 training
    - Restores RNG states for deterministic batches
    - Backward compatible with old checkpoints

    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load into
        optimizer: Optional optimizer to restore
        scheduler: Optional scheduler to restore
        scaler: Optional AMP grad scaler to restore
        restore_rng: Whether to restore RNG states
        device: Device to map checkpoint to

    Returns:
        (epoch, best_metric)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Check version for compatibility warnings
    version = checkpoint.get("version", "unknown")
    if version != CHECKPOINT_VERSION:
        logger.warning(
            f"[CHECKPOINT] Loading checkpoint version {version}, "
            f"current version is {CHECKPOINT_VERSION}"
        )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Restore AMP scaler (CRITICAL for FP16 training resume)
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        logger.debug("[CHECKPOINT] Restored AMP scaler state")
    elif scaler is not None:
        logger.warning("[CHECKPOINT] No scaler state in checkpoint (old checkpoint?)")

    # Restore RNG states for deterministic resume
    if restore_rng and "rng_state" in checkpoint:
        rng = checkpoint["rng_state"]
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng["torch_cuda"] is not None:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        logger.debug("[CHECKPOINT] Restored RNG states for deterministic resume")
    elif restore_rng:
        logger.warning("[CHECKPOINT] No RNG state in checkpoint (old checkpoint?)")

    epoch = checkpoint["epoch"]
    best_metric = checkpoint.get("best_metric", float("inf"))

    return epoch, best_metric
