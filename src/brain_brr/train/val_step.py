"""Validation epoch implementation extracted from loop.py.

Single Responsibility: Execute one validation epoch with proper metrics evaluation.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import suppress
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore[import-untyped]

from src.brain_brr.config.schemas import PostprocessingConfig
from src.brain_brr.utils.env import env

logger = logging.getLogger(__name__)


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    post_config: PostprocessingConfig,
    device: str = "cpu",
    fa_rates: list[float] | None = None,
) -> dict[str, Any]:
    """Validate model with streaming per-recording metrics (low memory).

    Args:
        model: SeizureDetector model
        dataloader: Validation DataLoader
        post_config: Post-processing configuration
        device: Device to evaluate on
        fa_rates: FA/24h targets for sensitivity

    Returns:
        Dictionary of metrics
    """
    from collections import defaultdict

    if fa_rates is None:
        fa_rates = [10, 5, 1]

    model.eval()
    device_obj = torch.device(device)
    criterion = nn.BCEWithLogitsLoss()

    recordings: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_loss = 0.0
    num_batches = 0

    n_val_batches = len(dataloader)
    logger.info(f"[VALIDATION] Starting streaming validation with {n_val_batches} batches...")

    use_tqdm = not env.disable_tqdm()
    progress_bar = None

    with torch.no_grad():
        if use_tqdm:
            try:
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
                    logger.warning(
                        "tqdm initialization failed in validation, using plain iteration"
                    )
                    iterator = dataloader
                else:
                    iterator = progress_bar
            except Exception as e:
                logger.warning(f"tqdm failed in validation ({e}), using plain iteration")
                iterator = dataloader
        else:
            iterator = dataloader

        try:
            last_heartbeat = time.time()
            heartbeat_interval = 120

            for batch_idx, batch in enumerate(iterator):
                windows = batch["window"].to(device_obj)
                labels = batch["label"].to(device_obj)
                file_ids = batch["file_id"]
                window_starts = batch["window_start_s"]

                if labels.dim() == 3:
                    labels = labels.max(dim=1)[0]

                logits = model(windows)
                loss = criterion(logits, labels)

                probs = torch.sigmoid(logits)

                for i, fid in enumerate(file_ids):
                    recordings[fid].append(
                        {
                            "start_s": float(window_starts[i]),
                            "probs": probs[i].cpu(),
                            "labels": labels[i].cpu(),
                        }
                    )

                total_loss += loss.item()
                num_batches += 1

                current_time = time.time()
                if current_time - last_heartbeat > heartbeat_interval:
                    avg_loss = total_loss / max(1, num_batches)
                    logger.info(
                        f"[VAL HEARTBEAT] Batch {batch_idx}/{len(dataloader)} | "
                        f"Avg Loss: {avg_loss:.4f} | Recordings: {len(recordings)}"
                    )
                    last_heartbeat = current_time
        finally:
            if progress_bar is not None and hasattr(progress_bar, "close"):
                with suppress(Exception):
                    progress_bar.close()

    logger.info(
        f"[VALIDATION] Completed {num_batches} batches, processing {len(recordings)} recordings..."
    )

    from src.brain_brr.eval.metrics import evaluate_predictions_streaming

    metrics = evaluate_predictions_streaming(
        recordings=recordings,
        fa_rates=fa_rates,
        post_cfg=post_config,
        sampling_rate=256,
    )

    metrics["val_loss"] = total_loss / max(1, num_batches)

    logger.info(f"[VALIDATION] Done! Val Loss: {metrics['val_loss']:.4f}")

    return metrics
