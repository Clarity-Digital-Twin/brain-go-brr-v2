"""Balanced sampling for imbalanced datasets.

Creates positive-aware samplers to handle extreme class imbalance in seizure detection.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
from torch.utils.data import WeightedRandomSampler

from src.brain_brr.constants import BALANCED_SAMPLER_SAMPLE_SIZE, EPSILON_ZERO_CHECK
from src.brain_brr.utils.env import env

logger = logging.getLogger(__name__)


def create_balanced_sampler(
    dataset: Any, sample_size: int = BALANCED_SAMPLER_SAMPLE_SIZE
) -> WeightedRandomSampler | None:
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

    if seizure_ratio < EPSILON_ZERO_CHECK:
        logger.info("[SAMPLER] WARNING: No seizures found in sample! Using uniform sampling.")
        return None

    # Calculate weight for positive samples (sqrt to prevent explosion)
    pos_weight = math.sqrt((1 - seizure_ratio) / seizure_ratio)

    # Assign weights based on VERIFIED seizure windows only
    # CRITICAL FIX: Do NOT randomly assign weights to unsampled windows
    # Previous behavior: randomly picked unsampled windows and labeled them "seizures"
    # This caused 92% false positives (most unsampled windows are background)
    # New behavior: Only assign high weight to windows we KNOW have seizures
    weights = torch.ones(len(dataset), dtype=torch.float32)

    # Known seizure windows get high weight
    weights[window_has_seizure > 0] = pos_weight

    # Unsampled windows keep weight=1.0 (neutral)
    # This is safer than guessing - if we need more accuracy, increase sample_size

    logger.info(f"[SAMPLER] Seizure ratio (from sample): {seizure_ratio:.2%}")
    logger.info(f"[SAMPLER] Positive weight: {pos_weight:.2f}")
    logger.info(f"[SAMPLER] Verified seizure windows: {(weights > 1).sum().item()}/{len(dataset)}")
    logger.info(f"[SAMPLER] Note: Using VERIFIED seizures only (no random extrapolation)")

    return WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=len(weights),
        replacement=True,
        generator=torch.Generator().manual_seed(42),
    )
