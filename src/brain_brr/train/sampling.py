"""Balanced sampling for imbalanced datasets.

Creates positive-aware samplers to handle extreme class imbalance in seizure detection.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
from torch.utils.data import WeightedRandomSampler

from src.brain_brr.constants import EPSILON_ZERO_CHECK
from src.brain_brr.utils.env import env

logger = logging.getLogger(__name__)


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

    if seizure_ratio < EPSILON_ZERO_CHECK:
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
