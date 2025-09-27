"""PR-4: Clamp monitoring utilities for gradual retirement.

This module provides monitoring wrappers to track which clamps are
actually needed, enabling systematic removal of redundant interventions.
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

# Track which clamps are safe to remove (populated during monitoring)
SAFE_TO_REMOVE: set[str] = set()

# Clamps that should always be kept for safety
ESSENTIAL_CLAMPS = {
    "tcn_input",  # Data quality
    "detector_output",  # Loss stability
    "detector_decoded",  # Pre-loss guard
    "cosine_similarity",  # Math bounds
    "correlation_similarity",  # Math bounds
    "division_safety_norms",  # Prevent div by zero
    "division_safety_denom",  # Prevent div by zero
}


def monitored_clamp(
    x: torch.Tensor,
    min_val: float | None = None,
    max_val: float | None = None,
    name: str = "unknown",
    config: dict | None = None,
) -> torch.Tensor:
    """Clamp with monitoring to track if it's needed.

    Args:
        x: Input tensor
        min_val: Minimum value (None = no lower bound)
        max_val: Maximum value (None = no upper bound)
        name: Identifier for this clamp location
        config: Clamp retirement configuration

    Returns:
        Clamped or unclamped tensor based on configuration
    """
    if config is None:
        config = {}

    # Check if values would be clamped
    would_clamp_low = False
    would_clamp_high = False
    num_low = 0
    num_high = 0

    if min_val is not None:
        mask_low = x < min_val
        would_clamp_low = bool(mask_low.any().item())
        if would_clamp_low:
            num_low = int(mask_low.sum().item())

    if max_val is not None:
        mask_high = x > max_val
        would_clamp_high = bool(mask_high.any().item())
        if would_clamp_high:
            num_high = int(mask_high.sum().item())

    # Log if monitoring is enabled
    if config.get("log_clamp_hits", False) and (would_clamp_low or would_clamp_high):
        logger.info(
            f"Clamp '{name}' would affect {num_low} low (<{min_val}), "
            f"{num_high} high (>{max_val}) out of {x.numel()} values"
        )

    # Decide whether to actually clamp
    should_remove = config.get("remove_intermediate_clamps", False) and name not in ESSENTIAL_CLAMPS

    if should_remove:
        # Don't clamp, but validate if configured
        if config.get("validate_finite", True) and not torch.isfinite(x).all():
            logger.warning(f"Non-finite values detected at '{name}' with clamps removed!")
        return x
    else:
        # Apply clamp as normal
        return torch.clamp(x, min=min_val, max=max_val)


def safe_clamp_min() -> float:
    """Get minimum clamp value from environment or default."""
    return float(os.environ.get("BGB_SAFE_CLAMP_MIN", "-10.0"))


def safe_clamp_max() -> float:
    """Get maximum clamp value from environment or default."""
    return float(os.environ.get("BGB_SAFE_CLAMP_MAX", "10.0"))


def should_use_safe_clamp() -> bool:
    """Check if safe clamping is enabled via environment."""
    return os.environ.get("BGB_SAFE_CLAMP", "").lower() in ("1", "true", "yes")


def monitored_nan_to_num(
    x: torch.Tensor,
    nan: float = 0.0,
    posinf: float | None = None,
    neginf: float | None = None,
    name: str = "unknown",
    config: dict | None = None,
) -> torch.Tensor:
    """Replace NaN/Inf with monitoring.

    Args:
        x: Input tensor
        nan: Value to replace NaN
        posinf: Value to replace positive infinity
        neginf: Value to replace negative infinity
        name: Identifier for this location
        config: Clamp retirement configuration

    Returns:
        Cleaned tensor or original based on configuration
    """
    if config is None:
        config = {}

    # Check for NaN/Inf
    has_nan = torch.isnan(x).any().item()
    has_inf = torch.isinf(x).any().item()

    if (has_nan or has_inf) and config.get("log_clamp_hits", False):
        num_nan = torch.isnan(x).sum().item()
        num_posinf = (x == float("inf")).sum().item()
        num_neginf = (x == float("-inf")).sum().item()
        logger.info(
            f"nan_to_num '{name}': {num_nan} NaN, {num_posinf} +Inf, "
            f"{num_neginf} -Inf out of {x.numel()} values"
        )

    # Essential nan_to_num locations that should always be kept
    essential_nan_to_num = {"pe_safety", "decoder_output", "final_output"}

    should_remove = config.get("remove_nan_to_num", False) and name not in essential_nan_to_num

    if should_remove:
        # Don't apply nan_to_num, but warn if issues detected
        if has_nan or has_inf:
            logger.warning(f"NaN/Inf detected at '{name}' with nan_to_num removed!")
        return x
    else:
        # Apply nan_to_num as normal
        return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)
