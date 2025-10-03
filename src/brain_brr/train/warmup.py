"""Warmup schedule utilities for gradient stabilization.

Production ML best practice: Gradually increase loss sensitivity during early training.
"""

from __future__ import annotations

from src.brain_brr.config.schemas import WarmupScheduleConfig


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
