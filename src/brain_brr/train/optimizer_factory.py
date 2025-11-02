"""Optimizer and scheduler factory functions.

Provides configurable creation of optimizers and learning rate schedulers.
"""

from __future__ import annotations

import logging
import math
import warnings

import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from src.brain_brr.config.schemas import SchedulerConfig, TrainingConfig
from src.brain_brr.constants import EPSILON_ADAMW

logger = logging.getLogger(__name__)


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

        return AdamW(param_groups, lr=config.learning_rate, betas=(0.9, 0.999), eps=EPSILON_ADAMW)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")


def create_scheduler(
    optimizer: Optimizer,
    config: SchedulerConfig,
    total_steps: int,
    steps_per_epoch: int | None = None,
) -> LRScheduler:
    """Create learning rate scheduler.

    Supports two modes:
    - cosine: Linear warmup followed by cosine decay to 0
    - cosine_restarts: Linear warmup followed by SGDR (Stochastic Gradient Descent with Warm Restarts)

    Both step once per optimization update.

    Args:
        optimizer: Optimizer being scheduled.
        config: Scheduler configuration.
        total_steps: Total number of optimizer steps planned.
        steps_per_epoch: Optimizer steps per epoch (required for cosine_restarts).
    """
    warmup_steps = max(1, int(config.warmup_ratio * total_steps))

    if config.type == "cosine":
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

        # Suppress PyTorch 1.1.0+ warning about scheduler.step() order
        # Our code correctly calls optimizer.step() before scheduler.step()
        # but PyTorch emits warning on scheduler creation before first training step
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Detected call of.*lr_scheduler")
            sched = LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=-1)
        # Reset any change at construction time.
        for g, lr in zip(optimizer.param_groups, initial_lrs, strict=False):
            g["lr"] = lr
        return sched

    if config.type == "cosine_restarts":
        # SGDR (Stochastic Gradient Descent with Warm Restarts)
        # References:
        # - Loshchilov & Hutter (2017): "SGDR: Stochastic Gradient Descent with Warm Restarts"
        # - https://arxiv.org/abs/1608.03983
        if config.t_initial is None:
            raise ValueError("t_initial is required for cosine_restarts scheduler")
        if steps_per_epoch is None or steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be provided and > 0 for cosine_restarts")

        # Convert cycle length (in epochs) to optimizer steps
        t_0_steps = max(1, int(config.t_initial * steps_per_epoch))
        initial_lrs = [g["lr"] for g in optimizer.param_groups]
        base_lr = initial_lrs[0] if initial_lrs else 1.0
        eta_min_value = max(0.0, config.eta_min if config.eta_min is not None else 0.0)
        # Convert absolute eta_min to multiplier relative to base_lr
        eta_min_ratio = min(eta_min_value / base_lr, 1.0) if base_lr > 0 else 0.0
        t_mult = config.t_mult if config.t_mult is not None else 1

        # For SGDR, we need to handle warmup separately then apply restarts
        # We'll use a custom lambda that does warmup first, then delegates to SGDR logic
        def lr_lambda(step: int) -> float:
            # Linear warmup
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)

            # After warmup, apply SGDR manually
            # This mimics CosineAnnealingWarmRestarts behavior
            post_warmup_step = step - warmup_steps
            t_cur = post_warmup_step
            t_i = t_0_steps

            # Find which cycle we're in
            while t_cur >= t_i:
                t_cur -= t_i
                t_i *= t_mult

            # Cosine annealing within current cycle
            progress = t_cur / t_i
            cosine_factor = (
                eta_min_ratio + (1.0 - eta_min_ratio) * (1.0 + math.cos(math.pi * progress)) / 2.0
            )
            return cosine_factor

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Detected call of.*lr_scheduler")
            sched = LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=-1)

        for g, lr in zip(optimizer.param_groups, initial_lrs, strict=False):
            g["lr"] = lr

        logger.info(
            "[SCHEDULER] SGDR with warmup: %s steps warmup, T_0=%s steps, "
            "T_mult=%s, eta_min=%s (ratio %.4f)",
            warmup_steps,
            t_0_steps,
            t_mult,
            eta_min_value,
            eta_min_ratio,
        )
        return sched

    raise ValueError(f"Unknown scheduler: {config.type}")
