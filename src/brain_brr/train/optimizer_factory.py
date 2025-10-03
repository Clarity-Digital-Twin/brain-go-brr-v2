"""Optimizer and scheduler factory functions.

Provides configurable creation of optimizers and learning rate schedulers.
"""

from __future__ import annotations

import logging
import math

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
) -> LRScheduler:
    """Create learning rate scheduler.

    Uses a LambdaLR for linear warmup followed by cosine decay.
    Designed to step once per optimization update.
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

        sched = LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=-1)
        # Reset any change at construction time.
        for g, lr in zip(optimizer.param_groups, initial_lrs, strict=False):
            g["lr"] = lr
        return sched
    else:
        raise ValueError(f"Unknown scheduler: {config.type}")
