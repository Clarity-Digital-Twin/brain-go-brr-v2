"""Loss functions for seizure detection.

Implements focal loss for handling class imbalance.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as tnf


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
        # Clamp logits to prevent overflow in BCE computation
        logits_clamped = logits.clamp(min=-100, max=100)
        bce = tnf.binary_cross_entropy_with_logits(
            logits_clamped, targets, reduction="none", pos_weight=pos_weight
        )
        # Probabilities (use clamped logits for numerical stability)
        p = torch.sigmoid(logits_clamped)
        # Critical: Clamp probabilities to avoid log(0) or log(1) issues
        p = p.clamp(min=1e-6, max=1 - 1e-6)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        # Class-balanced alpha
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        # Focal modulation with numerical stability
        # Clamp p_t away from 1 to prevent (1-p_t)^gamma from underflowing to 0
        p_t_stable = p_t.clamp(min=1e-7, max=1 - 1e-7)
        mod = (1.0 - p_t_stable).pow(self.gamma)
        focal_loss = alpha_t * mod * bce

        # Additional safety: clamp output to prevent extreme values
        focal_loss = focal_loss.clamp(max=100.0)  # Prevent loss explosion
        return cast(torch.Tensor, focal_loss)
