"""Scalar metric reducers - AUROC, ECE, TAES computation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.brain_brr.eval.metrics import calculate_ece, calculate_taes


@dataclass
class ScalarMetrics:
    """Container for scalar evaluation metrics."""

    taes: float
    auroc: float
    pr_auc: float
    ece: float


def compute_probability_metrics(probs: torch.Tensor, labels: torch.Tensor) -> ScalarMetrics:
    """Compute sample-level probability metrics (AUROC, PR-AUC, ECE).

    Args:
        probs: (N, T) probabilities in [0, 1]
        labels: (N, T) binary labels {0, 1}

    Returns:
        ScalarMetrics with AUROC, PR-AUC, ECE (TAES set to 0.0)

    Notes:
        - AUROC: Area under ROC curve (sample-level)
        - PR-AUC: Area under precision-recall curve
        - ECE: Expected calibration error with 10 bins
        - TAES is left as 0.0 (computed separately at event-level)
    """
    probs_flat = probs.cpu().numpy().flatten()
    labels_flat = labels.cpu().numpy().flatten()

    if np.unique(labels_flat).size < 2:
        auroc = 0.5
    else:
        auroc = float(roc_auc_score(labels_flat, probs_flat))

    if (labels_flat == 1).sum() == 0:
        pr_auc = 0.0
    else:
        pr_auc = float(average_precision_score(labels_flat, probs_flat))

    ece = calculate_ece(probs_flat, labels_flat, n_bins=10)

    return ScalarMetrics(
        taes=0.0,
        auroc=auroc,
        pr_auc=pr_auc,
        ece=ece,
    )


def compute_event_taes(
    pred_events: list[tuple[float, float]],
    ref_events: list[tuple[float, float]],
    alpha: float = 0.15,
) -> float:
    """Compute Time-Aligned Event Scoring (TAES).

    Wrapper around calculate_taes for consistency with other metric helpers.

    Args:
        pred_events: List of (start_s, end_s) predicted events
        ref_events: List of (start_s, end_s) reference events
        alpha: False alarm penalty weight (default 0.15)

    Returns:
        TAES score in [0, 1]

    Notes:
        TAES measures overlap between predicted and reference events,
        with a penalty for false alarms based on their total duration.
    """
    return calculate_taes(pred_events, ref_events, alpha) if ref_events else 0.0
