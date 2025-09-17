"""Evaluation metrics for seizure detection (TAES, FA/24h, sensitivity@FA)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy import ndimage

from src.experiment.schemas import PostprocessingConfig


def overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Return intersection length in seconds between [a0,a1] and [b0,b1]."""
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def calculate_taes(
    pred_events: list[tuple[float, float]],
    ref_events: list[tuple[float, float]],
    alpha: float = 0.15,
) -> float:
    """Calculate Time-Aligned Event Scoring (TAES).

    For each reference event r:
    - Compute overlap with all predicted events
    - Score = overlap_duration / ref_duration (capped at 1)

    False alarm penalty:
    - For predicted events with no overlap, accumulate duration
    - Penalty = alpha * (fp_duration / total_pred_duration)

    Args:
        pred_events: List of (start_s, end_s) predicted events
        ref_events: List of (start_s, end_s) reference events
        alpha: False alarm penalty weight (default 0.15)

    Returns:
        TAES score in [0, 1]
    """
    if not ref_events:
        return 0.0

    # Score each reference event
    per_ref_scores = []
    for ref_start, ref_end in ref_events:
        ref_dur = max(0.0, ref_end - ref_start)
        if ref_dur < 1e-8:
            continue

        # Total overlap with all predictions
        total_overlap = sum(
            overlap((ref_start, ref_end), (pred_start, pred_end))
            for pred_start, pred_end in pred_events
        )
        score = min(1.0, total_overlap / ref_dur)
        per_ref_scores.append(score)

    if not per_ref_scores:
        return 0.0

    # False alarm penalty
    fp_duration = 0.0
    for pred_start, pred_end in pred_events:
        # Check if this prediction overlaps any reference
        has_overlap = any(
            overlap((pred_start, pred_end), (ref_start, ref_end)) > 0
            for ref_start, ref_end in ref_events
        )
        if not has_overlap:
            fp_duration += max(0.0, pred_end - pred_start)

    total_pred_duration = sum(
        max(0.0, pred_end - pred_start) for pred_start, pred_end in pred_events
    )

    base_score = sum(per_ref_scores) / len(per_ref_scores)
    penalty = alpha * (fp_duration / max(total_pred_duration, 1e-8)) if total_pred_duration > 0 else 0
    taes = base_score - penalty

    return float(max(0.0, min(1.0, taes)))


def fa_per_24h(
    pred_events: list[list[tuple[float, float]]],
    ref_events: list[list[tuple[float, float]]],
    total_hours: float,
) -> float:
    """Calculate false alarms per 24 hours.

    Args:
        pred_events: Per-record predicted events (seconds)
        ref_events: Per-record reference events (seconds)
        total_hours: Total duration in hours

    Returns:
        False alarms per 24 hours
    """
    if total_hours < 1e-8:
        return 0.0

    fa_count = 0
    for preds, refs in zip(pred_events, ref_events, strict=False):
        for pred_start, pred_end in preds:
            # Check if this prediction overlaps any reference
            has_overlap = any(
                overlap((pred_start, pred_end), (ref_start, ref_end)) > 0
                for ref_start, ref_end in refs
            )
            if not has_overlap:
                fa_count += 1

    return (fa_count / total_hours) * 24.0


def batch_masks_to_events(
    masks: torch.Tensor, fs: int
) -> list[list[tuple[float, float]]]:
    """Convert binary masks to event intervals.

    Args:
        masks: (N, T) binary masks
        fs: Sampling rate (Hz)

    Returns:
        List of N lists, each containing (start_s, end_s) tuples
    """
    batch_events = []

    for mask in masks:
        mask_np = mask.cpu().numpy()

        # Find connected components (seizure regions)
        labeled, num_features = ndimage.label(mask_np)

        events = []
        for i in range(1, num_features + 1):
            indices = np.where(labeled == i)[0]
            if len(indices) > 0:
                start_s = float(indices[0]) / fs
                end_s = float(indices[-1] + 1) / fs
                events.append((start_s, end_s))

        batch_events.append(events)

    return batch_events


def batch_probs_to_events(
    probs: torch.Tensor,
    post_cfg: PostprocessingConfig,
    fs: int,
    threshold: float,
) -> list[list[tuple[float, float]]]:
    """Apply post-processing and convert to events.

    Args:
        probs: (N, T) probabilities in [0,1]
        post_cfg: Post-processing configuration
        fs: Sampling rate (Hz)
        threshold: Detection threshold

    Returns:
        List of N lists, each containing (start_s, end_s) tuples
    """
    batch_events = []

    for prob in probs:
        prob_np = prob.cpu().numpy()

        # Apply threshold
        binary = (prob_np > threshold).astype(float)

        # Apply hysteresis (two-threshold smoothing)
        tau_on = post_cfg.hysteresis.tau_on
        tau_off = post_cfg.hysteresis.tau_off

        hysteresis_mask = np.zeros_like(binary)
        in_event = False

        for i in range(len(prob_np)):
            if not in_event and prob_np[i] > tau_on:
                in_event = True
                hysteresis_mask[i] = 1
            elif in_event:
                if prob_np[i] < tau_off:
                    in_event = False
                else:
                    hysteresis_mask[i] = 1

        # Apply morphology
        kernel_size = post_cfg.morphology.get("kernel_size", 5)
        if kernel_size > 0:
            kernel = np.ones(kernel_size)
            if post_cfg.morphology.get("operation", "closing") == "closing":
                # Closing: dilation followed by erosion
                hysteresis_mask = ndimage.binary_dilation(hysteresis_mask, kernel).astype(float)
                hysteresis_mask = ndimage.binary_erosion(hysteresis_mask, kernel).astype(float)
            elif post_cfg.morphology.get("operation", "closing") == "opening":
                # Opening: erosion followed by dilation
                hysteresis_mask = ndimage.binary_erosion(hysteresis_mask, kernel).astype(float)
                hysteresis_mask = ndimage.binary_dilation(hysteresis_mask, kernel).astype(float)

        # Convert to events
        labeled, num_features = ndimage.label(hysteresis_mask)
        events = []

        for i in range(1, num_features + 1):
            indices = np.where(labeled == i)[0]
            if len(indices) > 0:
                start_s = float(indices[0]) / fs
                end_s = float(indices[-1] + 1) / fs
                duration = end_s - start_s

                # Apply minimum duration filter
                if duration >= post_cfg.min_duration:
                    events.append((start_s, end_s))

        batch_events.append(events)

    return batch_events


def find_threshold_for_fa_eventized(
    probs: torch.Tensor,
    post_cfg: PostprocessingConfig,
    ref_events: list[list[tuple[float, float]]],
    fa_target: float,
    total_hours: float,
    fs: int,
    max_iters: int = 20,
) -> float:
    """Binary search for threshold meeting FA target.

    Args:
        probs: (N, T) probabilities
        post_cfg: Post-processing configuration
        ref_events: Reference events for FA calculation
        fa_target: Target FA/24h rate
        total_hours: Total duration in hours
        fs: Sampling rate
        max_iters: Maximum iterations for binary search

    Returns:
        Threshold that meets FA target (conservative)
    """
    low, high = 0.0, 1.0
    best_threshold = 0.5

    for _ in range(max_iters):
        mid = (low + high) / 2
        pred_events = batch_probs_to_events(probs, post_cfg, fs, threshold=mid)
        fa_rate = fa_per_24h(pred_events, ref_events, total_hours)

        if fa_rate > fa_target:
            # Too many FA, increase threshold
            low = mid
        else:
            # At or below target, can go lower
            best_threshold = mid
            high = mid

        if abs(high - low) < 1e-4:
            break

    return best_threshold


def sensitivity_at_fa_rates(
    probs: torch.Tensor,
    labels: torch.Tensor,
    fa_targets: list[float],
    post_cfg: PostprocessingConfig,
    sampling_rate: int = 256,
) -> dict[str, float]:
    """Calculate sensitivity at specific FA/24h targets.

    Args:
        probs: (N, T) probabilities
        labels: (N, T) binary labels
        fa_targets: List of FA/24h targets (e.g., [10, 5, 1])
        post_cfg: Post-processing configuration
        sampling_rate: Sampling rate (Hz)

    Returns:
        Dict with sensitivity_at_Xfa keys
    """
    results = {}

    # Convert labels to events once
    ref_events = batch_masks_to_events(labels, sampling_rate)
    total_hours = labels.numel() / (sampling_rate * 3600)

    for fa_target in fa_targets:
        # Find threshold for this FA target
        threshold = find_threshold_for_fa_eventized(
            probs, post_cfg, ref_events, fa_target, total_hours, sampling_rate
        )

        # Get predictions at this threshold
        pred_events = batch_probs_to_events(
            probs, post_cfg, sampling_rate, threshold=threshold
        )

        # Calculate sensitivity (event-level)
        tp_count = 0
        total_ref_events = 0

        for refs, preds in zip(ref_events, pred_events, strict=False):
            total_ref_events += len(refs)
            for ref_start, ref_end in refs:
                # Check if any prediction overlaps this reference
                has_overlap = any(
                    overlap((ref_start, ref_end), (pred_start, pred_end)) > 0
                    for pred_start, pred_end in preds
                )
                if has_overlap:
                    tp_count += 1

        sensitivity = tp_count / max(total_ref_events, 1)
        results[f"sensitivity_at_{fa_target}fa"] = float(sensitivity)

    return results


def evaluate_predictions(
    probs: torch.Tensor,
    labels: torch.Tensor,
    fa_rates: list[float],
    post_cfg: PostprocessingConfig,
    sampling_rate: int = 256,
) -> dict[str, Any]:
    """Complete evaluation of predictions.

    Args:
        probs: (N, T) probabilities
        labels: (N, T) binary labels
        fa_rates: FA/24h targets for sensitivity calculation
        post_cfg: Post-processing configuration
        sampling_rate: Sampling rate (Hz)

    Returns:
        Dict with all metrics
    """
    # Convert to events for TAES
    ref_events = batch_masks_to_events(labels, sampling_rate)

    # Use default threshold for TAES (0.5)
    pred_events_taes = batch_probs_to_events(
        probs, post_cfg, sampling_rate, threshold=0.5
    )

    # Flatten events for TAES calculation
    all_ref = [evt for record in ref_events for evt in record]
    all_pred = [evt for record in pred_events_taes for evt in record]

    taes = calculate_taes(all_pred, all_ref) if all_ref else 0.0

    # AUROC (sample-level)
    probs_flat = probs.cpu().numpy().flatten()
    labels_flat = labels.cpu().numpy().flatten()

    try:
        auroc = float(roc_auc_score(labels_flat, probs_flat))
    except ValueError:
        # Handle case with single class
        auroc = 0.5

    # Sensitivity at FA rates
    sensitivity_results = sensitivity_at_fa_rates(
        probs, labels, fa_rates, post_cfg, sampling_rate
    )

    # Generate FA curve (10 points)
    fa_curve = []
    for fa in [0.5, 1, 2.5, 5, 10, 20, 50, 100]:
        sens = sensitivity_at_fa_rates(
            probs, labels, [fa], post_cfg, sampling_rate
        ).get(f"sensitivity_at_{fa}fa", 0.0)
        fa_curve.append((fa, sens))

    results = {
        "taes": taes,
        "auroc": auroc,
        "fa_curve": fa_curve,
    }
    results.update(sensitivity_results)

    return results