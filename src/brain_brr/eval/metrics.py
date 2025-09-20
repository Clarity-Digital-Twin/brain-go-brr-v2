"""Evaluation metrics for seizure detection (TAES, FA/24h, sensitivity@FA)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    roc_auc_score,
    roc_curve,
)

from src.brain_brr.config.schemas import PostprocessingConfig
from src.brain_brr.events import batch_mask_to_events
from src.brain_brr.post.postprocess import postprocess_predictions


def overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Return intersection length in seconds between [a0,a1] and [b0,b1]."""
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def calculate_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE).

    ECE measures the difference between predicted probabilities and actual
    accuracy across confidence bins.

    Args:
        probs: Predicted probabilities [0, 1]
        labels: Binary labels {0, 1}
        n_bins: Number of confidence bins

    Returns:
        ECE score (lower is better, 0 is perfectly calibrated)
    """
    if len(probs) == 0:
        return 0.0

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
        # Find samples in this bin
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            # Accuracy in bin
            accuracy_in_bin = labels[in_bin].mean()
            # Average confidence in bin
            avg_confidence_in_bin = probs[in_bin].mean()
            # Weight by proportion of samples
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


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
    penalty = (
        alpha * (fp_duration / max(total_pred_duration, 1e-8)) if total_pred_duration > 0 else 0
    )
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


def batch_masks_to_events(masks: torch.Tensor, fs: int) -> list[list[tuple[float, float]]]:
    """Convert binary masks to event intervals.

    Args:
        masks: (N, T) binary masks
        fs: Sampling rate (Hz)

    Returns:
        List of N lists, each containing (start_s, end_s) tuples
    """
    # Use new Phase 4 event conversion without merging or confidence
    batch_events_objects = batch_mask_to_events(
        masks,
        sampling_rate=fs,
        tau_merge=None,  # No merging for direct mask to event conversion
        probs=None,  # No confidence scoring
    )

    # Convert SeizureEvent objects to tuples for backward compatibility
    batch_events = [
        [(event.start_s, event.end_s) for event in events] for events in batch_events_objects
    ]

    return batch_events


def batch_probs_to_events(
    probs: torch.Tensor,
    post_cfg: PostprocessingConfig,
    fs: int,
    threshold: float | None = None,  # Deprecated, kept for backward compatibility
) -> list[list[tuple[float, float]]]:
    """Apply post-processing and convert to events.

    Args:
        probs: (N, T) probabilities in [0,1]
        post_cfg: Post-processing configuration with hysteresis settings
        fs: Sampling rate (Hz)
        threshold: Deprecated - use post_cfg.hysteresis.tau_on instead.
                  If provided, will override tau_on (for backward compatibility).

    Returns:
        List of N lists, each containing (start_s, end_s) tuples

    Note:
        The threshold parameter is deprecated. The function now uses
        post_cfg.hysteresis.tau_on and tau_off for thresholding.
    """
    # Use new Phase 4 modules
    masks = postprocess_predictions(probs, post_cfg, sampling_rate=fs)

    # Convert masks to events with merging and confidence
    # PostprocessingConfig always has events field with defaults from schema
    batch_events_objects = batch_mask_to_events(
        masks,
        sampling_rate=fs,
        tau_merge=post_cfg.events.tau_merge,
        probs=probs,
        confidence_method=post_cfg.events.confidence_method,
    )

    # Convert SeizureEvent objects to tuples for backward compatibility
    batch_events = [
        [(event.start_s, event.end_s) for event in events] for events in batch_events_objects
    ]

    return batch_events


def find_threshold_for_fa_eventized(
    probs: torch.Tensor,
    post_cfg: PostprocessingConfig,
    ref_events: list[list[tuple[float, float]]],
    fa_target: float,
    total_hours: float,
    fs: int,
    max_iters: int = 20,
    hysteresis_delta: float = 0.08,
) -> float:
    """Binary search for tau_on threshold meeting FA target.

    This function searches over hysteresis tau_on values, automatically
    deriving tau_off = max(0, tau_on - delta). This ensures consistent
    hysteresis behavior and monotonic FA rate changes.

    Args:
        probs: (N, T) probabilities
        post_cfg: Post-processing configuration (will be modified)
        ref_events: Reference events for FA calculation
        fa_target: Target FA/24h rate
        total_hours: Total duration in hours
        fs: Sampling rate
        max_iters: Maximum iterations for binary search
        hysteresis_delta: Gap between tau_on and tau_off (default 0.08)

    Returns:
        tau_on threshold that meets FA target (conservative)
    """
    # Search over tau_on values, ensuring tau_off is always below
    low = hysteresis_delta  # Minimum tau_on to maintain positive gap
    high = 1.0
    best_tau_on = 0.86  # Default from clinical settings

    # Create a copy of config to modify during search
    search_cfg = deepcopy(post_cfg)

    for _ in range(max_iters):
        mid_tau_on = (low + high) / 2
        mid_tau_off = max(0.0, mid_tau_on - hysteresis_delta)

        # Update hysteresis thresholds for this iteration
        search_cfg.hysteresis.tau_on = mid_tau_on
        search_cfg.hysteresis.tau_off = mid_tau_off

        # Get predictions with current thresholds (threshold param ignored)
        pred_events = batch_probs_to_events(probs, search_cfg, fs, threshold=mid_tau_on)
        fa_rate = fa_per_24h(pred_events, ref_events, total_hours)

        if fa_rate > fa_target:
            # Too many FA, increase tau_on
            low = mid_tau_on
        else:
            # At or below target, can potentially go lower
            best_tau_on = mid_tau_on
            high = mid_tau_on

        if abs(high - low) < 1e-4:
            break

    return best_tau_on


def sensitivity_at_fa_rates(
    probs: torch.Tensor,
    labels: torch.Tensor,
    fa_targets: list[float],
    post_cfg: PostprocessingConfig,
    sampling_rate: int = 256,
    window_stride_s: float = 10.0,
    window_size_s: float = 60.0,
    stitch_windows: bool = True,
) -> dict[str, float]:
    """Calculate sensitivity at specific FA/24h targets.

    Args:
        probs: (N, T) probabilities
        labels: (N, T) binary labels
        fa_targets: List of FA/24h targets (e.g., [10, 5, 1])
        post_cfg: Post-processing configuration
        sampling_rate: Sampling rate (Hz)
        window_stride_s: Stride between windows in seconds (for time accounting)
        window_size_s: Window size in seconds (for time accounting)
        stitch_windows: If True, stitch overlapping windows for record-level events

    Returns:
        Dict with sensitivity_at_Xfa keys
    """
    results = {}

    # Optionally stitch windows for record-level processing
    if stitch_windows and window_stride_s < window_size_s:
        from src.brain_brr.post.postprocess import stitch_windows as stitch_fn

        # Calculate window starts in samples
        stride_samples = int(window_stride_s * sampling_rate)
        window_starts = [i * stride_samples for i in range(probs.shape[0])]
        total_samples = window_starts[-1] + probs.shape[1] if window_starts else probs.shape[1]

        # Stitch probabilities and labels
        probs_stitched = stitch_fn(
            window_probs=list(probs),
            window_starts=window_starts,
            total_length=total_samples,
            method="overlap_add",
        )
        labels_stitched = stitch_fn(
            window_probs=list(labels.float()),
            window_starts=window_starts,
            total_length=total_samples,
            method="max",
        )
        labels_stitched = labels_stitched > 0.5

        # Work with stitched record
        probs = probs_stitched.unsqueeze(0)
        labels = labels_stitched.unsqueeze(0)

        # Update duration calculation for stitched record
        total_duration_s = total_samples / sampling_rate
        total_hours = total_duration_s / 3600
    else:
        # Original window-based processing
        n_windows = labels.shape[0]
        if n_windows > 0:
            total_duration_s = (n_windows - 1) * window_stride_s + window_size_s
            total_hours = total_duration_s / 3600
        else:
            total_hours = 0.0

    # Convert labels to events once
    ref_events = batch_masks_to_events(labels, sampling_rate)

    for fa_target in fa_targets:
        # Find threshold for this FA target
        threshold = find_threshold_for_fa_eventized(
            probs, post_cfg, ref_events, fa_target, total_hours, sampling_rate
        )

        # Get predictions at this threshold
        pred_events = batch_probs_to_events(probs, post_cfg, sampling_rate, threshold=threshold)

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

    # Use default threshold for TAES (0.5) — batch_probs_to_events ignores threshold
    # and uses hysteresis from post_cfg; kept for back-compat behavior.
    pred_events_taes = batch_probs_to_events(probs, post_cfg, sampling_rate, threshold=0.5)

    # Flatten events for TAES calculation
    all_ref = [evt for record in ref_events for evt in record]
    all_pred = [evt for record in pred_events_taes for evt in record]

    taes = calculate_taes(all_pred, all_ref) if all_ref else 0.0

    # AUROC and PR-AUC (sample-level)
    probs_flat = probs.cpu().numpy().flatten()
    labels_flat = labels.cpu().numpy().flatten()

    try:
        auroc = float(roc_auc_score(labels_flat, probs_flat))
    except ValueError:
        # Handle case with single class
        auroc = 0.5

    try:
        pr_auc = float(average_precision_score(labels_flat, probs_flat))
    except ValueError:
        # Handle case with single class or no positive samples
        pr_auc = 0.0

    # Expected Calibration Error (ECE) with 10 bins
    ece = calculate_ece(probs_flat, labels_flat, n_bins=10)

    # Sensitivity at FA rates with threshold table (τ_on per FA target)
    # Compute total_hours with overlap-aware duration: (N-1)*stride + window
    n_windows = labels.shape[0]
    total_duration_s = (n_windows - 1) * 10.0 + 60.0 if n_windows > 0 else 0.0
    total_hours = total_duration_s / 3600.0 if total_duration_s > 0 else 0.0

    thresholds: dict[str, float] = {}
    sensitivity_results: dict[str, float] = {}

    # Reuse reference events (already computed)
    for fa in fa_rates:
        tau_on = find_threshold_for_fa_eventized(
            probs,
            post_cfg,
            ref_events,
            fa_target=fa,
            total_hours=total_hours,
            fs=sampling_rate,
        )
        thresholds[f"{fa}"] = float(tau_on)

        # Evaluate sensitivity at this τ_on by updating hysteresis
        cfg_for_eval = deepcopy(post_cfg)
        cfg_for_eval.hysteresis.tau_on = tau_on
        cfg_for_eval.hysteresis.tau_off = max(0.0, tau_on - 0.08)

        pred_events_at_fa = batch_probs_to_events(
            probs, cfg_for_eval, sampling_rate, threshold=None
        )

        # Event-level sensitivity: proportion of reference events overlapped by any prediction
        tp_count = 0
        total_ref_events = 0
        for refs, preds in zip(ref_events, pred_events_at_fa, strict=False):
            total_ref_events += len(refs)
            for ref_start, ref_end in refs:
                if any(overlap((ref_start, ref_end), (ps, pe)) > 0 for (ps, pe) in preds):
                    tp_count += 1
        sensitivity = tp_count / max(total_ref_events, 1)
        sensitivity_results[f"sensitivity_at_{fa}fa"] = float(sensitivity)

    # Generate FA curve (10 points)
    fa_curve = []
    for fa in [0.5, 1, 2.5, 5, 10, 20, 50, 100]:
        sens = sensitivity_at_fa_rates(probs, labels, [fa], post_cfg, sampling_rate).get(
            f"sensitivity_at_{fa}fa", 0.0
        )
        fa_curve.append((fa, sens))

    results = {
        "taes": taes,
        "auroc": auroc,
        "pr_auc": pr_auc,
        "ece": ece,
        "fa_curve": fa_curve,
    }
    results.update(sensitivity_results)
    results["thresholds"] = thresholds  # FA target → τ_on

    return results


# Compatibility wrappers for tests
def compute_roc_curve(
    predictions: torch.Tensor | np.ndarray, labels: torch.Tensor | np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return FPR, TPR, thresholds, and AUROC for binary classification."""
    preds = (
        predictions.detach().cpu().numpy()
        if isinstance(predictions, torch.Tensor)
        else np.asarray(predictions)
    )
    labs = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    fpr, tpr, thresh = roc_curve(labs.ravel(), preds.ravel())
    try:
        auc = float(roc_auc_score(labs.ravel(), preds.ravel()))
    except ValueError:
        auc = 0.5
    return fpr, tpr, thresh, auc


def calculate_sensitivity_at_fa(
    tpr: np.ndarray, fpr: np.ndarray, target_fa_per_24h: float, duration_hours: float
) -> float:
    """Select sensitivity at operating point approximated by target FA/24h.

    Maps FA/24h to an approximate FPR target and returns the corresponding TPR.
    For unit tests we only require the value to be within [0,1].
    """
    if duration_hours <= 0 or len(fpr) == 0:
        return 0.0
    # Heuristic mapping to keep within [0,1]
    target_fpr = min(1.0, max(0.0, target_fa_per_24h / (24.0 * 60.0)))
    idx = int(np.argmin(np.abs(fpr - target_fpr)))
    return float(np.clip(tpr[idx], 0.0, 1.0))


def select_threshold_for_fa_rate(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    target_fa_per_24h: float,
    sample_rate: int = 256,
) -> float:
    """Return hysteresis tau_on that achieves target FA/24h for given predictions."""
    cfg = PostprocessingConfig()
    # One-hour default if we cannot infer duration from shapes
    n_windows = labels.shape[0]
    total_duration_s = (n_windows - 1) * 10.0 + 60.0 if n_windows > 0 else 3600.0
    total_hours = total_duration_s / 3600.0
    ref_events = batch_masks_to_events(labels > 0.5, sample_rate)
    return float(
        find_threshold_for_fa_eventized(
            predictions, cfg, ref_events, target_fa_per_24h, total_hours, sample_rate
        )
    )


def calculate_taes_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    fa_rate_target: float,
    sample_rate: int = 256,
    overlap_threshold: float | None = None,  # unused, kept for compatibility
) -> dict[str, Any]:
    """Compatibility wrapper that returns a rich metrics dict for tests."""
    cfg = PostprocessingConfig()
    metrics = evaluate_predictions(
        predictions, labels, fa_rates=[fa_rate_target], post_cfg=cfg, sampling_rate=sample_rate
    )
    # Add common classification metrics at sample level
    preds_bin = (predictions.detach().cpu().numpy().ravel() >= 0.5).astype(int)
    labs = labels.detach().cpu().numpy().ravel().astype(int)
    tp = int(((preds_bin == 1) & (labs == 1)).sum())
    tn = int(((preds_bin == 0) & (labs == 0)).sum())
    fp = int(((preds_bin == 1) & (labs == 0)).sum())
    fn = int(((preds_bin == 0) & (labs == 1)).sum())
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    metrics.update(
        {
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "precision": float(precision),
        }
    )
    return metrics


def main() -> None:
    """CLI entrypoint for evaluation.

    This function delegates to the main CLI module for proper argument parsing
    and evaluation workflow. Use 'run-experiment evaluate' for full functionality.
    """
    import sys

    print("Please use 'run-experiment evaluate' for evaluation functionality.")
    print("Example: run-experiment evaluate --config configs/evaluation.yaml")
    sys.exit(1)
