"""False alarm sweep helpers - threshold search for FA/24h targets."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch

from src.brain_brr.config.schemas import PostprocessingConfig
from src.brain_brr.constants import EPSILON_NUMERICAL
from src.brain_brr.eval.metrics import batch_probs_to_events
from src.brain_brr.events import batch_mask_to_events


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Return intersection length in seconds between intervals."""
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


@dataclass
class FASweepResult:
    """Result from FA sweep for a single FA target."""

    fa_target: float
    threshold_tau_on: float
    sensitivity: float
    threshold_unreachable: bool = False


def find_threshold_for_fa_target(
    timelines_probs: list[torch.Tensor],
    timelines_labels: list[torch.Tensor],
    fa_target: float,
    total_hours: float,
    all_ref_events: list[tuple[float, float]],
    post_cfg: PostprocessingConfig,
    sampling_rate: int,
    max_iters: int = 10,
) -> FASweepResult:
    """Binary search for tau_on threshold meeting FA/24h target.

    Args:
        timelines_probs: List of probability timelines (one per recording)
        timelines_labels: List of label timelines (one per recording)
        fa_target: Target FA/24h rate
        total_hours: Total duration in hours across all recordings
        all_ref_events: All reference events (for sensitivity calculation)
        post_cfg: Post-processing configuration
        sampling_rate: Sampling rate (Hz)
        max_iters: Maximum binary search iterations

    Returns:
        FASweepResult with:
            - threshold_tau_on: Best threshold found
            - sensitivity: Sensitivity at that threshold
            - threshold_unreachable: True if FA target cannot be reached even at τ=0.0

    Notes:
        FA counting uses overlap-aware logic: Only predictions that do NOT
        overlap with any reference event are counted as false alarms.
        This matches the original fa_per_24h behavior from v3.4.1.

        Search range expanded to [0.0, 1.0] to support low-confidence models
        that may require thresholds below 0.1 for high-FA operating points.
    """
    low, high = 0.0, 1.0
    best_tau_on = 0.86

    for _ in range(max_iters):
        mid_tau_on = (low + high) / 2
        mid_tau_off = max(0.0, mid_tau_on - 0.08)

        cfg_for_search = deepcopy(post_cfg)
        cfg_for_search.hysteresis.tau_on = mid_tau_on
        cfg_for_search.hysteresis.tau_off = mid_tau_off

        total_fa = 0
        for timeline_probs_rec, timeline_labels_rec in zip(
            timelines_probs, timelines_labels, strict=True
        ):
            pred_events_list = batch_probs_to_events(
                timeline_probs_rec.unsqueeze(0), cfg_for_search, sampling_rate
            )
            ref_events_list = batch_mask_to_events(timeline_labels_rec.unsqueeze(0), sampling_rate)

            search_preds: list[tuple[float, float]] = []
            if pred_events_list:
                search_preds = pred_events_list[0]

            search_refs: list[tuple[float, float]] = []
            if ref_events_list:
                for event_obj in ref_events_list[0]:
                    search_refs.append((float(event_obj.start_s), float(event_obj.end_s)))

            for pred_start, pred_end in search_preds:
                has_overlap = any(
                    _overlap((pred_start, pred_end), (ref_start, ref_end)) > 0
                    for ref_start, ref_end in search_refs
                )
                if not has_overlap:
                    total_fa += 1

        fa_rate = (total_fa / total_hours) * 24.0 if total_hours > 0 else 0.0

        if fa_rate > fa_target:
            low = mid_tau_on
        else:
            high = mid_tau_on
            best_tau_on = mid_tau_on

    cfg_for_eval = deepcopy(post_cfg)
    cfg_for_eval.hysteresis.tau_on = best_tau_on
    cfg_for_eval.hysteresis.tau_off = max(0.0, best_tau_on - 0.08)

    tp_count = 0
    total_ref_events = len(all_ref_events)

    for timeline_probs_rec, timeline_labels_rec in zip(
        timelines_probs, timelines_labels, strict=True
    ):
        ref_events_list = batch_mask_to_events(timeline_labels_rec.unsqueeze(0), sampling_rate)
        pred_events_list = batch_probs_to_events(
            timeline_probs_rec.unsqueeze(0), cfg_for_eval, sampling_rate
        )

        refs: list[tuple[float, float]] = []
        if ref_events_list:
            for event_obj in ref_events_list[0]:
                refs.append((float(event_obj.start_s), float(event_obj.end_s)))

        preds: list[tuple[float, float]] = []
        if pred_events_list:
            preds = pred_events_list[0]

        for ref_start, ref_end in refs:
            if any(_overlap((ref_start, ref_end), (ps, pe)) > 0 for (ps, pe) in preds):
                tp_count += 1

    sensitivity = tp_count / max(total_ref_events, 1)

    cfg_lowest = deepcopy(post_cfg)
    cfg_lowest.hysteresis.tau_on = EPSILON_NUMERICAL  # Essentially zero but valid (must be > tau_off)
    cfg_lowest.hysteresis.tau_off = 0.0

    total_fa_lowest = 0
    for timeline_probs_rec, timeline_labels_rec in zip(
        timelines_probs, timelines_labels, strict=True
    ):
        pred_events_list = batch_probs_to_events(
            timeline_probs_rec.unsqueeze(0), cfg_lowest, sampling_rate
        )
        ref_events_list = batch_mask_to_events(timeline_labels_rec.unsqueeze(0), sampling_rate)

        preds_lowest: list[tuple[float, float]] = []
        if pred_events_list:
            preds_lowest = pred_events_list[0]

        refs_lowest: list[tuple[float, float]] = []
        if ref_events_list:
            for event_obj in ref_events_list[0]:
                refs_lowest.append((float(event_obj.start_s), float(event_obj.end_s)))

        for pred_start, pred_end in preds_lowest:
            has_overlap = any(
                _overlap((pred_start, pred_end), (ref_start, ref_end)) > 0
                for ref_start, ref_end in refs_lowest
            )
            if not has_overlap:
                total_fa_lowest += 1

    fa_rate_lowest = (total_fa_lowest / total_hours) * 24.0 if total_hours > 0 else 0.0
    threshold_unreachable = fa_rate_lowest > fa_target

    return FASweepResult(
        fa_target=fa_target,
        threshold_tau_on=best_tau_on,
        sensitivity=sensitivity,
        threshold_unreachable=threshold_unreachable,
    )


def compute_fa_sweep(
    timelines_probs: list[torch.Tensor],
    timelines_labels: list[torch.Tensor],
    fa_targets: list[float],
    total_hours: float,
    all_ref_events: list[tuple[float, float]],
    post_cfg: PostprocessingConfig,
    sampling_rate: int,
) -> list[FASweepResult]:
    """Compute sensitivity at multiple FA/24h targets via threshold search.

    Args:
        timelines_probs: List of probability timelines (one per recording)
        timelines_labels: List of label timelines (one per recording)
        fa_targets: List of FA/24h targets (e.g., [10, 5, 1])
        total_hours: Total duration in hours across all recordings
        all_ref_events: All reference events (for sensitivity calculation)
        post_cfg: Post-processing configuration
        sampling_rate: Sampling rate (Hz)

    Returns:
        List of FASweepResult, one per FA target

    Notes:
        Uses binary search to find tau_on threshold meeting each FA target.
        Sensitivity is computed at the discovered threshold.
    """
    results: list[FASweepResult] = []

    for fa_target in fa_targets:
        result = find_threshold_for_fa_target(
            timelines_probs,
            timelines_labels,
            fa_target,
            total_hours,
            all_ref_events,
            post_cfg,
            sampling_rate,
        )
        results.append(result)

    return results
