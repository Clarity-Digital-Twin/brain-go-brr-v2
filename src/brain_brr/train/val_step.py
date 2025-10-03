"""Validation epoch implementation extracted from loop.py.

Single Responsibility: Execute one validation epoch with proper metrics evaluation.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import suppress
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore[import-untyped]

from src.brain_brr import constants
from src.brain_brr.config.schemas import PostprocessingConfig
from src.brain_brr.utils.env import env

logger = logging.getLogger(__name__)


def _process_recording(
    windows: list[dict[str, Any]],
    all_probs_flat: list[torch.Tensor],
    all_labels_flat: list[torch.Tensor],
    all_ref_events: list[tuple[float, float]],
    all_pred_events: list[tuple[float, float]],
    post_cfg: PostprocessingConfig,
    sampling_rate: int,
) -> float:
    """Process one complete recording, extract events, accumulate metrics.

    Args:
        windows: List of window dicts with keys: start_s, probs, labels
        all_probs_flat: Accumulator for flattened probabilities
        all_labels_flat: Accumulator for flattened labels
        all_ref_events: Accumulator for reference events
        all_pred_events: Accumulator for predicted events
        post_cfg: Post-processing configuration
        sampling_rate: Sampling rate in Hz

    Returns:
        Recording duration in hours
    """
    from src.brain_brr.eval.metrics import batch_probs_to_events, stitch_recording_timeline
    from src.brain_brr.events import batch_mask_to_events

    timeline_probs, timeline_labels = stitch_recording_timeline(windows, sampling_rate)

    ref_events_list = batch_mask_to_events(timeline_labels.unsqueeze(0), sampling_rate)
    pred_events_list = batch_probs_to_events(timeline_probs.unsqueeze(0), post_cfg, sampling_rate)

    if ref_events_list:
        for event_obj in ref_events_list[0]:
            all_ref_events.append((float(event_obj.start_s), float(event_obj.end_s)))
    if pred_events_list:
        all_pred_events.extend(pred_events_list[0])

    all_probs_flat.append(timeline_probs.flatten())
    all_labels_flat.append(timeline_labels.flatten())

    recording_end_s = windows[-1]["start_s"] + constants.WINDOW_SIZE_SEC
    recording_hours: float = recording_end_s / 3600.0

    del timeline_probs, timeline_labels

    return recording_hours


def _compute_final_metrics(
    all_probs_flat: list[torch.Tensor],
    all_labels_flat: list[torch.Tensor],
    all_ref_events: list[tuple[float, float]],
    all_pred_events: list[tuple[float, float]],
    total_hours: float,
    fa_rates: list[float],
    post_cfg: PostprocessingConfig,
    sampling_rate: int,
    num_recordings: int,
) -> dict[str, Any]:
    """Compute final metrics from accumulated data.

    Args:
        all_probs_flat: List of flattened probability tensors
        all_labels_flat: List of flattened label tensors
        all_ref_events: List of reference events
        all_pred_events: List of predicted events
        total_hours: Total duration in hours
        fa_rates: FA/24h targets (e.g., [10, 5, 1])
        post_cfg: Post-processing config
        sampling_rate: Sampling rate
        num_recordings: Number of recordings processed

    Returns:
        Dictionary with keys:
        - taes, auroc, pr_auc, ece: Standard metrics
        - sensitivity_at_<N>fa: Sensitivity at each FA target
        - thresholds: Dict mapping FA rates to tau_on values (e.g., {"10": 0.86})
        - num_recordings, total_hours: Dataset stats
    """
    from src.brain_brr.eval.metrics import calculate_ece, calculate_taes, overlap

    taes = calculate_taes(all_pred_events, all_ref_events) if all_ref_events else 0.0

    probs_flat = torch.cat(all_probs_flat).cpu().numpy()
    labels_flat = torch.cat(all_labels_flat).cpu().numpy()

    # Binarize labels (threshold at 0.5, matches original evaluate_predictions behavior)
    labels_flat = (labels_flat > 0.5).astype(np.float32)

    if np.unique(labels_flat).size < 2:
        auroc = 0.5
    else:
        auroc = float(roc_auc_score(labels_flat, probs_flat))

    if (labels_flat == 1).sum() == 0:
        pr_auc = 0.0
    else:
        pr_auc = float(average_precision_score(labels_flat, probs_flat))

    ece = calculate_ece(probs_flat, labels_flat, n_bins=10)

    fa_curve: list[tuple[float, float]] = []

    thresholds: dict[str, float] = {}
    sensitivity_results: dict[str, float] = {}

    for fa in fa_rates:
        low, high = THRESHOLD_SEARCH_LOW, THRESHOLD_SEARCH_HIGH
        best_tau_on = 0.86

        for _ in range(10):
            mid_tau_on = (low + high) / 2
            mid_tau_off = max(0.0, mid_tau_on - 0.08)

            cfg_for_search = deepcopy(post_cfg)
            cfg_for_search.hysteresis.tau_on = mid_tau_on
            cfg_for_search.hysteresis.tau_off = mid_tau_off

            from src.brain_brr.eval.metrics import batch_probs_to_events

            num_pred_events = 0
            for i in range(len(all_probs_flat)):
                pred_events = batch_probs_to_events(
                    all_probs_flat[i].unsqueeze(0), cfg_for_search, sampling_rate
                )
                num_pred_events += len(pred_events[0]) if pred_events else 0

            fa_24h = (num_pred_events / total_hours) * 24.0 if total_hours > 0 else 0.0

            if fa_24h > fa:
                low = mid_tau_on
            else:
                high = mid_tau_on

        thresholds[f"{fa}"] = best_tau_on

        cfg_for_eval = deepcopy(post_cfg)
        cfg_for_eval.hysteresis.tau_on = best_tau_on
        cfg_for_eval.hysteresis.tau_off = max(0.0, best_tau_on - 0.08)

        total_ref_events = len(all_ref_events)
        tp_count = 0

        for i in range(len(all_probs_flat)):
            from src.brain_brr.eval.metrics import batch_probs_to_events
            from src.brain_brr.events import batch_mask_to_events

            ref_events_list = batch_mask_to_events(all_labels_flat[i].unsqueeze(0), sampling_rate)
            pred_events_list = batch_probs_to_events(
                all_probs_flat[i].unsqueeze(0), cfg_for_eval, sampling_rate
            )

            refs: list[tuple[float, float]] = []
            if ref_events_list:
                for event_obj in ref_events_list[0]:
                    refs.append((float(event_obj.start_s), float(event_obj.end_s)))

            preds: list[tuple[float, float]] = []
            if pred_events_list:
                preds = pred_events_list[0]

            for ref_start, ref_end in refs:
                if any(overlap((ref_start, ref_end), (ps, pe)) > 0 for (ps, pe) in preds):
                    tp_count += 1

        sensitivity = tp_count / max(total_ref_events, 1)
        sensitivity_results[f"sensitivity_at_{fa}fa"] = sensitivity

    results = {
        "taes": taes,
        "auroc": auroc,
        "pr_auc": pr_auc,
        "ece": ece,
        "fa_curve": fa_curve,
        "num_recordings": num_recordings,
        "total_hours": total_hours,
    }
    results.update(sensitivity_results)
    results["thresholds"] = thresholds

    return results


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    post_config: PostprocessingConfig,
    device: str = "cpu",
    fa_rates: list[float] | None = None,
) -> dict[str, Any]:
    """Validate model with true streaming per-recording processing (low memory).

    CRITICAL: Requires validation files to be sorted by file_id for incremental processing.

    Args:
        model: SeizureDetector model
        dataloader: Validation DataLoader (MUST be sorted by file_id)
        post_config: Post-processing configuration
        device: Device to evaluate on
        fa_rates: FA/24h targets for sensitivity

    Returns:
        Dictionary of metrics
    """
    if fa_rates is None:
        fa_rates = [10, 5, 1]

    model.eval()
    device_obj = torch.device(device)
    criterion = nn.BCEWithLogitsLoss()

    current_file_id: str | None = None
    current_windows: list[dict[str, Any]] = []

    all_probs_flat: list[torch.Tensor] = []
    all_labels_flat: list[torch.Tensor] = []
    all_ref_events: list[tuple[float, float]] = []
    all_pred_events: list[tuple[float, float]] = []
    total_hours = 0.0
    total_loss = 0.0
    num_batches = 0
    num_recordings = 0

    n_val_batches = len(dataloader)
    logger.info(f"[VALIDATION] Starting incremental streaming validation ({n_val_batches} batches)")

    use_tqdm = not env.disable_tqdm()
    progress_bar = None

    with torch.no_grad():
        if use_tqdm:
            try:
                progress_bar = tqdm(
                    dataloader,
                    desc="Validating",
                    leave=False,
                    file=sys.stderr,
                    ascii=True,
                    ncols=80,
                    disable=None,
                )
                if progress_bar is None or not hasattr(progress_bar, "__iter__"):
                    logger.warning(
                        "tqdm initialization failed in validation, using plain iteration"
                    )
                    iterator = dataloader
                else:
                    iterator = progress_bar
            except Exception as e:
                logger.warning(f"tqdm failed in validation ({e}), using plain iteration")
                iterator = dataloader
        else:
            iterator = dataloader

        try:
            last_heartbeat = time.time()
            heartbeat_interval = 120

            for batch_idx, batch in enumerate(iterator):
                windows = batch["window"].to(device_obj)
                labels = batch["label"].to(device_obj)
                file_ids = batch["file_id"]
                window_starts = batch["window_start_s"]

                if labels.dim() == 3:
                    labels = labels.max(dim=1)[0]

                logits = model(windows)
                loss = criterion(logits, labels)

                probs = torch.sigmoid(logits)

                for i, fid in enumerate(file_ids):
                    if fid != current_file_id and current_windows:
                        recording_hours = _process_recording(
                            current_windows,
                            all_probs_flat,
                            all_labels_flat,
                            all_ref_events,
                            all_pred_events,
                            post_config,
                            256,
                        )
                        total_hours += recording_hours
                        num_recordings += 1
                        current_windows = []

                    current_file_id = fid
                    current_windows.append(
                        {
                            "start_s": float(window_starts[i]),
                            "probs": probs[i].cpu(),
                            "labels": labels[i].cpu(),
                        }
                    )

                total_loss += loss.item()
                num_batches += 1

                current_time = time.time()
                if current_time - last_heartbeat > heartbeat_interval:
                    avg_loss = total_loss / max(1, num_batches)
                    logger.info(
                        f"[VAL HEARTBEAT] Batch {batch_idx}/{len(dataloader)} | "
                        f"Avg Loss: {avg_loss:.4f} | Recordings processed: {num_recordings}"
                    )
                    last_heartbeat = current_time
        finally:
            if progress_bar is not None and hasattr(progress_bar, "close"):
                with suppress(Exception):
                    progress_bar.close()

    if current_windows:
        recording_hours = _process_recording(
            current_windows,
            all_probs_flat,
            all_labels_flat,
            all_ref_events,
            all_pred_events,
            post_config,
            256,
        )
        total_hours += recording_hours
        num_recordings += 1

    logger.info(f"[VALIDATION] Processed {num_recordings} recordings, computing final metrics...")

    metrics = _compute_final_metrics(
        all_probs_flat,
        all_labels_flat,
        all_ref_events,
        all_pred_events,
        total_hours,
        fa_rates,
        post_config,
        256,
        num_recordings,
    )

    metrics["val_loss"] = total_loss / max(1, num_batches)

    logger.info(f"[VALIDATION] Done! Val Loss: {metrics['val_loss']:.4f}")

    return metrics
