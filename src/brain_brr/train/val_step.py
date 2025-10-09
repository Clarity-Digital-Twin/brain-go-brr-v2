"""Validation epoch implementation extracted from loop.py.

Single Responsibility: Execute one validation epoch with proper metrics evaluation.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# sklearn type stubs incomplete (known third-party issue)
from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore[attr-defined]
from torch.utils.data import DataLoader

# tqdm has no type stubs (third-party library)
from tqdm import tqdm  # type: ignore[import-untyped]

from src.brain_brr import constants
from src.brain_brr.config.schemas import PostprocessingConfig
from src.brain_brr.constants import ECE_NUM_BINS, format_sensitivity_key
from src.brain_brr.eval.helpers.false_alarm import FASweepResult, find_threshold_for_fa_target
from src.brain_brr.eval.metrics import (
    batch_probs_to_events,
    calculate_taes,
    stitch_recording_timeline,
)
from src.brain_brr.events import batch_mask_to_events
from src.brain_brr.train.recording_storage import RecordingStorage
from src.brain_brr.utils.env import env

logger = logging.getLogger(__name__)


def _process_recording(
    file_id: str,
    windows: list[dict[str, Any]],
    storage: RecordingStorage,
    all_ref_events: list[tuple[float, float]],
    all_pred_events: list[tuple[float, float]],
    post_cfg: PostprocessingConfig,
    sampling_rate: int,
) -> float:
    """Process one recording (9MB transient, 0GB resident).

    Memory Contract:
    - Compute timeline: 9MB temporary
    - Write to disk: 0GB after write
    - Return: 0GB resident

    Args:
        file_id: Unique identifier for the recording
        windows: List of window dicts with keys: start_s, probs, labels
        storage: Disk-backed storage for validation data
        all_ref_events: Accumulator for reference events
        all_pred_events: Accumulator for predicted events
        post_cfg: Post-processing configuration
        sampling_rate: Sampling rate in Hz

    Returns:
        Recording duration in hours
    """
    timeline_probs, timeline_labels = stitch_recording_timeline(windows, sampling_rate)

    ref_events_list = batch_mask_to_events(timeline_labels.unsqueeze(0), sampling_rate)
    pred_events_list = batch_probs_to_events(timeline_probs.unsqueeze(0), post_cfg, sampling_rate)

    if ref_events_list:
        for event_obj in ref_events_list[0]:
            all_ref_events.append((float(event_obj.start_s), float(event_obj.end_s)))
    if pred_events_list:
        all_pred_events.extend(pred_events_list[0])

    probs_flat = timeline_probs.flatten()
    labels_flat = timeline_labels.flatten()

    storage.write_recording(file_id, probs_flat, labels_flat)

    recording_end_s = windows[-1]["start_s"] + constants.WINDOW_SIZE_SEC
    recording_hours: float = recording_end_s / constants.SECONDS_PER_HOUR

    del timeline_probs, timeline_labels, probs_flat, labels_flat

    return recording_hours


def _compute_ece_streaming(storage: RecordingStorage, n_bins: int = 10) -> float:
    """Compute ECE with true streaming (O(1) memory).

    This IS actually streaming - only stores bin statistics (10 bins x 24 bytes).

    Args:
        storage: Disk-backed storage
        n_bins: Number of calibration bins

    Returns:
        Expected Calibration Error
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_sums = np.zeros(n_bins, dtype=np.float64)
    bin_label_sums = np.zeros(n_bins, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=np.int64)

    for probs, labels in storage.iter_recordings():
        labels_binary = (labels > 0.5).astype(np.float32)
        bin_indices = np.digitize(probs, bin_edges[1:-1])

        np.add.at(bin_sums, bin_indices, probs)
        np.add.at(bin_label_sums, bin_indices, labels_binary)
        np.add.at(bin_counts, bin_indices, 1)

    total = bin_counts.sum()
    if total == 0:
        return 1.0

    ece = 0.0
    for i in range(n_bins):
        if bin_counts[i] > 0:
            avg_prob = bin_sums[i] / bin_counts[i]
            avg_label = bin_label_sums[i] / bin_counts[i]
            weight = bin_counts[i] / total
            ece += weight * abs(avg_prob - avg_label)

    return float(ece)


def _compute_final_metrics(
    storage: RecordingStorage,
    all_ref_events: list[tuple[float, float]],
    all_pred_events: list[tuple[float, float]],
    total_hours: float,
    fa_rates: list[float],
    post_cfg: PostprocessingConfig,
    sampling_rate: int,
    num_recordings: int,
) -> dict[str, Any]:
    """Compute exact metrics with staged memory loading (39GB peak).

    Memory Strategy:
    1. Load all data for AUROC/PR-AUC (34GB + 5GB overhead = 39GB)
    2. Compute metrics (exact sklearn algorithms)
    3. Explicitly free (0GB)
    4. Compute ECE streaming (<1MB)
    5. Reload for FA sweep (<10MB zero-copy mmap)

    Peak: 39GB (well within 96GB Modal limit, 2.5x safety margin)

    Args:
        storage: Disk-backed storage with validation data
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
    if num_recordings == 0:
        logger.warning("[METRICS] No validation data; returning defaults.")
        default_results = {
            "taes": 0.0,
            "auroc": 0.5,
            "pr_auc": 0.0,
            "ece": 1.0,
            "fa_curve": [],
            "num_recordings": 0,
            "total_hours": 0.0,
            "thresholds": {},
        }
        for fa in fa_rates:
            default_results[format_sensitivity_key(fa)] = 0.0
        return default_results

    taes = calculate_taes(all_pred_events, all_ref_events) if all_ref_events else 0.0

    logger.info("[METRICS] Loading validation data for AUROC/PR-AUC computation...")
    probs_all, labels_all = storage.get_all_concatenated()

    labels_binary = (labels_all > 0.5).astype(np.int32)

    if np.unique(labels_binary).size < 2:
        auroc = 0.5
    else:
        auroc = float(roc_auc_score(labels_binary, probs_all))

    if (labels_binary == 1).sum() == 0:
        pr_auc = 0.0
    else:
        pr_auc = float(average_precision_score(labels_binary, probs_all))

    logger.info(f"[METRICS] AUROC: {auroc:.4f}, PR-AUC: {pr_auc:.4f}")

    del probs_all, labels_all, labels_binary
    import gc

    gc.collect()
    logger.info("[METRICS] Freed AUROC/PR-AUC memory (39GB → 0GB)")

    logger.info("[METRICS] Computing ECE (streaming)...")
    ece = _compute_ece_streaming(storage, n_bins=ECE_NUM_BINS)
    logger.info(f"[METRICS] ECE: {ece:.4f}")

    logger.info("[METRICS] Starting FA sweep (zero-copy mmap)...")
    fa_curve: list[tuple[float, float]] = []
    thresholds: dict[str, float] = {}
    sensitivity_results: dict[str, float] = {}

    timelines_probs, timelines_labels = storage.get_all_as_torch_tensors()

    for fa in fa_rates:
        result: FASweepResult = find_threshold_for_fa_target(
            timelines_probs=timelines_probs,
            timelines_labels=timelines_labels,
            fa_target=fa,
            total_hours=total_hours,
            all_ref_events=all_ref_events,
            post_cfg=post_cfg,
            sampling_rate=sampling_rate,
            max_iters=constants.THRESHOLD_SEARCH_MAX_ITERS,
        )

        thresholds[f"{fa}"] = result.threshold_tau_on
        sensitivity_results[format_sensitivity_key(fa)] = result.sensitivity
        fa_curve.append((fa, result.sensitivity))

        logger.info(
            f"[FA] {fa} FA/24h → τ={result.threshold_tau_on:.3f}, "
            f"sensitivity={result.sensitivity:.3f}"
        )

    del timelines_probs, timelines_labels
    gc.collect()
    logger.info("[METRICS] Freed FA sweep memory (<10MB → 0MB)")

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
    focal_alpha: float | None = None,
    focal_gamma: float | None = None,
    save_predictions: bool = False,
    save_plots: bool = False,
    output_dir: str | Path | None = None,
    epoch: int | None = None,
) -> dict[str, Any]:
    """Validate model with disk-backed storage (39GB peak).

    Memory Profile:
    - Loop: 0GB accumulation (writes to disk)
    - Metrics: 39GB peak (staged loading)
    - Total: 39GB peak (2.5x safety margin on 96GB)

    CRITICAL: Requires validation files to be sorted by file_id for incremental processing.

    Args:
        model: SeizureDetector model
        dataloader: Validation DataLoader (MUST be sorted by file_id)
        post_config: Post-processing configuration
        device: Device to evaluate on
        fa_rates: FA/24h targets for sensitivity
        focal_alpha: Focal loss alpha (if None, skip focal loss computation)
        focal_gamma: Focal loss gamma (if None, skip focal loss computation)

    Returns:
        Dictionary of metrics (includes val_loss and optionally val_loss_focal)
    """
    if fa_rates is None:
        fa_rates = [10, 5, 1]

    model.eval()
    device_obj = torch.device(device)

    current_file_id: str | None = None
    current_windows: list[dict[str, Any]] = []

    all_ref_events: list[tuple[float, float]] = []
    all_pred_events: list[tuple[float, float]] = []
    total_hours = 0.0
    total_loss = 0.0
    num_batches = 0
    num_recordings = 0

    n_val_batches = len(dataloader)
    logger.info(f"[VALIDATION] Starting disk-backed validation ({n_val_batches} batches)")

    use_tqdm = not env.disable_tqdm()
    progress_bar = None

    with RecordingStorage() as storage, torch.no_grad():
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
                probs = torch.sigmoid(logits)

                assert focal_alpha is not None
                assert focal_gamma is not None
                pt = labels * probs + (1 - labels) * (1 - probs)
                at = labels * focal_alpha + (1 - labels) * (1 - focal_alpha)
                focal_weight = at * ((1 - pt) ** focal_gamma)
                bce = nn.functional.binary_cross_entropy_with_logits(
                    logits, labels, reduction="none"
                )
                loss = (focal_weight * bce).mean()
                total_loss += loss.item()

                for i, fid in enumerate(file_ids):
                    if fid != current_file_id and current_windows:
                        assert current_file_id is not None
                        recording_hours = _process_recording(
                            current_file_id,
                            current_windows,
                            storage,
                            all_ref_events,
                            all_pred_events,
                            post_config,
                            constants.SAMPLING_RATE,
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
            assert current_file_id is not None
            recording_hours = _process_recording(
                current_file_id,
                current_windows,
                storage,
                all_ref_events,
                all_pred_events,
                post_config,
                constants.SAMPLING_RATE,
            )
            total_hours += recording_hours
            num_recordings += 1

        logger.info(
            f"[VALIDATION] Processed {num_recordings} recordings, computing final metrics..."
        )

        metrics = _compute_final_metrics(
            storage,
            all_ref_events,
            all_pred_events,
            total_hours,
            fa_rates,
            post_config,
            constants.SAMPLING_RATE,
            num_recordings,
        )

    metrics["val_loss"] = total_loss / max(1, num_batches)
    logger.info(f"[VALIDATION] Done! Val Loss (Focal): {metrics['val_loss']:.4f}")

    if save_predictions and output_dir:
        logger.warning("[SAVE] Prediction saving temporarily disabled (disk-backed validation)")

    if save_plots and output_dir:
        logger.warning("[SAVE] Plot saving temporarily disabled (disk-backed validation)")

    return metrics
