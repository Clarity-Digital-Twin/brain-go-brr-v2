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
from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore[attr-defined]
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore[import-untyped]

from src.brain_brr import constants
from src.brain_brr.config.schemas import PostprocessingConfig
from src.brain_brr.eval.helpers.false_alarm import FASweepResult, find_threshold_for_fa_target
from src.brain_brr.eval.metrics import (
    batch_probs_to_events,
    calculate_ece,
    calculate_taes,
    stitch_recording_timeline,
)
from src.brain_brr.events import batch_mask_to_events
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
    recording_hours: float = recording_end_s / constants.SECONDS_PER_HOUR

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
    if not all_probs_flat or not all_labels_flat:
        logger.warning("[METRICS] No validation outputs; returning default metrics.")
        default_results = {
            "taes": 0.0,
            "auroc": 0.5,
            "pr_auc": 0.0,
            "ece": 1.0,
            "fa_curve": [],
            "num_recordings": num_recordings,
            "total_hours": total_hours,
            "thresholds": {},
        }
        for fa in fa_rates:
            default_results[f"sensitivity_at_{fa}fa"] = 0.0
        return default_results

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

    timelines_probs = [prob.cpu() for prob in all_probs_flat]
    timelines_labels = [label.cpu() for label in all_labels_flat]

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
        sensitivity_results[f"sensitivity_at_{fa}fa"] = result.sensitivity
        fa_curve.append((fa, result.sensitivity))

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
    """Validate model with true streaming per-recording processing (low memory).

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
    criterion = nn.BCEWithLogitsLoss()

    use_focal = focal_alpha is not None and focal_gamma is not None

    current_file_id: str | None = None
    current_windows: list[dict[str, Any]] = []

    all_probs_flat: list[torch.Tensor] = []
    all_labels_flat: list[torch.Tensor] = []
    all_ref_events: list[tuple[float, float]] = []
    all_pred_events: list[tuple[float, float]] = []
    total_hours = 0.0
    total_loss = 0.0
    total_loss_focal = 0.0
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
                probs = torch.sigmoid(logits)

                loss_bce = criterion(logits, labels)
                total_loss += loss_bce.item()

                if use_focal:
                    assert focal_alpha is not None
                    assert focal_gamma is not None
                    pt = labels * probs + (1 - labels) * (1 - probs)
                    at = labels * focal_alpha + (1 - labels) * (1 - focal_alpha)
                    focal_weight = at * ((1 - pt) ** focal_gamma)
                    bce = nn.functional.binary_cross_entropy_with_logits(
                        logits, labels, reduction="none"
                    )
                    loss_focal = (focal_weight * bce).mean()
                    total_loss_focal += loss_focal.item()

                for i, fid in enumerate(file_ids):
                    if fid != current_file_id and current_windows:
                        recording_hours = _process_recording(
                            current_windows,
                            all_probs_flat,
                            all_labels_flat,
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
        recording_hours = _process_recording(
            current_windows,
            all_probs_flat,
            all_labels_flat,
            all_ref_events,
            all_pred_events,
            post_config,
            constants.SAMPLING_RATE,
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
        constants.SAMPLING_RATE,
        num_recordings,
    )

    metrics["val_loss"] = total_loss / max(1, num_batches)

    if use_focal:
        metrics["val_loss_focal"] = total_loss_focal / max(1, num_batches)
        logger.info(
            f"[VALIDATION] Done! Val Loss (BCE): {metrics['val_loss']:.4f} | "
            f"Val Loss (Focal): {metrics['val_loss_focal']:.4f}"
        )
    else:
        logger.info(f"[VALIDATION] Done! Val Loss: {metrics['val_loss']:.4f}")

    if save_predictions and output_dir:
        if not all_probs_flat or not all_labels_flat:
            logger.warning("[SAVE] No validation outputs to save; skipping predictions.")
        else:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            probs_flat = torch.cat(all_probs_flat).cpu().numpy()
            labels_flat = torch.cat(all_labels_flat).cpu().numpy()

            epoch_suffix = f"_epoch{epoch}" if epoch is not None else ""
            pred_file = output_path / f"predictions{epoch_suffix}.npy"
            label_file = output_path / f"labels{epoch_suffix}.npy"

            np.save(pred_file, probs_flat)
            np.save(label_file, labels_flat)
            logger.info(f"[SAVE] Predictions saved to {pred_file} and {label_file}")

    if save_plots and output_dir:
        if not all_probs_flat or not all_labels_flat:
            logger.warning("[SAVE] No validation outputs for plots; skipping.")
        else:
            try:
                from sklearn.metrics import (  # type: ignore[attr-defined]
                    precision_recall_curve,
                    roc_curve,
                )
            except (ImportError, AttributeError):
                logger.warning("[SAVE] sklearn not available; skipping diagnostic plots.")
                return metrics

            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            probs_flat = torch.cat(all_probs_flat).cpu().numpy()
            labels_flat = torch.cat(all_labels_flat).cpu().numpy()
            labels_binary = (labels_flat > 0.5).astype(np.float32)

            epoch_suffix = f"_epoch{epoch}" if epoch is not None else ""

            if np.unique(labels_binary).size >= 2:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                auroc = float(metrics.get("auroc", float("nan")))
                pr_auc = float(metrics.get("pr_auc", float("nan")))

                fpr, tpr, _ = roc_curve(labels_binary, probs_flat)
                label_roc = f"ROC (AUC={auroc:.3f})" if np.isfinite(auroc) else "ROC"
                axes[0].plot(fpr, tpr, label=label_roc)
                axes[0].plot([0, 1], [0, 1], "k--", label="Random")
                axes[0].set_xlabel("False Positive Rate")
                axes[0].set_ylabel("True Positive Rate")
                axes[0].set_title("ROC Curve")
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)

                precision, recall, _ = precision_recall_curve(labels_binary, probs_flat)
                label_pr = f"PR (AUC={pr_auc:.3f})" if np.isfinite(pr_auc) else "PR"
                axes[1].plot(recall, precision, label=label_pr)
                axes[1].set_xlabel("Recall")
                axes[1].set_ylabel("Precision")
                axes[1].set_title("Precision-Recall Curve")
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

                plot_file = output_path / f"diagnostic_plots{epoch_suffix}.png"
                plt.tight_layout()
                plt.savefig(plot_file, dpi=150, bbox_inches="tight")
                plt.close(fig)
                logger.info(f"[SAVE] Diagnostic plots saved to {plot_file}")
            else:
                logger.warning("[SAVE] Skipping plots - insufficient label diversity")

    return metrics
