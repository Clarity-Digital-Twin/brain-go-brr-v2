"""Evaluation service - business logic for model evaluation CLI."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.brain_brr.config.schemas import Config, PostprocessingConfig
from src.brain_brr.constants import HYSTERESIS_DELTA, HYSTERESIS_TAU_ON
from src.brain_brr.data import EEGWindowDataset
from src.brain_brr.eval.metrics import batch_probs_to_events
from src.brain_brr.events import SeizureEvent
from src.brain_brr.events.export import export_csv_bi
from src.brain_brr.models import SeizureDetector
from src.brain_brr.train.loop import validate_epoch


def create_threshold_config(
    base_config: PostprocessingConfig,
    threshold: float,
    tau_off_delta: float = HYSTERESIS_DELTA,
) -> PostprocessingConfig:
    """Create a postprocessing config with updated hysteresis thresholds.

    Args:
        base_config: Base postprocessing configuration to copy
        threshold: tau_on threshold value
        tau_off_delta: Delta to subtract from tau_on for tau_off (default from constants)

    Returns:
        New PostprocessingConfig with updated thresholds

    Notes:
        This helper ensures config immutability by creating a deep copy.
        Used when exporting events with calibrated thresholds.
    """
    cfg_copy = deepcopy(base_config)
    cfg_copy.hysteresis.tau_on = threshold
    cfg_copy.hysteresis.tau_off = max(0.0, threshold - tau_off_delta)
    return cfg_copy


@dataclass
class EvaluationRequest:
    """Evaluation request parameters."""

    checkpoint_path: Path
    data_path: Path
    config_path: Path | None
    device: str
    output_json: Path | None
    output_csv_bi: Path | None


@dataclass
class EvaluationResult:
    """Evaluation result with metrics and optional exports."""

    metrics: dict[str, Any]
    checkpoint_path: str
    data_path: str
    device: str


def load_checkpoint_and_config(
    checkpoint_path: Path, config_path: Path | None
) -> tuple[dict[str, Any], Config]:
    """Load checkpoint and configuration.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Optional config override

    Returns:
        Tuple of (checkpoint_dict, config)

    Raises:
        FileNotFoundError: If checkpoint not found
        ValueError: If config not found in checkpoint or provided
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if config_path:
        cfg = Config.from_yaml(config_path)
    elif "config" in checkpoint and checkpoint["config"] is not None:
        cfg = Config(**checkpoint["config"])
    else:
        raise ValueError("No config found in checkpoint or provided. Please provide --config")

    return checkpoint, cfg


def create_dataloader(
    data_path: Path, cfg: Config, device: str
) -> tuple[DataLoader[Any], list[Path]]:
    """Create evaluation dataloader from EDF files.

    Args:
        data_path: Path to directory containing EDF files
        cfg: Configuration
        device: Target device (for pin_memory optimization)

    Returns:
        Tuple of (dataloader, edf_files_list)

    Raises:
        FileNotFoundError: If data path doesn't exist
        ValueError: If no EDF files found
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    edf_files = list(data_path.glob("**/*.edf"))
    if len(edf_files) == 0:
        raise ValueError(f"No EDF files found under: {data_path}")

    dataset = EEGWindowDataset(
        edf_files,
        cache_dir=Path(cfg.data.cache_dir) / "test",
    )

    dataloader: DataLoader[Any] = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=(device == "cuda"),
    )

    return dataloader, edf_files


def run_evaluation(request: EvaluationRequest) -> EvaluationResult:
    """Run model evaluation on test data.

    Delegates to existing validate_epoch for core evaluation logic.
    Handles checkpoint loading, model creation, dataset construction,
    and optional export to JSON/CSV_BI formats.

    Args:
        request: Evaluation request parameters

    Returns:
        EvaluationResult with metrics and metadata

    Raises:
        FileNotFoundError: If checkpoint or data path not found
        ValueError: If configuration invalid or no EDF files found
    """
    checkpoint, cfg = load_checkpoint_and_config(request.checkpoint_path, request.config_path)

    model = SeizureDetector.from_config(cfg.model)
    model.load_state_dict(checkpoint["model_state_dict"])

    device = (
        request.device
        if request.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(device)

    dataloader, _edf_files = create_dataloader(request.data_path, cfg, device)

    focal_alpha = cfg.training.focal_alpha if cfg.training.loss == "focal" else None
    focal_gamma = cfg.training.focal_gamma if cfg.training.loss == "focal" else None

    metrics = validate_epoch(
        model,
        dataloader,
        cfg.postprocessing,
        device=device,
        fa_rates=cfg.evaluation.fa_rates,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        save_predictions=cfg.evaluation.save_predictions,
        save_plots=cfg.evaluation.save_plots,
        output_dir=cfg.experiment.output_dir,
    )

    if request.output_json:
        _export_metrics_json(metrics, request, device)

    if request.output_csv_bi:
        _export_events_csv_bi(model, dataloader, metrics, cfg, device, request.output_csv_bi)

    return EvaluationResult(
        metrics=metrics,
        checkpoint_path=str(request.checkpoint_path),
        data_path=str(request.data_path),
        device=device,
    )


def _export_metrics_json(metrics: dict[str, Any], request: EvaluationRequest, device: str) -> None:
    """Export metrics to JSON file with metadata."""
    assert request.output_json is not None

    metrics["metadata"] = {
        "checkpoint": str(request.checkpoint_path),
        "data_path": str(request.data_path),
        "timestamp": datetime.now().isoformat(),
        "device": device,
    }

    request.output_json.parent.mkdir(parents=True, exist_ok=True)
    with request.output_json.open("w") as f:
        json.dump(metrics, f, indent=2, default=str)


def _export_events_csv_bi(
    model: torch.nn.Module,
    dataloader: DataLoader[Any],
    metrics: dict[str, Any],
    cfg: Config,
    device: str,
    output_path: Path,
) -> None:
    """Export predicted events to CSV_BI format."""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for windows, _labels in dataloader:
            inputs = windows.to(device)
            logits = model(inputs)
            probs_batch = torch.sigmoid(logits)
            all_probs.append(probs_batch.cpu())

    probs = torch.cat(all_probs, dim=0)

    thresholds_dict = metrics.get("thresholds", {})
    if isinstance(thresholds_dict, dict):
        if "10" in thresholds_dict:
            best_threshold = thresholds_dict["10"]
        elif 10 in thresholds_dict:
            best_threshold = thresholds_dict[10]
        elif 10.0 in thresholds_dict:
            best_threshold = thresholds_dict[10.0]
        else:
            best_threshold = HYSTERESIS_TAU_ON
    else:
        best_threshold = HYSTERESIS_TAU_ON

    cfg_for_export = create_threshold_config(cfg.postprocessing, best_threshold)

    pred_events = batch_probs_to_events(probs, cfg_for_export, cfg.data.sampling_rate)

    stride_s = cfg.data.stride
    window_s = cfg.data.window_size
    seizure_events = []
    for window_idx, record_events in enumerate(pred_events):
        window_start_s = window_idx * stride_s
        for start_s, end_s in record_events:
            seizure_events.append(
                SeizureEvent(
                    start_s=window_start_s + start_s,
                    end_s=window_start_s + end_s,
                    confidence=0.9,
                )
            )

    total_duration = (len(probs) - 1) * stride_s + window_s if len(probs) > 0 else 0.0
    export_csv_bi(
        seizure_events,
        output_path,
        patient_id="test",
        recording_id="eval",
        duration_s=total_duration,
    )
