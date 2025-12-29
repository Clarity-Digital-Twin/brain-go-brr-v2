"""SzCORE inference runner for Brain-Go-Brr."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.brain_brr import constants
from src.brain_brr.config.schemas import Config
from src.brain_brr.data.preprocess import preprocess_recording
from src.brain_brr.events.events import SeizureEvent, mask_to_events
from src.brain_brr.post.postprocess import postprocess_predictions, stitch_windows

from .hed_score import write_hed_score_tsv
from .loader import load_szcore_edf

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SzcorePaths:
    config_yaml: Path
    checkpoint_pt: Path


def _default_paths() -> SzcorePaths:
    docker = SzcorePaths(
        config_yaml=Path("/app/configs/szcore_inference.yaml"),
        checkpoint_pt=Path("/app/checkpoints/best.pt"),
    )
    if docker.config_yaml.exists() and docker.checkpoint_pt.exists():
        return docker

    repo_root = Path(__file__).resolve().parents[3]
    return SzcorePaths(
        config_yaml=repo_root / "configs/szcore_inference.yaml",
        checkpoint_pt=repo_root / "results/local_fla_exp4_cyclic/checkpoints/best.pt",
    )


def _gpu_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def _run_cpu_heuristic(
    x: np.ndarray, fs: int, *, min_event_s: float = 3.0, smooth_s: float = 1.0
) -> list[SeizureEvent]:
    """Very lightweight CPU fallback detector.

    This is intended primarily to keep SzCORE PR CI (CPU-only) from failing when the
    main GPU model stack is unavailable. It is a real detector (not a stub), but
    it is not expected to be competitive.
    """
    if x.size == 0:
        return []

    score = np.mean(np.abs(x), axis=0)  # (T,)
    win = max(1, round(smooth_s * fs))
    if win > 1:
        kernel = np.ones(win, dtype=np.float32) / float(win)
        score = np.convolve(score.astype(np.float32), kernel, mode="same")

    med = float(np.median(score))
    mad = float(np.median(np.abs(score - med))) + 1e-6
    thr = med + 8.0 * mad  # conservative
    mask = score > thr

    min_samples = round(min_event_s * fs)
    return mask_to_events(mask, sampling_rate=fs, min_samples=max(1, min_samples))


def _window_starts(n_samples: int, window: int, stride: int) -> list[int]:
    if n_samples <= window:
        return [0]
    starts = list(range(0, n_samples - window + 1, stride))
    last = n_samples - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def run_szcore(
    input_edf: Path,
    output_tsv: Path,
    *,
    paths: SzcorePaths | None = None,
) -> None:
    """Run SzCORE inference end-to-end and write HED-SCORE TSV output."""
    if paths is None:
        paths = _default_paths()

    rec = load_szcore_edf(input_edf)
    if float(rec.fs) <= 0.0:
        raise ValueError(f"Invalid sampling rate {rec.fs} for {input_edf}")

    fs_target = int(constants.SAMPLING_RATE)
    x = preprocess_recording(
        rec.data_uv,
        fs_original=rec.fs,
        target_fs=fs_target,
        bandpass=(constants.BANDPASS_LOW_HZ, constants.BANDPASS_HIGH_HZ),
        notch_freq=constants.NOTCH_FILTER_HZ,
        normalize=True,
    )

    if _gpu_available():
        events = _run_gpu_model(x, fs_target, paths)
    else:
        logger.warning("[SzCORE] No GPU detected; using CPU heuristic fallback (CI-safe).")
        events = _run_cpu_heuristic(x, fs_target)

    write_hed_score_tsv(
        output_tsv,
        events,
        recording_duration_s=rec.duration_s,
        recording_start_dt=rec.start_dt,
    )


def _run_gpu_model(x: np.ndarray, fs: int, paths: SzcorePaths) -> list[SeizureEvent]:
    """Run the Brain-Go-Brr model stack on GPU.

    Imports torch/model code lazily to avoid breaking CPU-only CI environments.
    """
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyTorch is required for GPU inference but failed to import") from e

    if not torch.cuda.is_available():
        logger.warning("[SzCORE] torch.cuda.is_available() is False; using CPU heuristic fallback.")
        return _run_cpu_heuristic(x, fs)

    from src.brain_brr.models import SeizureDetector

    cfg = Config.from_yaml(paths.config_yaml)

    device = torch.device("cuda")
    model = SeizureDetector.from_config(cfg.model, warmup_schedule=cfg.training.warmup_schedule).to(device)

    checkpoint = torch.load(paths.checkpoint_pt, map_location=device)
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "model", "state_dict"):
            if key in checkpoint:
                state_dict = checkpoint[key]
                break

    if not isinstance(state_dict, dict):  # pragma: no cover
        raise RuntimeError(f"Unexpected checkpoint format at {paths.checkpoint_pt}: {type(state_dict)}")

    # Handle known dynamic buffers that can have shape mismatches (e.g., gnn.last_valid_pe).
    model_state = model.state_dict()
    for key in list(state_dict.keys()):
        if key not in model_state:
            continue
        try:
            ckpt_shape = state_dict[key].shape
            model_shape = model_state[key].shape
        except Exception:
            continue
        if ckpt_shape == model_shape:
            continue
        if key.endswith(".last_valid_pe"):
            logger.info(
                f"[SzCORE] Skipping dynamic buffer with shape mismatch: {key} "
                f"(checkpoint: {ckpt_shape}, model: {model_shape})"
            )
            del state_dict[key]
            continue
        raise RuntimeError(
            f"[SzCORE] Shape mismatch for {key}: checkpoint {ckpt_shape}, model {model_shape}. "
            "This likely indicates an architecture/config mismatch."
        )

    incompatible = model.load_state_dict(state_dict, strict=False)

    # Enforce "no silent mismatch" while allowing known dynamic buffers to be recomputed.
    allowed_missing = {"gnn.last_valid_pe"}
    missing_filtered = [k for k in incompatible.missing_keys if k not in allowed_missing]
    unexpected_filtered = [k for k in incompatible.unexpected_keys if not k.endswith(".last_valid_pe")]

    if missing_filtered or unexpected_filtered:
        raise RuntimeError(
            "[SzCORE] Checkpoint does not match the inference model.\n"
            f"Missing (unexpected): {missing_filtered}\n"
            f"Unexpected: {unexpected_filtered}\n"
            "This likely indicates a dependency or config/architecture mismatch."
        )

    if incompatible.missing_keys:
        logger.info(f"[SzCORE] Missing keys (allowed): {incompatible.missing_keys}")
    model.eval()

    window = constants.WINDOW_SAMPLES
    stride = constants.STRIDE_SAMPLES
    starts = _window_starts(x.shape[1], window, stride)

    window_probs: list[torch.Tensor] = []
    with torch.inference_mode():
        batch_size = 8
        for i in range(0, len(starts), batch_size):
            batch_starts = starts[i : i + batch_size]
            batch = np.zeros((len(batch_starts), x.shape[0], window), dtype=np.float32)
            for j, start in enumerate(batch_starts):
                chunk = x[:, start : start + window]
                batch[j, :, : chunk.shape[1]] = chunk

            inp = torch.from_numpy(batch).to(device=device, dtype=torch.float32)
            logits = model(inp)  # (B, window)
            probs = torch.sigmoid(logits).to(torch.float32).detach().cpu()
            for j in range(probs.shape[0]):
                window_probs.append(probs[j])

    probs = stitch_windows(window_probs, starts, total_length=x.shape[1], method="overlap_add")
    probs_b = probs.unsqueeze(0)  # (1, T)
    masks = postprocess_predictions(probs_b, cfg.postprocessing, sampling_rate=fs)
    mask = masks[0].cpu()

    # postprocess_predictions already applies duration filtering; keep mask_to_events simple.
    return mask_to_events(mask, sampling_rate=fs, min_samples=1)
