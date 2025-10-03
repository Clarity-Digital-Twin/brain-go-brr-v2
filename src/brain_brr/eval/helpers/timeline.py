"""Timeline assembly helpers - stitch windows into recording timelines."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch

from src.brain_brr import constants


@dataclass
class RecordingTimeline:
    """Stitched timeline for a single recording."""

    file_id: str
    timeline_probs: torch.Tensor
    timeline_labels: torch.Tensor
    duration_s: float


def build_recording_timelines(
    probs: torch.Tensor,
    labels: torch.Tensor,
    file_ids: list[str],
    window_starts: list[float],
    sampling_rate: int,
) -> list[RecordingTimeline]:
    """Build per-recording timelines from overlapping windows.

    Groups windows by file_id, stitches overlaps with averaging, and
    returns structured timeline data for each recording.

    Args:
        probs: (N, T) probabilities
        labels: (N, T) binary labels
        file_ids: List of N file IDs (one per window)
        window_starts: List of N window start times in seconds
        sampling_rate: Sampling rate (Hz)

    Returns:
        List of RecordingTimeline objects, one per unique file_id

    Notes:
        - Windows are sorted by start time before stitching
        - Overlapping regions use average of contributing windows
        - Recording duration computed as: last_window_start + WINDOW_SIZE_SEC
    """
    recordings: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for i, (fid, start_s) in enumerate(zip(file_ids, window_starts, strict=False)):
        recordings[fid].append(
            {
                "start_s": float(start_s),
                "probs": probs[i],
                "labels": labels[i],
            }
        )

    timelines: list[RecordingTimeline] = []

    for fid, windows in recordings.items():
        windows.sort(key=lambda x: x["start_s"])

        recording_end_s = windows[-1]["start_s"] + constants.WINDOW_SIZE_SEC
        timeline_length = int(recording_end_s * sampling_rate)

        device = windows[0]["probs"].device
        timeline_probs = torch.zeros(timeline_length, dtype=torch.float32, device=device)
        timeline_labels = torch.zeros(timeline_length, dtype=torch.float32, device=device)
        timeline_counts = torch.zeros(timeline_length, dtype=torch.float32, device=device)

        for w in windows:
            start_idx = int(w["start_s"] * sampling_rate)
            end_idx = min(start_idx + len(w["probs"]), timeline_length)
            window_len = end_idx - start_idx

            timeline_probs[start_idx:end_idx] += w["probs"][:window_len]
            timeline_labels[start_idx:end_idx] += w["labels"][:window_len]
            timeline_counts[start_idx:end_idx] += 1.0

        mask = timeline_counts > 0
        timeline_probs[mask] /= timeline_counts[mask]
        timeline_labels[mask] /= timeline_counts[mask]

        timelines.append(
            RecordingTimeline(
                file_id=fid,
                timeline_probs=timeline_probs,
                timeline_labels=timeline_labels,
                duration_s=recording_end_s,
            )
        )

    return timelines
