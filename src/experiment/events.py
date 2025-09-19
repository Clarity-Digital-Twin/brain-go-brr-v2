"""Event processing for seizure detection.

DEPRECATED: This module has been moved to brain_brr.events
Please update your imports to use the new location.

Handles conversion from masks to events, event merging, and confidence scoring.
"""

import warnings

warnings.warn(
    "Importing from 'src.experiment.events' is deprecated. "
    "Please use 'from brain_brr.events import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Import everything from new location for compatibility
from src.brain_brr.events import *  # noqa: F403, F401

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from scipy import ndimage  # type: ignore[import-untyped]


@dataclass
class SeizureEvent:
    """Represents a single seizure event."""

    start_s: float
    end_s: float
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        """Event duration in seconds."""
        return self.end_s - self.start_s

    def overlaps(self, other: SeizureEvent) -> bool:
        """Check if this event overlaps with another."""
        return not (self.end_s <= other.start_s or self.start_s >= other.end_s)

    def merge(self, other: SeizureEvent) -> SeizureEvent:
        """Merge with another event, keeping max confidence."""
        return SeizureEvent(
            start_s=min(self.start_s, other.start_s),
            end_s=max(self.end_s, other.end_s),
            confidence=max(self.confidence, other.confidence),
        )


def mask_to_events(
    mask: torch.Tensor | np.ndarray,
    sampling_rate: int = 256,
    min_samples: int = 1,
) -> list[SeizureEvent]:
    """Convert binary mask to list of events.

    Args:
        mask: Binary mask (T,) as bool or float
        sampling_rate: Sampling rate in Hz
        min_samples: Minimum event length in samples (filter out shorter)

    Returns:
        List of SeizureEvent objects
    """
    mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask

    # Ensure binary
    if mask_np.dtype != bool:
        mask_np = mask_np > 0.5

    # Find connected components
    labeled, num_features = ndimage.label(mask_np)
    events = []

    for i in range(1, num_features + 1):
        indices = np.where(labeled == i)[0]
        if len(indices) >= min_samples:
            start_s = float(indices[0]) / sampling_rate
            end_s = float(indices[-1] + 1) / sampling_rate
            events.append(SeizureEvent(start_s=start_s, end_s=end_s))

    return events


def merge_events(
    events: list[SeizureEvent],
    tau_merge: float = 2.0,
) -> list[SeizureEvent]:
    """Merge events with gaps smaller than tau_merge.

    Args:
        events: List of events (should be sorted by start time)
        tau_merge: Maximum gap to merge (seconds)

    Returns:
        Merged list of events
    """
    if not events:
        return []

    # Sort by start time
    events = sorted(events, key=lambda e: e.start_s)
    merged = [events[0]]

    for event in events[1:]:
        last = merged[-1]
        gap = event.start_s - last.end_s

        if gap <= tau_merge:
            # Merge events
            merged[-1] = last.merge(event)
        else:
            # Keep as separate event
            merged.append(event)

    return merged


def calculate_event_confidence(
    probs: torch.Tensor | np.ndarray,
    event: SeizureEvent,
    sampling_rate: int = 256,
    method: Literal["mean", "peak", "percentile"] = "mean",
    percentile: float = 0.75,
) -> float:
    """Calculate confidence score for an event.

    Args:
        probs: Probability sequence (T,) in [0, 1]
        event: SeizureEvent with start/end times
        sampling_rate: Sampling rate in Hz
        method: Confidence calculation method
        percentile: Percentile value if method="percentile"

    Returns:
        Confidence score in [0, 1]
    """
    probs_np = probs.cpu().numpy() if isinstance(probs, torch.Tensor) else probs

    # Get indices for this event
    start_idx = int(event.start_s * sampling_rate)
    end_idx = int(event.end_s * sampling_rate)

    # Bounds checking
    start_idx = max(0, start_idx)
    end_idx = min(len(probs_np), end_idx)

    if start_idx >= end_idx:
        return 0.0

    event_probs = probs_np[start_idx:end_idx]

    if method == "mean":
        confidence = float(np.mean(event_probs))
    elif method == "peak":
        confidence = float(np.max(event_probs))
    elif method == "percentile":
        confidence = float(np.percentile(event_probs, percentile * 100))
    else:
        raise ValueError(f"Unknown confidence method: {method}")

    return float(np.clip(confidence, 0.0, 1.0))


def add_confidence_scores(
    events: list[SeizureEvent],
    probs: torch.Tensor | np.ndarray,
    sampling_rate: int = 256,
    method: Literal["mean", "peak", "percentile"] = "mean",
    percentile: float = 0.75,
) -> list[SeizureEvent]:
    """Add confidence scores to events.

    Args:
        events: List of events without confidence
        probs: Probability sequence (T,)
        sampling_rate: Sampling rate in Hz
        method: Confidence calculation method
        percentile: Percentile value if method="percentile"

    Returns:
        Events with confidence scores added
    """
    for event in events:
        event.confidence = calculate_event_confidence(
            probs, event, sampling_rate, method, percentile
        )
    return events


def events_to_mask(
    events: list[SeizureEvent],
    length: int,
    sampling_rate: int = 256,
) -> np.ndarray:
    """Convert events back to binary mask.

    Args:
        events: List of SeizureEvent objects
        length: Total length of mask in samples
        sampling_rate: Sampling rate in Hz

    Returns:
        Binary mask array (length,) as bool
    """
    mask = np.zeros(length, dtype=bool)

    for event in events:
        start_idx = int(event.start_s * sampling_rate)
        end_idx = int(event.end_s * sampling_rate)
        start_idx = max(0, start_idx)
        end_idx = min(length, end_idx)
        mask[start_idx:end_idx] = True

    return mask


def batch_mask_to_events(
    masks: torch.Tensor,
    sampling_rate: int = 256,
    tau_merge: float | None = None,
    probs: torch.Tensor | None = None,
    confidence_method: Literal["mean", "peak", "percentile"] = "mean",
    confidence_percentile: float = 0.75,
) -> list[list[SeizureEvent]]:
    """Convert batch of masks to events with optional merging and confidence.

    Args:
        masks: Binary masks (B, T) as bool
        sampling_rate: Sampling rate in Hz
        tau_merge: Optional gap threshold for merging
        probs: Optional probabilities for confidence scoring (B, T)
        confidence_method: Method for confidence calculation
        confidence_percentile: Percentile if method="percentile"

    Returns:
        List of event lists, one per batch item
    """
    batch_events = []
    batch_size = masks.shape[0]

    for b in range(batch_size):
        # Convert mask to events
        events = mask_to_events(masks[b], sampling_rate)

        # Merge if requested
        if tau_merge is not None and tau_merge > 0:
            events = merge_events(events, tau_merge)

        # Add confidence if probabilities provided
        if probs is not None:
            events = add_confidence_scores(
                events, probs[b], sampling_rate, confidence_method, confidence_percentile
            )

        batch_events.append(events)

    return batch_events
