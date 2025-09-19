"""Event detection and processing utilities.

This module contains:
- intervals.py: Mask to interval conversion
- merge.py: Event merging logic
- confidence.py: Confidence scoring
- events.py: Core event classes and utilities
"""

# Import from the moved module
from .events import (
    SeizureEvent,
    mask_to_intervals,
    intervals_to_mask,
    merge_events,
    compute_event_confidence,
)

__all__ = [
    "SeizureEvent",
    "mask_to_intervals",
    "intervals_to_mask",
    "merge_events",
    "compute_event_confidence",
]