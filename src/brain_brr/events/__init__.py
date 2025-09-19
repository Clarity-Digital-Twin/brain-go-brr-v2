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
    mask_to_events,
    merge_events,
    calculate_event_confidence,
    add_confidence_scores,
    events_to_mask,
    batch_mask_to_events,
)

__all__ = [
    "SeizureEvent",
    "mask_to_events",
    "merge_events",
    "calculate_event_confidence",
    "add_confidence_scores",
    "events_to_mask",
    "batch_mask_to_events",
]
