"""Helper modules for metrics evaluation (SRP compliance)."""

from .false_alarm import compute_fa_sweep
from .scalar_metrics import compute_event_taes, compute_probability_metrics
from .timeline import build_recording_timelines

__all__ = [
    "build_recording_timelines",
    "compute_event_taes",
    "compute_fa_sweep",
    "compute_probability_metrics",
]
