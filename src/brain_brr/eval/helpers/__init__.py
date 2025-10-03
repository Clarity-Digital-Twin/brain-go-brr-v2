"""Helper modules for metrics evaluation (SRP compliance)."""

from .false_alarm import compute_fa_sweep
from .timeline import build_recording_timelines

__all__ = [
    "build_recording_timelines",
    "compute_fa_sweep",
]
