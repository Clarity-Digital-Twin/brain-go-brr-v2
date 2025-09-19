"""Post-processing for seizure predictions.

This module will contain:
- hysteresis.py: Dual-threshold hysteresis
- morphology.py: Opening/closing operations
- duration.py: Min/max duration filtering
- stitch.py: Window stitching and overlap handling
"""

# Import from moved module
from .postprocess import (
    apply_hysteresis,
    apply_morphology,
    filter_duration,
    postprocess_predictions,
    stitch_windows,
)

__all__ = [
    "apply_hysteresis",
    "apply_morphology",
    "filter_duration",
    "postprocess_predictions",
    "stitch_windows",
]
