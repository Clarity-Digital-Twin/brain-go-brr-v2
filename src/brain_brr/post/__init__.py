"""Post-processing for seizure predictions.

This module will contain:
- hysteresis.py: Dual-threshold hysteresis
- morphology.py: Opening/closing operations
- duration.py: Min/max duration filtering
- stitch.py: Window stitching and overlap handling
"""

# During migration, re-export from experiment
try:
    from src.experiment.postprocess import (
        apply_hysteresis,
        apply_morphology,
        apply_duration_filter,
        PostProcessor,
    )

    __all__ = [
        "apply_hysteresis",
        "apply_morphology",
        "apply_duration_filter",
        "PostProcessor",
    ]
except ImportError:
    # Clean-slate imports will go here after migration
    pass