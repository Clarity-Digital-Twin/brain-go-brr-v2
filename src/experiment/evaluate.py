"""Model evaluation utilities.

DEPRECATED: This module has been moved to src.brain_brr.eval
Please update your imports to use the new location:
- from src.brain_brr.eval import evaluate_predictions, calculate_taes, etc.
"""

# Import everything from new location for compatibility (imports first per E402)
import warnings

from src.brain_brr.eval import *  # noqa: F403

warnings.warn(
    "Importing from 'src.experiment.evaluate' is deprecated. "
    "Please use 'from src.brain_brr.eval import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)
