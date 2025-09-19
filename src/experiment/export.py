"""Export utilities for seizure predictions.

DEPRECATED: This module has been moved to brain_brr.events.export
Please update your imports to use the new location.
"""

import warnings

warnings.warn(
    "Importing from 'src.experiment.export' is deprecated. "
    "Please use 'from src.brain_brr.events.export import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Import everything from new location for compatibility
from src.brain_brr.events.export import *  # noqa: F403
