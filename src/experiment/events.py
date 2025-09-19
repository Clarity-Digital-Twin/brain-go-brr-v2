"""Event processing for seizure detection.

DEPRECATED: This module has been moved to brain_brr.events
Please update your imports to use the new location.
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