"""Streaming inference utilities.

DEPRECATED: This module has been moved to brain_brr.streaming
Please update your imports to use the new location.
"""

import warnings

warnings.warn(
    "Importing from 'src.experiment.streaming' is deprecated. "
    "Please use 'from src.brain_brr.streaming import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Import everything from new location for compatibility
from src.brain_brr.streaming.streaming import *  # noqa: F403
