"""Event processing for seizure detection.

DEPRECATED: This module has been moved to brain_brr.events
Please update your imports to use the new location.
"""

# Import everything from new location for compatibility (imports first per E402)
import warnings

from src.brain_brr.events import *  # noqa: F403

warnings.warn(
    "Importing from 'src.experiment.events' is deprecated. "
    "Please use 'from src.brain_brr.events import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)
