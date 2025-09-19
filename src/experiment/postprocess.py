"""Post-processing pipeline for seizure predictions.

DEPRECATED: This module has been moved to brain_brr.post
Please update your imports to use the new location.
"""

# Import everything from new location for compatibility (imports first per E402)
import warnings

from src.brain_brr.post.postprocess import *  # noqa: F403

warnings.warn(
    "Importing from 'src.experiment.postprocess' is deprecated. "
    "Please use 'from src.brain_brr.post import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)
