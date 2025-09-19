"""EEG constants for Phase 1 data pipeline.

These constants define the canonical 10-20 channel order and windowing params.

DEPRECATED: This module has been moved to brain_brr.constants
Please update your imports to use the new location.
"""

from __future__ import annotations

import warnings

# Import everything from new location for compatibility (imports first per E402)
from src.brain_brr.constants import *  # noqa: F403

warnings.warn(
    "Importing from 'src.experiment.constants' is deprecated. "
    "Please use 'from src.brain_brr.constants import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)
