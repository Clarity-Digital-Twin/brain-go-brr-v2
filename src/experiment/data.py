"""EEG data I/O, preprocessing, windowing, and dataset utilities (Phase 1).

DEPRECATED: This module has been split and moved to src.brain_brr.data
Please update your imports to use the new locations:
- from src.brain_brr.data import load_edf_file, parse_tusz_csv
- from src.brain_brr.data import preprocess_recording, extract_windows
- from src.brain_brr.data import EEGWindowDataset
"""

# Import everything from new location for compatibility (imports first per E402)
import warnings

from src.brain_brr.data import *  # noqa: F403
from src.brain_brr.data.io import _read_raw_edf, _repair_edf_header_inplace

warnings.warn(
    "Importing from 'src.experiment.data' is deprecated. "
    "Please use 'from src.brain_brr.data import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)
