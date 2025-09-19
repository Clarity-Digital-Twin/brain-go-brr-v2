"""Brain-Go-Brr: Bi-Mamba-2 + U-Net + ResCNN for TUSZ seizure detection.

This is the refactored package structure. During migration, all imports
are re-exported from src.experiment for backwards compatibility.
"""

__version__ = "2.0.0-alpha"

# Compatibility re-exports during migration
# These will be replaced with direct imports once migration is complete

import warnings


def _compatibility_warning(old_path: str, new_path: str) -> None:
    """Issue deprecation warning for old import paths."""
    warnings.warn(
        f"Importing from '{old_path}' is deprecated. "
        f"Please use '{new_path}' instead.",
        DeprecationWarning,
        stacklevel=3,
    )


# For now, re-export everything from experiment to maintain compatibility
try:
    from src.experiment import *  # noqa: F403, F401
except ImportError:
    pass  # During initial setup, experiment might not exist yet