"""Centralized environment variable management for Brain-Go-Brr.

This module provides typed access to all BGB_* environment variables
used throughout the codebase, preventing scattered os.getenv() calls
and providing clear documentation of each variable's purpose.

NOTE: Environment variables are cached at module import time to support
torch.compile which cannot trace through os.getenv() calls.
"""

import os

# Cache environment variables at import time for torch.compile compatibility
# Model/forward pass variables
_EDGE_CLAMP = os.getenv("BGB_EDGE_CLAMP", "1") == "1"
_EDGE_CLAMP_MIN = float(os.getenv("BGB_EDGE_CLAMP_MIN", "-5.0"))
_EDGE_CLAMP_MAX = float(os.getenv("BGB_EDGE_CLAMP_MAX", "5.0"))
_DEBUG_FINITE = os.getenv("BGB_DEBUG_FINITE", "0") == "1"
_FORCE_MAMBA_FALLBACK = os.getenv("SEIZURE_MAMBA_FORCE_FALLBACK", "0") == "1"
_FORCE_TCN_EXT = os.getenv("BGB_FORCE_TCN_EXT", "0") == "1"

# Training/data variables (not used in forward pass, can stay dynamic)
_SMOKE_TEST = os.getenv("BGB_SMOKE_TEST", "0") == "1"
_LIMIT_FILES = os.getenv("BGB_LIMIT_FILES")
_FORCE_MANIFEST_REBUILD = os.getenv("BGB_FORCE_MANIFEST_REBUILD", "0") == "1"
_DISABLE_TQDM = os.getenv("BGB_DISABLE_TQDM", "0") == "1"
_DISABLE_TENSORBOARD = os.getenv("BGB_DISABLE_TB", "0") == "1"
_MID_EPOCH_MINUTES = os.getenv("BGB_MID_EPOCH_MINUTES")
_MID_EPOCH_KEEP = int(os.getenv("BGB_MID_EPOCH_KEEP", "2"))
_NAN_DEBUG = os.getenv("BGB_NAN_DEBUG", "0") == "1"
_NAN_DEBUG_MAX = int(os.getenv("BGB_NAN_DEBUG_MAX", "10"))
_SANITIZE_INPUTS = os.getenv("BGB_SANITIZE_INPUTS", "0") == "1"
_SANITIZE_GRADS = os.getenv("BGB_SANITIZE_GRADS", "0") == "1"
_SKIP_OPT_STEP_ON_NAN = os.getenv("BGB_SKIP_OPT_STEP_ON_NAN", "0") == "1"
_ANOMALY_DETECT = os.getenv("BGB_ANOMALY_DETECT", "0") == "1"
_PERF_ALLOW_GPU = os.getenv("BGB_PERF_ALLOW_GPU", "0") == "1"
_PERF_THREADS = os.getenv("BGB_PERF_THREADS")

# Performance test tolerance configuration
_PERF_TOLERANCE_FACTOR = float(os.getenv("BGB_PERF_TOLERANCE_FACTOR", "1.2"))  # 20% slack
_PERF_STRICT_MODE = os.getenv("BGB_PERF_STRICT_MODE", "0") == "1"  # Disable slack


class EnvConfig:
    """Typed accessor for Brain-Go-Brr environment variables."""

    # Model/Streaming toggles
    @staticmethod
    def edge_clamp() -> bool:
        """Enable edge value clamping in V3 path (default: True for stability)."""
        return _EDGE_CLAMP

    @staticmethod
    def edge_clamp_min() -> float:
        """Minimum value for edge clamping (default: -5.0)."""
        return _EDGE_CLAMP_MIN

    @staticmethod
    def edge_clamp_max() -> float:
        """Maximum value for edge clamping (default: 5.0)."""
        return _EDGE_CLAMP_MAX

    @staticmethod
    def debug_finite() -> bool:
        """Enable finite value assertions in models (default: False)."""
        return _DEBUG_FINITE

    @staticmethod
    def force_mamba_fallback() -> bool:
        """Force Mamba to use Conv1d fallback instead of CUDA kernel."""
        return _FORCE_MAMBA_FALLBACK

    @staticmethod
    def force_tcn_ext() -> bool:
        """Force use of TCNExt instead of standard TCN."""
        return _FORCE_TCN_EXT

    # Data/Training controls
    @staticmethod
    def smoke_test() -> bool:
        """Enable smoke test mode (limit to 3 files)."""
        return _SMOKE_TEST

    @staticmethod
    def limit_files() -> int | None:
        """Limit number of files to load (None for no limit)."""
        return int(_LIMIT_FILES) if _LIMIT_FILES else None

    @staticmethod
    def force_manifest_rebuild() -> bool:
        """Force rebuild of cache manifest."""
        return _FORCE_MANIFEST_REBUILD

    @staticmethod
    def disable_tqdm() -> bool:
        """Disable tqdm progress bars."""
        return _DISABLE_TQDM

    @staticmethod
    def disable_tensorboard() -> bool:
        """Disable TensorBoard logging."""
        return _DISABLE_TENSORBOARD

    @staticmethod
    def mid_epoch_minutes() -> int | None:
        """Save checkpoint every N minutes during epoch."""
        return int(_MID_EPOCH_MINUTES) if _MID_EPOCH_MINUTES else None

    @staticmethod
    def mid_epoch_keep() -> int:
        """Number of mid-epoch checkpoints to keep (default: 2)."""
        return _MID_EPOCH_KEEP

    # NaN/Debug controls
    @staticmethod
    def nan_debug() -> bool:
        """Enable NaN debugging mode."""
        return _NAN_DEBUG

    @staticmethod
    def nan_debug_max() -> int:
        """Maximum NaN occurrences before stopping (default: 10)."""
        return _NAN_DEBUG_MAX

    @staticmethod
    def sanitize_inputs() -> bool:
        """Enable input sanitization (replace NaN/Inf with 0)."""
        return _SANITIZE_INPUTS

    @staticmethod
    def sanitize_grads() -> bool:
        """Enable gradient sanitization."""
        return _SANITIZE_GRADS

    @staticmethod
    def skip_opt_step_on_nan() -> bool:
        """Skip optimizer step when NaN detected."""
        return _SKIP_OPT_STEP_ON_NAN

    @staticmethod
    def anomaly_detect() -> bool:
        """Enable PyTorch anomaly detection."""
        return _ANOMALY_DETECT

    # Performance testing
    @staticmethod
    def perf_allow_gpu() -> bool:
        """Allow GPU usage in performance tests."""
        return _PERF_ALLOW_GPU

    @staticmethod
    def perf_threads() -> int | None:
        """Number of threads for performance tests."""
        return int(_PERF_THREADS) if _PERF_THREADS else None

    # Performance tolerance configuration
    @staticmethod
    def perf_tolerance_factor() -> float:
        """Tolerance factor for performance tests (default: 1.2 = 20% slack)."""
        return _PERF_TOLERANCE_FACTOR

    @staticmethod
    def perf_strict_mode() -> bool:
        """Strict mode disables tolerance in performance tests."""
        return _PERF_STRICT_MODE


# Global instance for convenience
env = EnvConfig()
