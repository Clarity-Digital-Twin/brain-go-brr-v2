"""Centralized environment variable management for Brain-Go-Brr.

This module provides typed access to all BGB_* environment variables
used throughout the codebase, preventing scattered os.getenv() calls
and providing clear documentation of each variable's purpose.

NOTE: Environment variables are cached at module import time to support
torch.compile which cannot trace through os.getenv() calls.
"""

import os


# Cache environment variables at import time for torch.compile compatibility
_EDGE_CLAMP = os.getenv("BGB_EDGE_CLAMP", "1") == "1"
_EDGE_CLAMP_MIN = float(os.getenv("BGB_EDGE_CLAMP_MIN", "-5.0"))
_EDGE_CLAMP_MAX = float(os.getenv("BGB_EDGE_CLAMP_MAX", "5.0"))


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
        return os.getenv("BGB_DEBUG_FINITE", "0") == "1"

    @staticmethod
    def force_mamba_fallback() -> bool:
        """Force Mamba to use Conv1d fallback instead of CUDA kernel."""
        return os.getenv("SEIZURE_MAMBA_FORCE_FALLBACK", "0") == "1"

    @staticmethod
    def force_tcn_ext() -> bool:
        """Force use of TCNExt instead of standard TCN."""
        return os.getenv("BGB_FORCE_TCN_EXT", "0") == "1"

    # Data/Training controls
    @staticmethod
    def smoke_test() -> bool:
        """Enable smoke test mode (limit to 3 files)."""
        return os.getenv("BGB_SMOKE_TEST", "0") == "1"

    @staticmethod
    def limit_files() -> int | None:
        """Limit number of files to load (None for no limit)."""
        val = os.getenv("BGB_LIMIT_FILES")
        return int(val) if val else None

    @staticmethod
    def force_manifest_rebuild() -> bool:
        """Force rebuild of cache manifest."""
        return os.getenv("BGB_FORCE_MANIFEST_REBUILD", "0") == "1"

    @staticmethod
    def disable_tqdm() -> bool:
        """Disable tqdm progress bars."""
        return os.getenv("BGB_DISABLE_TQDM", "0") == "1"

    @staticmethod
    def disable_tensorboard() -> bool:
        """Disable TensorBoard logging."""
        return os.getenv("BGB_DISABLE_TB", "0") == "1"

    @staticmethod
    def mid_epoch_minutes() -> int | None:
        """Save checkpoint every N minutes during epoch."""
        val = os.getenv("BGB_MID_EPOCH_MINUTES")
        return int(val) if val else None

    @staticmethod
    def mid_epoch_keep() -> int:
        """Number of mid-epoch checkpoints to keep (default: 2)."""
        return int(os.getenv("BGB_MID_EPOCH_KEEP", "2"))

    # NaN/Debug controls
    @staticmethod
    def nan_debug() -> bool:
        """Enable NaN debugging mode."""
        return os.getenv("BGB_NAN_DEBUG", "0") == "1"

    @staticmethod
    def nan_debug_max() -> int:
        """Maximum NaN occurrences before stopping (default: 10)."""
        return int(os.getenv("BGB_NAN_DEBUG_MAX", "10"))

    @staticmethod
    def sanitize_inputs() -> bool:
        """Enable input sanitization (replace NaN/Inf with 0)."""
        return os.getenv("BGB_SANITIZE_INPUTS", "0") == "1"

    @staticmethod
    def sanitize_grads() -> bool:
        """Enable gradient sanitization."""
        return os.getenv("BGB_SANITIZE_GRADS", "0") == "1"

    @staticmethod
    def skip_opt_step_on_nan() -> bool:
        """Skip optimizer step when NaN detected."""
        return os.getenv("BGB_SKIP_OPT_STEP_ON_NAN", "0") == "1"

    @staticmethod
    def anomaly_detect() -> bool:
        """Enable PyTorch anomaly detection."""
        return os.getenv("BGB_ANOMALY_DETECT", "0") == "1"

    # Performance testing
    @staticmethod
    def perf_allow_gpu() -> bool:
        """Allow GPU usage in performance tests."""
        return os.getenv("BGB_PERF_ALLOW_GPU", "0") == "1"

    @staticmethod
    def perf_threads() -> int | None:
        """Number of threads for performance tests."""
        val = os.getenv("BGB_PERF_THREADS")
        return int(val) if val else None


# Global instance for convenience
env = EnvConfig()
