"""Training utility functions."""

from __future__ import annotations

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_memory_stats() -> dict[str, float]:
    """Get current memory usage statistics (GPU + system RAM).

    Returns:
        Dictionary with memory stats in GB.
    """
    stats = {}

    if torch.cuda.is_available():
        stats["gpu_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
        stats["gpu_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
        stats["gpu_max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9

    try:
        import psutil

        process = psutil.Process()
        mem_info = process.memory_info()
        stats["ram_used_gb"] = mem_info.rss / 1e9

        sys_mem = psutil.virtual_memory()
        stats["ram_total_gb"] = sys_mem.total / 1e9
        stats["ram_available_gb"] = sys_mem.available / 1e9
        stats["swap_used_gb"] = psutil.swap_memory().used / 1e9
    except ImportError:
        pass

    return stats


def worker_init_fn(worker_id: int) -> None:
    """Initialize worker seeds for DataLoader determinism."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
