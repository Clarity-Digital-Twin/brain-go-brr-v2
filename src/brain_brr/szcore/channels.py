"""Channel definitions and mappings for SzCORE EDF inputs."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from src.brain_brr.constants import CHANNEL_NAMES_10_20

# Official SzCORE order (19 channels) as documented on epilepsybenchmarks.com and arXiv:2402.13005.
SZCORE_CHANNELS_AVG: list[str] = [
    "Fp1-Avg",
    "F3-Avg",
    "C3-Avg",
    "P3-Avg",
    "O1-Avg",
    "F7-Avg",
    "T3-Avg",
    "T5-Avg",
    "Fz-Avg",
    "Cz-Avg",
    "Pz-Avg",
    "Fp2-Avg",
    "F4-Avg",
    "C4-Avg",
    "P4-Avg",
    "O2-Avg",
    "F8-Avg",
    "T4-Avg",
    "T6-Avg",
]

# Mapping semantics:
#   SZCORE_TO_OURS[szcore_idx] == our_idx
#
# IMPORTANT: This list is NOT directly usable as `data[SZCORE_TO_OURS]` because that expression expects
# "output_index -> input_index". Use `remap_szcore_to_ours()` which implements the correct scatter.
SZCORE_TO_OURS: list[int] = [0, 1, 2, 3, 7, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 18, 15, 16, 17]


def _validate_mapping() -> None:
    szcore_clean = [ch.replace("-Avg", "") for ch in SZCORE_CHANNELS_AVG]
    computed = [CHANNEL_NAMES_10_20.index(ch) for ch in szcore_clean]
    if computed != SZCORE_TO_OURS:
        raise ValueError(f"SZCORE_TO_OURS mismatch: computed={computed} expected={SZCORE_TO_OURS}")


_validate_mapping()


def remap_szcore_to_ours(data: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Remap EEG array from SzCORE channel order to our training order.

    Args:
        data: (19, T) or (B, 19, T) array in SzCORE order

    Returns:
        Array with same shape in `CHANNEL_NAMES_10_20` order.
    """
    if data.ndim == 2:
        out = np.empty_like(data)
        out[SZCORE_TO_OURS] = data
        return out
    if data.ndim == 3:
        out = np.empty_like(data)
        out[:, SZCORE_TO_OURS, :] = data
        return out
    raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")

