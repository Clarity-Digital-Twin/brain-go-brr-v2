"""EDF loading utilities for SzCORE inputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from src.brain_brr.constants import SAMPLING_RATE

from .channels import SZCORE_CHANNELS_AVG, remap_szcore_to_ours


@dataclass(frozen=True)
class SzcoreRecording:
    data_uv: npt.NDArray[np.float32]  # (19, T) in our channel order
    fs: float
    duration_s: float
    start_dt: datetime | None


def _read_raw_edf(edf_path: Path) -> Any:  # pragma: no cover
    import mne  # type: ignore[import-untyped]

    return mne.io.read_raw_edf(edf_path, preload=True, verbose="WARNING")


def load_szcore_edf(edf_path: Path) -> SzcoreRecording:
    """Load SzCORE EDF and return EEG in our canonical channel order.

    SzCORE guarantees:
    - 19 EEG channels in a fixed order
    - channel names with `-Avg` suffix (common average reference)
    - sampling rate 256 Hz
    """
    raw = _read_raw_edf(edf_path)

    if len(raw.ch_names) < 19:
        raise ValueError(f"SzCORE EDF must contain >=19 channels, found {len(raw.ch_names)}: {raw.ch_names}")

    expected = SZCORE_CHANNELS_AVG
    ch0 = list(raw.ch_names[:19])

    if ch0 != expected:
        # Robust fallback: locate expected channels anywhere in the EDF and pick them in the official order.
        indices: list[int] = []
        missing: list[str] = []
        for name in expected:
            try:
                indices.append(raw.ch_names.index(name))
            except ValueError:
                missing.append(name)

        if missing:
            raise ValueError(
                "SzCORE channel mismatch. Expected first 19 channels to be:\n"
                f"{expected}\n"
                f"Found first 19 channels:\n{ch0}\n"
                f"Missing expected channels: {missing}"
            )

        if hasattr(raw, "pick"):
            raw.pick(indices)  # preserves the order of indices
        else:  # pragma: no cover
            # MNE Raw always has .pick; keep a defensive fallback.
            raw.pick_channels(expected, ordered=True)

    # Extract in SzCORE order (volts)
    data_volts = raw.get_data()[:19]
    fs = float(raw.info.get("sfreq", SAMPLING_RATE))
    n_samples = int(data_volts.shape[1])
    duration_s = float(n_samples / fs) if n_samples > 0 else 0.0

    # MNE meas_date can be datetime | float | None
    meas_date = raw.info.get("meas_date")
    start_dt: datetime | None
    if isinstance(meas_date, datetime):
        start_dt = meas_date
    else:
        start_dt = None

    # Convert to µV and remap to our canonical order
    data_uv = (data_volts * 1e6).astype(np.float32)
    data_uv = remap_szcore_to_ours(data_uv)

    return SzcoreRecording(data_uv=data_uv, fs=fs, duration_s=duration_s, start_dt=start_dt)

