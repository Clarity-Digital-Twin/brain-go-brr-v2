"""EEG constants for Phase 1 data pipeline.

These constants define the canonical 10-20 channel order and windowing params.
"""

from __future__ import annotations

# Canonical 10-20 montage order (19 channels)
CHANNEL_NAMES_10_20: list[str] = [
    "Fp1",
    "F3",
    "C3",
    "P3",
    "F7",
    "T3",
    "T5",
    "O1",
    # Midline
    "Fz",
    "Cz",
    "Pz",
    # Right hemisphere
    "Fp2",
    "F4",
    "C4",
    "P4",
    "F8",
    "T4",
    "T6",
    "O2",
]

# Synonyms observed in various datasets (map alternative → canonical)
CHANNEL_SYNONYMS: dict[str, str] = {
    "T7": "T3",
    "T8": "T4",
    "P7": "T5",
    "P8": "T6",
}

# Sampling / windowing
SAMPLING_RATE: int = 256
WINDOW_SIZE_SEC: int = 60
STRIDE_SIZE_SEC: int = 10

WINDOW_SAMPLES: int = WINDOW_SIZE_SEC * SAMPLING_RATE  # 15360
STRIDE_SAMPLES: int = STRIDE_SIZE_SEC * SAMPLING_RATE  # 2560

