"""EEG constants for Phase 1 data pipeline.

These constants define the canonical 10-20 channel order and windowing params.
"""

from __future__ import annotations

# Canonical 10-20 montage order (19 channels)
# CRITICAL CONTRACT: The model is trained assuming this EXACT channel order.
# Channel position defines spatial relationships that U-Net convolutions learn.
# Upstream code MUST map any dataset names to these and preserve this order.
# Breaking this order will cause complete model failure as spatial patterns
# become meaningless (e.g., frontal activity interpreted as occipital).
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

# ==============================================================================
# Clinical Thresholds - Threshold Search Configuration
# ==============================================================================

# Binary search bounds for FA rate calibration
# v3.5.0: Expanded from [0.1, 1.0] to [0.0, 1.0] to support low-confidence models
# This allows the search to find thresholds for models that output low probabilities
THRESHOLD_SEARCH_LOW: float = 0.0
THRESHOLD_SEARCH_HIGH: float = 1.0
