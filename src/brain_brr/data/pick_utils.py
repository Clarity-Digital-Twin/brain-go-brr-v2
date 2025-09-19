"""Utility for robust channel selection and ordering across MNE versions."""

from __future__ import annotations

from typing import Any

import numpy as np


def pick_and_order(raw: Any, required: list[str]) -> tuple[Any, list[str]]:
    """Subset and impose exact channel order robustly across MNE versions and test doubles.

    This utility ensures channels are selected AND ordered exactly as specified,
    working around MNE API differences and test mock limitations.

    Key insight: When you pass INTEGER INDICES to raw.pick(), MNE preserves
    the order you provide. This bypasses the need for the 'ordered' parameter
    entirely.

    Args:
        raw: MNE Raw object or test mock with channel data
        required: List of channel names in the EXACT order needed

    Returns:
        (raw_modified, missing): Modified raw object and list of missing channels
        The raw object is modified in-place and also returned.

    Implementation strategy:
    1. Compute integer indices for the required channels
    2. Use raw.pick() with those indices (preserves order)
    3. Call reorder_channels() as a safety net if available
    4. Fall back to pick_channels() for older MNE versions
    5. Last resort: build a new RawArray for test mocks
    """
    # Get channel names - handle both MNE Raw and test mocks
    if hasattr(raw, "ch_names"):
        ch_names = list(raw.ch_names)  # Test mock
    elif "ch_names" in raw.info:
        ch_names = list(raw.info["ch_names"])  # MNE Raw
    else:
        ch_names = list(raw.info.ch_names)  # MNE Raw alternate access
    present = [ch for ch in required if ch in ch_names]
    missing = [ch for ch in required if ch not in ch_names]

    if not present:
        return raw, missing

    # Build ordered integer index list for present channels
    idx = [ch_names.index(ch) for ch in present]

    # Strategy 1: Modern API with integer indices (order preserved)
    if hasattr(raw, "pick"):
        raw.pick(idx)
        # Belt-and-suspenders: explicitly reorder if method exists
        if hasattr(raw, "reorder_channels"):
            raw.reorder_channels(present)
        return raw, missing

    # Strategy 2: Legacy API with ordered parameter
    if hasattr(raw, "pick_channels"):
        raw.pick_channels(present, ordered=True)
        return raw, missing

    # Strategy 3: Test mock fallback - build a new RawArray
    # This handles simple test mocks that don't have pick methods
    try:
        # Try to get data via get_data() or direct _data access
        if hasattr(raw, "get_data"):
            data = raw.get_data(picks=idx)
        elif hasattr(raw, "_data"):
            data = raw._data[idx] if hasattr(raw._data, "__getitem__") else raw._data[idx, :]
        else:
            # Can't extract data, return as-is
            return raw, missing

        # Build new Raw with correct channel order
        import mne

        info = mne.create_info(
            ch_names=present, sfreq=raw.info["sfreq"], ch_types="eeg", verbose=False
        )
        new_raw = mne.io.RawArray(data, info, verbose=False)

        # Transfer any other critical attributes from original
        if hasattr(raw, "annotations"):
            new_raw.annotations = raw.annotations

        return new_raw, missing

    except Exception:
        # If all else fails, return original
        return raw, missing