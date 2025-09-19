"""EEG data I/O and preprocessing utilities."""

from __future__ import annotations

import contextlib
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from src.brain_brr import constants

logger = logging.getLogger(__name__)


def _read_raw_edf(file_path: Path) -> Any:  # pragma: no cover - thin wrapper
    """Read EDF via MNE with preload.

    Split out for easy monkeypatching in tests (avoids depending on MNE).
    """
    import mne  # type: ignore[import-untyped]  # MNE has no stubs

    return mne.io.read_raw_edf(file_path, preload=True, verbose="WARNING")


def _repair_edf_header_inplace(file_path: Path) -> bool:
    """Repair common EDF header issues in-place (dates with wrong separators).

    Returns True if repair was needed and successful, False otherwise.
    """
    try:
        with open(file_path, "r+b") as f:
            # Check header for date format issues (byte 168-175: DD.MM.YY)
            f.seek(168)
            date_bytes = f.read(8)

            # Fix colons to periods in date field
            if b":" in date_bytes:
                fixed_date = date_bytes.replace(b":", b".")
                f.seek(168)
                f.write(fixed_date)
                return True
    except Exception:
        pass
    return False


def load_edf_file(
    file_path: Path,
    target_channels: list[str] | None = None,
    apply_montage: bool = True,
    channel_synonyms: dict[str, str] | None = None,
) -> tuple[npt.NDArray[np.float32], float]:
    """Load an EDF and return signals in canonical order and original fs.

    Args:
        file_path: Path to EDF file
        target_channels: Desired channel order; defaults to 10-20 list
        apply_montage: If True, set standard_1020 montage (best-effort)
        channel_synonyms: Mapping of alternate names to canonical names

    Returns:
        data: (n_channels, n_samples) float32 in microvolts (µV)
        fs: Original sampling frequency (Hz)

    Raises:
        ValueError: If required channels are missing after synonym remapping

    Note:
        MNE is generally permissive with malformed headers (e.g., TUSZ files
        with colon date separators). If MNE fails on header issues, we attempt
        a repair on a temp copy and retry.
    """
    if target_channels is None:
        target_channels = constants.CHANNEL_NAMES_10_20
    if channel_synonyms is None:
        channel_synonyms = constants.CHANNEL_SYNONYMS

    # First attempt with MNE (already permissive)
    try:
        raw = _read_raw_edf(file_path)
    except Exception as e:
        # If MNE fails with header/date error, try repair on temp copy
        if "startdate" in str(e) or "header" in str(e).lower():
            with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                shutil.copy2(file_path, tmp_path)

                try:
                    _repair_edf_header_inplace(tmp_path)
                    raw = _read_raw_edf(tmp_path)
                finally:
                    with contextlib.suppress(Exception):
                        tmp_path.unlink()
        else:
            raise

    # Normalize channel names - standardize to canonical case
    # Step 1: Handle TUSZ-specific naming (e.g., "EEG FP1-LE" -> "FP1")
    def clean_tusz_name(name: str) -> str:
        """Clean TUSZ channel names by removing prefixes and suffixes."""
        # Remove spaces first
        name = name.replace(" ", "")

        # Remove 'EEG' prefix if present (common in TUSZ)
        if name.startswith("EEG"):
            name = name[3:].lstrip()

        # Remove reference suffixes like '-LE', '-REF', '-AR' (TUSZ montages)
        for suffix in ["-LE", "-REF", "-AR"]:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break

        return name

    raw.rename_channels(lambda x: clean_tusz_name(x))

    # Step 2: Apply standard casing for known channels
    # Create case-insensitive mapping to canonical names
    canonical_map = {}
    for ch_name in raw.ch_names:
        # Check if this matches any target channel (case-insensitive)
        for target in target_channels:
            if ch_name.upper() == target.upper():
                canonical_map[ch_name] = target
                break

    if canonical_map:
        raw.rename_channels(canonical_map)

    # Step 3: Apply channel synonyms (e.g., "T7" -> "T3")
    if channel_synonyms:
        present_map = {k: v for k, v in channel_synonyms.items() if k in raw.ch_names}
        if present_map:
            raw.rename_channels(present_map)

    # Filter to target channels only; consider midline interpolation if needed
    available = [ch for ch in target_channels if ch in raw.ch_names]
    missing = set(target_channels) - set(available)

    # Handle special case: interpolate Fz/Pz if montage is available
    midline = {"Fz", "Pz"}
    missing_midline = sorted(missing & midline)
    if missing and set(missing_midline) == missing:
        if not apply_montage:
            raise ValueError("Cannot interpolate missing midline channels when montage disabled")

        # Ensure montage is applied before interpolation
        _apply_montage_best_effort(raw)

        # Add missing channels with zeros then interpolate
        import mne

        n_times = raw.get_data().shape[1]
        for ch in missing_midline:
            info_new = mne.create_info(ch_names=[ch], sfreq=raw.info["sfreq"], ch_types="eeg")
            zero = np.zeros((1, n_times), dtype=np.float64)  # MNE expects volts float64
            raw_new = mne.io.RawArray(zero, info_new)
            raw.add_channels([raw_new], force_update_info=True)

        # Re-apply montage so newly added channels receive positions
        _apply_montage_best_effort(raw)

        # Mark as bads and interpolate based on montage positions
        raw.info["bads"] = missing_midline.copy()
        with contextlib.suppress(Exception):
            raw.interpolate_bads(reset_bads=True)

        logger.warning(
            "Interpolated channels %s for file %s",
            str(missing_midline),
            str(file_path),
        )

        # Recompute availability after interpolation
        available = [ch for ch in target_channels if ch in raw.ch_names]
        missing = set(target_channels) - set(available)

    # Final check after optional interpolation
    if missing:
        raise ValueError(f"Missing required channels: {missing}")

    # Reorder/pick channels
    # NOTE: We use pick_channels() despite deprecation warning because:
    # - pick() doesn't support ordered=True parameter (checked MNE 1.10.1)
    # - We need ordered=True to ensure channel order matches REQUIRED_CHANNELS
    # - Until MNE adds ordered param to pick(), we must use pick_channels()
    raw.pick_channels(available, ordered=True)

    # Best-effort montage (permissive - won't fail if some positions missing)
    if apply_montage:
        _apply_montage_best_effort(raw)

    # Extract data and sampling rate
    data_volts = raw.get_data()  # (n_channels, n_samples) in volts
    data_microvolts = data_volts * 1e6  # Convert to µV
    fs = raw.info["sfreq"]

    return data_microvolts.astype(np.float32), fs


def _apply_montage_best_effort(raw: Any) -> None:
    """Apply standard_1020 montage without failing if positions missing.

    Modifies raw in-place.
    """
    import mne

    with contextlib.suppress(Exception):
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="ignore", verbose=False)


def parse_tusz_csv(csv_path: Path) -> tuple[float, list[tuple[float, float, str]]]:
    """Parse TUSZ-style annotations to get duration and seizure events.

    Args:
        csv_path: Path to annotations CSV file

    Returns:
        duration_seconds: Total recording duration
        events: List of (start_sec, end_sec, label) tuples

    Example CSV format:
        ```
        version,tse_v2.0.0
        duration_sec,1800.0000
        # More metadata lines...
        0.0000,16.0000,bckg
        16.0000,256.0000,seiz
        256.0000,1800.0000,bckg
        ```
    """
    events = []
    duration_seconds = 0.0

    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split(",")
            if len(parts) < 2:
                continue

            # Extract duration from header
            if parts[0] == "duration_sec":
                duration_seconds = float(parts[1])
                continue

            # Parse event rows
            if len(parts) >= 3:
                try:
                    start = float(parts[0])
                    end = float(parts[1])
                    label = parts[2].strip()
                    events.append((start, end, label))
                except ValueError:
                    continue  # Skip malformed lines

    return duration_seconds, events


def events_to_binary_mask(
    events: list[tuple[float, float, str]],
    duration_sec: float,
    fs: float,
    seizure_labels: set[str] | None = None,
) -> npt.NDArray[np.float32]:
    """Convert event list to binary seizure mask.

    Args:
        events: List of (start_sec, end_sec, label) tuples
        duration_sec: Total duration in seconds
        fs: Sampling frequency (Hz)
        seizure_labels: Set of labels to treat as seizures (default: {"seiz"})

    Returns:
        Binary mask of shape (n_samples,) with 1.0 for seizure, 0.0 for background
    """
    if seizure_labels is None:
        seizure_labels = {"seiz"}

    n_samples = int(duration_sec * fs)
    mask = np.zeros(n_samples, dtype=np.float32)

    for start_sec, end_sec, label in events:
        if label in seizure_labels:
            start_idx = int(start_sec * fs)
            end_idx = int(end_sec * fs)
            mask[start_idx:end_idx] = 1.0

    return mask
