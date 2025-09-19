import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch

from src.experiment import constants
from src.experiment import data as data_mod


class FakeRaw:
    """Lightweight stand-in for mne.io.Raw for testing load_edf_file."""

    def __init__(self, ch_names: list[str], data: np.ndarray, sfreq: float = 256.0):
        self.ch_names = list(ch_names)
        self._data = data
        self.info: dict[str, float] = {"sfreq": float(sfreq)}

    def rename_channels(self, mapping: dict[str, str] | Callable) -> None:
        if callable(mapping):
            # Handle lambda function
            self.ch_names = [mapping(ch) for ch in self.ch_names]
        else:
            # Handle dictionary
            for i, name in enumerate(self.ch_names):
                if name in mapping:
                    self.ch_names[i] = mapping[name]

    def pick_channels(self, picks: list[str], ordered: bool = True) -> None:
        idx = [self.ch_names.index(ch) for ch in picks]
        self.ch_names = [self.ch_names[i] for i in idx]
        self._data = self._data[idx]

    def get_data(self) -> np.ndarray:
        return self._data


@pytest.mark.unit
def test_load_edf_orders_channels_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    # Build a fake recording with shuffled/synonym channels
    chans = constants.CHANNEL_NAMES_10_20.copy()
    # Introduce synonyms: replace T3→T7, T4→T8
    chans_syn = ["T7" if c == "T3" else "T8" if c == "T4" else c for c in chans]
    rng = np.random.default_rng(123)
    sig = rng.standard_normal((len(chans_syn), constants.WINDOW_SAMPLES)).astype(np.float32)
    raw = FakeRaw(ch_names=chans_syn, data=sig, sfreq=256.0)

    # Patch the actual module where the function is defined
    import src.brain_brr.data.io as io_mod

    monkeypatch.setattr(io_mod, "_read_raw_edf", lambda p: raw)

    arr, fs = data_mod.load_edf_file(Path("dummy.edf"))
    assert isinstance(fs, float)
    assert fs == 256.0
    assert arr.shape == (19, constants.WINDOW_SAMPLES)
    # Verify channel order matches canonical list after synonym renames
    assert raw.ch_names == constants.CHANNEL_NAMES_10_20


@pytest.mark.unit
def test_load_edf_missing_channels_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Remove a required channel
    required = constants.CHANNEL_NAMES_10_20.copy()
    missing = required[:-1]
    rng = np.random.default_rng(0)
    sig = rng.standard_normal((len(missing), 1000)).astype(np.float32)
    raw = FakeRaw(ch_names=missing, data=sig, sfreq=250.0)

    # Patch the actual module where the function is defined
    import src.brain_brr.data.io as io_mod

    monkeypatch.setattr(io_mod, "_read_raw_edf", lambda p: raw)
    with pytest.raises(ValueError, match="Missing required channels"):
        _ = data_mod.load_edf_file(Path("dummy.edf"))


@pytest.mark.unit
def test_load_edf_header_repair_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that malformed EDF headers trigger repair fallback."""
    # Create a mock EDF file with bad header (colons in date field)
    edf_path = tmp_path / "bad_header.edf"
    with open(edf_path, "wb") as f:
        # Write minimal EDF header with bad date
        f.write(b"0       ")  # version
        f.write(b" " * 80)  # patient ID
        f.write(b" " * 80)  # recording ID
        f.write(b"01:01:85")  # date with colons (bad!)
        f.write(b"00.00.00")  # time
        f.write(b" " * (256 - f.tell()))  # pad to 256

    # First call should fail with header error
    call_count = 0

    def mock_read_edf(path: Path) -> FakeRaw:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First attempt fails with header error
            raise RuntimeError("the file is not EDF(+) compliant, the startdate is incorrect")
        else:
            # Second attempt (after repair) succeeds
            sig = np.random.randn(19, 15360).astype(np.float32)
            return FakeRaw(ch_names=constants.CHANNEL_NAMES_10_20, data=sig, sfreq=256.0)

    # Patch the actual module where the function is defined
    import src.brain_brr.data.io as io_mod

    monkeypatch.setattr(io_mod, "_read_raw_edf", mock_read_edf)

    # Should succeed after repair
    arr, fs = data_mod.load_edf_file(edf_path)
    assert arr.shape == (19, 15360)
    assert fs == 256.0
    assert call_count == 2  # Called twice: failed, then succeeded after repair


@pytest.mark.unit
def test_preprocess_shapes_and_dtype() -> None:
    # 2 channels, 4 seconds @ 512 Hz → resample to 256 Hz yields 1024 samples
    fs_in = 512.0
    t = np.arange(0, 4.0, 1.0 / fs_in)
    x = np.vstack(
        [
            np.sin(2 * math.pi * 10 * t),
            np.cos(2 * math.pi * 20 * t),
        ]
    )
    x = x.astype(np.float32)

    y = data_mod.preprocess_recording(x, fs_original=fs_in, target_fs=256)
    assert y.dtype == np.float32
    assert y.shape[0] == 2
    expected_len = x.shape[1] // 2  # 512 → 256 halves the samples
    assert y.shape[1] == expected_len
    assert np.isfinite(y).all()


@pytest.mark.unit
def test_extract_windows_counts_and_metadata() -> None:
    rng = np.random.default_rng(42)
    n_samples = constants.WINDOW_SAMPLES + 2 * constants.STRIDE_SAMPLES  # 3 windows
    x = rng.standard_normal((19, n_samples)).astype(np.float32)

    windows, y, meta = data_mod.extract_windows(
        x, constants.WINDOW_SAMPLES, constants.STRIDE_SAMPLES
    )
    assert windows.shape == (3, 19, constants.WINDOW_SAMPLES)
    assert y is None
    assert meta["start_samples"] == [0, constants.STRIDE_SAMPLES, 2 * constants.STRIDE_SAMPLES]


@pytest.mark.unit
def test_extract_windows_with_labels_alignment() -> None:
    n_samples = constants.WINDOW_SAMPLES + constants.STRIDE_SAMPLES  # 2 windows
    x = np.zeros((19, n_samples), dtype=np.float32)
    y = np.zeros((n_samples,), dtype=np.float32)
    # Mark the second half of the record as 1s
    y[n_samples // 2 :] = 1.0

    windows, labels, _ = data_mod.extract_windows(
        x, constants.WINDOW_SAMPLES, constants.STRIDE_SAMPLES, labels=y
    )
    assert windows.shape[0] == 2
    assert labels is not None
    assert labels.shape == (2, constants.WINDOW_SAMPLES)
    # First window should overlap with some 1s only if the half threshold is crossed
    assert labels[1].mean() > labels[0].mean()


@pytest.mark.integration
def test_dataset_len_and_item_shapes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Monkeypatch load_edf_file to avoid MNE dependency
    def _fake_load(_path: Path):
        # Build a deterministic synthetic signal (exactly 3 windows @ 256 Hz)
        n = constants.WINDOW_SAMPLES + 2 * constants.STRIDE_SAMPLES
        sig = np.zeros((19, n), dtype=np.float32)
        return sig, float(constants.SAMPLING_RATE)

    # Patch the actual call site used inside EEGWindowDataset
    import src.brain_brr.data.datasets as ds_mod

    monkeypatch.setattr(ds_mod, "load_edf_file", _fake_load)

    edf_files = [tmp_path / "a.edf", tmp_path / "b.edf"]
    ds = data_mod.EEGWindowDataset(edf_files=edf_files, cache_dir=None)

    # 3 windows per file → 6 total
    assert len(ds) == 6
    x0 = ds[0]
    if isinstance(x0, tuple):
        x0 = x0[0]
    assert isinstance(x0, torch.Tensor)
    assert x0.shape == (19, constants.WINDOW_SAMPLES)
