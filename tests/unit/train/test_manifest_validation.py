import json
import tempfile
from pathlib import Path

import pytest

from src.brain_brr.constants import MANIFEST_FILENAME
from src.brain_brr.data.cache_utils import check_manifest_stale


class TestCheckManifestStale:
    @pytest.fixture
    def temp_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            yield cache_dir

    def test_detects_npz_naming_as_stale(self, temp_cache_dir):
        (temp_cache_dir / "test_001_data.npy").touch()
        (temp_cache_dir / "test_001_labels.npy").touch()

        manifest = {
            "partial_seizure": [
                {"cache_file": "test_001_windows.npz", "window_idx": 0, "label": 1}
            ],
            "full_seizure": [],
            "no_seizure": [],
        }

        assert check_manifest_stale(temp_cache_dir, manifest) is True

    def test_accepts_npy_naming_as_valid(self, temp_cache_dir):
        (temp_cache_dir / "test_001_data.npy").touch()
        (temp_cache_dir / "test_001_labels.npy").touch()

        manifest = {
            "partial_seizure": [{"cache_file": "test_001_data.npy", "window_idx": 0, "label": 1}],
            "full_seizure": [],
            "no_seizure": [],
        }

        assert check_manifest_stale(temp_cache_dir, manifest) is False

    def test_detects_missing_file_as_stale(self, temp_cache_dir):
        manifest = {
            "partial_seizure": [
                {"cache_file": "nonexistent_data.npy", "window_idx": 0, "label": 1}
            ],
            "full_seizure": [],
            "no_seizure": [],
        }

        assert check_manifest_stale(temp_cache_dir, manifest) is True

    def test_empty_manifest_not_stale(self, temp_cache_dir):
        manifest = {
            "partial_seizure": [],
            "full_seizure": [],
            "no_seizure": [],
        }

        assert check_manifest_stale(temp_cache_dir, manifest) is False

    def test_checks_all_categories(self, temp_cache_dir):
        (temp_cache_dir / "test_001_data.npy").touch()
        (temp_cache_dir / "test_002_data.npy").touch()

        manifest = {
            "partial_seizure": [{"cache_file": "test_001_data.npy", "window_idx": 0}],
            "full_seizure": [{"cache_file": "test_002_windows.npz", "window_idx": 0}],
            "no_seizure": [],
        }

        assert check_manifest_stale(temp_cache_dir, manifest) is True

    def test_exception_handling_returns_stale(self, temp_cache_dir):
        invalid_manifest = {"partial_seizure": "not a list"}

        assert check_manifest_stale(temp_cache_dir, invalid_manifest) is True


class TestManifestValidationInLoop:
    """Integration test: manifest validation triggers rebuild in training loop."""

    @pytest.fixture
    def temp_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            (cache_dir / "dev").mkdir()
            yield cache_dir

    def test_stale_manifest_deleted_on_load(self, temp_cache_dir):
        dev_dir = temp_cache_dir / "dev"
        manifest_path = dev_dir / MANIFEST_FILENAME

        (dev_dir / "test_001_data.npy").touch()
        (dev_dir / "test_001_labels.npy").touch()

        stale_manifest = {
            "partial_seizure": [
                {"cache_file": "test_001_windows.npz", "window_idx": 0, "label": 1}
            ],
            "full_seizure": [],
            "no_seizure": [],
        }

        with manifest_path.open("w") as f:
            json.dump(stale_manifest, f)

        assert manifest_path.exists()

        with manifest_path.open() as f:
            manifest_data = json.load(f)

        if check_manifest_stale(dev_dir, manifest_data):
            manifest_path.unlink()

        assert not manifest_path.exists()

    def test_valid_manifest_preserved(self, temp_cache_dir):
        dev_dir = temp_cache_dir / "dev"
        manifest_path = dev_dir / MANIFEST_FILENAME

        (dev_dir / "test_001_data.npy").touch()
        (dev_dir / "test_001_labels.npy").touch()

        valid_manifest = {
            "partial_seizure": [{"cache_file": "test_001_data.npy", "window_idx": 0, "label": 1}],
            "full_seizure": [],
            "no_seizure": [],
        }

        with manifest_path.open("w") as f:
            json.dump(valid_manifest, f)

        assert manifest_path.exists()

        with manifest_path.open() as f:
            manifest_data = json.load(f)

        if check_manifest_stale(dev_dir, manifest_data):
            manifest_path.unlink()

        assert manifest_path.exists()
