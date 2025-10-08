import tempfile
from pathlib import Path

import pytest


class TestManifestNPZDetection:
    @pytest.fixture
    def temp_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            yield cache_dir

    def test_detects_npz_naming_in_manifest(self, temp_cache_dir):
        data_file = temp_cache_dir / "test_001_data.npy"
        labels_file = temp_cache_dir / "test_001_labels.npy"
        data_file.touch()
        labels_file.touch()

        manifest = {
            "partial_seizure": [
                {"cache_file": "test_001_windows.npz", "window_idx": 0, "label": 1}
            ],
            "full_seizure": [],
            "no_seizure": [],
        }

        first_entry = manifest["partial_seizure"][0]
        cache_file_ref = first_entry.get("cache_file", "")

        assert "_windows.npz" in cache_file_ref or ".npz" in cache_file_ref

    def test_does_not_detect_npy_naming(self, temp_cache_dir):
        data_file = temp_cache_dir / "test_001_data.npy"
        labels_file = temp_cache_dir / "test_001_labels.npy"
        data_file.touch()
        labels_file.touch()

        manifest = {
            "partial_seizure": [{"cache_file": "test_001_data.npy", "window_idx": 0, "label": 1}],
            "full_seizure": [],
            "no_seizure": [],
        }

        first_entry = manifest["partial_seizure"][0]
        cache_file_ref = first_entry.get("cache_file", "")

        assert "_windows.npz" not in cache_file_ref
        assert ".npz" not in cache_file_ref


class TestManifestFileExistence:
    @pytest.fixture
    def temp_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            yield cache_dir

    def test_detects_missing_file(self, temp_cache_dir):
        manifest = {
            "partial_seizure": [
                {"cache_file": "nonexistent_data.npy", "window_idx": 0, "label": 1}
            ],
            "full_seizure": [],
            "no_seizure": [],
        }

        first_entry = manifest["partial_seizure"][0]
        cache_file_ref = first_entry.get("cache_file", "")
        cache_file_path = temp_cache_dir / cache_file_ref

        assert not cache_file_path.exists()

    def test_detects_existing_file(self, temp_cache_dir):
        data_file = temp_cache_dir / "test_001_data.npy"
        labels_file = temp_cache_dir / "test_001_labels.npy"
        data_file.touch()
        labels_file.touch()

        manifest = {
            "partial_seizure": [{"cache_file": "test_001_data.npy", "window_idx": 0, "label": 1}],
            "full_seizure": [],
            "no_seizure": [],
        }

        first_entry = manifest["partial_seizure"][0]
        cache_file_ref = first_entry.get("cache_file", "")
        cache_file_path = temp_cache_dir / cache_file_ref

        assert cache_file_path.exists()


class TestWhitelistMatching:
    @pytest.fixture
    def temp_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            edf_dir = Path(tmpdir) / "edf" / "dev"
            edf_dir.mkdir(parents=True)
            yield cache_dir, edf_dir

    def test_whitelist_mismatch_filters_all(self, temp_cache_dir):
        cache_dir, edf_dir = temp_cache_dir

        data_file = cache_dir / "patient_001_data.npy"
        labels_file = cache_dir / "patient_001_labels.npy"
        data_file.touch()
        labels_file.touch()

        edf_file = edf_dir / "patient_002.edf"
        edf_file.touch()

        manifest = {
            "partial_seizure": [
                {"cache_file": "patient_001_data.npy", "window_idx": 0, "label": 1}
            ],
            "full_seizure": [],
            "no_seizure": [],
        }

        edf_files = list(edf_dir.glob("**/*.edf"))
        allowed_cache_files = {f"{edf.stem}_data.npy" for edf in edf_files}

        matched = 0
        for category in ["partial_seizure", "full_seizure", "no_seizure"]:
            entries = manifest.get(category, [])
            for entry in entries:
                if entry.get("cache_file") in allowed_cache_files:
                    matched += 1

        assert matched == 0

    def test_whitelist_match_succeeds(self, temp_cache_dir):
        cache_dir, edf_dir = temp_cache_dir

        data_file = cache_dir / "patient_001_data.npy"
        labels_file = cache_dir / "patient_001_labels.npy"
        data_file.touch()
        labels_file.touch()

        edf_file = edf_dir / "patient_001.edf"
        edf_file.touch()

        manifest = {
            "partial_seizure": [
                {"cache_file": "patient_001_data.npy", "window_idx": 0, "label": 1}
            ],
            "full_seizure": [],
            "no_seizure": [],
        }

        edf_files = list(edf_dir.glob("**/*.edf"))
        allowed_cache_files = {f"{edf.stem}_data.npy" for edf in edf_files}

        matched = 0
        for category in ["partial_seizure", "full_seizure", "no_seizure"]:
            entries = manifest.get(category, [])
            for entry in entries:
                if entry.get("cache_file") in allowed_cache_files:
                    matched += 1

        assert matched == 1

    def test_npz_naming_causes_whitelist_mismatch(self, temp_cache_dir):
        cache_dir, edf_dir = temp_cache_dir

        data_file = cache_dir / "patient_001_data.npy"
        labels_file = cache_dir / "patient_001_labels.npy"
        data_file.touch()
        labels_file.touch()

        edf_file = edf_dir / "patient_001.edf"
        edf_file.touch()

        manifest = {
            "partial_seizure": [
                {"cache_file": "patient_001_windows.npz", "window_idx": 0, "label": 1}
            ],
            "full_seizure": [],
            "no_seizure": [],
        }

        edf_files = list(edf_dir.glob("**/*.edf"))
        allowed_cache_files = {f"{edf.stem}_data.npy" for edf in edf_files}

        matched = 0
        for category in ["partial_seizure", "full_seizure", "no_seizure"]:
            entries = manifest.get(category, [])
            for entry in entries:
                if entry.get("cache_file") in allowed_cache_files:
                    matched += 1

        assert matched == 0
