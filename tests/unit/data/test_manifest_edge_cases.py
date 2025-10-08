import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.brain_brr.constants import MANIFEST_FILENAME
from src.brain_brr.data.cache_utils import scan_existing_cache, validate_manifest


class TestManifestBuilding:
    @pytest.fixture
    def temp_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            yield cache_dir

    @pytest.fixture
    def create_npy_files(self, temp_cache_dir):
        def _create(file_stems, num_windows=10):
            for stem in file_stems:
                data_file = temp_cache_dir / f"{stem}_data.npy"
                labels_file = temp_cache_dir / f"{stem}_labels.npy"

                data = np.random.randn(num_windows, 19, 15360).astype(np.float32)
                labels = np.random.randint(0, 3, size=(num_windows,), dtype=np.int64)

                np.save(data_file, data)
                np.save(labels_file, labels)

        return _create

    def test_build_manifest_from_npy_files(self, temp_cache_dir, create_npy_files):
        create_npy_files(["patient_001", "patient_002"])

        with patch("src.brain_brr.data.cache_utils.logger"):
            manifest = scan_existing_cache(temp_cache_dir)

        assert manifest is not None
        total_windows = (
            len(manifest["partial_seizure"])
            + len(manifest["full_seizure"])
            + len(manifest["no_seizure"])
        )
        assert total_windows == 20

    def test_build_manifest_empty_cache(self, temp_cache_dir):
        with patch("src.brain_brr.data.cache_utils.logger"):
            manifest = scan_existing_cache(temp_cache_dir)

        assert manifest is not None
        assert len(manifest["partial_seizure"]) == 0
        assert len(manifest["full_seizure"]) == 0
        assert len(manifest["no_seizure"]) == 0

    def test_build_manifest_categorizes_labels(self, temp_cache_dir):
        data_file = temp_cache_dir / "test_data.npy"
        labels_file = temp_cache_dir / "test_labels.npy"

        data = np.random.randn(6, 19, 15360).astype(np.float32)
        labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)

        np.save(data_file, data)
        np.save(labels_file, labels)

        with patch("src.brain_brr.data.cache_utils.logger"):
            manifest = scan_existing_cache(temp_cache_dir)

        assert len(manifest["no_seizure"]) == 2
        assert len(manifest["partial_seizure"]) == 2
        assert len(manifest["full_seizure"]) == 2

    def test_build_manifest_saves_to_disk(self, temp_cache_dir, create_npy_files):
        create_npy_files(["patient_001"])

        manifest_path = temp_cache_dir / MANIFEST_FILENAME
        assert not manifest_path.exists()

        with patch("src.brain_brr.data.cache_utils.logger"):
            scan_existing_cache(temp_cache_dir)

        assert manifest_path.exists()

    def test_build_manifest_window_indices(self, temp_cache_dir):
        data_file = temp_cache_dir / "test_data.npy"
        labels_file = temp_cache_dir / "test_labels.npy"

        data = np.random.randn(5, 19, 15360).astype(np.float32)
        labels = np.array([0, 0, 0, 0, 0], dtype=np.int64)

        np.save(data_file, data)
        np.save(labels_file, labels)

        with patch("src.brain_brr.data.cache_utils.logger"):
            manifest = scan_existing_cache(temp_cache_dir)

        window_indices = [entry["window_idx"] for entry in manifest["no_seizure"]]
        assert window_indices == [0, 1, 2, 3, 4]

    def test_build_manifest_ignores_npz_files_when_npy_exists(
        self, temp_cache_dir, create_npy_files
    ):
        create_npy_files(["patient_001"])

        npz_file = temp_cache_dir / "patient_002_windows.npz"
        npz_file.touch()

        with patch("src.brain_brr.data.cache_utils.logger"):
            manifest = scan_existing_cache(temp_cache_dir)

        total_windows = (
            len(manifest["partial_seizure"])
            + len(manifest["full_seizure"])
            + len(manifest["no_seizure"])
        )
        assert total_windows == 10

    def test_build_manifest_handles_npz_when_no_npy(self, temp_cache_dir):
        npz_file = temp_cache_dir / "test_windows.npz"

        data = np.random.randn(5, 19, 15360).astype(np.float32)
        labels = np.zeros((5,), dtype=np.int64)

        np.savez(npz_file, windows=data, labels=labels)

        with patch("src.brain_brr.data.cache_utils.logger"):
            manifest = scan_existing_cache(temp_cache_dir)

        assert len(manifest["no_seizure"]) == 5

    def test_build_manifest_skips_missing_labels_npy(self, temp_cache_dir):
        data_file = temp_cache_dir / "test_data.npy"

        data = np.random.randn(5, 19, 15360).astype(np.float32)
        np.save(data_file, data)

        with patch("src.brain_brr.data.cache_utils.logger"):
            with pytest.warns(UserWarning, match="has NO LABELS FILE"):
                manifest = scan_existing_cache(temp_cache_dir)

        total_windows = (
            len(manifest["partial_seizure"])
            + len(manifest["full_seizure"])
            + len(manifest["no_seizure"])
        )
        assert total_windows == 0


class TestManifestValidation:
    @pytest.fixture
    def temp_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            yield cache_dir

    def test_validate_manifest_valid_npy(self, temp_cache_dir):
        data_file = temp_cache_dir / "test_data.npy"
        labels_file = temp_cache_dir / "test_labels.npy"
        data_file.touch()
        labels_file.touch()

        manifest = {
            "partial_seizure": [],
            "full_seizure": [],
            "no_seizure": [{"cache_file": "test_data.npy", "window_idx": 0, "label": 0}],
        }

        assert validate_manifest(temp_cache_dir, manifest)

    def test_validate_manifest_missing_files(self, temp_cache_dir):
        manifest = {
            "partial_seizure": [],
            "full_seizure": [],
            "no_seizure": [{"cache_file": "nonexistent_data.npy", "window_idx": 0, "label": 0}],
        }

        assert not validate_manifest(temp_cache_dir, manifest)

    def test_validate_manifest_empty(self, temp_cache_dir):
        manifest = {"partial_seizure": [], "full_seizure": [], "no_seizure": []}

        assert not validate_manifest(temp_cache_dir, manifest)

    def test_validate_manifest_partial_missing(self, temp_cache_dir):
        data_file = temp_cache_dir / "test_data.npy"
        labels_file = temp_cache_dir / "test_labels.npy"
        data_file.touch()
        labels_file.touch()

        manifest = {
            "partial_seizure": [],
            "full_seizure": [],
            "no_seizure": [
                {"cache_file": "test_data.npy", "window_idx": 0, "label": 0},
                {"cache_file": "missing_data.npy", "window_idx": 0, "label": 0},
            ],
        }

        assert validate_manifest(temp_cache_dir, manifest)

    def test_validate_manifest_too_many_missing(self, temp_cache_dir):
        data_file = temp_cache_dir / "test_data.npy"
        labels_file = temp_cache_dir / "test_labels.npy"
        data_file.touch()
        labels_file.touch()

        manifest = {
            "partial_seizure": [],
            "full_seizure": [],
            "no_seizure": [{"cache_file": "test_data.npy", "window_idx": 0, "label": 0}]
            + [
                {"cache_file": f"missing_{i}_data.npy", "window_idx": 0, "label": 0}
                for i in range(20)
            ],
        }

        assert not validate_manifest(temp_cache_dir, manifest)

    def test_validate_manifest_npz_format(self, temp_cache_dir):
        npz_file = temp_cache_dir / "test_windows.npz"
        npz_file.touch()

        manifest = {
            "partial_seizure": [],
            "full_seizure": [],
            "no_seizure": [{"cache_file": "test_windows.npz", "window_idx": 0, "label": 0}],
        }

        assert validate_manifest(temp_cache_dir, manifest)

    def test_validate_manifest_mixed_npz_npy(self, temp_cache_dir):
        npy_data_file = temp_cache_dir / "test1_data.npy"
        npy_labels_file = temp_cache_dir / "test1_labels.npy"
        npy_data_file.touch()
        npy_labels_file.touch()

        npz_file = temp_cache_dir / "test2_windows.npz"
        npz_file.touch()

        manifest = {
            "partial_seizure": [],
            "full_seizure": [],
            "no_seizure": [
                {"cache_file": "test1_data.npy", "window_idx": 0, "label": 0},
                {"cache_file": "test2_windows.npz", "window_idx": 0, "label": 0},
            ],
        }

        assert validate_manifest(temp_cache_dir, manifest)


class TestManifestCategorizationLogic:
    @pytest.fixture
    def temp_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            yield cache_dir

    def test_no_seizure_categorization(self, temp_cache_dir):
        data_file = temp_cache_dir / "test_data.npy"
        labels_file = temp_cache_dir / "test_labels.npy"

        data = np.random.randn(3, 19, 15360).astype(np.float32)
        labels = np.array([0, 0, 0], dtype=np.int64)

        np.save(data_file, data)
        np.save(labels_file, labels)

        with patch("src.brain_brr.data.cache_utils.logger"):
            manifest = scan_existing_cache(temp_cache_dir)

        assert len(manifest["no_seizure"]) == 3
        assert len(manifest["partial_seizure"]) == 0
        assert len(manifest["full_seizure"]) == 0

    def test_full_seizure_categorization(self, temp_cache_dir):
        data_file = temp_cache_dir / "test_data.npy"
        labels_file = temp_cache_dir / "test_labels.npy"

        data = np.random.randn(3, 19, 15360).astype(np.float32)
        labels = np.ones((3,), dtype=np.int64)

        np.save(data_file, data)
        np.save(labels_file, labels)

        with patch("src.brain_brr.data.cache_utils.logger"):
            manifest = scan_existing_cache(temp_cache_dir)

        assert len(manifest["no_seizure"]) == 0
        assert len(manifest["partial_seizure"]) == 0
        assert len(manifest["full_seizure"]) == 3

    def test_partial_seizure_categorization(self, temp_cache_dir):
        data_file = temp_cache_dir / "test_data.npy"
        labels_file = temp_cache_dir / "test_labels.npy"

        data = np.random.randn(3, 19, 15360).astype(np.float32)
        labels = np.array(
            [
                np.concatenate([np.zeros(7680), np.ones(7680)]),
                np.concatenate([np.zeros(7680), np.ones(7680)]),
                np.concatenate([np.zeros(7680), np.ones(7680)]),
            ]
        ).astype(np.int64)

        np.save(data_file, data)
        np.save(labels_file, labels)

        with patch("src.brain_brr.data.cache_utils.logger"):
            manifest = scan_existing_cache(temp_cache_dir)

        assert len(manifest["no_seizure"]) == 0
        assert len(manifest["partial_seizure"]) == 3
        assert len(manifest["full_seizure"]) == 0

    def test_mixed_categorization(self, temp_cache_dir):
        data_file = temp_cache_dir / "test_data.npy"
        labels_file = temp_cache_dir / "test_labels.npy"

        data = np.random.randn(4, 19, 15360).astype(np.float32)
        labels = np.array(
            [
                np.zeros(15360),
                np.ones(15360),
                np.concatenate([np.zeros(7680), np.ones(7680)]),
                np.zeros(15360),
            ]
        ).astype(np.int64)

        np.save(data_file, data)
        np.save(labels_file, labels)

        with patch("src.brain_brr.data.cache_utils.logger"):
            manifest = scan_existing_cache(temp_cache_dir)

        assert len(manifest["no_seizure"]) == 2
        assert len(manifest["partial_seizure"]) == 1
        assert len(manifest["full_seizure"]) == 1
