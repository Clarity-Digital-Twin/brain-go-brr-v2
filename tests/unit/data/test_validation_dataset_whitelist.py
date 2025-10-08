import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.brain_brr.data.datasets import ValidationDataset


class TestValidationDatasetWhitelist:
    @pytest.fixture
    def temp_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            yield cache_dir

    @pytest.fixture
    def create_cache_files(self, temp_cache_dir):
        def _create(file_stems):
            for stem in file_stems:
                data_file = temp_cache_dir / f"{stem}_data.npy"
                labels_file = temp_cache_dir / f"{stem}_labels.npy"

                np.save(data_file, np.random.randn(10, 19, 15360).astype(np.float32))
                np.save(labels_file, np.zeros((10,), dtype=np.int64))

        return _create

    @pytest.fixture
    def create_manifest(self, temp_cache_dir):
        def _create(entries):
            manifest = {"partial_seizure": [], "full_seizure": [], "no_seizure": []}

            for cache_file, category, count in entries:
                for i in range(count):
                    manifest[category].append(
                        {"cache_file": cache_file, "window_idx": i, "label": 0}
                    )

            manifest_path = temp_cache_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            return manifest_path

        return _create

    def test_whitelist_filters_all_entries(
        self, temp_cache_dir, create_cache_files, create_manifest
    ):
        create_cache_files(["patient_001", "patient_002"])
        create_manifest(
            [
                ("patient_001_data.npy", "no_seizure", 5),
                ("patient_002_data.npy", "no_seizure", 5),
            ]
        )

        allowed_cache_files = {"patient_003_data.npy"}

        with patch("src.brain_brr.data.datasets.logger"):
            dataset = ValidationDataset(
                cache_dir=str(temp_cache_dir),
                allowed_cache_files=allowed_cache_files,
            )

            assert len(dataset) == 0

    def test_whitelist_allows_matching_entries(
        self, temp_cache_dir, create_cache_files, create_manifest
    ):
        create_cache_files(["patient_001", "patient_002"])
        create_manifest(
            [
                ("patient_001_data.npy", "no_seizure", 5),
                ("patient_002_data.npy", "no_seizure", 5),
            ]
        )

        allowed_cache_files = {"patient_001_data.npy"}

        with patch("src.brain_brr.data.datasets.logger"):
            dataset = ValidationDataset(
                cache_dir=str(temp_cache_dir),
                allowed_cache_files=allowed_cache_files,
            )

            assert len(dataset) == 5

    def test_whitelist_partial_match(self, temp_cache_dir, create_cache_files, create_manifest):
        create_cache_files(["patient_001", "patient_002", "patient_003"])
        create_manifest(
            [
                ("patient_001_data.npy", "no_seizure", 3),
                ("patient_002_data.npy", "no_seizure", 4),
                ("patient_003_data.npy", "no_seizure", 5),
            ]
        )

        allowed_cache_files = {"patient_001_data.npy", "patient_003_data.npy"}

        with patch("src.brain_brr.data.datasets.logger"):
            dataset = ValidationDataset(
                cache_dir=str(temp_cache_dir),
                allowed_cache_files=allowed_cache_files,
            )

            assert len(dataset) == 8

    def test_whitelist_none_allows_all(self, temp_cache_dir, create_cache_files, create_manifest):
        create_cache_files(["patient_001", "patient_002"])
        create_manifest(
            [
                ("patient_001_data.npy", "no_seizure", 5),
                ("patient_002_data.npy", "no_seizure", 5),
            ]
        )

        with patch("src.brain_brr.data.datasets.logger"):
            dataset = ValidationDataset(
                cache_dir=str(temp_cache_dir),
                allowed_cache_files=None,
            )

            assert len(dataset) == 10

    def test_whitelist_empty_set_filters_all(
        self, temp_cache_dir, create_cache_files, create_manifest
    ):
        create_cache_files(["patient_001", "patient_002"])
        create_manifest(
            [
                ("patient_001_data.npy", "no_seizure", 5),
                ("patient_002_data.npy", "no_seizure", 5),
            ]
        )

        allowed_cache_files = set()

        with patch("src.brain_brr.data.datasets.logger"):
            dataset = ValidationDataset(
                cache_dir=str(temp_cache_dir),
                allowed_cache_files=allowed_cache_files,
            )

            assert len(dataset) == 0

    def test_whitelist_with_multiple_categories(
        self, temp_cache_dir, create_cache_files, create_manifest
    ):
        create_cache_files(["patient_001", "patient_002", "patient_003"])

        manifest = {
            "partial_seizure": [
                {"cache_file": "patient_001_data.npy", "window_idx": 0, "label": 1}
            ],
            "full_seizure": [{"cache_file": "patient_002_data.npy", "window_idx": 0, "label": 2}],
            "no_seizure": [{"cache_file": "patient_003_data.npy", "window_idx": 0, "label": 0}],
        }

        manifest_path = temp_cache_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        allowed_cache_files = {"patient_001_data.npy", "patient_003_data.npy"}

        with patch("src.brain_brr.data.datasets.logger"):
            dataset = ValidationDataset(
                cache_dir=str(temp_cache_dir),
                allowed_cache_files=allowed_cache_files,
            )

            assert len(dataset) == 2

    def test_whitelist_case_sensitivity(self, temp_cache_dir, create_cache_files, create_manifest):
        create_cache_files(["patient_001"])
        create_manifest([("patient_001_data.npy", "no_seizure", 5)])

        allowed_cache_files = {"PATIENT_001_DATA.NPY"}

        with patch("src.brain_brr.data.datasets.logger"):
            dataset = ValidationDataset(
                cache_dir=str(temp_cache_dir),
                allowed_cache_files=allowed_cache_files,
            )

            assert len(dataset) == 0

    def test_whitelist_with_npz_manifest_entries(
        self, temp_cache_dir, create_cache_files, create_manifest
    ):
        create_cache_files(["patient_001"])

        manifest = {
            "partial_seizure": [],
            "full_seizure": [],
            "no_seizure": [
                {"cache_file": "patient_001_windows.npz", "window_idx": 0, "label": 0},
                {"cache_file": "patient_001_windows.npz", "window_idx": 1, "label": 0},
            ],
        }

        manifest_path = temp_cache_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        allowed_cache_files = {"patient_001_data.npy"}

        with patch("src.brain_brr.data.datasets.logger"):
            dataset = ValidationDataset(
                cache_dir=str(temp_cache_dir),
                allowed_cache_files=allowed_cache_files,
            )

            assert len(dataset) == 0

    def test_getitem_returns_correct_window(
        self, temp_cache_dir, create_cache_files, create_manifest
    ):
        create_cache_files(["patient_001"])
        create_manifest([("patient_001_data.npy", "no_seizure", 5)])

        allowed_cache_files = {"patient_001_data.npy"}

        with patch("src.brain_brr.data.datasets.logger"):
            dataset = ValidationDataset(
                cache_dir=str(temp_cache_dir),
                allowed_cache_files=allowed_cache_files,
            )

            assert len(dataset) == 5

            result = dataset[0]

            assert isinstance(result, dict)
            assert "window" in result
            assert "label" in result
            assert "file_id" in result
            assert "window_start_s" in result

            assert isinstance(result["window"], torch.Tensor)
            assert isinstance(result["label"], torch.Tensor)
            assert result["window"].shape == (19, 15360)
            assert result["label"].shape == (15360,)

    def test_whitelist_maintains_window_indices(
        self, temp_cache_dir, create_cache_files, create_manifest
    ):
        create_cache_files(["patient_001", "patient_002"])

        manifest = {
            "partial_seizure": [],
            "full_seizure": [],
            "no_seizure": [
                {"cache_file": "patient_001_data.npy", "window_idx": 0, "label": 0},
                {"cache_file": "patient_002_data.npy", "window_idx": 0, "label": 0},
                {"cache_file": "patient_001_data.npy", "window_idx": 1, "label": 0},
                {"cache_file": "patient_002_data.npy", "window_idx": 1, "label": 0},
            ],
        }

        manifest_path = temp_cache_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        allowed_cache_files = {"patient_001_data.npy"}

        with patch("src.brain_brr.data.datasets.logger"):
            dataset = ValidationDataset(
                cache_dir=str(temp_cache_dir),
                allowed_cache_files=allowed_cache_files,
            )

            assert len(dataset) == 2


class TestValidationDatasetEdgeCases:
    @pytest.fixture
    def temp_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            yield cache_dir

    def test_missing_manifest_builds_automatically(self, temp_cache_dir):
        data_file = temp_cache_dir / "patient_001_data.npy"
        labels_file = temp_cache_dir / "patient_001_labels.npy"
        np.save(data_file, np.random.randn(5, 19, 15360).astype(np.float32))
        np.save(labels_file, np.zeros((5,), dtype=np.int64))

        allowed_cache_files = {"patient_001_data.npy"}

        with patch("src.brain_brr.data.datasets.logger"):
            dataset = ValidationDataset(
                cache_dir=str(temp_cache_dir),
                allowed_cache_files=allowed_cache_files,
            )

            assert len(dataset) == 5

    def test_corrupted_manifest_raises_error(self, temp_cache_dir):
        manifest_path = temp_cache_dir / "manifest.json"
        manifest_path.write_text("{invalid json")

        allowed_cache_files = {"patient_001_data.npy"}

        with patch("src.brain_brr.data.datasets.logger"):
            with pytest.raises(json.JSONDecodeError):
                ValidationDataset(
                    cache_dir=str(temp_cache_dir),
                    allowed_cache_files=allowed_cache_files,
                )

    def test_empty_manifest_creates_empty_dataset(self, temp_cache_dir):
        manifest = {"partial_seizure": [], "full_seizure": [], "no_seizure": []}

        manifest_path = temp_cache_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        allowed_cache_files = {"patient_001_data.npy"}

        with patch("src.brain_brr.data.datasets.logger"):
            dataset = ValidationDataset(
                cache_dir=str(temp_cache_dir),
                allowed_cache_files=allowed_cache_files,
            )

            assert len(dataset) == 0

    def test_manifest_with_missing_categories(self, temp_cache_dir):
        manifest = {"partial_seizure": []}

        manifest_path = temp_cache_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        allowed_cache_files = {"patient_001_data.npy"}

        with patch("src.brain_brr.data.datasets.logger"):
            dataset = ValidationDataset(
                cache_dir=str(temp_cache_dir),
                allowed_cache_files=allowed_cache_files,
            )

            assert len(dataset) == 0


class TestValidationDatasetLogging:
    @pytest.fixture
    def temp_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            yield cache_dir

    def test_logs_filtered_count(self, temp_cache_dir):
        manifest = {
            "partial_seizure": [
                {"cache_file": "patient_001_data.npy", "window_idx": 0, "label": 1}
            ],
            "full_seizure": [],
            "no_seizure": [{"cache_file": "patient_002_data.npy", "window_idx": 0, "label": 0}],
        }

        manifest_path = temp_cache_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        data_file = temp_cache_dir / "patient_001_data.npy"
        labels_file = temp_cache_dir / "patient_001_labels.npy"
        np.save(data_file, np.random.randn(10, 19, 15360).astype(np.float32))
        np.save(labels_file, np.zeros((10,), dtype=np.int64))

        allowed_cache_files = {"patient_001_data.npy"}

        with patch("src.brain_brr.data.datasets.logger") as mock_logger:
            dataset = ValidationDataset(
                cache_dir=str(temp_cache_dir),
                allowed_cache_files=allowed_cache_files,
            )

            assert len(dataset) == 1

    def test_logs_zero_windows_warning(self, temp_cache_dir):
        manifest = {
            "partial_seizure": [
                {"cache_file": "patient_001_data.npy", "window_idx": 0, "label": 1}
            ],
            "full_seizure": [],
            "no_seizure": [],
        }

        manifest_path = temp_cache_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        data_file = temp_cache_dir / "patient_001_data.npy"
        labels_file = temp_cache_dir / "patient_001_labels.npy"
        np.save(data_file, np.random.randn(10, 19, 15360).astype(np.float32))
        np.save(labels_file, np.zeros((10,), dtype=np.int64))

        allowed_cache_files = {"patient_002_data.npy"}

        with patch("src.brain_brr.data.datasets.logger") as mock_logger:
            dataset = ValidationDataset(
                cache_dir=str(temp_cache_dir),
                allowed_cache_files=allowed_cache_files,
            )

            assert len(dataset) == 0
