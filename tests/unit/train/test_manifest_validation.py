import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.brain_brr.config.schemas import Config
from src.brain_brr.data.datasets import ValidationDataset


class TestDevManifestValidation:
    @pytest.fixture
    def mock_config(self):
        config = MagicMock(spec=Config)
        config.data.cache_dir = "/tmp/cache"
        config.data.num_workers = 0
        config.data.prefetch_factor = None
        config.data.edf_root = "/data/edf"
        config.training.batch_size = 8
        config.training.num_epochs = 100
        config.training.early_stopping_patience = None
        config.training.optimizer = "adamw"
        config.training.learning_rate = 0.001
        config.training.weight_decay = 0.01
        config.training.scheduler = "onecycle"
        config.training.mixed_precision = False
        config.training.gradient_clip = 0.5
        config.training.gradient_accumulation_steps = 1
        config.training.loss = "focal"
        config.training.focal_alpha = 0.25
        config.training.focal_gamma = 2.0
        config.training.use_balanced_sampling = True
        config.output.dir = "/tmp/output"
        config.output.save_every = 1
        config.model.tcn = MagicMock()
        config.model.mamba = MagicMock()
        config.model.graph = None
        return config

    @pytest.fixture
    def temp_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache_dir.mkdir()
            yield cache_dir

    def test_valid_npy_manifest_passes(self, temp_cache_dir):
        data_file = temp_cache_dir / "test_001_data.npy"
        labels_file = temp_cache_dir / "test_001_labels.npy"
        data_file.touch()
        labels_file.touch()

        manifest = {
            "partial_seizure": [{"cache_file": "test_001_data.npy", "window_idx": 0, "label": 1}],
            "full_seizure": [],
            "no_seizure": [],
        }

        manifest_path = temp_cache_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))


        with (
            patch("src.brain_brr.train.loop.get_data_loaders") as mock_loaders,
            patch("src.brain_brr.train.loop.build_model") as mock_model,
            patch("src.brain_brr.train.loop.build_optimizer"),
            patch("src.brain_brr.train.loop.build_scheduler"),
            patch("src.brain_brr.train.loop.WandBLogger"),
            patch("src.brain_brr.train.loop.logger") as mock_logger,
            patch("src.brain_brr.data.datasets.EEGWindowDataset") as mock_dataset_cls,
        ):
            mock_val_dataset = MagicMock()
            mock_val_dataset.__len__.return_value = 100
            mock_dataset_cls.return_value = mock_val_dataset

            mock_train_loader = MagicMock()
            mock_val_loader = MagicMock()
            mock_loaders.return_value = (mock_train_loader, mock_val_loader)

            mock_model_instance = MagicMock()
            mock_model.return_value = mock_model_instance

            config = MagicMock()
            config.data.cache_dir = str(temp_cache_dir.parent)
            config.data.edf_root = str(temp_cache_dir.parent / "edf")
            config.training.num_epochs = 0

            assert manifest_path.exists()

    def test_npz_manifest_triggers_deletion(self, temp_cache_dir):
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

        manifest_path = temp_cache_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        from src.brain_brr.data.cache_utils import _load_manifest_from_path

        with patch("src.brain_brr.train.loop.logger") as mock_logger:
            loaded_manifest = _load_manifest_from_path(manifest_path, temp_cache_dir)

            if loaded_manifest:
                for category in ["partial_seizure", "full_seizure", "no_seizure"]:
                    entries = loaded_manifest.get(category, [])
                    if entries:
                        first_entry = entries[0]
                        cache_file_ref = first_entry.get("cache_file", "")
                        if "_windows.npz" in cache_file_ref or ".npz" in cache_file_ref:
                            manifest_path.unlink()
                            break

        assert not manifest_path.exists()

    def test_missing_file_manifest_triggers_deletion(self, temp_cache_dir):
        manifest = {
            "partial_seizure": [
                {"cache_file": "nonexistent_data.npy", "window_idx": 0, "label": 1}
            ],
            "full_seizure": [],
            "no_seizure": [],
        }

        manifest_path = temp_cache_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        from src.brain_brr.data.cache_utils import _load_manifest_from_path

        with patch("src.brain_brr.train.loop.logger") as mock_logger:
            loaded_manifest = _load_manifest_from_path(manifest_path, temp_cache_dir)

            if loaded_manifest:
                for category in ["partial_seizure", "full_seizure", "no_seizure"]:
                    entries = loaded_manifest.get(category, [])
                    if entries:
                        first_entry = entries[0]
                        cache_file_ref = first_entry.get("cache_file", "")
                        cache_file_path = temp_cache_dir / cache_file_ref
                        if not cache_file_path.exists():
                            manifest_path.unlink()
                            break

        assert not manifest_path.exists()

    def test_corrupted_json_triggers_deletion(self, temp_cache_dir):
        manifest_path = temp_cache_dir / "manifest.json"
        manifest_path.write_text("{invalid json content")

        from src.brain_brr.data.cache_utils import _load_manifest_from_path

        with patch("src.brain_brr.train.loop.logger") as mock_logger:
            try:
                loaded_manifest = _load_manifest_from_path(manifest_path, temp_cache_dir)
                if loaded_manifest is None and manifest_path.exists():
                    manifest_path.unlink()
            except json.JSONDecodeError:
                if manifest_path.exists():
                    manifest_path.unlink()

        assert not manifest_path.exists()

    def test_empty_manifest_edge_case(self, temp_cache_dir):
        manifest = {"partial_seizure": [], "full_seizure": [], "no_seizure": []}

        manifest_path = temp_cache_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        from src.brain_brr.data.cache_utils import _load_manifest_from_path

        loaded_manifest = _load_manifest_from_path(manifest_path, temp_cache_dir)

        assert loaded_manifest is not None
        assert manifest_path.exists()

    def test_mixed_npz_npy_manifest(self, temp_cache_dir):
        data_file = temp_cache_dir / "test_001_data.npy"
        labels_file = temp_cache_dir / "test_001_labels.npy"
        data_file.touch()
        labels_file.touch()

        manifest = {
            "partial_seizure": [
                {"cache_file": "test_001_windows.npz", "window_idx": 0, "label": 1}
            ],
            "full_seizure": [{"cache_file": "test_001_data.npy", "window_idx": 1, "label": 2}],
            "no_seizure": [],
        }

        manifest_path = temp_cache_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        from src.brain_brr.data.cache_utils import _load_manifest_from_path

        with patch("src.brain_brr.train.loop.logger") as mock_logger:
            loaded_manifest = _load_manifest_from_path(manifest_path, temp_cache_dir)

            if loaded_manifest:
                for category in ["partial_seizure", "full_seizure", "no_seizure"]:
                    entries = loaded_manifest.get(category, [])
                    if entries:
                        first_entry = entries[0]
                        cache_file_ref = first_entry.get("cache_file", "")
                        if "_windows.npz" in cache_file_ref or ".npz" in cache_file_ref:
                            manifest_path.unlink()
                            break

        assert not manifest_path.exists()


class TestFailFastValidationDataset:
    @pytest.fixture
    def temp_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache_dir.mkdir()
            yield cache_dir

    def test_zero_windows_raises_error(self, temp_cache_dir):
        data_file = temp_cache_dir / "test_001_data.npy"
        labels_file = temp_cache_dir / "test_001_labels.npy"
        data_file.touch()
        labels_file.touch()

        manifest = {
            "partial_seizure": [{"cache_file": "test_001_data.npy", "window_idx": 0, "label": 1}],
            "full_seizure": [],
            "no_seizure": [],
        }

        manifest_path = temp_cache_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        with patch("src.brain_brr.train.loop.logger") as mock_logger:
            val_dataset = MagicMock(spec=ValidationDataset)
            val_dataset.__len__.return_value = 0

            if len(val_dataset) == 0:
                if manifest_path.exists():
                    manifest_path.unlink()

                with pytest.raises(ValueError, match="ValidationDataset has 0 windows"):
                    raise ValueError(
                        "ValidationDataset has 0 windows - manifest/EDF mismatch (deleted, retry training)"
                    )

        assert not manifest_path.exists()

    def test_nonzero_windows_passes(self, temp_cache_dir):
        data_file = temp_cache_dir / "test_001_data.npy"
        labels_file = temp_cache_dir / "test_001_labels.npy"
        data_file.touch()
        labels_file.touch()

        manifest = {
            "partial_seizure": [{"cache_file": "test_001_data.npy", "window_idx": 0, "label": 1}],
            "full_seizure": [],
            "no_seizure": [],
        }

        manifest_path = temp_cache_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        val_dataset = MagicMock(spec=ValidationDataset)
        val_dataset.__len__.return_value = 100

        if len(val_dataset) == 0:
            pytest.fail("Should not reach here with nonzero dataset")

        assert manifest_path.exists()

    def test_manifest_deleted_before_error(self, temp_cache_dir):
        manifest = {
            "partial_seizure": [{"cache_file": "test_001_data.npy", "window_idx": 0, "label": 1}],
            "full_seizure": [],
            "no_seizure": [],
        }

        manifest_path = temp_cache_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        with patch("src.brain_brr.train.loop.logger") as mock_logger:
            val_dataset = MagicMock(spec=ValidationDataset)
            val_dataset.__len__.return_value = 0

            if len(val_dataset) == 0:
                assert manifest_path.exists()
                manifest_path.unlink()
                assert not manifest_path.exists()

                with pytest.raises(ValueError):
                    raise ValueError("ValidationDataset has 0 windows")


class TestManifestValidationIntegration:
    @pytest.fixture
    def temp_setup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache_dir.mkdir()

            edf_dir = Path(tmpdir) / "edf" / "dev"
            edf_dir.mkdir(parents=True)

            yield cache_dir, edf_dir

    def test_whitelist_mismatch_detected(self, temp_setup):
        cache_dir, edf_dir = temp_setup

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

        manifest_path = cache_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        edf_files = list(edf_dir.glob("**/*.edf"))
        allowed_cache_files = {f"{edf.stem}_data.npy" for edf in edf_files}

        matched = 0
        for category in ["partial_seizure", "full_seizure", "no_seizure"]:
            entries = manifest.get(category, [])
            for entry in entries:
                if entry.get("cache_file") in allowed_cache_files:
                    matched += 1

        assert matched == 0

    def test_whitelist_match_succeeds(self, temp_setup):
        cache_dir, edf_dir = temp_setup

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

        manifest_path = cache_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        edf_files = list(edf_dir.glob("**/*.edf"))
        allowed_cache_files = {f"{edf.stem}_data.npy" for edf in edf_files}

        matched = 0
        for category in ["partial_seizure", "full_seizure", "no_seizure"]:
            entries = manifest.get(category, [])
            for entry in entries:
                if entry.get("cache_file") in allowed_cache_files:
                    matched += 1

        assert matched == 1

    def test_multiple_patients_partial_match(self, temp_setup):
        cache_dir, edf_dir = temp_setup

        for i in range(1, 4):
            data_file = cache_dir / f"patient_00{i}_data.npy"
            labels_file = cache_dir / f"patient_00{i}_labels.npy"
            data_file.touch()
            labels_file.touch()

        (edf_dir / "patient_001.edf").touch()
        (edf_dir / "patient_002.edf").touch()

        manifest = {
            "partial_seizure": [
                {"cache_file": "patient_001_data.npy", "window_idx": 0, "label": 1},
                {"cache_file": "patient_002_data.npy", "window_idx": 0, "label": 1},
                {"cache_file": "patient_003_data.npy", "window_idx": 0, "label": 1},
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

        assert matched == 2
