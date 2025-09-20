"""Root test configuration and shared fixtures for Brain-Go-Brr v2."""

import tempfile
import time
from collections.abc import Generator
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import yaml
from click.testing import CliRunner


@pytest.fixture(scope="session")
def sample_edf_data():
    """Generate valid 19-channel EDF test data."""
    from src.brain_brr.constants import CHANNEL_NAMES_10_20

    duration = 600  # 10 minutes
    fs = 256
    n_samples = duration * fs

    # Generate realistic EEG with 1/f characteristics
    data = np.random.randn(19, n_samples) * 10
    for i in range(19):
        # Add realistic frequency components
        t = np.arange(n_samples) / fs
        data[i] += 10 * np.sin(2 * np.pi * 10 * t)  # Alpha
        data[i] += 5 * np.sin(2 * np.pi * 20 * t)  # Beta
        data[i] += 2 * np.sin(2 * np.pi * 40 * t)  # Gamma

    return {
        "data": data.astype(np.float32),
        "channels": CHANNEL_NAMES_10_20,
        "fs": fs,
        "duration": duration,
    }


@pytest.fixture
def mock_raw_edf(sample_edf_data):
    """Mock MNE Raw object for EDF testing."""
    raw = Mock()
    raw.ch_names = list(sample_edf_data["channels"])
    raw.info = {"sfreq": sample_edf_data["fs"]}
    raw.get_data = Mock(return_value=sample_edf_data["data"])
    raw.reorder_channels = Mock()
    raw.pick_channels = Mock()
    raw.n_times = sample_edf_data["data"].shape[1]
    raw.times = np.arange(raw.n_times) / sample_edf_data["fs"]
    return raw


@pytest.fixture
def trained_model(tmp_path):
    """Lightweight pre-trained model for testing."""
    from src.brain_brr.config.schemas import (
        DecoderConfig,
        EncoderConfig,
        MambaConfig,
        ModelConfig,
        ResCNNConfig,
    )
    from src.brain_brr.models import SeizureDetector

    config = ModelConfig(
        encoder=EncoderConfig(channels=[64, 128, 256, 512], stages=4),
        rescnn=ResCNNConfig(n_blocks=3, kernel_sizes=[3, 5, 7]),
        # Use fewer Mamba layers to keep clinical pipeline test under CI timeout
        mamba=MambaConfig(n_layers=1, d_model=512, d_state=16, conv_kernel=5),
        decoder=DecoderConfig(stages=4, kernel_size=4),
    )

    model = SeizureDetector.from_config(config)

    # Initialize with known weights for reproducibility
    torch.manual_seed(42)
    for param in model.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
    model.eval()  # evaluation mode for deterministic BatchNorm/Dropout and speed

    return model


@pytest.fixture
def minimal_model():
    """Minimal model for fast testing."""
    from src.brain_brr.config.schemas import (
        DecoderConfig,
        EncoderConfig,
        MambaConfig,
        ModelConfig,
        ResCNNConfig,
    )
    from src.brain_brr.models import SeizureDetector

    config = ModelConfig(
        encoder=EncoderConfig(channels=[64, 128, 256, 512], stages=4),
        rescnn=ResCNNConfig(n_blocks=3, kernel_sizes=[3, 5, 7]),
        mamba=MambaConfig(n_layers=1, d_model=512, d_state=16, conv_kernel=5),
        decoder=DecoderConfig(stages=4, kernel_size=4),
    )

    return SeizureDetector.from_config(config)


@pytest.fixture
def cli_runner():
    """Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def valid_config_yaml(tmp_path: Path) -> Path:
    """Create a valid configuration YAML file."""
    config_path = tmp_path / "valid_config.yaml"
    config_data = {
        "experiment": {
            "name": "test",
            "seed": 42,
            "output_dir": str(tmp_path / "output"),
            "cache_dir": str(tmp_path / "cache"),
        },
        "data": {
            "dataset": "tuh_eeg",
            "data_dir": str(tmp_path / "data"),
            "sampling_rate": 256,
            "n_channels": 19,
            "window_size": 60,
            "stride": 10,
            "num_workers": 0,
        },
        "model": {
            "encoder": {"channels": [64, 128, 256, 512], "stages": 4},
            "rescnn": {"n_blocks": 3, "kernel_sizes": [3, 5, 7]},
            "mamba": {"n_layers": 6, "d_model": 512, "d_state": 16, "conv_kernel": 5},
            "decoder": {"stages": 4, "kernel_size": 4},
        },
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 1e-3,
            "optimizer": "adamw",
            "gradient_clip": 1.0,
            "validation_split": 0.2,
        },
        "postprocessing": {
            "hysteresis": {"tau_on": 0.86, "tau_off": 0.78},
            "morphology": {"kernel_size": 5},
            "duration": {"min_duration_s": 1.0, "max_duration_s": 300.0},
        },
        "evaluation": {"fa_rates": [10, 5, 1], "overlap_threshold": 0.5},
    }

    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    return config_path


@pytest.fixture
def sample_windows():
    """Sample window data for testing."""
    batch_size = 2
    n_channels = 19
    window_samples = 15360  # 60s at 256Hz

    windows = torch.randn(batch_size, n_channels, window_samples)
    labels = torch.tensor([0, 1], dtype=torch.float32)

    return windows, labels


@pytest.fixture
def sample_predictions():
    """Sample model predictions for evaluation."""
    n_windows = 100
    window_size = 15360

    predictions = torch.sigmoid(torch.randn(n_windows, window_size))
    labels = torch.zeros(n_windows, window_size)

    # Add some seizure events
    labels[10, 1000:2000] = 1
    labels[25, 5000:7000] = 1
    labels[50, 10000:12000] = 1

    return predictions, labels


@pytest.fixture
def temp_checkpoint(tmp_path: Path, minimal_model) -> Path:
    """Create a temporary checkpoint file."""
    checkpoint_path = tmp_path / "test_checkpoint.pt"

    state = {
        "model_state_dict": minimal_model.state_dict(),
        "epoch": 10,
        "best_metric": 0.85,
        "config": None,
    }

    torch.save(state, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def mock_dataloader(sample_windows):
    """Mock DataLoader for testing."""

    class MockDataLoader:
        def __init__(self, data):
            self.data = data
            self.dataset = Mock()
            self.dataset.__len__ = Mock(return_value=10)

        def __iter__(self):
            return iter([self.data])

        def __len__(self):
            return 1

    return MockDataLoader(sample_windows)


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("SEIZURE_MAMBA_FORCE_FALLBACK", "1")
    monkeypatch.setenv("BGB_LIMIT_FILES", "2")
    monkeypatch.setenv("PYTHONFAULTHANDLER", "1")


@pytest.fixture
def benchmark_timer():
    """Simple timer for performance testing."""

    class Timer:
        def __init__(self):
            self.times = []

        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.times.append(time.perf_counter() - self.start)

        @property
        def median(self):
            return np.median(self.times) if self.times else 0

        @property
        def p95(self):
            return np.percentile(self.times, 95) if self.times else 0

    return Timer()


# Test markers
def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance benchmarks")
    config.addinivalue_line("markers", "clinical: Clinical validation tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "gpu: GPU-specific tests")


# Utility functions for tests
def create_temp_config(**overrides) -> Generator[Path, None, None]:
    """Create temporary config file with overrides."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        base_config = {
            "experiment": {"name": "test", "seed": 42},
            "data": {"dataset": "tuh_eeg", "data_dir": "tests/fixtures/data"},
            "model": {
                "encoder": {"channels": [64, 128, 256, 512], "stages": 4},
                "rescnn": {"n_blocks": 3, "kernel_sizes": [3, 5, 7]},
                "mamba": {"n_layers": 6, "d_model": 512, "d_state": 16},
                "decoder": {"stages": 4, "kernel_size": 4},
            },
            "training": {"epochs": 1, "batch_size": 2},
            "postprocessing": {"hysteresis": {"tau_on": 0.86, "tau_off": 0.78}},
        }

        # Deep merge overrides
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        config = deep_update(base_config, overrides)
        yaml.dump(config, f)
        path = Path(f.name)

    yield path
    path.unlink()


def assert_tensor_close(a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-8):
    """Helper for comparing tensors with tolerance."""
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    assert torch.allclose(a, b, rtol=rtol, atol=atol)
