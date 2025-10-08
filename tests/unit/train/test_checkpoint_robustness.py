"""Tests for checkpoint robustness features.

Testing philosophy: Test BEHAVIOR, not implementation.
Focus on new v3.10.0 features: atomic saves, scaler/RNG persistence, best_metric fallback.
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.amp import GradScaler

from src.brain_brr.train.checkpoint import load_checkpoint, save_checkpoint


class DummyModel(nn.Module):
    """Minimal model for checkpoint testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


class TestCheckpointAtomicSaves:
    """Test atomic save behavior (temp file + rename)."""

    def test_checkpoint_save_succeeds(self):
        """Atomic save should create checkpoint file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"

            model = DummyModel()
            optimizer = torch.optim.AdamW(model.parameters())

            save_checkpoint(
                model, optimizer, epoch=5, best_metric=0.95, checkpoint_path=checkpoint_path
            )

            assert checkpoint_path.exists()

    def test_checkpoint_no_temp_file_remains(self):
        """Temp file should be removed after successful save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"

            model = DummyModel()
            optimizer = torch.optim.AdamW(model.parameters())

            save_checkpoint(
                model, optimizer, epoch=5, best_metric=0.95, checkpoint_path=checkpoint_path
            )

            temp_path = checkpoint_path.with_suffix(".tmp")
            assert not temp_path.exists()

    def test_checkpoint_integrity_after_save(self):
        """Saved checkpoint should be loadable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"

            model = DummyModel()
            optimizer = torch.optim.AdamW(model.parameters())

            save_checkpoint(
                model, optimizer, epoch=5, best_metric=0.95, checkpoint_path=checkpoint_path
            )

            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            assert "model_state_dict" in ckpt
            assert "optimizer_state_dict" in ckpt
            assert ckpt["epoch"] == 5
            assert ckpt["best_metric"] == 0.95


class TestCheckpointScalerPersistence:
    """Test AMP scaler state persistence."""

    def test_scaler_saved_when_provided(self):
        """Scaler state should be saved in checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"

            model = DummyModel()
            optimizer = torch.optim.AdamW(model.parameters())
            scaler = GradScaler(enabled=True)

            scaler._scale = torch.tensor(1024.0)

            save_checkpoint(
                model,
                optimizer,
                epoch=5,
                best_metric=0.95,
                checkpoint_path=checkpoint_path,
                scaler=scaler,
            )

            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            assert "scaler_state_dict" in ckpt

    def test_scaler_not_saved_when_none(self):
        """Scaler state should not be in checkpoint when scaler=None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"

            model = DummyModel()
            optimizer = torch.optim.AdamW(model.parameters())

            save_checkpoint(
                model,
                optimizer,
                epoch=5,
                best_metric=0.95,
                checkpoint_path=checkpoint_path,
                scaler=None,
            )

            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            assert "scaler_state_dict" not in ckpt

    def test_scaler_restored_correctly(self):
        """Scaler state should be restored on load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"

            model = DummyModel()
            optimizer = torch.optim.AdamW(model.parameters())
            scaler = GradScaler(enabled=True)

            scaler._scale = torch.tensor(1024.0)

            save_checkpoint(
                model,
                optimizer,
                epoch=5,
                best_metric=0.95,
                checkpoint_path=checkpoint_path,
                scaler=scaler,
            )

            new_model = DummyModel()
            new_optimizer = torch.optim.AdamW(new_model.parameters())
            new_scaler = GradScaler(enabled=True)

            load_checkpoint(
                checkpoint_path,
                new_model,
                new_optimizer,
                scaler=new_scaler,
                device="cpu",
            )

            assert new_scaler._scale == scaler._scale


class TestCheckpointRNGPersistence:
    """Test RNG state persistence for deterministic resume."""

    def test_rng_states_saved_when_requested(self):
        """All RNG states should be saved when save_rng=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"

            model = DummyModel()
            optimizer = torch.optim.AdamW(model.parameters())

            save_checkpoint(
                model,
                optimizer,
                epoch=5,
                best_metric=0.95,
                checkpoint_path=checkpoint_path,
                save_rng=True,
            )

            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            assert "rng_state" in ckpt
            assert "torch" in ckpt["rng_state"]
            assert "numpy" in ckpt["rng_state"]
            assert "python" in ckpt["rng_state"]

    def test_rng_states_not_saved_when_disabled(self):
        """RNG states should not be saved when save_rng=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"

            model = DummyModel()
            optimizer = torch.optim.AdamW(model.parameters())

            save_checkpoint(
                model,
                optimizer,
                epoch=5,
                best_metric=0.95,
                checkpoint_path=checkpoint_path,
                save_rng=False,
            )

            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            assert "rng_state" not in ckpt

    def test_rng_states_restored_correctly(self):
        """Restored RNG should produce same random sequence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"

            torch.manual_seed(42)
            np.random.seed(42)
            random.seed(42)

            original_torch = torch.rand(5)
            original_numpy = np.random.rand(5)
            original_python = [random.random() for _ in range(5)]

            model = DummyModel()
            optimizer = torch.optim.AdamW(model.parameters())

            save_checkpoint(
                model,
                optimizer,
                epoch=5,
                best_metric=0.95,
                checkpoint_path=checkpoint_path,
                save_rng=True,
            )

            torch.manual_seed(999)
            np.random.seed(999)
            random.seed(999)

            new_model = DummyModel()
            new_optimizer = torch.optim.AdamW(new_model.parameters())

            load_checkpoint(
                checkpoint_path,
                new_model,
                new_optimizer,
                restore_rng=True,
                device="cpu",
            )

            restored_torch = torch.rand(5)
            restored_numpy = np.random.rand(5)
            restored_python = [random.random() for _ in range(5)]

            assert torch.allclose(original_torch, restored_torch)
            assert np.allclose(original_numpy, restored_numpy)
            assert original_python == restored_python


class TestCheckpointBestMetricFallback:
    """Test best_metric fallback logic for backward compatibility."""

    def test_loads_best_metric_when_present(self):
        """Should load best_metric field when present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"

            model = DummyModel()
            optimizer = torch.optim.AdamW(model.parameters())

            save_checkpoint(
                model, optimizer, epoch=5, best_metric=0.95, checkpoint_path=checkpoint_path
            )

            new_model = DummyModel()
            new_optimizer = torch.optim.AdamW(new_model.parameters())

            epoch, best_metric = load_checkpoint(
                checkpoint_path, new_model, new_optimizer, device="cpu"
            )

            assert epoch == 5
            assert best_metric == 0.95

    def test_falls_back_to_metric_when_best_metric_missing(self):
        """Should fall back to 'metric' field when 'best_metric' missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"

            model = DummyModel()
            optimizer = torch.optim.AdamW(model.parameters())

            ckpt = {
                "epoch": 5,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metric": 0.88,
            }
            torch.save(ckpt, checkpoint_path)

            new_model = DummyModel()
            new_optimizer = torch.optim.AdamW(new_model.parameters())

            epoch, best_metric = load_checkpoint(
                checkpoint_path, new_model, new_optimizer, device="cpu"
            )

            assert epoch == 5
            assert best_metric == 0.88

    def test_defaults_to_zero_when_both_missing(self):
        """Should default to 0.0 when both best_metric and metric missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"

            model = DummyModel()
            optimizer = torch.optim.AdamW(model.parameters())

            ckpt = {
                "epoch": 5,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(ckpt, checkpoint_path)

            new_model = DummyModel()
            new_optimizer = torch.optim.AdamW(new_model.parameters())

            epoch, best_metric = load_checkpoint(
                checkpoint_path, new_model, new_optimizer, device="cpu"
            )

            assert epoch == 5
            assert best_metric == 0.0


class TestCheckpointFullState:
    """Test complete state persistence (model + optimizer + scheduler + scaler + RNG)."""

    def test_full_state_roundtrip(self):
        """Complete state should survive save/load cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"

            torch.manual_seed(42)
            np.random.seed(42)
            random.seed(42)

            model = DummyModel()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            scaler = GradScaler(enabled=True)
            scaler._scale = torch.tensor(512.0)

            model_state_before = model.state_dict()
            optim_state_before = optimizer.state_dict()

            save_checkpoint(
                model,
                optimizer,
                epoch=10,
                best_metric=0.92,
                checkpoint_path=checkpoint_path,
                scaler=scaler,
                save_rng=True,
            )

            torch.manual_seed(999)
            np.random.seed(999)
            random.seed(999)

            new_model = DummyModel()
            new_optimizer = torch.optim.AdamW(new_model.parameters())
            new_scaler = GradScaler(enabled=True)

            epoch, best_metric = load_checkpoint(
                checkpoint_path,
                new_model,
                new_optimizer,
                scaler=new_scaler,
                restore_rng=True,
                device="cpu",
            )

            model_state_after = new_model.state_dict()
            optim_state_after = new_optimizer.state_dict()

            assert epoch == 10
            assert best_metric == 0.92
            assert new_scaler._scale == scaler._scale

            for key in model_state_before:
                assert torch.allclose(model_state_before[key], model_state_after[key])


class TestCheckpointEdgeCases:
    """Test edge cases and error handling."""

    def test_load_nonexistent_checkpoint_raises(self):
        """Loading nonexistent checkpoint should raise error."""
        model = DummyModel()
        optimizer = torch.optim.AdamW(model.parameters())

        with pytest.raises(FileNotFoundError):
            load_checkpoint(Path("/nonexistent/checkpoint.pt"), model, optimizer, device="cpu")

    def test_checkpoint_with_extra_metadata(self):
        """Should preserve extra metadata fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"

            model = DummyModel()
            optimizer = torch.optim.AdamW(model.parameters())

            save_checkpoint(
                model,
                optimizer,
                epoch=5,
                best_metric=0.95,
                checkpoint_path=checkpoint_path,
                extra={"batch_idx": 123, "kind": "mid_epoch"},
            )

            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            assert ckpt["batch_idx"] == 123
            assert ckpt["kind"] == "mid_epoch"
