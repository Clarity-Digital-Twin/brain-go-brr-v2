"""Unit tests for timeline assembly helpers."""

from __future__ import annotations

import pytest
import torch

from src.brain_brr.eval.helpers.timeline import RecordingTimeline, build_recording_timelines


class TestBuildRecordingTimelines:
    """Test timeline assembly from overlapping windows."""

    def test_single_recording_single_window(self) -> None:
        """Single recording with one window should create exact copy timeline."""
        probs = torch.rand(1, 15360)
        labels = torch.randint(0, 2, (1, 15360)).float()
        file_ids = ["rec_001"]
        window_starts = [0.0]
        sampling_rate = 256

        timelines = build_recording_timelines(probs, labels, file_ids, window_starts, sampling_rate)

        assert len(timelines) == 1
        assert timelines[0].file_id == "rec_001"
        assert timelines[0].timeline_probs.shape == probs[0].shape
        assert timelines[0].timeline_labels.shape == labels[0].shape
        assert torch.allclose(timelines[0].timeline_probs, probs[0])
        assert torch.allclose(timelines[0].timeline_labels, labels[0])
        assert timelines[0].duration_s == 60.0

    def test_single_recording_overlapping_windows(self) -> None:
        """Overlapping windows should average probabilities in overlap region."""
        probs = torch.tensor(
            [
                [0.2] * 2560,
                [0.8] * 2560,
            ]
        )
        labels = torch.tensor(
            [
                [0.0] * 2560,
                [1.0] * 2560,
            ]
        )
        file_ids = ["rec_001", "rec_001"]
        window_starts = [0.0, 5.0]
        sampling_rate = 256

        timelines = build_recording_timelines(probs, labels, file_ids, window_starts, sampling_rate)

        assert len(timelines) == 1
        timeline = timelines[0]

        overlap_start_idx = int(5.0 * sampling_rate)
        window_0_end_idx = int(10.0 * sampling_rate)

        assert timeline.timeline_probs[:overlap_start_idx].mean() == pytest.approx(0.2, abs=1e-6)
        overlap_probs = timeline.timeline_probs[overlap_start_idx:window_0_end_idx]
        assert overlap_probs.mean() == pytest.approx(0.5, abs=1e-6)

        assert timeline.timeline_labels[:overlap_start_idx].mean() == pytest.approx(0.0, abs=1e-6)
        overlap_labels = timeline.timeline_labels[overlap_start_idx:window_0_end_idx]
        assert overlap_labels.mean() == pytest.approx(0.5, abs=1e-6)

    def test_multiple_recordings(self) -> None:
        """Multiple recordings should be grouped correctly."""
        probs = torch.rand(4, 15360)
        labels = torch.randint(0, 2, (4, 15360)).float()
        file_ids = ["rec_001", "rec_002", "rec_001", "rec_003"]
        window_starts = [0.0, 0.0, 10.0, 0.0]
        sampling_rate = 256

        timelines = build_recording_timelines(probs, labels, file_ids, window_starts, sampling_rate)

        assert len(timelines) == 3

        timeline_fids = {t.file_id for t in timelines}
        assert timeline_fids == {"rec_001", "rec_002", "rec_003"}

        rec_001 = next(t for t in timelines if t.file_id == "rec_001")
        assert rec_001.duration_s == 70.0

        rec_002 = next(t for t in timelines if t.file_id == "rec_002")
        assert rec_002.duration_s == 60.0

    def test_window_sorting(self) -> None:
        """Windows should be sorted by start time regardless of input order."""
        probs = torch.tensor(
            [
                [0.8] * 2560,
                [0.2] * 2560,
            ]
        )
        labels = torch.zeros(2, 2560)
        file_ids = ["rec_001", "rec_001"]
        window_starts = [10.0, 0.0]
        sampling_rate = 256

        timelines = build_recording_timelines(probs, labels, file_ids, window_starts, sampling_rate)

        assert len(timelines) == 1
        timeline = timelines[0]

        first_window_region = timeline.timeline_probs[:2560]
        assert first_window_region.mean() == pytest.approx(0.2, abs=1e-6)

    def test_device_preservation(self) -> None:
        """Timeline should preserve device of input tensors."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        probs = torch.rand(2, 15360, device=device)
        labels = torch.randint(0, 2, (2, 15360), device=device).float()
        file_ids = ["rec_001", "rec_001"]
        window_starts = [0.0, 10.0]
        sampling_rate = 256

        timelines = build_recording_timelines(probs, labels, file_ids, window_starts, sampling_rate)

        assert len(timelines) == 1
        assert timelines[0].timeline_probs.device.type == device.type
        assert timelines[0].timeline_labels.device.type == device.type

    def test_empty_input(self) -> None:
        """Empty input should return empty timeline list."""
        probs = torch.empty(0, 15360)
        labels = torch.empty(0, 15360)
        file_ids: list[str] = []
        window_starts: list[float] = []
        sampling_rate = 256

        timelines = build_recording_timelines(probs, labels, file_ids, window_starts, sampling_rate)

        assert len(timelines) == 0

    def test_recording_timeline_dataclass(self) -> None:
        """RecordingTimeline dataclass should have expected attributes."""
        timeline = RecordingTimeline(
            file_id="test_001",
            timeline_probs=torch.rand(15360),
            timeline_labels=torch.zeros(15360),
            duration_s=60.0,
        )

        assert timeline.file_id == "test_001"
        assert timeline.timeline_probs.shape == (15360,)
        assert timeline.timeline_labels.shape == (15360,)
        assert timeline.duration_s == 60.0
