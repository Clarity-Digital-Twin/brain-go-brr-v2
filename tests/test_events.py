"""Tests for event processing."""

import numpy as np
import torch

from src.experiment.events import (
    SeizureEvent,
    batch_mask_to_events,
    calculate_event_confidence,
    mask_to_events,
    merge_events,
)


class TestSeizureEvent:
    """Test SeizureEvent dataclass."""

    def test_event_properties(self):
        """Test basic event properties."""
        event = SeizureEvent(start_s=10.0, end_s=15.0, confidence=0.9)
        assert event.duration == 5.0

    def test_event_overlap(self):
        """Test overlap detection."""
        event1 = SeizureEvent(start_s=10.0, end_s=20.0)
        event2 = SeizureEvent(start_s=15.0, end_s=25.0)  # Overlaps
        event3 = SeizureEvent(start_s=25.0, end_s=30.0)  # No overlap

        assert event1.overlaps(event2)
        assert event2.overlaps(event1)
        assert not event1.overlaps(event3)

    def test_event_merge(self):
        """Test event merging."""
        event1 = SeizureEvent(start_s=10.0, end_s=20.0, confidence=0.8)
        event2 = SeizureEvent(start_s=15.0, end_s=25.0, confidence=0.9)

        merged = event1.merge(event2)
        assert merged.start_s == 10.0
        assert merged.end_s == 25.0
        assert merged.confidence == 0.9  # Max confidence


class TestMaskToEvents:
    """Test mask to event conversion."""

    def test_single_event(self):
        """Test single continuous event."""
        mask = np.zeros(256)  # 1 second at 256 Hz
        mask[50:150] = 1  # Event from ~0.195s to ~0.586s

        events = mask_to_events(mask, sampling_rate=256)
        assert len(events) == 1
        assert abs(events[0].start_s - 50 / 256) < 0.01
        assert abs(events[0].end_s - 150 / 256) < 0.01

    def test_multiple_events(self):
        """Test multiple separate events."""
        mask = np.zeros(512)
        mask[50:100] = 1  # First event
        mask[200:300] = 1  # Second event

        events = mask_to_events(mask, sampling_rate=256)
        assert len(events) == 2

    def test_min_samples_filter(self):
        """Test filtering by minimum samples."""
        mask = np.zeros(100)
        mask[10:12] = 1  # 2 samples (short)
        mask[50:60] = 1  # 10 samples (keep)

        events = mask_to_events(mask, sampling_rate=100, min_samples=5)
        assert len(events) == 1  # Only long event kept


class TestMergeEvents:
    """Test event merging."""

    def test_merge_close_events(self):
        """Test merging events with small gap."""
        events = [
            SeizureEvent(start_s=10.0, end_s=20.0),
            SeizureEvent(start_s=21.0, end_s=30.0),  # 1s gap
            SeizureEvent(start_s=35.0, end_s=40.0),  # 5s gap
        ]

        merged = merge_events(events, tau_merge=2.0)
        assert len(merged) == 2  # First two merged
        assert merged[0].end_s == 30.0
        assert merged[1].start_s == 35.0

    def test_no_merge_large_gap(self):
        """Test events with large gaps stay separate."""
        events = [
            SeizureEvent(start_s=10.0, end_s=20.0),
            SeizureEvent(start_s=30.0, end_s=40.0),  # 10s gap
        ]

        merged = merge_events(events, tau_merge=2.0)
        assert len(merged) == 2  # Stay separate


class TestConfidence:
    """Test confidence calculation."""

    def test_mean_confidence(self):
        """Test mean confidence calculation."""
        probs = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3, 0.1])
        event = SeizureEvent(start_s=2 / 7, end_s=5 / 7)  # Covers indices 2-4

        conf = calculate_event_confidence(probs, event, sampling_rate=7, method="mean")
        expected = np.mean([0.8, 0.9, 0.7])
        assert abs(conf - expected) < 0.01

    def test_peak_confidence(self):
        """Test peak confidence calculation."""
        probs = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3, 0.1])
        event = SeizureEvent(start_s=2 / 7, end_s=5 / 7)

        conf = calculate_event_confidence(probs, event, sampling_rate=7, method="peak")
        assert conf == 0.9

    def test_percentile_confidence(self):
        """Test percentile confidence calculation."""
        probs = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3, 0.1])
        event = SeizureEvent(start_s=2 / 7, end_s=5 / 7)

        conf = calculate_event_confidence(
            probs, event, sampling_rate=7, method="percentile", percentile=0.75
        )
        expected = np.percentile([0.8, 0.9, 0.7], 75)
        assert abs(conf - expected) < 0.01


class TestBatchProcessing:
    """Test batch processing functions."""

    def test_batch_mask_to_events(self):
        """Test batch conversion with all features."""
        masks = torch.zeros(2, 256, dtype=torch.bool)
        masks[0, 50:100] = True
        masks[0, 120:170] = True  # Two events, close together
        masks[1, 200:250] = True  # One event

        probs = torch.rand(2, 256) * 0.5 + 0.5  # Random [0.5, 1.0]

        batch_events = batch_mask_to_events(
            masks,
            sampling_rate=256,
            tau_merge=0.1,  # Merge if gap < 0.1s
            probs=probs,
            confidence_method="mean",
        )

        assert len(batch_events) == 2
        assert len(batch_events[0]) == 1  # Two events merged
        assert len(batch_events[1]) == 1
        assert batch_events[0][0].confidence > 0  # Has confidence
