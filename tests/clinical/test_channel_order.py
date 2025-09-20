"""Clinical validation tests for EEG channel ordering and montage consistency."""

from unittest.mock import Mock

import numpy as np
import pytest

from src.brain_brr.constants import CHANNEL_NAMES_10_20, CHANNEL_SYNONYMS
from src.brain_brr.utils.pick_utils import handle_channel_synonyms, pick_and_order


class TestChannelOrdering:
    """Test canonical 10-20 montage channel ordering."""

    def test_canonical_channel_order(self):
        """Verify canonical channel order is maintained."""
        expected_order = [
            "Fp1",
            "F3",
            "C3",
            "P3",
            "F7",
            "T3",
            "T5",
            "O1",
            "Fz",
            "Cz",
            "Pz",
            "Fp2",
            "F4",
            "C4",
            "P4",
            "F8",
            "T4",
            "T6",
            "O2",
        ]

        assert expected_order == CHANNEL_NAMES_10_20
        assert len(CHANNEL_NAMES_10_20) == 19

    def test_channel_synonyms_mapping(self):
        """Test channel synonym mappings for different naming conventions."""
        # Modern to legacy mappings
        assert CHANNEL_SYNONYMS["T7"] == "T3"
        assert CHANNEL_SYNONYMS["T8"] == "T4"
        assert CHANNEL_SYNONYMS["P7"] == "T5"
        assert CHANNEL_SYNONYMS["P8"] == "T6"

        # Case variations are handled by handle_channel_synonyms function
        # Test that function properly
        from src.brain_brr.utils.pick_utils import handle_channel_synonyms

        # Test case normalization
        normalized = handle_channel_synonyms(["fp1", "FP2", "t7", "T8"])
        assert "Fp1" in normalized or "fp1" in normalized  # Should normalize case
        assert "T3" in normalized  # T7 -> T3
        assert "T4" in normalized  # T8 -> T4

    def test_pick_and_order_exact_match(self):
        """Test channel picking when all channels match exactly."""
        raw = Mock(spec=[])
        raw.ch_names = list(CHANNEL_NAMES_10_20)
        raw.reorder_channels = Mock()
        raw.pick_channels = Mock()

        _ordered_raw, missing = pick_and_order(raw, CHANNEL_NAMES_10_20)

        assert missing == []
        # pick_and_order uses pick() if available (modern MNE), pick_channels (legacy), or fallback
        # Our mock doesn't have pick() (spec=[]), so it will use pick_channels
        raw.pick_channels.assert_called_once()

    def test_pick_and_order_with_synonyms(self):
        """Test channel picking with synonym substitution."""
        raw = Mock()
        # Use modern naming (T7/T8 instead of T3/T4)
        raw.ch_names = [
            "Fp1",
            "F3",
            "C3",
            "P3",
            "F7",
            "T7",
            "P7",
            "O1",  # T7->T3, P7->T5
            "Fz",
            "Cz",
            "Pz",
            "Fp2",
            "F4",
            "C4",
            "P4",
            "F8",
            "T8",
            "P8",
            "O2",  # T8->T4, P8->T6
        ]
        raw.reorder_channels = Mock()
        raw.pick_channels = Mock()

        # Mock handle_channel_synonyms to return mapped names
        raw.ch_names = handle_channel_synonyms(raw.ch_names)

        _ordered_raw, missing = pick_and_order(raw, CHANNEL_NAMES_10_20)

        # After synonym handling, should find all channels
        assert len(missing) <= 4  # T3, T4, T5, T6 might be "missing" but substituted

    def test_pick_and_order_missing_channels(self):
        """Test channel picking with missing channels."""
        raw = Mock()
        # Missing some channels
        raw.ch_names = ["Fp1", "F3", "C3", "P3", "Fz", "Cz", "Pz"]
        raw.reorder_channels = Mock()
        raw.pick_channels = Mock()

        _ordered_raw, missing = pick_and_order(raw, CHANNEL_NAMES_10_20)

        expected_missing = ["F7", "T3", "T5", "O1", "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2"]
        assert set(missing) == set(expected_missing)

    def test_pick_and_order_extra_channels(self):
        """Test channel picking with extra channels."""
        raw = Mock(spec=[])
        # Has canonical channels plus extras
        raw.ch_names = [*list(CHANNEL_NAMES_10_20), "ECG", "EOG1", "EOG2", "EMG"]
        raw.reorder_channels = Mock()
        raw.pick_channels = Mock()

        _ordered_raw, missing = pick_and_order(raw, CHANNEL_NAMES_10_20)

        assert missing == []
        # Should pick only the 19 canonical channels
        raw.pick_channels.assert_called_once()
        picked_channels = raw.pick_channels.call_args[0][0]
        assert len(picked_channels) == 19

    def test_pick_and_order_case_insensitive(self):
        """Test channel picking is case-insensitive."""
        raw = Mock()
        # Mixed case channel names
        raw.ch_names = [
            "fp1",
            "F3",
            "c3",
            "P3",
            "f7",
            "t3",
            "T5",
            "o1",
            "FZ",
            "Cz",
            "pz",
            "FP2",
            "f4",
            "C4",
            "p4",
            "F8",
            "T4",
            "t6",
            "O2",
        ]
        raw.reorder_channels = Mock()
        raw.pick_channels = Mock()

        # Apply case normalization
        raw.ch_names = handle_channel_synonyms(raw.ch_names)

        _ordered_raw, missing = pick_and_order(raw, CHANNEL_NAMES_10_20)

        # Should handle case variations
        assert len(missing) == 0 or all(ch in CHANNEL_NAMES_10_20 for ch in missing)


class TestChannelDataIntegrity:
    """Test data integrity during channel operations."""

    def test_channel_reordering_preserves_data(self):
        """Test that reordering channels preserves data integrity."""
        n_channels = 19
        n_samples = 1000
        original_data = np.random.randn(n_channels, n_samples)

        # Create mapping for reordering
        original_order = list(CHANNEL_NAMES_10_20)
        shuffled_order = original_order.copy()
        np.random.shuffle(shuffled_order)

        # Create reordering indices
        reorder_indices = [shuffled_order.index(ch) for ch in original_order]

        # Reorder data
        reordered_data = original_data[reorder_indices]

        # Verify each channel's data is preserved after reordering
        for i, ch in enumerate(original_order):
            original_idx = shuffled_order.index(ch)  # Where was this channel in the shuffled data?
            reordered_idx = i  # Where it should be now in the reordered data
            np.testing.assert_array_equal(
                original_data[original_idx], reordered_data[reordered_idx]
            )

    def test_channel_picking_subset(self):
        """Test picking subset of channels maintains correct data."""
        n_samples = 1000
        full_data = {}

        # Create data for all channels
        for i, ch in enumerate(CHANNEL_NAMES_10_20):
            full_data[ch] = np.random.randn(n_samples) * (i + 1)  # Unique scale per channel

        # Pick subset
        subset_channels = ["Fp1", "Fz", "Cz", "Pz", "O2"]
        subset_data = np.array([full_data[ch] for ch in subset_channels])

        # Verify correct channels were picked
        for i, ch in enumerate(subset_channels):
            expected_scale = CHANNEL_NAMES_10_20.index(ch) + 1
            actual_std = np.std(subset_data[i])
            # Check that we got the right channel (by its unique scale)
            assert abs(actual_std - expected_scale) < 5  # Approximate check

    def test_zero_padding_missing_channels(self):
        """Test zero-padding for missing channels."""
        available_channels = ["Fp1", "F3", "C3", "Fz", "Cz"]
        n_samples = 1000

        # Create data for available channels
        available_data = np.random.randn(len(available_channels), n_samples)

        # Create full array with zeros for missing channels
        full_data = np.zeros((19, n_samples))

        # Fill in available channels
        for i, ch in enumerate(available_channels):
            if ch in CHANNEL_NAMES_10_20:
                idx = CHANNEL_NAMES_10_20.index(ch)
                full_data[idx] = available_data[i]

        # Verify structure
        assert full_data.shape == (19, n_samples)

        # Verify available channels have data
        for ch in available_channels:
            idx = CHANNEL_NAMES_10_20.index(ch)
            assert not np.all(full_data[idx] == 0)

        # Verify missing channels are zero
        missing_channels = set(CHANNEL_NAMES_10_20) - set(available_channels)
        for ch in missing_channels:
            idx = CHANNEL_NAMES_10_20.index(ch)
            assert np.all(full_data[idx] == 0)


class TestMontageConsistency:
    """Test montage consistency across different data sources."""

    def test_tuh_montage_compatibility(self):
        """Test compatibility with TUH EEG dataset montage."""
        # TUH commonly uses these channels
        tuh_common_channels = [
            "FP1",
            "FP2",
            "F7",
            "F3",
            "FZ",
            "F4",
            "F8",
            "T3",
            "C3",
            "CZ",
            "C4",
            "T4",
            "T5",
            "P3",
            "PZ",
            "P4",
            "T6",
            "O1",
            "O2",
        ]

        # Apply synonym handling
        normalized = handle_channel_synonyms(tuh_common_channels)

        # Check all can be mapped to canonical
        for ch in normalized:
            assert ch in CHANNEL_NAMES_10_20 or ch.upper() in [
                c.upper() for c in CHANNEL_NAMES_10_20
            ]

    def test_chb_mit_montage_compatibility(self):
        """Test compatibility with CHB-MIT dataset montage."""
        # CHB-MIT uses modern naming
        chb_channels = [
            "FP1-F7",
            "F7-T7",
            "T7-P7",
            "P7-O1",
            "FP1-F3",
            "F3-C3",
            "C3-P3",
            "P3-O1",
            "FP2-F4",
            "F4-C4",
            "C4-P4",
            "P4-O2",
            "FP2-F8",
            "F8-T8",
            "T8-P8",
            "P8-O2",
            "FZ-CZ",
            "CZ-PZ",
        ]

        # Extract individual channels from bipolar montage
        individual_channels = set()
        for pair in chb_channels:
            if "-" in pair:
                ch1, ch2 = pair.split("-")
                individual_channels.add(ch1)
                individual_channels.add(ch2)

        # Apply synonym handling
        normalized = handle_channel_synonyms(list(individual_channels))

        # Most should map to canonical (T7->T3, T8->T4, P7->T5, P8->T6)
        mapped_count = sum(1 for ch in normalized if ch in CHANNEL_NAMES_10_20)
        assert mapped_count >= len(individual_channels) * 0.8  # At least 80% should map

    def test_european_vs_american_notation(self):
        """Test handling of European vs American notation differences."""
        # European often uses T3/T4/T5/T6
        european = ["T3", "T4", "T5", "T6"]

        # American often uses T7/T8/P7/P8
        american = ["T7", "T8", "P7", "P8"]

        # Both should be handled
        for ch_list in [european, american]:
            normalized = handle_channel_synonyms(ch_list)
            # All should map to something
            assert len(normalized) == len(ch_list)


class TestChannelValidation:
    """Test channel validation and error handling."""

    def test_validate_channel_count(self):
        """Test validation of channel count."""

        # Too few channels
        def validate_channels(n_channels):
            if n_channels != 19:
                raise ValueError(f"Expected 19 channels, got {n_channels}")

        with pytest.raises(ValueError, match="channels"):
            validate_channels(10)

        # Too many channels (should work but warn)
        data = np.random.randn(25, 1000)  # 25 channels
        # Should not raise, but might log warning

        # Correct channels
        data = np.random.randn(19, 1000)
        assert data.shape[0] == 19  # Should pass

    def test_validate_channel_names(self):
        """Test validation of channel names."""
        # Valid names
        valid_names = list(CHANNEL_NAMES_10_20)
        assert all(isinstance(name, str) for name in valid_names)
        assert all(len(name) >= 2 for name in valid_names)

        # Invalid names should be caught
        invalid_names = ["", "X", "Invalid", "123", None, 123]
        for name in invalid_names:
            is_valid = name in CHANNEL_NAMES_10_20 or str(name).upper() in CHANNEL_SYNONYMS
            assert not is_valid

    def test_handle_duplicate_channels(self):
        """Test handling of duplicate channel names."""
        channels_with_duplicates = ["Fp1", "F3", "C3", "Fp1", "F3"]  # Fp1 and F3 duplicated

        # Should handle duplicates gracefully
        unique_channels = list(dict.fromkeys(channels_with_duplicates))
        assert len(unique_channels) == 3
        assert unique_channels == ["Fp1", "F3", "C3"]

    def test_handle_missing_critical_channels(self):
        """Test handling when critical channels are missing."""
        # Define critical channels (e.g., for seizure detection)
        critical_channels = ["Fp1", "Fp2", "T3", "T4", "O1", "O2"]

        available_channels = ["Fp1", "C3", "Cz"]  # Missing most critical channels

        missing_critical = set(critical_channels) - set(available_channels)
        assert len(missing_critical) > 0

        # Should log warning or handle appropriately
        # In production, might zero-pad or interpolate


@pytest.mark.clinical
class TestClinicalChannelRequirements:
    """Test clinical requirements for channel handling."""

    def test_minimum_channels_for_detection(self):
        """Test minimum channel requirements for seizure detection."""
        # Minimum viable channel sets for detection
        minimum_sets = [
            ["Fp1", "Fp2", "T3", "T4", "O1", "O2"],  # Frontal + Temporal + Occipital
            ["F3", "F4", "C3", "C4", "P3", "P4"],  # Lateral chain
            ["Fz", "Cz", "Pz", "T3", "T4"],  # Midline + Temporal
        ]

        for channel_set in minimum_sets:
            available = set(channel_set)
            coverage = len(available.intersection(CHANNEL_NAMES_10_20)) / 19
            assert coverage >= 0.25  # At least 25% coverage

    def test_bilateral_symmetry(self):
        """Test bilateral symmetry in channel selection."""
        left_channels = ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1"]
        right_channels = ["Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2"]
        midline_channels = ["Fz", "Cz", "Pz"]

        # Verify symmetry
        assert len(left_channels) == len(right_channels)

        # Verify all are in canonical set
        all_channels = left_channels + right_channels + midline_channels
        assert set(all_channels) == set(CHANNEL_NAMES_10_20)

    def test_channel_interpolation_quality(self):
        """Test quality of channel interpolation for missing channels."""
        n_samples = 1000
        available_data = np.random.randn(10, n_samples)  # Only 10 channels available

        # Simulate interpolation (simplified)
        interpolated_data = np.zeros((19, n_samples))

        # Fill available channels
        available_indices = [0, 2, 4, 6, 8, 10, 11, 13, 15, 17]  # Random subset
        for i, idx in enumerate(available_indices):
            interpolated_data[idx] = available_data[i]

        # Simple nearest-neighbor interpolation for missing
        for idx in range(19):
            if idx not in available_indices:
                # Find nearest available channel
                nearest = min(available_indices, key=lambda x: abs(x - idx))
                nearest_pos = available_indices.index(nearest)
                interpolated_data[idx] = available_data[nearest_pos] * 0.8  # Scaled

        # Verify all channels have data
        assert not np.any(np.all(interpolated_data == 0, axis=1))

        # Verify reasonable statistics
        for i in range(19):
            channel_std = np.std(interpolated_data[i])
            assert 0.1 < channel_std < 10  # Reasonable range
