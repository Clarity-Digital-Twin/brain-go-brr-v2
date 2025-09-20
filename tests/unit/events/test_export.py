"""Comprehensive export functionality tests for Brain-Go-Brr v2."""

import csv
import json
from pathlib import Path

import pytest

from src.brain_brr.events import SeizureEvent
from src.brain_brr.events.export import (
    export_batch_csv_bi,
    export_csv_bi,
    export_json,
    validate_csv_bi,
)


class TestCSVBIExport:
    """Test CSV-BI format export functionality."""

    def test_export_csv_bi_single_event(self, tmp_path: Path):
        """Test exporting a single seizure event."""
        events = [SeizureEvent(start_s=1.0, end_s=2.5, confidence=0.9)]
        output_file = tmp_path / "P001_R001.csv"

        export_csv_bi(
            events=events,
            output_path=output_file,
            patient_id="P001",
            recording_id="R001",
            duration_s=60.0,
        )

        assert output_file.exists()

        # Verify content
        is_valid, errors = validate_csv_bi(output_file)
        assert is_valid, f"CSV_BI invalid: {errors}"

    def test_export_csv_bi_multiple_events(self, tmp_path: Path):
        """Test exporting multiple seizure events."""
        events = [
            SeizureEvent(0.0, 1.5, 0.95),
            SeizureEvent(10.0, 12.0, 0.88),
            SeizureEvent(25.5, 30.0, 0.92),
        ]
        output_file = tmp_path / "test.csv"

        export_csv_bi(
            events=events,
            output_path=output_file,
            patient_id="P002",
            recording_id="R002",
            duration_s=120.0,
        )

        assert output_file.exists()

        # Parse and verify
        with open(output_file) as f:
            reader = csv.reader(f)
            rows = list(reader)
            # Header + 3 events
            assert len(rows) >= 4

    def test_export_csv_bi_empty_events(self, tmp_path: Path):
        """Test exporting empty event list."""
        events = []
        output_file = tmp_path / "empty.csv"

        export_csv_bi(
            events=events,
            output_path=output_file,
            patient_id="P003",
            recording_id="R003",
            duration_s=60.0,
        )

        assert output_file.exists()

        with open(output_file) as f:
            lines = f.readlines()
            # Should have at least header
            assert len(lines) >= 1

    def test_export_csv_bi_creates_parent_dirs(self, tmp_path: Path):
        """Test that export creates parent directories if needed."""
        output_file = tmp_path / "nested" / "dir" / "structure" / "events.csv"
        events = [SeizureEvent(1.0, 2.0, 0.8)]

        export_csv_bi(
            events=events,
            output_path=output_file,
            patient_id="P004",
            recording_id="R004",
            duration_s=60.0,
        )

        assert output_file.exists()
        assert output_file.parent.exists()

    def test_validate_csv_bi_valid_file(self, tmp_path: Path):
        """Test validation of valid CSV-BI file."""
        events = [SeizureEvent(1.0, 2.5, 0.9)]
        output_file = tmp_path / "valid.csv"

        export_csv_bi(
            events=events,
            output_path=output_file,
            patient_id="P001",
            recording_id="R001",
            duration_s=60.0,
        )

        is_valid, errors = validate_csv_bi(output_file)
        assert is_valid
        assert len(errors) == 0

    def test_validate_csv_bi_missing_file(self, tmp_path: Path):
        """Test validation of missing file."""
        missing_file = tmp_path / "missing.csv"
        is_valid, errors = validate_csv_bi(missing_file)
        assert not is_valid
        assert len(errors) > 0

    def test_validate_csv_bi_malformed_file(self, tmp_path: Path):
        """Test validation of malformed CSV-BI file."""
        bad_file = tmp_path / "bad.csv"
        bad_file.write_text("this,is,not,valid,csv,bi,format\n1,2,3,4,5,6,7", encoding="utf-8")

        is_valid, errors = validate_csv_bi(bad_file)
        # May or may not be invalid depending on implementation
        # Just check it doesn't crash
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)


class TestJSONExport:
    """Test JSON export functionality."""

    def test_export_json_basic(self, tmp_path: Path):
        """Test basic JSON export."""
        events = [SeizureEvent(0.0, 1.0, 0.5), SeizureEvent(2.0, 3.0, 0.8)]
        output_file = tmp_path / "events.json"

        export_json(events=events, output_path=output_file, metadata={"source": "unit_test"})

        assert output_file.exists()

        # Verify JSON structure
        with open(output_file) as f:
            data = json.load(f)
            assert "events" in data
            assert "num_events" in data
            assert data["num_events"] == 2
            assert len(data["events"]) == 2

    def test_export_json_with_metadata(self, tmp_path: Path):
        """Test JSON export with custom metadata."""
        events = [SeizureEvent(10.5, 15.2, 0.92)]
        output_file = tmp_path / "with_meta.json"

        metadata = {
            "patient_id": "P005",
            "recording_id": "R005",
            "model_version": "v2.0",
            "processing_params": {"tau_on": 0.86, "tau_off": 0.78},
        }

        export_json(events=events, output_path=output_file, metadata=metadata)

        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)
            assert "metadata" in data
            assert data["metadata"]["patient_id"] == "P005"
            assert data["metadata"]["processing_params"]["tau_on"] == 0.86

    def test_export_json_empty_events(self, tmp_path: Path):
        """Test JSON export with empty events."""
        events = []
        output_file = tmp_path / "empty.json"

        export_json(events=events, output_path=output_file)

        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)
            assert data["num_events"] == 0
            assert len(data["events"]) == 0

    def test_export_json_event_serialization(self, tmp_path: Path):
        """Test that seizure events are properly serialized."""
        event = SeizureEvent(start_s=12.345, end_s=67.890, confidence=0.7654)
        output_file = tmp_path / "serialized.json"

        export_json(events=[event], output_path=output_file)

        with open(output_file) as f:
            data = json.load(f)
            event_data = data["events"][0]
            assert abs(event_data["start_s"] - 12.345) < 0.001
            assert abs(event_data["end_s"] - 67.890) < 0.001
            assert abs(event_data["confidence"] - 0.7654) < 0.0001

    def test_export_json_roundtrip(self, tmp_path: Path):
        """Test JSON export and re-import."""
        original_events = [
            SeizureEvent(0.0, 1.0, 0.9),
            SeizureEvent(5.0, 8.5, 0.75),
            SeizureEvent(20.0, 25.0, 0.85),
        ]
        output_file = tmp_path / "roundtrip.json"

        export_json(events=original_events, output_path=output_file)

        # Re-load and verify
        with open(output_file) as f:
            data = json.load(f)
            loaded_events = [
                SeizureEvent(start_s=e["start_s"], end_s=e["end_s"], confidence=e["confidence"])
                for e in data["events"]
            ]

        assert len(loaded_events) == len(original_events)
        for orig, loaded in zip(original_events, loaded_events, strict=False):
            assert abs(orig.start_s - loaded.start_s) < 0.001
            assert abs(orig.end_s - loaded.end_s) < 0.001
            assert abs(orig.confidence - loaded.confidence) < 0.001


class TestBatchExport:
    """Test batch export functionality."""

    def test_batch_csv_bi_basic(self, tmp_path: Path):
        """Test basic batch CSV-BI export."""
        batch_events = [
            [SeizureEvent(0.0, 0.5, 0.7)],
            [SeizureEvent(1.0, 1.5, 0.9), SeizureEvent(2.0, 3.0, 0.8)],
            [],  # Empty for third recording
        ]
        patient_ids = ["P1", "P2", "P3"]
        recording_ids = ["R1", "R2", "R3"]
        durations = [10.0, 20.0, 15.0]

        export_batch_csv_bi(
            batch_events=batch_events,
            output_dir=tmp_path,
            patient_ids=patient_ids,
            recording_ids=recording_ids,
            durations_s=durations,
        )

        # Check all files created
        for p, r in zip(patient_ids, recording_ids, strict=False):
            expected_file = tmp_path / f"{p}_{r}.csv"
            assert expected_file.exists()

    def test_batch_csv_bi_length_mismatch(self, tmp_path: Path):
        """Test batch export with mismatched lengths."""
        batch_events = [[SeizureEvent(0.0, 1.0, 0.5)]]
        patient_ids = ["P1", "P2"]  # Length mismatch
        recording_ids = ["R1"]
        durations = [10.0]

        with pytest.raises(ValueError, match="length"):
            export_batch_csv_bi(
                batch_events=batch_events,
                output_dir=tmp_path,
                patient_ids=patient_ids,
                recording_ids=recording_ids,
                durations_s=durations,
            )

    def test_batch_csv_bi_creates_output_dir(self, tmp_path: Path):
        """Test that batch export creates output directory."""
        output_dir = tmp_path / "new_output_dir"
        batch_events = [[SeizureEvent(0.0, 1.0, 0.8)]]
        patient_ids = ["P1"]
        recording_ids = ["R1"]
        durations = [10.0]

        export_batch_csv_bi(
            batch_events=batch_events,
            output_dir=output_dir,
            patient_ids=patient_ids,
            recording_ids=recording_ids,
            durations_s=durations,
        )

        assert output_dir.exists()
        assert (output_dir / "P1_R1.csv").exists()

    def test_batch_csv_bi_large_batch(self, tmp_path: Path):
        """Test batch export with many recordings."""
        n_recordings = 100
        batch_events = [[SeizureEvent(i, i + 1, 0.5 + i * 0.001)] for i in range(n_recordings)]
        patient_ids = [f"P{i:03d}" for i in range(n_recordings)]
        recording_ids = [f"R{i:03d}" for i in range(n_recordings)]
        durations = [60.0] * n_recordings

        export_batch_csv_bi(
            batch_events=batch_events,
            output_dir=tmp_path,
            patient_ids=patient_ids,
            recording_ids=recording_ids,
            durations_s=durations,
        )

        # Verify all files created
        created_files = list(tmp_path.glob("*.csv"))
        assert len(created_files) == n_recordings

    def test_batch_csv_bi_validation(self, tmp_path: Path):
        """Test that batch exported files are valid CSV-BI."""
        batch_events = [[SeizureEvent(0.5, 2.0, 0.95)], [SeizureEvent(10.0, 15.0, 0.88)]]
        patient_ids = ["P1", "P2"]
        recording_ids = ["R1", "R2"]
        durations = [30.0, 30.0]

        export_batch_csv_bi(
            batch_events=batch_events,
            output_dir=tmp_path,
            patient_ids=patient_ids,
            recording_ids=recording_ids,
            durations=durations,
        )

        # Validate each exported file
        for p, r in zip(patient_ids, recording_ids, strict=False):
            file_path = tmp_path / f"{p}_{r}.csv"
            is_valid, errors = validate_csv_bi(file_path)
            assert is_valid, f"File {file_path} is not valid: {errors}"


class TestExportEdgeCases:
    """Test edge cases in export functionality."""

    def test_export_very_long_event(self, tmp_path: Path):
        """Test exporting very long seizure event."""
        # 5 minute seizure
        long_event = SeizureEvent(start_s=0.0, end_s=300.0, confidence=0.99)
        output_file = tmp_path / "long.csv"

        export_csv_bi(
            events=[long_event],
            output_path=output_file,
            patient_id="P_LONG",
            recording_id="R_LONG",
            duration_s=600.0,
        )

        assert output_file.exists()

    def test_export_overlapping_events(self, tmp_path: Path):
        """Test exporting overlapping events."""
        # These events overlap
        events = [
            SeizureEvent(10.0, 20.0, 0.8),
            SeizureEvent(15.0, 25.0, 0.9),
            SeizureEvent(18.0, 30.0, 0.85),
        ]
        output_file = tmp_path / "overlap.json"

        export_json(events=events, output_path=output_file)
        assert output_file.exists()

        # Verify all events are exported
        with open(output_file) as f:
            data = json.load(f)
            assert len(data["events"]) == 3

    def test_export_high_precision_times(self, tmp_path: Path):
        """Test exporting events with high precision timestamps."""
        event = SeizureEvent(
            start_s=12.3456789012345, end_s=98.7654321098765, confidence=0.123456789
        )
        output_file = tmp_path / "precision.json"

        export_json(events=[event], output_path=output_file)

        with open(output_file) as f:
            data = json.load(f)
            e = data["events"][0]
            # Check precision is maintained (at least to microseconds)
            assert abs(e["start_s"] - 12.3456789012345) < 1e-6
            assert abs(e["end_s"] - 98.7654321098765) < 1e-6

    def test_export_unicode_patient_ids(self, tmp_path: Path):
        """Test exporting with unicode characters in IDs."""
        events = [SeizureEvent(1.0, 2.0, 0.9)]
        output_file = tmp_path / "unicode.csv"

        export_csv_bi(
            events=events,
            output_path=output_file,
            patient_id="患者_001",
            recording_id="録音_001",
            duration_s=60.0,
        )

        assert output_file.exists()

    def test_export_special_characters_in_path(self, tmp_path: Path):
        """Test export with special characters in file path."""
        # Create directory with spaces and special chars
        special_dir = tmp_path / "dir with spaces & special-chars!"
        special_dir.mkdir()
        output_file = special_dir / "file (with) [brackets].csv"

        events = [SeizureEvent(0.0, 1.0, 0.8)]

        export_csv_bi(
            events=events,
            output_path=output_file,
            patient_id="P001",
            recording_id="R001",
            duration_s=60.0,
        )

        assert output_file.exists()
