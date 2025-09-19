"""Export functions for seizure detection results.

Handles CSV_BI (Temple-compliant) and JSON export formats.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.experiment.events import SeizureEvent


def export_csv_bi(
    events: list[SeizureEvent],
    output_path: Path | str,
    patient_id: str,
    recording_id: str,
    duration_s: float,
    montage_file: str = "nedc_eas_default_montage.txt",
) -> None:
    """Export events in Temple-compliant CSV_BI format.

    Args:
        events: List of SeizureEvent objects
        output_path: Output file path
        patient_id: Patient identifier
        recording_id: Recording identifier
        duration_s: Total recording duration in seconds
        montage_file: Montage file name (default for NEDC)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build header (exact format required)
    lines = [
        "# version = csv_v1.0.0\n",
        f"# bname = {patient_id}_{recording_id}\n",
        f"# duration = {duration_s:.4f} secs\n",
        f"# montage_file = {montage_file}\n",
        "#\n",
        "channel,start_time,stop_time,label,confidence\n",
    ]

    # Add events sorted by start time
    sorted_events = sorted(events, key=lambda e: e.start_s)
    for event in sorted_events:
        lines.append(f"TERM,{event.start_s:.4f},{event.end_s:.4f},seiz,{event.confidence:.4f}\n")

    # Write to file
    output_path.write_text("".join(lines), encoding="utf-8")


def export_json(
    events: list[SeizureEvent],
    output_path: Path | str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Export events in JSON format.

    Args:
        events: List of SeizureEvent objects
        output_path: Output file path
        metadata: Optional metadata to include
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert events to dictionaries
    events_data = [
        {
            "start_s": event.start_s,
            "end_s": event.end_s,
            "duration_s": event.duration,
            "confidence": event.confidence,
        }
        for event in events
    ]

    # Build output structure
    output = {"events": events_data, "num_events": len(events)}

    if metadata:
        output["metadata"] = metadata

    # Add summary statistics
    if events:
        output["summary"] = {
            "total_duration_s": sum(e.duration for e in events),
            "mean_duration_s": sum(e.duration for e in events) / len(events),
            "mean_confidence": sum(e.confidence for e in events) / len(events),
            "min_confidence": min(e.confidence for e in events),
            "max_confidence": max(e.confidence for e in events),
        }

    # Write JSON
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


def export_batch_csv_bi(
    batch_events: list[list[SeizureEvent]],
    output_dir: Path | str,
    patient_ids: list[str],
    recording_ids: list[str],
    durations_s: list[float],
    montage_file: str = "nedc_eas_default_montage.txt",
) -> None:
    """Export batch of events to multiple CSV_BI files.

    Args:
        batch_events: List of event lists
        output_dir: Output directory
        patient_ids: List of patient identifiers
        recording_ids: List of recording identifiers
        durations_s: List of recording durations
        montage_file: Montage file name
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (len(batch_events) == len(patient_ids) == len(recording_ids) == len(durations_s)):
        raise ValueError("All input lists must have the same length")

    for _i, (events, patient_id, recording_id, duration) in enumerate(
        zip(batch_events, patient_ids, recording_ids, durations_s, strict=False)
    ):
        output_file = output_dir / f"{patient_id}_{recording_id}.csv"
        export_csv_bi(events, output_file, patient_id, recording_id, duration, montage_file)


def validate_csv_bi(file_path: Path | str) -> tuple[bool, list[str]]:
    """Validate a CSV_BI file for Temple compliance.

    Args:
        file_path: Path to CSV_BI file

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    file_path = Path(file_path)
    errors = []

    if not file_path.exists():
        return False, ["File does not exist"]

    lines = file_path.read_text(encoding="utf-8").splitlines()

    # Check minimum lines
    if len(lines) < 6:
        errors.append("File too short (minimum 6 lines required)")
        return False, errors

    # Check header format
    expected_headers = [
        "# version = csv_v1.0.0",
        "# bname = ",  # Prefix check
        "# duration = ",  # Prefix check
        "# montage_file = ",  # Prefix check
        "#",
        "channel,start_time,stop_time,label,confidence",
    ]

    for i, expected in enumerate(expected_headers):
        if i in [1, 2, 3]:  # Prefix checks
            if not lines[i].startswith(expected):
                errors.append(f"Line {i + 1}: Expected to start with '{expected}'")
        else:
            if lines[i] != expected:
                errors.append(f"Line {i + 1}: Expected '{expected}', got '{lines[i]}'")

    # Check data rows
    for i, line in enumerate(lines[6:], start=7):
        parts = line.split(",")
        if len(parts) != 5:
            errors.append(f"Line {i}: Expected 5 columns, got {len(parts)}")
            continue

        # Check channel is TERM
        if parts[0] != "TERM":
            errors.append(f"Line {i}: Channel must be 'TERM', got '{parts[0]}'")

        # Check times are floats
        try:
            start = float(parts[1])
            stop = float(parts[2])
            if stop <= start:
                errors.append(f"Line {i}: Stop time must be > start time")
        except ValueError:
            errors.append(f"Line {i}: Times must be valid floats")

        # Check label is 'seiz'
        if parts[3] != "seiz":
            errors.append(f"Line {i}: Label must be 'seiz', got '{parts[3]}'")

        # Check confidence is float in [0, 1]
        try:
            conf = float(parts[4])
            if not (0 <= conf <= 1):
                errors.append(f"Line {i}: Confidence must be in [0, 1], got {conf}")
        except ValueError:
            errors.append(f"Line {i}: Confidence must be valid float")

    return len(errors) == 0, errors
