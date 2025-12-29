"""HED-SCORE TSV writer for SzCORE submissions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.brain_brr.events.events import SeizureEvent

HED_SCORE_HEADER = "onset\tduration\teventType\tconfidence\tchannels\tdateTime\trecordingDuration\n"


@dataclass(frozen=True)
class HedScoreRow:
    onset_s: float
    duration_s: float
    event_type: str
    confidence: str
    channels: str
    date_time: str
    recording_duration_s: float


def _format_datetime(dt: datetime | None) -> str:
    if dt is None:
        dt = datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def write_hed_score_tsv(
    output_path: Path,
    events: list[SeizureEvent],
    recording_duration_s: float,
    recording_start_dt: datetime | None,
    *,
    confidence_value: str = "n/a",
    channels_value: str = "n/a",
) -> None:
    """Write SzCORE-compliant HED-SCORE TSV.

    Notes:
    - SzCORE ignores `confidence` and `channels` in scoring, but the columns must exist.
    - For seizure-free recordings, SzCORE allows a single `bckg` row whose duration equals the
      recording duration.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dt_str = _format_datetime(recording_start_dt)

    rows: list[HedScoreRow] = []
    if not events:
        rows.append(
            HedScoreRow(
                onset_s=0.0,
                duration_s=float(recording_duration_s),
                event_type="bckg",
                confidence=confidence_value,
                channels=channels_value,
                date_time=dt_str,
                recording_duration_s=float(recording_duration_s),
            )
        )
    else:
        for event in sorted(events, key=lambda e: e.start_s):
            onset_s = max(0.0, float(event.start_s))
            end_s = min(float(recording_duration_s), float(event.end_s))
            duration_s = max(0.0, end_s - onset_s)
            if duration_s <= 0.0:
                continue
            rows.append(
                HedScoreRow(
                    onset_s=onset_s,
                    duration_s=duration_s,
                    event_type="sz",
                    confidence=confidence_value,
                    channels=channels_value,
                    date_time=dt_str,
                    recording_duration_s=float(recording_duration_s),
                )
            )

    with output_path.open("w", encoding="utf-8") as f:
        f.write(HED_SCORE_HEADER)
        for r in rows:
            f.write(
                f"{r.onset_s:.3f}\t{r.duration_s:.3f}\t{r.event_type}\t{r.confidence}\t"
                f"{r.channels}\t{r.date_time}\t{r.recording_duration_s:.2f}\n"
            )

