from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from src.brain_brr.constants import CHANNEL_NAMES_10_20
from src.brain_brr.events.events import SeizureEvent
from src.brain_brr.szcore.channels import SZCORE_CHANNELS_AVG, SZCORE_TO_OURS, remap_szcore_to_ours
from src.brain_brr.szcore.hed_score import HED_SCORE_HEADER, write_hed_score_tsv


def test_szcore_channel_list_has_19() -> None:
    assert len(SZCORE_CHANNELS_AVG) == 19
    assert all(ch.endswith("-Avg") for ch in SZCORE_CHANNELS_AVG)


def test_szcore_mapping_matches_channel_names() -> None:
    szcore_clean = [ch.replace("-Avg", "") for ch in SZCORE_CHANNELS_AVG]
    computed = [CHANNEL_NAMES_10_20.index(ch) for ch in szcore_clean]
    assert computed == SZCORE_TO_OURS


def test_remap_szcore_to_ours_2d() -> None:
    data = np.zeros((19, 5), dtype=np.float32)
    for i in range(19):
        data[i] = float(i)

    remapped = remap_szcore_to_ours(data)

    assert remapped.shape == (19, 5)
    for szcore_idx, our_idx in enumerate(SZCORE_TO_OURS):
        assert float(remapped[our_idx, 0]) == float(szcore_idx)


def test_remap_szcore_to_ours_3d() -> None:
    data = np.zeros((2, 19, 3), dtype=np.float32)
    for b in range(2):
        for i in range(19):
            data[b, i] = float(i + 100 * b)

    remapped = remap_szcore_to_ours(data)

    assert remapped.shape == (2, 19, 3)
    for b in range(2):
        for szcore_idx, our_idx in enumerate(SZCORE_TO_OURS):
            assert float(remapped[b, our_idx, 0]) == float(szcore_idx + 100 * b)


def test_hed_score_header_exact() -> None:
    assert HED_SCORE_HEADER == "onset\tduration\teventType\tconfidence\tchannels\tdateTime\trecordingDuration\n"


def test_write_hed_score_tsv_bckg(tmp_path) -> None:
    out = tmp_path / "out.tsv"
    write_hed_score_tsv(out, events=[], recording_duration_s=12.34, recording_start_dt=None)

    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == HED_SCORE_HEADER.rstrip("\n")
    assert lines[1].startswith("0.000\t12.340\tbckg\tn/a\tn/a\t1970-01-01 00:00:00\t12.34")


def test_write_hed_score_tsv_sz_rows_sorted_and_clipped(tmp_path) -> None:
    out = tmp_path / "out.tsv"
    events = [
        SeizureEvent(start_s=9.0, end_s=20.0),
        SeizureEvent(start_s=-5.0, end_s=2.0),
        SeizureEvent(start_s=25.0, end_s=30.0),
    ]
    start_dt = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    write_hed_score_tsv(out, events=events, recording_duration_s=26.0, recording_start_dt=start_dt)

    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == HED_SCORE_HEADER.rstrip("\n")
    assert len(lines) == 4  # header + 3 events (one clipped to recording end)
    assert lines[1].startswith("0.000\t2.000\tsz\tn/a\tn/a\t2020-01-01 00:00:00\t26.00")
    assert lines[2].startswith("9.000\t11.000\tsz\tn/a\tn/a\t2020-01-01 00:00:00\t26.00")
    assert lines[3].startswith("25.000\t1.000\tsz\tn/a\tn/a\t2020-01-01 00:00:00\t26.00")
