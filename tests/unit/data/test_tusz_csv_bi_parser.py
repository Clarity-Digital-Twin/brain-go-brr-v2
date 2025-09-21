from pathlib import Path

import numpy as np

from src.brain_brr.data.io import events_to_binary_mask, parse_tusz_csv


def test_parse_tusz_csv_bi_and_mask(tmp_path: Path) -> None:
    csv = tmp_path / "sample.csv"
    content = (
        "# version = csv_v1.0.0\n"
        "# bname = aaaaaa_s001_t000\n"
        "# duration = 300.00 secs\n"
        "channel,start_time,stop_time,label,confidence\n"
        "FP1-F7,0.0000,36.0000,bckg,1.0000\n"
        "FP1-F7,36.0000,180.0000,cpsz,1.0000\n"
        "FP1-F7,180.0000,300.0000,bckg,1.0000\n"
    )
    csv.write_text(content)

    dur, events = parse_tusz_csv(csv)
    assert abs(dur - 300.0) < 1e-3
    # Expect 3 rows, 1 seizure
    assert len(events) == 3
    assert any(lbl == "cpsz" for _, _, lbl in events)

    # Build mask at 256 Hz across full duration
    fs = 256.0
    mask = events_to_binary_mask(events, duration_sec=dur, fs=fs)
    assert mask.shape[0] == int(dur * fs)
    # Check seizure region (36s..180s) has ones, and pre/post are zeros
    s0 = int(36.0 * fs)
    s1 = int(180.0 * fs)
    assert np.all(mask[: s0 - 1] == 0.0)
    assert np.all(mask[s0:s1] == 1.0)
    assert np.all(mask[s1 + 1 :] == 0.0)
