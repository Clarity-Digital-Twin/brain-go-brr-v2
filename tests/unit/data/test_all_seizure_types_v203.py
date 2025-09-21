import numpy as np

from src.brain_brr.data.io import events_to_binary_mask


def test_events_to_binary_mask_includes_all_tusz_v203_labels() -> None:
    # Build synthetic non-overlapping 1-second events for each seizure label
    labels = [
        "gnsz",
        "fnsz",
        "cpsz",
        "absz",
        "spsz",
        "tcsz",
        "tnsz",
        "mysz",
    ]
    fs = 256.0
    # Place each event one second apart starting at t=0
    events: list[tuple[float, float, str]] = []
    for i, lbl in enumerate(labels):
        start = float(i)
        end = float(i + 1)
        events.append((start, end, lbl))

    duration_sec = float(len(labels) + 1)
    mask = events_to_binary_mask(events, duration_sec=duration_sec, fs=fs)

    assert mask.shape[0] == int(duration_sec * fs)

    # Verify each interval is marked as seizure
    for i in range(len(labels)):
        s0 = int(i * fs)
        s1 = int((i + 1) * fs)
        segment = mask[s0:s1]
        assert np.all(segment == 1.0), f"Label at index {i} not marked as seizure"
