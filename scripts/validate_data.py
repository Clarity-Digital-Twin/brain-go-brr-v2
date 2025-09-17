from __future__ import annotations

from pathlib import Path

import numpy as np

from src.experiment.constants import STRIDE_SAMPLES, WINDOW_SAMPLES
from src.experiment.data import extract_windows, load_edf_file, preprocess_recording


def validate_data_pipeline() -> None:
    # Example sample files (update paths as needed)
    test_files = [
        Path("data/samples/tuh_sample.edf"),
        Path("data/samples/chb_sample.edf"),
    ]

    for file_path in test_files:
        print("=" * 60)
        print(f"Testing: {file_path}")

        if not file_path.exists():
            print("⚠️  Skipping (file not found)")
            continue

        data_uv, fs = load_edf_file(file_path)
        print(f"✅ Loaded: {data_uv.shape} @ {fs} Hz (µV)")

        proc = preprocess_recording(data_uv, fs_original=fs)
        print(f"✅ Preprocessed: {proc.shape} @ target fs")

        windows, _, _ = extract_windows(proc, WINDOW_SAMPLES, STRIDE_SAMPLES)
        print(f"✅ Windows: {windows.shape}")

        # Basic checks
        assert windows.dtype == np.float32
        assert windows.shape[1] == 19
        assert windows.shape[2] == WINDOW_SAMPLES
        assert np.isfinite(windows).all()

        print("🎉 Validation OK")

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    validate_data_pipeline()

