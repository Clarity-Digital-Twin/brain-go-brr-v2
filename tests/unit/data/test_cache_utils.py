from pathlib import Path

import numpy as np

from src.brain_brr.data.cache_utils import cache_file_path, check_cache_completeness


def test_check_cache_completeness(tmp_path: Path) -> None:
    edf_root = tmp_path / "edf"
    cache_root = tmp_path / "cache"
    edf_root.mkdir()
    cache_root.mkdir()

    edf_files = [edf_root / f"rec_{i:03d}.edf" for i in range(5)]
    for p in edf_files:
        p.write_bytes(b"")

    # Create cache only for some files (NPY format)
    present = {0, 2, 4}
    for i, p in enumerate(edf_files):
        if i in present:
            # Create NPY files (mmap format)
            data_path = cache_file_path(cache_root, p)
            labels_path = data_path.parent / data_path.name.replace("_data.npy", "_labels.npy")
            np.save(data_path, np.zeros((1, 19, 10), dtype=np.float32))
            np.save(labels_path, np.zeros((1, 10), dtype=np.float32))

    status = check_cache_completeness(edf_files, cache_root)
    assert status.total_files == 5
    assert status.cached_files == 3
    assert status.missing_files == 2
    missing_stems = sorted([m.stem for m in status.missing])
    assert missing_stems == ["rec_001", "rec_003"]
