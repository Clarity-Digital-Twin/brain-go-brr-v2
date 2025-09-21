# Test Plan (Canonical)

Last updated: 2025-09-21

Purpose: Provide a practical, current testing strategy aligned with the codebase after the TUSZ CSV_BI parser and SeizureTransformer-style balancing fixes.

## Goals

- Verify data integrity end-to-end (EDF → labels → cache → manifest → dataset → batch).
- Ensure training never starts with zero-seizure datasets.
- Keep tests fast locally; defer heavy runs to smoke tests.

## Test Layers

- Unit
  - data/io.py: CSV_BI parsing (channel,start,stop,label,confidence), seizure label set, mask boundaries.
  - data/cache_utils.py: scan-existing-cache categorization (partial/full/no), corrupted NPZ handling, missing labels → no_seizure.
  - data/datasets.py: BalancedSeizureDataset composition and reproducibility.
- Integration (lightweight)
  - Mini-cache build on a dozen EDF/CSV files with known seizures; manifest shows partial>0 or full>0.
  - Training smoke test (1 epoch, tiny batch) verifies non-zero seizure prevalence in batches.
- Performance (optional, local only)
  - Dataset iteration speed on ~10k entries; no memory growth over 3 epochs.

## Required Tests (now)

- Unit
  - tests/unit/data/test_tusz_csv_bi_parser.py
    - Parses duration from header comments.
    - Reads label codes: {seiz, gnsz, fnsz, spsz, cpsz, absz, tnsz, tcsz, spkz}.
    - Builds binary mask at fs=256 with correct boundaries.
  - tests/unit/data/test_manifest_and_balanced.py
    - scan_existing_cache builds manifest with correct counts.
    - BalancedSeizureDataset: ALL partial + 0.3× full + 2.5× background; deterministic with seed.

- Integration
  - tests/integration/data/test_manifest_real_subset.py (todo)
    - Build tiny cache from seizure-positive CSVs; assert manifest partial>0 or full>0.
  - tests/integration/train/test_smoke_training.py (todo)
    - 1 epoch run; assert batches contain seizures and training completes.

## Commands

- Fast checks
  - make q
  - .venv/bin/pytest -q tests/unit

- Mini-cache preflight (local data required)
  - python -m src scan-cache --cache-dir cache/tusz/train
  - Expect: partial>0 or full>0; otherwise FAIL and stop.

- Smoke training
  - python -m src train configs/smoke_test.yaml

## Guardrails

- Training exits if BalancedSeizureDataset has zero windows.
- scan-cache warns on corrupted NPZ and missing labels; missing labels are categorized as no_seizure.

## Future Work

- Add integration tests on a small real TUSZ subset.
- Add performance tests for dataset iteration and memory stability.

