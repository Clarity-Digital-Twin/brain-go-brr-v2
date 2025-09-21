Preflight & Troubleshooting (TUSZ)

Preflight (must pass before any training)

1) Verify CSV_BI parsing on a known seizure file
   - Expect correct duration, non-zero seizure events, reasonable mask ratio
2) Build or verify cache
   - `python -m src build-cache --data-dir <edf_root> --cache-dir <cache_dir>`
   - Or reuse existing cache
3) Create/scan manifest
   - `python -m src scan-cache --cache-dir <cache_dir>`
   - Expect partial > 0 or full > 0; otherwise STOP
4) Instantiate BalancedSeizureDataset
   - `from src.brain_brr.data import BalancedSeizureDataset; len(BalancedSeizureDataset(Path('<cache_dir>')))`
   - Expect len > 0

Training (smoke → full)

- Smoke: `python -m src train configs/smoke_test.yaml`
- Full: `python -m src train configs/tusz_train_wsl2.yaml`

What went wrong (our real failures)

- CSV_BI mis-parsed as simple CSV → 0% seizures across 250k+ windows
- Label set too narrow (only `seiz`) → missed `cpsz`, etc.
- No manifest/guards → training proceeded despite zero seizures

If scan shows zero seizures

- Confirm CSV paths align with EDFs (co-located `.csv` next to `.edf`)
- Open a CSV with known seizure events; verify codes (e.g., `cpsz`)
- Rebuild cache after fixing parser; rescan; only train when manifest has seizures
- If MNE fails to read an EDF (header parse errors), see `EDF_HEADER_REPAIR.md`

WSL2 and environment tips

- Use `num_workers=0` to avoid multiprocessing hangs
- Ensure pin_memory=false for CPU-only runs on WSL2

CUDA/Mamba SSM notes

- d_conv coerced internally; set `SEIZURE_MAMBA_FORCE_FALLBACK=1` to force Conv1d fallback when needed

Cache hygiene

- Old caches built before CSV_BI fix are invalid → delete and rebuild
- Manifest uses relative filenames; keep it next to NPZs under the cache dir

Code anchors

- src/brain_brr/data/io.py (parser + mask)
- src/brain_brr/data/cache_utils.py (manifest)
- src/brain_brr/train/loop.py (guards and dataset selection)
