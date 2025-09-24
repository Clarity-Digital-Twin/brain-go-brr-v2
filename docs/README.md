# Brain-Go-Brr Docs Home (SSOT)

This is the canonical, current documentation for the codebase. Historical material lives in `docs/archive/`. TUH-specific guides live in `docs/tusz/` (do not modify from here).

Sections

- 00-Overview — quick orientation and goals
- 01-Installation — environment and GPU stack
- 02-Data — dataset overview and preprocessing
- 03-Configuration — config schema and presets
- 04-Model — V3 architecture and components
  - Time–frequency hybrid: `docs/04-model/time-frequency-hybrid.md`
- 05-Training — local and Modal workflows
- 06-Evaluation — metrics and outputs
- 07-CLI-Tools — CLI and Makefile
- 08-Operations — troubleshooting and performance
  - Dynamic PE stability — see `docs/08-operations/troubleshooting.md#dynamic-pe-nans`
- 09-Development — standards, testing, versioning

Start here: `docs/00-overview/overview.md`.

Source of truth pointers

- Architecture: `docs/04-model/v3-architecture.md`
- Code: `src/brain_brr/models/detector.py`, `src/brain_brr/models/edge_features.py`, `src/brain_brr/models/gnn_pyg.py`, `src/brain_brr/models/mamba.py`, `src/brain_brr/models/tcn.py`, `src/brain_brr/train/loop.py`, `src/brain_brr/data/loader.py`, `src/brain_brr/data/dataset.py`

Quick local stability tips

- If local training hangs (WSL2): set `data.num_workers: 0`.
- If NaNs on RTX 4090: set `training.mixed_precision: false`, reduce `batch_size` or `learning_rate`.

Archival note

- Legacy root docs have been archived into `docs/archive/`. See `docs/ARCHIVE_MAPPING.md` for where each topic now lives in the 0X structure.
