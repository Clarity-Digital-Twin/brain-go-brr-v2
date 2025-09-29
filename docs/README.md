# Brain-Go-Brr Docs Home (SSOT)

This is the canonical, current documentation for the codebase. Historical drafts in `docs/archive/` have been retired and their relevant content integrated here. TUH-specific guides live in `docs/tusz/` (do not modify from here).

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
- Code: `src/brain_brr/models/detector.py`, `src/brain_brr/models/gnn_pyg.py`, `src/brain_brr/models/mamba.py`, `src/brain_brr/models/tcn.py`, `src/brain_brr/train/loop.py`, `src/brain_brr/data/datasets.py`, `src/brain_brr/data/io.py`, `src/brain_brr/data/preprocess.py`, `src/brain_brr/data/windows.py`, `src/brain_brr/data/tusz_splits.py`

Stability (implemented)

- PR‑1 boundary norms: `model.norms.*` toggles normalization at seams.
- PR‑2 bounded edge lift: `graph.edge_lift_activation|edge_lift_norm`, `edge_lift_init_gain: 0.1`.
- PR‑3 adjacency conditioning: `graph.adj_row_softmax|adj_ema_beta|adj_force_symmetric|laplacian_eps`.
- PR‑5 clamp at source: `graph.edge_similarity_margin` keeps cosine/corr within `[-1+margin, 1‑margin]`.

Dataset strategy (CRITICAL)

- **Training**: Uses `BalancedSeizureDataset` with manifest to oversample seizures (8% → ~30%)
- **Validation**: Uses `EEGWindowDataset` with natural distribution (~8% seizures)
- **Why different**: Train on balanced data to learn patterns, validate on real distribution for true metrics
- **Manifest**: train/manifest.json REQUIRED, dev/manifest.json optional (validation doesn't use it)

Quick local stability tips

- If local training hangs (WSL2): set `data.num_workers: 0`.
- If NaNs on RTX 4090: set `training.mixed_precision: false`, reduce `batch_size` or `learning_rate`.

Archival note

- Legacy drafts have been retired. Any previously archived content that remained relevant has been merged into the sections above.
