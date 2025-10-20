# Project Overview

- Mission: O(N) clinical seizure detection via TCN → Dual-Stream Mamba → Vectorized GNN → Post-processing.
- Input: 19-channel EEG (60s at 256Hz)
- Output: Per-sample seizure probability `(B, 15360)` and clinical post-processed events

Current architecture (V4.0.0 - October 2025)

**Dual Production Stacks** (v4.0.0 milestone):
- **BiMamba2**: Modal A100-80GB training (PAUSED at Epoch 6 due to costs)
- **FLA (Gated DeltaNet)**: Local RTX 4090 training (ongoing, 100 epochs planned)

Both use V3 dual-stream architecture:
- V3: TCN → Node Mamba(64) + Edge Mamba(16/32) → Learned adjacency → Vectorized PyG GNN + Laplacian PE (dynamic by default) → Projection/Upsample → Detection
  (Legacy V2 heuristic graph path has been removed; kept in docs only for historical context.)
  - Edge similarity is clamped at the source with a configurable margin (`edge_similarity_margin`, default 0.01) to prevent ±1.0 boundary issues.

Model size and stack

- ~31M parameters overall
- TCN: 8 layers, channels [64,128,256,512]
- Node Mamba: 6 layers, d_model=64, headdim=8, d_state=16, expand=2
- Edge Mamba: 2 layers, d_model=16 (BiMamba2) or 32 (FLA), headdim=4/8, d_state=8, expand=2
- GNN: SSGConv×2, α=0.05, Laplacian PE k=16 (dynamic by default; static optional)

Authoritative sources

- Architecture: `docs/04-model/v3-architecture.md`
- Core code: `src/brain_brr/models/detector.py`, `src/brain_brr/models/edge_features.py`, `src/brain_brr/models/gnn_pyg.py`, `src/brain_brr/models/mamba.py`, `src/brain_brr/models/tcn.py`

Data and preprocessing

- Corpus: TUH EEG Seizure Corpus; strict 10-20 montage (19 channels)
- Preprocessing: bandpass 0.5-120Hz, 60Hz notch, resample 256Hz, 60s windows with 10s stride, per-channel z-score
- Cache: memory-mapped NPY pairs in `cache/tusz_mmap/{train,dev}` (and `/results/cache/tusz_mmap` on Modal) with `manifest.json`; BalancedSeizureDataset uses the manifest to avoid random decompression.

**WSL2 CRITICAL**: Cache MUST be on native ext4 filesystem (NOT Windows drives like `/mnt/c` or `/mnt/d`). Windows-hosted mmap causes page evictions → SIGBUS crashes during FLA training. See `docs/08-operations/wsl2-sigbus-fix.md` for details.

Channel order (must maintain)

- ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz", "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2"]

Training quick start

**Local (RTX 4090)**:
- BiMamba2 smoke: `make smoke-bimamba` (or `make smoke-local`)
- BiMamba2 full: `make train-bimamba` (or `make train-local`)
- FLA smoke: `make smoke-fla`
- FLA full: `make train-fla`

**Modal (A100-80GB)**:
- BiMamba2 smoke: `modal run deploy/modal/app.py --action train --config configs/modal/smoke_bimamba.yaml`
- BiMamba2 full (with auto-restart): `modal run --detach deploy/modal/app.py --action schedule-training --config configs/modal/train_bimamba.yaml`
- FLA smoke: `modal run deploy/modal/app.py --action train --config configs/modal/smoke_fla.yaml`
- FLA full (with auto-restart): `modal run --detach deploy/modal/app.py --action schedule-training --config configs/modal/train_fla.yaml`

**CRITICAL**: Use `--action schedule-training` for 100-epoch production runs (auto-restart every 23h), NOT `--action train` (manual restart required).

Environment and versions (exact)

- PyTorch `2.5.0+cu124`, CUDA Toolkit `12.4`, mamba-ssm `2.2.5` (A100 fix), causal-conv1d `1.5.2`, torch-geometric `2.6.1`, numpy `1.26.4`

Where to go next

- Architecture details: `docs/00-overview/architecture-summary.md`
- Configuration: `docs/03-configuration/config-schema.md`
- Training: `docs/05-training/local.md`, `docs/05-training/modal.md`
- Performance targets: `docs/00-overview/performance-targets.md`
- WSL2 SIGBUS fix: `docs/08-operations/wsl2-sigbus-fix.md`
