# Project Overview

- Mission: O(N) clinical seizure detection via TCN → Dual‑Stream Mamba → Vectorized GNN → Post‑processing.
- Input: 19‑channel EEG (60s at 256Hz)
- Output: Per‑sample seizure probability `(B, 15360)` and clinical post‑processed events

What’s implemented

- V2 (TCN path): TCN → Bi‑Mamba2(512) → Projection/Upsample → Detection; optional heuristic GNN
- V3 (Dual‑stream): TCN → Node Mamba(64) + Edge Mamba(16) → Learned adjacency → Vectorized PyG GNN + Laplacian PE (dynamic by default) → Projection/Upsample → Detection

Model size and stack

- ~31M parameters overall
- TCN: 8 layers, channels [64,128,256,512]
- Node Mamba: 6 layers, d_model=64, headdim=8, d_state=16, expand=2
- Edge Mamba: 2 layers, d_model=16, headdim=4, d_state=8, expand=2
- GNN: SSGConv×2, α=0.05, Laplacian PE k=16 (dynamic by default; static optional)

Authoritative sources

- Ground truth: `V3_ARCHITECTURE_AS_IMPLEMENTED.md`
- Core code: `src/brain_brr/models/detector.py`, `src/brain_brr/models/edge_features.py`, `src/brain_brr/models/gnn_pyg.py`, `src/brain_brr/models/mamba.py`, `src/brain_brr/models/tcn.py`

Data and preprocessing

- Corpus: TUH EEG Seizure Corpus; strict 10–20 montage (19 channels)
- Preprocessing: bandpass 0.5–120Hz, 60Hz notch, resample 256Hz, 60s windows with 10s stride, per‑channel z‑score
- Cache: `cache/tusz/{train,val}` with NPZ files and `manifest.json`; balanced sampling uses manifest (no random I/O)

Channel order (must maintain)

- ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz", "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2"]

Training quick start

- Local smoke: `make s`
- Full local: `make train-local`
- Modal smoke: `modal run deploy/modal/app.py --action train --config configs/modal/smoke.yaml`
- Modal full (detached): `modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml`

Environment and versions (exact)

- PyTorch `2.2.2+cu121`, CUDA Toolkit `12.1`, mamba‑ssm `2.2.2`, causal‑conv1d `1.4.0`, torch‑geometric `2.6.1`, numpy `1.26.4`

Where to go next

- Architecture details: `docs/00-overview/architecture-summary.md`
- Configuration: `docs/03-configuration/config-schema.md`
- Training: `docs/05-training/local.md`, `docs/05-training/modal.md`
- Performance targets: `docs/00-overview/performance-targets.md`
