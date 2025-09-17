# CLAUDE.md — AI Assistant Operating Rules

Single source of truth for ALL AI coding assistants (Claude, Copilot, Cursor, etc.). Follow this file for repo‑specific behavior. ✨

## 🧠 Project Overview

Brain‑Go‑Brr v2: First Bi‑Mamba‑2 + U‑Net + ResCNN for clinical EEG seizure detection — O(N) sequence modeling with bidirectional SSM.

Why this is different:
- Transformers struggle on long EEG (O(N²) cost)
- Pure CNNs miss global temporal context
- Bidirectional Mamba‑2 brings O(N) global context efficiently

## 🏗️ Architecture

| Component | Specification |
|-----------|--------------|
| Input | 19‑channel EEG @ 256 Hz |
| U‑Net Encoder | [64, 128, 256, 512] channels, ×16 downsample |
| ResCNN | 3 blocks, kernels [3, 5, 7] |
| Bi‑Mamba‑2 | 6 layers, d_model=512, d_state=16 |
| Hysteresis | τ_on=0.86, τ_off=0.78 |
| Output | Per‑timestep probabilities |

## 🎯 Clinical Targets (TAES)

- 10 FA/24h: >95% sensitivity
- 5 FA/24h: >90% sensitivity
- 1 FA/24h: >75% sensitivity

## 📁 Project Structure

```
src/experiment/     # Core modules (preserve public APIs)
├── schemas.py      # Pydantic configs (planned)
├── data.py        # EEG preprocessing (planned)
├── models.py      # Bi‑Mamba‑2 architecture (planned)
├── pipeline.py    # Orchestration (stub)
└── infra.py       # Infra utils (planned)

configs/           # YAML experiments
tests/             # Pytest suite
data/              # Datasets (git‑ignored)
results/           # Outputs (git‑ignored)
```

## ⚡ Essential Commands

| Command | Purpose |
|---------|---------|
| `make q` | Quality check (lint+format+mypy) ✅ |
| `make t` | Fast tests (no coverage) |
| `make l` / `make f` | Lint / Format with Ruff |
| `make setup` | Initial setup (uv, hooks) |
| `make train-local` | Local training config |
| `uv sync -E gpu` | GPU extra (Mamba‑SSM) |

## 🔧 Development Rules

1) Quality first: run `make q` after EVERY change 🧹
2) Type everything: full type hints required
3) Tests required for new functions (unit/integration markers)
4) No comments unless explicitly requested
5) Follow neighboring file patterns; preserve `src/experiment/` APIs

Code style
- Python 3.11+, 4‑space indent, Ruff line length 100
- Imports: stdlib → third‑party → first‑party (sorted)

## 🔬 Tech Stack

- Python 3.11+ (UV package manager)
- PyTorch ≥2.5.0
- MNE ≥1.5.0
- Ruff (lint/format), mypy (strict typing)
- mamba‑ssm (GPU extra only): install with `uv sync -E gpu`

## 📊 Data Pipeline

1) Read EDF via MNE; 10‑20 montage
2) Bandpass 0.5–120 Hz; 60 Hz notch
3) Resample to 256 Hz
4) Window 60s with 10s stride
5) Per‑channel z‑score normalization

## 🚀 Training Strategy

- Train: TUH EEG Seizure Corpus
- Validate: CHB‑MIT
- Evaluate: epilepsybenchmarks.com
- No pretrained weights (novel architecture)

## ⚠️ Critical Notes

- Caching keys depend on config; edit config to invalidate cache
- WSL tip: `export UV_LINK_MODE=copy` (Makefile sets this by default) ⚙️
- CI uses `uv sync` (no extras) to avoid GPU builds on non‑CUDA runners
- Keep README, Makefile, pyproject, and configs in sync with this file

---
Mission: Shock the world with O(N) clinical seizure detection. 🚀
