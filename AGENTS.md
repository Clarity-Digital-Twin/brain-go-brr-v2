# AGENTS.md - Universal AI Agent Instructions

Universal instructions for ALL AI coding assistants (Claude, GitHub Copilot, Cursor, etc.)

## 🧠 Project Overview

**Brain-Go-Brr v2**: World's first Bi-Mamba-2 + U-Net + ResCNN for clinical EEG seizure detection

**Why Revolutionary**:
- Transformers fail on long EEG (O(N²) complexity)
- CNNs miss global context
- We achieve O(N) with bidirectional SSM

## 🏗️ Architecture

| Component | Specification |
|-----------|--------------|
| Input | 19-channel EEG @ 256 Hz |
| U-Net Encoder | [64, 128, 256, 512] channels, ×16 downsample |
| ResCNN | 3 blocks, kernels [3, 5, 7] |
| Bi-Mamba-2 | 6 layers, d_model=512, d_state=16 |
| Hysteresis | τ_on=0.86, τ_off=0.78 |
| Output | Per-timestep probabilities |

## 🎯 Clinical Targets (TAES)

- **10 FA/24h**: >95% sensitivity
- **5 FA/24h**: >90% sensitivity
- **1 FA/24h**: >75% sensitivity

## 📁 Project Structure

```
src/experiment/     # Core modules
├── schemas.py      # Pydantic configs
├── data.py        # EEG preprocessing
├── models.py      # Bi-Mamba-2 architecture
├── pipeline.py    # Training orchestration
└── infra.py       # Infrastructure

configs/           # YAML experiments
tests/            # Pytest suite
data/            # Datasets (git-ignored)
results/         # Outputs (git-ignored)
```

## ⚡ Essential Commands

| Command | Purpose |
|---------|---------|
| `make q` | Quality check (MUST run after code changes) |
| `make t` | Fast tests |
| `make train-local` | Local training |
| `make setup` | Initial setup |
| `uv sync -E gpu` | GPU support |

## 🔧 Development Rules

1. **Quality First**: Run `make q` after EVERY change
2. **Type Everything**: Full type hints required
3. **Test Coverage**: Unit tests for new functions
4. **No Comments**: Unless explicitly requested
5. **Follow Patterns**: Check neighboring files first

## 🔬 Technical Stack

- **Python 3.11+** with UV package manager
- **PyTorch 2.0+** for deep learning
- **mamba-ssm 2.0+** for Bi-Mamba-2
- **MNE 1.5+** for EEG I/O
- **Ruff** for formatting/linting
- **mypy** for strict typing

## 📊 Data Pipeline

1. Read EDF files with MNE
2. Bandpass 0.5-120 Hz, notch 60 Hz
3. Resample to 256 Hz
4. Window: 60s with 10s stride
5. Per-channel z-score normalization

## 🚀 Training Strategy

- **Dataset**: TUH EEG Seizure Corpus
- **Validation**: CHB-MIT
- **Evaluation**: epilepsybenchmarks.com
- **No pretrained weights** (novel architecture)

## ⚠️ Critical Notes

- Cache invalidation via config changes
- WSL users: `export UV_LINK_MODE=copy`
- Conventional commits (feat:, fix:, test:)
- Preserve `src/experiment/` APIs

---
**Mission: Shock the world with O(N) clinical seizure detection**