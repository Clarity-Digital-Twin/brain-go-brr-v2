# 🧠 Brain-Go-Brr v2: Next-Generation Seizure Detection with Bi-Mamba State Space Models

## 🎯 Mission Statement

This project pioneers the first systematic evaluation of **bidirectional Mamba-2 state space models** combined with **U-Net CNN encoders** and **Residual CNN stacks** for clinical-grade seizure detection. While transformers have dominated recent EEG research, their O(N²) complexity limits real-time deployment. We propose a novel architecture achieving O(N) complexity while maintaining or exceeding transformer performance.

**No one has tested this specific architecture combination before.**

## Why This Matters

Current seizure detection systems suffer from:
- **High false alarm rates** (>10 FA/24h) making them clinically unusable
- **Poor cross-dataset generalization** limiting real-world deployment
- **Computational complexity** preventing real-time edge deployment
- **Lack of standardized evaluation** making progress assessment difficult

We address all four challenges with a unified approach.

## Novel Architecture: U-Net + ResCNN + Bi-Mamba-2

```
EEG Input (19ch, 256Hz)
    ↓
[U-Net Encoder (1D CNN)]  ← Multi-scale morphology extraction
    ↓
[ResCNN Stack (3 blocks)] ← Local pattern enhancement
    ↓
[Bi-Mamba-2 (6 layers)]   ← Long-range temporal dependencies (O(N))
    ↓
[U-Net Decoder]           ← Multi-resolution reconstruction
    ↓
[Sigmoid + Hysteresis]    ← Dual-threshold (τ_on=0.86, τ_off=0.78)
    ↓
[TAES Scoring]            ← Time-aligned clinical evaluation
```

### Key Innovations

1. **First Bi-Mamba-2 for Seizure Detection**: Replacing transformers with state space models for linear complexity
2. **Unified Multi-Scale Architecture**: Combining CNN locality with SSM global context
3. **Hysteresis Post-Processing**: Clinically-inspired dual thresholds reducing false alarms
4. **TAES-First Evaluation**: Focusing on Time-Aligned Event Scoring for clinical relevance

## Performance Targets

Based on NEDC benchmarking standards and epilepsybenchmarks.com:

| Metric | Target | Current SOTA | Why It Matters |
|--------|--------|--------------|----------------|
| Sensitivity @ 10 FA/24h | >95% | ~90% | Clinical usability threshold |
| Sensitivity @ 5 FA/24h | >90% | ~82% | ICU deployment standard |
| Sensitivity @ 1 FA/24h | >75% | ~65% | Home monitoring goal |
| TAES Score | >0.85 | ~0.75 | Temporal alignment quality |
| Cross-dataset AUC | >0.95 | ~0.92 | Generalization capability |

## 🛠️ Technical Stack (2025 Best Practices)

- **Python 3.11+** with UV package manager (10-100x faster than pip)
- **PyTorch 2.5+** with FlashAttention-3 and torch.compile
- **MNE-Python** for robust EDF/BDF file I/O and montage handling
- **Ruff** for blazing-fast linting/formatting (replacing Black/isort/flake8)
- **Pre-commit hooks** ensuring code quality
- **Apache 2.0 License** for open collaboration

## 🚀 Quick Start

```bash
# Install UV (modern Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup project
make setup

# Optional: install GPU/SSM extras (Mamba SSM)
# uv sync -E gpu

# Run local training (reduced data)
make train-local

# Full training with wandb logging
make train
```

Note for WSL/cross-filesystem users ⚙️
- If you ever see a uv hardlink warning during installs, it’s harmless. For silence and stability, you can export these (already defaulted in the Makefile):
  - `export UV_LINK_MODE=copy`
  - `export UV_CACHE_DIR=.uv_cache`

## 📚 Literature Foundation

Our approach synthesizes insights from key papers:

- **FEMBA (2024)**: 21,000-hour pre-trained bidirectional Mamba for EEG
- **EEGMamba (2024)**: Mixture of Experts SSM achieving 98.5% on CHB-MIT
- **SeizureTransformer (2022)**: U-Net + ResCNN baseline architecture
- **NEDC/TAES (2021)**: Clinical evaluation metrics and scoring methodology

## Preprocessing Pipeline

MNE-hybrid approach optimized for seizure morphology:

1. **Robust EDF reading** with MNE's battle-tested parser
2. **19-channel 10-20 montage** standardization
3. **Bandpass filtering** (0.5-120 Hz) preserving seizure signatures
4. **60 Hz notch filter** removing line noise
5. **Sliding windows** (60s, 10s stride) for temporal context

## Model Architecture Details

### U-Net Encoder (1D CNN)
- 4 stages: [64, 128, 256, 512] channels
- Kernel size 5, stride 2 downsampling
- Captures multi-scale EEG morphology

### ResCNN Stack
- 3 residual blocks at bottleneck
- Multi-kernel [3, 5, 7] for frequency diversity
- Dropout 0.1 for regularization

### Bi-Mamba-2 Core
- 6 bidirectional layers
- d_model=512, d_state=16
- Selective state spaces with hardware-aware implementation
- O(N) complexity vs Transformer's O(N²)

### Post-Processing
- **Hysteresis**: τ_on=0.86, τ_off=0.78 (empirically optimized)
- **Morphological filtering**: 5-sample kernel
- **Minimum duration**: 3 seconds (clinical standard)

## 📊 Evaluation Strategy

### Primary Metrics (NEDC Standard)
- TAES @ [10, 5, 2.5, 1] FA/24h
- Sensitivity/Specificity curves
- Cross-dataset generalization

### Benchmark Datasets
1. **TUH EEG Seizure Corpus** (primary)
2. **CHB-MIT** (pediatric validation)
3. **epilepsybenchmarks.com** (final evaluation)

## 🗂️ Project Structure

```
brain-go-brr-v2/
├── src/
│   └── experiment/
│       ├── schemas.py    # Pydantic config models (planned)
│       ├── data.py       # EDF loading, preprocessing (planned)
│       ├── models.py     # U-Net, ResCNN, Bi-Mamba-2 (planned)
│       ├── pipeline.py   # Orchestration entrypoint (stub)
│       └── infra.py      # Caching, logging, utils (planned)
├── configs/              # YAML experiment configs (EEG-focused)
├── literature/           # Converted papers, references
├── tests/                # Pytest suites
└── Makefile              # Automation commands
```

## Why This Will Work

1. **Proven Components**: Each piece (U-Net, ResCNN, Mamba) has shown success individually
2. **Novel Combination**: First to combine all three for seizure detection
3. **Clinical Focus**: TAES-first design prioritizing false alarm reduction
4. **Modern Engineering**: 2025 best practices ensuring reproducibility
5. **Open Science**: Apache 2.0 license enabling collaboration

## Citation

If you use this code, please cite:

```bibtex
@software{brain-go-brr-v2,
  title={Brain-Go-Brr v2: Bi-Mamba State Space Models for Seizure Detection},
  author={Clarity Digital Twin Project},
  year={2025},
  url={https://github.com/clarity-digital-twin/brain-go-brr-v2},
  license={Apache-2.0}
}
```

## ✉️ Contact

For questions, issues, or collaboration:
- GitHub Issues: [Create an issue](https://github.com/clarity-digital-twin/brain-go-brr-v2/issues)
- Email: [Contact maintainers]

---

*"Reducing false alarms from 100+ to <1 per day will transform epilepsy care."*
