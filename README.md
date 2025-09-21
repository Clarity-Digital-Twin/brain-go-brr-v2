# 🧠 Brain-Go-Brr v2: Bi-Mamba-2 + U-Net + ResCNN for Clinical EEG Seizure Detection

**Pioneering O(N) complexity seizure detection with bidirectional state space models**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![PyTorch 2.2.2](https://img.shields.io/badge/pytorch-2.2.2-red.svg)](https://pytorch.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

<details>
<summary><strong>Status: Active Development (Benchmarks Pending)</strong></summary>

- Architecture and pipeline are implemented; training/evaluation are in progress.
- No published results yet — benchmarks and clinical metrics will be added once validated.
- For local runs use `configs/tusz_train_wsl2.yaml`; for Modal A100 use `configs/tusz_train_a100.yaml`.
- Data split discipline: train for training; dev for tuning; eval for final one‑shot testing.
- Last updated: 2025‑09‑20.

</details>

## 🎯 Mission

We are investigating whether combining bidirectional state space models (Bi‑Mamba‑2) with multi‑scale CNNs (U‑Net + ResCNN) can reduce false alarms while maintaining sensitivity on long clinical EEG. Current systems often trigger **>10 false alarms per day**, and while transformers perform well, their O(N²) cost hinders real‑time use on long recordings. This project explores an O(N) alternative; benchmarking is pending.

Note (literature, 2025‑09): we are not aware of a published evaluation of this exact architecture (U‑Net + ResCNN + bidirectional Mamba‑2 for clinical seizure detection). While recent work includes SeizureTransformer (U-Net + ResCNN + Transformer) and EEGMamba (bidirectional Mamba for EEG tasks), none combine all three components in our specific configuration.

**Our approach**: A novel architecture combining bidirectional Mamba-2 SSMs with U-Net CNNs and residual convolutions, achieving **O(N) complexity** with a research goal of reducing false alarms to clinically acceptable rates.

## 🏗️ Architecture

```
EEG Input (19ch, 256Hz, 60s windows)
         ↓
[U-Net Encoder]     → Multi-scale feature extraction (4 stages)
         ↓
[ResCNN Stack]      → Local pattern enhancement (3 blocks)
         ↓
[Bi-Mamba-2 SSM]    → Long-range dependencies (6 layers, O(N))
         ↓
[U-Net Decoder]     → Multi-resolution reconstruction
         ↓
[Detection Head]    → Per-timestep seizure probabilities
         ↓
[Post-Processing]   → Hysteresis (τ_on=0.86, τ_off=0.78) + Morphology
         ↓
[TAES Evaluation]   → Clinical performance metrics
```

**Key Specifications:**
- **Input**: 19-channel 10-20 montage @ 256 Hz
- **Model**: ~25M parameters (varies by config)
- **GPU Requirements**: NVIDIA RTX 4090 (24GB VRAM) minimum for practical training
- **Training Time**: ~16-20 hours for 100 epochs on RTX 4090 (CPU training not recommended: ~4 years)
- **Complexity**: O(N) vs Transformer's O(N²)
- **Window**: 60s with 10s stride (83% overlap)

→ Full architecture details: [`docs/architecture/CANONICAL_ARCHITECTURE_SPEC.md`](docs/architecture/CANONICAL_ARCHITECTURE_SPEC.md)

## ⚡ Quick Start

### Installation

```bash
# Install UV package manager (10-100x faster than pip)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/clarity-digital-twin/brain-go-brr-v2.git
cd brain-go-brr-v2
make setup

# Optional: GPU support for Mamba-SSM CUDA kernels
uv sync -E gpu
```

### Training

```bash
# Local smoke test (1 epoch, small batch)
make train-local

# Full training (local, WSL2-safe settings)
python -m src train configs/tusz_train_wsl2.yaml

# With custom config (dev)
python -m src train configs/local.yaml
```

### 🌩️ Cloud Training (Modal.com)

```bash
# Install Modal CLI
pip install --upgrade modal
modal setup

# Train on Modal (A100-80GB, optimized)
modal run --detach deploy/modal/app.py -- --action train --config configs/tusz_train_a100.yaml

# Quick smoke on Modal
modal run --detach deploy/modal/app.py -- --action train --config configs/smoke_test.yaml
```

→ Full guide: [`docs/deployment/MODAL_DEPLOYMENT_COMPLETE_GUIDE.md`](docs/deployment/MODAL_DEPLOYMENT_COMPLETE_GUIDE.md)

## 📊 Performance Goals (pending validation)

These are target goals, not achieved results. Formal validation and benchmarking are pending.

| Metric | Goal (pending) | Clinical Need |
|--------|-----------------|---------------|
| Sensitivity @ 10 FA/24h | >95%          | ICU monitoring |
| Sensitivity @ 5 FA/24h  | >90%          | General ward   |
| Sensitivity @ 1 FA/24h  | >75%          | Home monitoring|
| TAES Score              | >0.85         | Temporal accuracy |
| Inference Speed         | <100ms        | Real-time capable |

## 📁 Project Structure

```
brain-go-brr-v2/
├── src/
│   └── brain_brr/           # Core modules
│       ├── models/           # Neural networks
│       │   ├── detector.py  # Main SeizureDetector
│       │   ├── unet.py      # U-Net encoder/decoder
│       │   ├── rescnn.py    # Residual CNN blocks
│       │   ├── mamba.py     # Bidirectional Mamba-2
│       │   └── layers.py    # Shared components
│       ├── data/             # Data pipeline
│       │   ├── io.py        # EDF/MNE loading
│       │   ├── datasets.py  # PyTorch datasets
│       │   ├── preprocess.py # Filtering/normalization
│       │   └── windows.py   # Window extraction
│       ├── train/            # Training loop
│       │   └── loop.py      # Training/validation
│       ├── post/             # Post-processing
│       │   └── postprocess.py # Hysteresis/morphology
│       ├── events/           # Event handling
│       │   ├── events.py    # Event generation
│       │   └── export.py    # CSV_BI export
│       ├── eval/             # Evaluation
│       │   └── metrics.py   # TAES, FA curves
│       ├── streaming/        # Real-time inference
│       │   └── streaming.py # Online processing
│       ├── config/           # Configuration
│       │   └── schemas.py   # Pydantic schemas
│       ├── cli/              # CLI interface
│       │   └── cli.py       # Main entry point
│       └── constants.py      # Global constants
├── configs/                  # YAML configurations
├── docs/                     # Documentation
│   ├── architecture/         # Technical specs
│   ├── deployment/           # Cloud guides
│   ├── implementation/       # Setup notes
│   └── phases/              # Development plans
├── tests/                    # Test suite
├── deploy/
│   └── modal/
│       └── app.py           # Modal deployment entrypoint
└── Makefile                 # Automation commands
```

## 🛠️ Development

```bash
# Quality checks (run after every change!)
make q          # Lint + format + type check

# Testing
make t          # Fast tests without coverage
make test       # Full tests with coverage
make test-gpu   # GPU-specific tests

# Training
make train-local  # Local development
make train        # Full training run
```

## 📚 Key References

### Papers
- **FEMBA (2024)**: 21,000-hour pre-trained bidirectional Mamba for EEG
- **EEGMamba (2024)**: Mixture of Experts SSM achieving 98.5% on CHB-MIT
- **SeizureTransformer (2022)**: U-Net + ResCNN baseline architecture
- **NEDC/TAES (2021)**: Clinical evaluation metrics

### Code References
- [`nedc-bench`](https://github.com/Clarity-Digital-Twin/nedc-bench): Our TAES implementation
- [`mamba-ssm`](https://github.com/state-spaces/mamba): Official Mamba-2
- [`SeizureTransformer`](reference_repos/SeizureTransformer): Architecture patterns

## 🔗 Documentation

| Document | Description |
|----------|-------------|
| [`CANONICAL_ARCHITECTURE_SPEC.md`](docs/architecture/CANONICAL_ARCHITECTURE_SPEC.md) | Complete technical specification |
| [`MODAL_DEPLOYMENT_SSOT.md`](docs/deployment/MODAL_DEPLOYMENT_SSOT.md) | Cloud deployment guide (Single Source of Truth) |
| [`PHASE5_EVALUATION.md`](docs/phases/PHASE5_EVALUATION.md) | Evaluation methodology |
| [`SETUP_NOTES.md`](docs/implementation/SETUP_NOTES.md) | Development setup |

## 🔧 Requirements

- **Python**: 3.11+ (3.12 supported)
- **PyTorch**: 2.2.2 (required for mamba-ssm)
- **CUDA**: 11.8+ (optional, for GPU acceleration)
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: 24GB+ VRAM for full training

**Note**: NumPy must be <2.0 for mamba-ssm compatibility

## 📈 Benchmarks

Results are pending. We will publish full benchmarks after rigorous evaluation on held-out datasets.

| Dataset        | Status   |
|----------------|----------|
| TUH EEG v2.0.0 | Pending  |
| CHB-MIT        | Pending  |

## Citation

```bibtex
@software{brain-go-brr-v2,
  title={Brain-Go-Brr v2: Bi-Mamba State Space Models for Seizure Detection},
  author={Clarity Digital Twin Project},
  year={2025},
  url={https://github.com/clarity-digital-twin/brain-go-brr-v2},
  license={Apache-2.0}
}
```

## License

Apache License 2.0 - See [LICENSE](LICENSE)

## Contact

- **Issues**: [GitHub Issues](https://github.com/clarity-digital-twin/brain-go-brr-v2/issues)
- **Discussions**: [GitHub Discussions](https://github.com/clarity-digital-twin/brain-go-brr-v2/discussions)

---

*"Reducing false alarms from 100+ to <1 per day will transform epilepsy care."*
