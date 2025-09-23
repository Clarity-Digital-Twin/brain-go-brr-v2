# 🧠 Brain-Go-Brr v2: TCN + Bi-Mamba for Clinical EEG Seizure Detection

**Pioneering O(N) complexity seizure detection with TCN-Mamba hybrid architecture**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![PyTorch 2.2.2](https://img.shields.io/badge/pytorch-2.2.2-red.svg)](https://pytorch.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

<details>
<summary><strong>Status: v2.3.0 - TCN Architecture Deployed</strong></summary>

- **TCN + Mamba hybrid** architecture fully implemented and training on Modal A100
- **Fixed**: OOM test crashes, LR scheduler warnings, cache isolation
- Local: `configs/local/train.yaml`; Modal A100: `configs/modal/train.yaml`
- Data split: train → dev (tuning) → eval (final one‑shot)
- Last updated: 2025‑09‑22

</details>

## 🎯 Mission

We're investigating whether **Temporal Convolutional Networks (TCN)** combined with **bidirectional Mamba-2 SSMs** can reduce false alarms while maintaining sensitivity on long clinical EEG. Current systems trigger **>10 false alarms per day**, and while transformers work well, their O(N²) cost hinders real-time deployment. Our TCN-Mamba hybrid achieves **O(N) complexity** with superior temporal modeling.

**Why TCN + Mamba?**
- TCNs excel at multi-scale temporal features with dilated convolutions
- Mamba-2 captures long-range dependencies with O(N) efficiency
- Together: Local patterns (TCN) + Global context (Mamba) = Optimal detection

**Our approach**: TCN encoder (8 layers, 16× downsample) → Bidirectional Mamba‑2 (6 layers) → Projection (512→19) + Upsample (960→15360), achieving ~35M parameters with clinically‑focused post‑processing. v2.6 adds an optional Dynamic GNN stage (SSGConv + Laplacian PE) driven by learned adjacency from an edge‑Mamba stream (no heuristic cosine/correlation graphs).

## 🏗️ Architecture

```
EEG Input (19ch, 256Hz, 60s windows)
         ↓
[TCN Encoder]       → Dilated convolutions (8 layers, stride_down=16)
         ↓           Channels: [64, 128, 256, 512]
[Bi-Mamba-2 SSM]    → Long-range dependencies (6 layers, d_model=512)
         ↓           O(N) complexity, d_state=16
[Projection + Upsample] → Restore original resolution (512→19, 960→15360)
         ↓
[Detection Head]    → Per-timestep seizure probabilities
         ↓
[Post-Processing]   → Hysteresis (τ_on=0.86, τ_off=0.78) + Morphology
         ↓
[TAES Evaluation]   → Clinical performance metrics
```

Design decisions (v2.6):
- Pure learned adjacency via edge Mamba stream; no heuristic cosine/correlation graph builder.
- PyG SSGConv (α=0.05) with Laplacian PE (k=16) as the only GNN backend.

**Key Specifications:**
- **Input**: 19-channel 10-20 montage @ 256 Hz
- **Model**: 34.8M parameters (TCN + Mamba hybrid)
- **GPU Requirements**: NVIDIA RTX 4090 (24GB VRAM) minimum for practical training
- **Training Time**: ~16-20 hours for 100 epochs on RTX 4090 (CPU training not recommended: ~4 years)
- **Complexity**: O(N) vs Transformer's O(N²)
- **Window**: 60s with 10s stride (83% overlap)

→ Full architecture details: `docs/02-model/architecture/canonical-spec.md`
→ Current runtime path: `docs/02-model/architecture/current-state.md`

## ⚡ Quick Start

### Installation

```bash
# Install UV package manager (10-100x faster than pip)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/clarity-digital-twin/brain-go-brr-v2.git
cd brain-go-brr-v2
make setup

# Optional: GPU support for Mamba‑SSM CUDA kernels (requires CUDA 12.1)
make setup-gpu
```

### Training

```bash
# Local smoke test (1 epoch, small batch)
make train-local

# Full training (local, WSL2-safe settings)
python -m src train configs/local/train.yaml

# With custom config (dev)
python -m src train configs/local/dev.yaml
```

### 🌩️ Cloud Training (Modal.com)

```bash
# Install Modal CLI
pip install --upgrade modal
modal setup

# Train on Modal (A100-80GB, optimized)
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml

# Quick smoke on Modal
modal run --detach deploy/modal/app.py --action train --config configs/modal/smoke.yaml
```

→ Full guide: `docs/03-deployment/modal/deploy.md` (see also `docs/03-deployment/README.md`)

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

See `docs/README.md` for the full documentation index. Core code lives under `src/brain_brr/` (models, data, train, post, eval, config).

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
| `docs/02-model/architecture/canonical-spec.md` | Canonical technical specification |
| `docs/03-deployment/README.md` | Deployment guides (Modal/local, ops) |
| `docs/03-deployment/operations/evaluation.md` | Evaluation methodology (TAES, FA curves) |
| `docs/03-deployment/local/setup.md` | Local development setup |
| `docs/03-deployment/operations/training.md` | Training procedures and LR scheduling notes |
| `../v2_6_dynamic_gnn_lpe_plan.md` | v2.6 Dynamic GNN + LPE (edge Mamba + learned adjacency) plan |

## 🔧 Requirements

- **Python**: 3.11+ (3.12 supported)
- **PyTorch**: 2.2.2 (required for mamba‑ssm 2.2.2)
- **CUDA**: 12.1 (for GPU acceleration and mamba‑ssm build)
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

## Future Research Direction

See `docs/04-research/future/CANONICAL-ROADMAP.md` for the roadmap and `../v2_6_dynamic_gnn_lpe_plan.md` for the next step (Dynamic GNN + Laplacian PE after Mamba with learned adjacency from an edge stream). ConvNeXt is optional as a local refiner only if metrics indicate a gap; it does not replace the active TCN path.

## Contact

- **Issues**: [GitHub Issues](https://github.com/clarity-digital-twin/brain-go-brr-v2/issues)
- **Discussions**: [GitHub Discussions](https://github.com/clarity-digital-twin/brain-go-brr-v2/discussions)

---

*"Reducing false alarms from 100+ to <1 per day will transform epilepsy care."*
