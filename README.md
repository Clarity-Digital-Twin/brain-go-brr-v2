# Brain-Go-Brr V3

**EEG seizure detection. TCN + BiMamba + GNN. 31M parameters.**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![PyTorch 2.2.2](https://img.shields.io/badge/pytorch-2.2.2-red.svg)](https://pytorch.org)
[![CUDA 12.1](https://img.shields.io/badge/cuda-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)

## Architecture

```
Input: 19 channels × 60 seconds @ 256Hz = (B, 19, 15360)
   ↓
TCN: 8 layers, dilations [1,2,4,8,16,32,64,128], stride=16
   → (B, 512, 960)
   ↓
architecture="tcn" path:              architecture="v3" path:
BiMamba: 6 layers                     Projection → (B, 19×64, 960)
   → (B, 512, 960)                       ↓
   ↓                                  Node Mamba: 19 parallel (B, 64, 960)
Decoder: 4 stages                     Edge Mamba: 171 parallel (B, 1→16→1, 960)
   → (B, 19, 15360)                      ↓
   ↓                                  GNN: 2-layer SSGConv + LPE
Detection head                           ↓
   → (B, 15360)                      Back-projection → Decoder → Detection
```

## Implementation Status

✅ **Working:**
- TCN encoder with dilated convolutions
- Bidirectional Mamba-2 (6 layers, d_state=16)
- V3 dual-stream path (node + edge Mambas)
- Dynamic Laplacian PE (configurable interval)
- Focal loss with class balancing
- Hysteresis post-processing

⚠️ **In Progress:**
- Training on TUSZ (4667 train, 1832 dev files)
- Cache currently building (~50GB NPZ files)

❌ **Not Implemented:**
- STFT side-branch (planned, see FUTURE_WORK_STFT_ENHANCEMENT.md)
- Real-time inference optimization
- ONNX export

## Setup

```bash
# Requirements: CUDA 12.1, 24GB+ VRAM
git clone https://github.com/Clarity-Digital-Twin/brain-go-brr-v2
cd brain-go-brr-v2

# Install (exact versions matter)
make setup          # UV environment
make setup-gpu      # Mamba CUDA + PyG

# Test
make smoke          # 1 epoch, 3 files
```

## Training

### Local (RTX 4090)
```yaml
# configs/local/train.yaml
architecture: v3    # or "tcn" for v2
batch_size: 4       # 16GB VRAM usage
mixed_precision: false  # MUST be false or NaNs
semi_dynamic_interval: 5  # PE every 5 timesteps
```

```bash
tmux new -s train
make train-local    # ~200 hours total
```

### Cloud (Modal A100)
```yaml
# configs/modal/train.yaml
batch_size: 64
mixed_precision: true
use_dynamic_pe: true  # Full dynamic
```

```bash
modal run --detach deploy/modal/app.py \
  --action train --config configs/modal/train.yaml
# ~100 hours, $319
```

## Data Pipeline

1. **TUSZ EDF files** → MNE preprocessing
2. **Resample** 256Hz, **bandpass** 0.5-120Hz, **notch** 60Hz
3. **Window** 60s @ 10s stride → 15360 samples/window
4. **Cache** as NPZ: `cache/tusz/{train,dev}/*.npz`
5. **Balanced sampling** for 12:1 class imbalance

## Model Details

```python
# src/brain_brr/models/detector.py
class SeizureDetector(nn.Module):
    def __init__(self, cfg):
        # TCN: Multi-scale temporal extraction
        self.tcn_encoder = TCNEncoder(...)  # 8 layers

        # V2 path: single Mamba
        self.bidirectional_mamba = BiMamba2(...)  # 6 layers

        # V3 path: dual-stream
        self.node_mamba = nn.ModuleList([Mamba2(...) for _ in range(19)])
        self.edge_mamba = nn.ModuleList([Mamba2(...) for _ in range(171)])

        # Optional GNN (both paths)
        if cfg.graph.enabled:
            self.gnn = VectorizedGNN(...)  # SSGConv, α=0.05
            self.lpe = LaplacianPE(k=16)
```

**Parameters:** 31,475,722 (counted via `sum(p.numel())`)

## Post-Processing

```python
# src/brain_brr/post/postprocess.py
hysteresis: tau_on=0.86, tau_off=0.78
morphology: opening=11, closing=31
duration: 3-600s valid
merging: within 2s
```

## Critical Issues

1. **RTX 4090**: `mixed_precision: false` or instant NaN
2. **WSL2**: `num_workers: 0` or multiprocess deadlock
3. **First epoch**: 30-60min cache build (expected)
4. **Modal**: Needs `cpu: 24` in resources or bottlenecks
5. **Patient splits**: Must use `split_policy: official_tusz`

## Files

```
src/brain_brr/
├── models/
│   ├── detector.py      # Main model, both v2/v3 paths
│   ├── tcn.py          # TCN encoder
│   ├── mamba.py        # Mamba wrappers
│   ├── gnn_pyg.py      # Vectorized GNN
│   └── edge_features.py # V3 edge stream
├── data/
│   ├── loader.py       # EDF→tensor pipeline
│   ├── dataset.py      # Balanced sampling
│   └── tusz_splits.py  # Patient-disjoint splits
└── train/
    └── loop.py         # Training orchestration

configs/
├── local/train.yaml    # RTX 4090 config
└── modal/train.yaml    # A100 config
```

## Documentation

- [INSTALLATION.md](INSTALLATION.md) - Exact dependency versions
- [ARCHITECTURE_EVOLUTION.md](ARCHITECTURE_EVOLUTION.md) - Design decisions
- [configs/README.md](configs/README.md) - All parameters explained
- [docs/](docs/) - Technical deep dives

## License

Apache 2.0