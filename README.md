# ⚡🧠 Brain-Go-Brr V3

**31M params. O(N) complexity. Dual-stream Mamba. Dynamic graphs. Zero bullshit.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![PyTorch 2.2.2](https://img.shields.io/badge/pytorch-2.2.2-red.svg)](https://pytorch.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

## What Actually Works

✅ **V3 Dual-Stream**: 19 node Mambas + 171 edge Mambas learn in parallel
✅ **Dynamic PE**: Laplacian eigendecomposition every N timesteps (memory/accuracy tradeoff)
✅ **31M params**: TCN→BiMamba→GNN stack that actually trains
✅ **NaN-proof**: Clamped decoder, fixed focal loss, numerical safeguards everywhere
✅ **Currently training**: RTX 4090 (batch=4) and A100 (batch=64)

## The Architecture That Actually Ships

**V3 = Dual streams learning what matters:**
- Node stream: 19 electrode Mambas processing temporal dynamics
- Edge stream: 171 connection Mambas learning adjacency from scratch
- Dynamic graphs: Brain topology evolves every timestep
- Vectorized ops: All 960 timesteps processed in parallel (10× speedup)

## Real Architecture (Not Marketing)

```
EEG Input (B, 19, 15360) @ 256Hz
         ↓
[TCN Encoder]           8 layers, [64,128,256,512], stride_down=16
         ↓              Output: (B, 512, 960)
[Projection]            512 → 19×64 electrode features
         ↓
    ┌────┴────┐
[Node Mamba]  [Edge Mamba]     PARALLEL DUAL-STREAM
19× BiMamba2  171× BiMamba2    Node: (B×19, 64, 960)
    │         │                 Edge: (B×171, 16, 960)
    │    [Adjacency]           Learned per timestep
    └────┬────┘
         ↓
[Vectorized GNN]        2-layer SSGConv (α=0.05)
+ Dynamic LPE           k=16 eigenvectors, computed every N steps
         ↓              Process all 960 timesteps at once
[Back-Projection]       19×64 → 512 bottleneck
         ↓
[Decoder + Upsample]    4 stages, restore to (B, 19, 15360)
         ↓
[Detection Head]        Per-sample logits with clamping
         ↓
[Post-Processing]       Hysteresis + Morphology
```

### Actual Numbers (Measured, Not Guessed)
- **Parameters**: 31,475,722 exactly
- **RTX 4090**: 16GB VRAM @ batch_size=4, PE interval=5
- **A100**: 60GB VRAM @ batch_size=64, full dynamic PE
- **Speed**: 2-3h/epoch (4090), 1h/epoch (A100), ~$319 total
- **Data**: 4667 train files, 1832 dev files (patient-disjoint)

See [ARCHITECTURE_EVOLUTION.md](ARCHITECTURE_EVOLUTION.md) for why we built it this way.

## Get It Running

```bash
# Requirements: CUDA 12.1, 24GB+ VRAM, patience
git clone https://github.com/clarity-digital-twin/brain-go-brr-v2.git
cd brain-go-brr-v2
make setup && make setup-gpu

# Smoke test (verify nothing explodes)
make smoke

# Real training
tmux new -s train
make train-local
# Ctrl+B, D to detach
```

### Cloud (Modal)

```bash
modal setup  # One-time auth
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml
# That's it. ~100 hours, $319.
```

## Code That Matters

```
src/brain_brr/models/
├── detector_v3.py       # V3 dual-stream orchestrator
├── tcn.py              # 8-layer TCN (actually works)
├── mamba.py            # Bidirectional Mamba2 wrapper
├── gnn_pyg.py          # Vectorized GNN + dynamic LPE
├── edge_features.py    # Edge Mamba stream (171 channels)
└── laplacian_pe.py     # Eigendecomposition with NaN protection

configs/
├── local/train.yaml    # RTX 4090 settings that don't OOM
└── modal/train.yaml    # A100 settings ($3.19/hour)
```

## Performance Targets

| FA/24h | Sensitivity | Status |
|--------|------------|--------|
| 10 | >95% | Training |
| 5 | >90% | Training |
| 1 | >75% | Training |

## What's Actually Implemented

**Data**: TUSZ corpus → 256Hz → 60s windows → NPZ cache
**Training**: Focal loss, balanced sampling, AdamW, cosine schedule
**Post**: Hysteresis (0.86/0.78) + morphology (11/31 kernels)

**Critical configs that work:**
```yaml
# RTX 4090 (no NaNs)
batch_size: 4
mixed_precision: false
semi_dynamic_interval: 5

# A100 (fast)
batch_size: 64
mixed_precision: true
use_dynamic_pe: true
```

## Dev Workflow

```bash
make q      # Lint/format/type check before commit
make t      # Fast tests
make smoke  # 1-epoch sanity check
```

## Real Docs (Not README Bloat)

- [INSTALLATION.md](INSTALLATION.md) - Exact versions that work
- [ARCHITECTURE_EVOLUTION.md](ARCHITECTURE_EVOLUTION.md) - Why V3 exists
- [docs/04-model/v3-architecture.md](docs/04-model/v3-architecture.md) - Dual-stream details
- [configs/README.md](configs/README.md) - Every parameter explained
- [CLAUDE.md](CLAUDE.md) - AI pair programming setup

## Known Issues That Matter

- **RTX 4090**: Mixed precision = NaN explosion. Keep it off.
- **WSL2**: Set num_workers=0 or hang forever
- **First epoch**: 30-60min cache build. Normal.
- **Modal**: Need 24 CPU cores or data loading bottlenecks

---

**Built to prove O(N) > O(N²) for brain signals.**
