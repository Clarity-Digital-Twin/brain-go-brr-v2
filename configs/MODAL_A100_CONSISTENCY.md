# Modal A100 Config Consistency (v4.0.0) ✅

Config namespace:

```
configs/modal/
  smoke_bimamba.yaml   # BiMamba2: 50 files, 1 epoch (smoke)
  train_bimamba.yaml   # BiMamba2: 100 epochs (production)
  smoke_fla.yaml       # FLA (Gated DeltaNet) smoke
  train_fla.yaml       # FLA full training
```

BiMamba2 and FLA configs are identical apart from the temporal blocks /
associated safeguards (`temporal_type*`, `gdn_*`, `edge_mamba_d_model`).

Canonical stack: TCN + Dual-Stream Temporal Blocks + Vectorized GNN
(PyG SSGConv + Dynamic Laplacian PE)

GPU‑optimized loading
```yaml
num_workers: 4                # SAFE: 8 caused overhead
pin_memory: true
persistent_workers: false     # CRITICAL: Prevents spawn delay + memory leaks
prefetch_factor: 2            # SAFE: 4/8 caused OOM
```

Shared model
```yaml
model:
  architecture: v3  # V3 dual-stream architecture
  tcn: { num_layers: 8, kernel_size: 7, stride_down: 16 }
  mamba: { n_layers: 6, d_model: 512, d_state: 16, conv_kernel: 4 }
  graph:
    enabled: true
    # PyG is required; no explicit toggle
    # V3 edge stream (learned adjacency)
    edge_features: cosine
    edge_top_k: 3
    edge_threshold: 1.0e-4
    edge_mamba_layers: 2
    edge_mamba_d_state: 8
    edge_mamba_d_model: 16  # BiMamba2 default
    edge_similarity_margin: 0.01  # v3.2.0: Safety margin from ±1 boundaries
    # GNN
    n_layers: 2
    dropout: 0.1
    use_residual: true
    alpha: 0.05
    k_eigenvectors: 16
    use_dynamic_pe: true  # Dynamic PE (recomputed per timestep)
    semi_dynamic_interval: 5  # OPTIMAL: 192 eigendecomps (matches local)
```

Batch sizing
- `smoke_bimamba.yaml` / `smoke_fla.yaml`: batch_size 48
- `train_bimamba.yaml` / `train_fla.yaml`: batch_size 48 (≈58GB VRAM peak, verified)

Precision & stability
- A100: `mixed_precision: true` (Tensor Cores, 3.8× faster)
- Gradient clipping: `gradient_clip: 0.5` (NaN protection)
- Learning rate: `8.0e-5` (batch-size scaled from 1e-4)

Mid-epoch checkpointing (train.yaml only)
```yaml
checkpoint_interval: 1
mid_checkpoint_interval_s: 1800  # Save every 30 min (CRITICAL for 6-7h epochs)
mid_epoch_keep: 3                # Keep last 3 mid-epoch snapshots
```

Paths (Modal)
```yaml
data:
  data_dir: /data/edf  # Parent dir containing train/dev/eval
  cache_dir: /results/cache/tusz_mmap  # Persistent SSD volume (NPY mmap)
experiment:
  output_dir: /results/v3_full_training  # (smoke configs use /results/smoke)
  device: cuda
```

## FLA-specific differences (`*_fla.yaml`)
```yaml
model:
  mamba:
    temporal_type: gated_deltanet
    temporal_type_node: gated_deltanet
    temporal_type_edge: gated_deltanet
    gdn_fusion_mode: sum
    gdn_allow_neg_eigval: false
    gdn_edge_num_heads: 3
    gdn_edge_headdim: 8
graph:
  edge_mamba_d_model: 32        # Required for FLA causal_conv1d kernels
```
All other hyperparameters remain identical to the BiMamba2 configs to enable direct
apples-to-apples comparisons.

Validation UX
- Long validation is normal: ~800+ batches. We print:
  - `[VALIDATION] Starting validation …`
  - `[VAL HEARTBEAT] Batch i/N | Avg Loss: …` (every ~2 min)
  - `[VALIDATION] Completed …, computing metrics…` then final metrics.

Notes
- PyG is required; ensure graph wheels match your Torch/CUDA.
- Smoke test: `deploy/modal/app.py` sets `BGB_LIMIT_FILES=50` automatically
- V3 requires headdim parameters (handled in detector.py)
- Mid-epoch checkpoints critical for Modal (epochs can be 6-7 hours)
- Resume: Automatically loads newest `mid_epoch_*.pt`, `timeout_exit.pt`, then `last.pt`
