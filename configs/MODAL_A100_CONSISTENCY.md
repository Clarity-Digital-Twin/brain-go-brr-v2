# Modal A100 Config Consistency (V3.6.1) ✅

Canonical stack: TCN + Dual-Stream BiMamba + GNN (PyG SSGConv + Laplacian PE)

Files
- configs/modal/smoke.yaml — 1 epoch, A100 smoke; balanced sampling; full stack
- configs/modal/train.yaml — full training on A100; balanced sampling; full stack

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
    edge_mamba_d_model: 16  # multiple of 8
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
- smoke.yaml: batch_size: 48 (same as train for consistency)
- train.yaml: batch_size: 48 (PRODUCTION: ~58GB peak verified stable)

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
  cache_dir: /results/cache/tusz  # Persistent SSD volume
experiment:
  output_dir: /results/v3_full_training  # Or /results/smoke for smoke test
  device: cuda
```

Validation UX
- Long validation is normal: ~800+ batches. We print:
  - `[VALIDATION] Starting validation …`
  - `[VAL HEARTBEAT] Batch i/N | Avg Loss: …` (every ~2 min)
  - `[VALIDATION] Completed …, computing metrics…` then final metrics.

Notes
- PyG is required; ensure graph wheels match your Torch/CUDA.
- Smoke test: app.py sets BGB_LIMIT_FILES=50 automatically
- V3 requires headdim parameters (handled in detector.py)
- Mid-epoch checkpoints critical for Modal (epochs can be 6-7 hours)
- Resume: Automatically loads newest mid_epoch_*.pt or last.pt
