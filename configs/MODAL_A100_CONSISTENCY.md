# Modal A100 Config Consistency (V3) ✅

Canonical stack: TCN + Dual-Stream BiMamba + GNN (PyG SSGConv + Laplacian PE)

Files
- configs/modal/smoke.yaml — 1 epoch, A100 smoke; balanced sampling; full stack
- configs/modal/train.yaml — full training on A100; balanced sampling; full stack

GPU‑optimized loading
```yaml
num_workers: 8
pin_memory: true
persistent_workers: true
prefetch_factor: 4
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
    semi_dynamic_interval: 1  # Full dynamic on A100
```

Batch sizing
- smoke.yaml: batch_size: 32 (reduced for V3 dual-stream)
- train.yaml: batch_size: 64 (optimized for A100-80GB with V3)

Precision & stability
- A100: `mixed_precision: true` (Tensor Cores)
- Gradient clipping: `gradient_clip: 0.5` (stronger for NaN protection)

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
- Removed legacy `configs/modal/train_gnn.yaml` (pre‑refactor schema).
- Smoke test: app.py sets BGB_LIMIT_FILES=50 automatically
- V3 requires headdim parameters (handled in detector.py)
