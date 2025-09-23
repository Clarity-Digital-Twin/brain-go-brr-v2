# Modal A100 Config Consistency (v2.6) ✅

Canonical stack: TCN + Bi‑Mamba + Dynamic GNN (PyG SSGConv + Laplacian PE)

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
  architecture: tcn
  tcn: { num_layers: 8, channels: [64,128,256,512], kernel_size: 7, stride_down: 16 }
  mamba: { n_layers: 6, d_model: 512, d_state: 16, conv_kernel: 4 }
  graph:
    enabled: true
    use_pyg: true
    similarity: cosine
    top_k: 3
    threshold: 1.0e-4
    temperature: 0.1
    n_layers: 2
    dropout: 0.1
    use_residual: true
    alpha: 0.05
    k_eigenvectors: 16
```

Batch sizing
- smoke.yaml: batch_size: 64 (conservative with GNN)
- train.yaml: batch_size: 64 (A100‑80GB default)

Precision & stability
- A100: `mixed_precision: true` (Tensor Cores)
- Gradient clipping: `gradient_clip: 1.0`

Paths (Modal)
```yaml
data:
  data_dir: /data/edf/train
  cache_dir: /results/cache/tusz
experiment:
  output_dir: /results/tcn_full_100ep
  device: cuda
```

Validation UX
- Long validation is normal: ~800+ batches. We print:
  - `[VALIDATION] Starting validation …`
  - `[VAL HEARTBEAT] Batch i/N | Avg Loss: …` (every ~2 min)
  - `[VALIDATION] Completed …, computing metrics…` then final metrics.

Notes
- PyG is required (`use_pyg: true`); ensure graph wheels match your Torch/CUDA.
- Removed legacy `configs/modal/train_gnn.yaml` (pre‑refactor schema).
