# Local Config Consistency (v2.6) ✅

Canonical stack: TCN + Bi‑Mamba + Dynamic GNN (PyG SSGConv + Laplacian PE)

Files
- configs/local/smoke.yaml — 1 epoch, small batch, fast validation
- configs/local/train.yaml — full training, balanced sampling

Shared model
```yaml
model:
  architecture: tcn
  tcn: { num_layers: 8, channels: [64, 128, 256, 512], kernel_size: 7, stride_down: 16, dropout: 0.15 }
  mamba: { n_layers: 6, d_model: 512, d_state: 16, conv_kernel: 4, dropout: 0.1 }
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

Shared data
```yaml
data:
  dataset: tuh_eeg
  sampling_rate: 256
  n_channels: 19
  window_size: 60
  stride: 10
```

Smoke (safe + fast)
```yaml
epochs: 1
batch_size: 4
use_balanced_sampling: false
mixed_precision: false
```

Train (4090‑ready defaults)
```yaml
epochs: 100
batch_size: 12
use_balanced_sampling: true
mixed_precision: false  # enable after NaN‑free smoke
learning_rate: 1.5e-4
gradient_clip: 1.0
```

WSL2 note
- Prefer `num_workers: 0`, `pin_memory: false`, `persistent_workers: false` if training under WSL2.
