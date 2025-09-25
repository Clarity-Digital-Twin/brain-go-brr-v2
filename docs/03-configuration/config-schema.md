# Configuration Schema

Source of truth: `src/brain_brr/config/schemas.py` (Pydantic v2).

Top-level sections

- `data` — dataset roots, cache, batching
- `preprocessing` — filters, normalization, montage
- `model` — architecture (`tcn` or `v3`), TCN/Mamba/GNN
- `postprocessing` — hysteresis, morphology, durations, stitching
- `training` — epochs, batch size, loss, LR, scheduler
- `evaluation` — metrics and FA rates
- `experiment` — device, output dirs, W&B, logging level
- `logging` — gradient/weight logging options
- `resources` — optional compute settings

Data

- `dataset: tuh_eeg|chb_mit` (default `tuh_eeg`)
- `data_dir: path` — raw EDF root
- `cache_dir: path` — NPZ cache root
- `use_balanced_sampling: bool` — use manifest‑driven balanced dataset
- `sampling_rate: 256`, `n_channels: 19`, `window_size: 60`, `stride: 10`
- `num_workers: 0..32` (WSL2: prefer 0), `pin_memory`, `persistent_workers`, `prefetch_factor`, `validation_split: 0..0.5`
- Limits: `max_samples`, `max_hours`

Preprocessing

- `montage: "10-20"|"standard_1020"`
- `bandpass: [0.5, 120.0]` — low < high; within [0.1, 200]
- `notch_freq: 50|60`
- `normalize: true` — per‑channel z‑score
- `use_mne: true` — use MNE EDF loader

Model

- `architecture: tcn|v3`
- `tcn`: `num_layers: 8`, `kernel_size: 7`, `dropout: 0.15`, `causal: false`, `stride_down: 16`
- `mamba`: `n_layers: 6`, `d_model: 512`, `d_state: 16`, `conv_kernel: 4`, `dropout: 0.1`
- `graph` (optional): see Graph below. Required for v3.

Graph (GNN + adjacency)

- Enable: `enabled: true`
- Edge stream: `edge_features: cosine|correlation`, `edge_top_k: 3`, `edge_threshold: 1e-4`, `edge_mamba_layers: 2`, `edge_mamba_d_state: 8`, `edge_mamba_d_model: 16`
- GNN: `n_layers: 2`, `dropout: 0.1`, `use_residual: true`, `alpha: 0.05`, `k_eigenvectors: 16`
- Dynamic PE: `use_dynamic_pe: true|false` (schema default true for V3)
- Semi-dynamic update interval: `semi_dynamic_interval: 1` (1 = fully dynamic)
- Sign consistency: `pe_sign_consistency: true` (prevent random eigenvector sign flips)

Postprocessing

- `hysteresis: { tau_on: 0.86, tau_off: 0.78, min_onset_samples: 128, min_offset_samples: 256 }`
- `morphology: { opening_kernel: 11, closing_kernel: 31 }` (odd sizes)
- `duration: { min_duration_s: 3.0, max_duration_s: 600.0 }`
- `events: { tau_merge: 2.0, confidence_method: mean|peak|percentile, confidence_percentile: 0.75 }`
- `stitching: { method: overlap_add|overlap_add_weighted|max, window_size: 15360, stride: 2560 }`

Training

- `epochs`, `batch_size`
- `loss: bce|focal` with `focal_alpha` (0.5 neutral), `focal_gamma`
- `learning_rate`, `weight_decay`, `optimizer: adamw|adam|sgd`
- `scheduler: { type: cosine|linear|constant, warmup_ratio }`
- `gradient_clip`, `mixed_precision`, `resume`, `early_stopping`

Evaluation

- `metrics: [taes, sensitivity, specificity, auroc]`
- `fa_rates: [10, 5, 2.5, 1]`
- `save_predictions`, `save_plots`

Experiment / Logging / Resources

- `experiment: { name, description, seed, device: auto|cuda|cpu|mps, output_dir, cache_dir, log_level, save_model, save_best_only, wandb{…} }`
- `logging: { log_every_n_steps, log_gradients, log_weights }`
- `resources: { max_memory_gb, distributed, mixed_precision }` (optional)

Constraints and validations

- TCN channels must be `[64,128,256,512]`
- Mamba `d_model` is `512` in the bottleneck path
- Graph `edge_mamba_d_model` must be multiple of 8 (default 16)
- Hysteresis `tau_on > tau_off`
- Morphology kernels must be odd
- Duration `max >= min`

Examples

- Local training: `configs/local/train.yaml`
- Modal training: `configs/modal/train.yaml`
- Smoke tests: `configs/local/smoke.yaml`, `configs/modal/smoke.yaml`

Validation

- `python -m src validate configs/local/train.yaml`
- Optional phase checks: `--phase data|model|training`
