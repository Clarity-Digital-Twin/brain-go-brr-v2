# Configuration Schema

See `src/brain_brr/config/schemas.py` for the Pydantic definitions.

Highlights

- Model `architecture: v3` activates dual-stream and GNN path
- Graph params: `edge_features`, `edge_top_k`, `edge_threshold`, `n_layers`, `alpha`, `k_eigenvectors`
- Training: `batch_size`, `mixed_precision`, `use_balanced_sampling`

Examples

- Local: `configs/local/train.yaml`
- Modal: `configs/modal/train.yaml`
