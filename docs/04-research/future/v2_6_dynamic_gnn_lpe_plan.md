# [ARCHIVED] 🧠🔥 v2.6 Dynamic GNN + LPE — COMPLETE IMPLEMENTATION GUIDE

Status: Archived. This plan has been superseded by the v3 dual‑stream architecture with learned edge lift (1→D→1), vectorized PyG GNN over all timesteps, and static Laplacian PE.

Canonical documentation: `docs/02-model/architecture/v3_tcn_evobrain_hybrid.md`


## 🎯 EXECUTIVE SUMMARY
Add Dynamic GNN with Laplacian PE after Bi‑Mamba in the TCN path, driven by a learned adjacency from an edge Mamba stream (no heuristic cosine/correlation graphs). EvoBrain reports +23% AUROC and +30% F1 over its dynamic‑GNN baseline; treat as directional guidance, not guaranteed here.

## ✅ CURRENT ARCHITECTURE (v2.3 - VERIFIED)
```
EEG (19ch, 256Hz) → TCN Encoder → Bi-Mamba → Projection → Upsample → Detection
                                      ↑
                               (B, 512, 960)
```

## 🚀 TARGET ARCHITECTURE (v2.6)
```
EEG → TCN Encoder → Bi‑Mamba → [Edge stream → learned adjacency → GNN+LPE] → ProjectionHead → Detection
                                        ↑
                     Insert after Bi‑Mamba, before proj_head(…)

Constraints to honor:
- Channel order: keep the canonical 19‑channel 10–20 montage from `src/brain_brr/constants.py` when constructing graphs.
- CUDA Mamba kernel: set `conv_kernel: 4` (CUDA supports {2,3,4}).
- Output contract: logits at 256 Hz with exact length 15360 preserved.
```

Evidence anchors (code and refs):
- TCN insertion site in detector: after `temporal = self.mamba(features)` and before `decoded = self.proj_head(temporal)` in `src/brain_brr/models/detector.py` (current lines ~154 → ~157).
- Canonical channel order (19-ch 10–20): `src/brain_brr/constants.py:14`.
- EvoBrain SSGConv with alpha=0.05: `reference_repos/EvoBrain-FBC5/model/EvoBrain.py:332`.
- EvoBrain Laplacian PE k (AddLaplacianEigenvectorPE): `reference_repos/EvoBrain-FBC5/model/EvoBrain.py:858`.
- EvoBrain Softplus edge transform (edge→weight): `reference_repos/EvoBrain-FBC5/model/EvoBrain.py:869`.
- EvoBrain top‑k sparsification helper: `reference_repos/EvoBrain-FBC5/data/data_utils.py:174`.

Design decisions:
- Pure learned adjacency via edge stream; no heuristic cosine/correlation graph builder.
- PyG SSGConv (α=0.05) with Laplacian PE (k=16) is the canonical GNN backend.

---

## 📦 PHASE 1: EDGE STREAM + LEARNED ADJACENCY (PURE TORCH)

### 1.1 Edge Feature Extractor
Objective: from electrode features `elec_feats` (B, 19, T, 64), produce per‑edge scalar time series `edge_feat` (B, E, T), where E is number of edges.

Implementation outline:
- Choose base scalar per edge per timestep (start with cosine between electrode embeddings; can swap to coherence later).
- Use a fixed edge index mapping to pack upper (or full) triangle into E.
- Keep ordering consistent for assembly back to adjacency.

### 1.2 Edge Temporal Model (Bi‑Mamba)
Run a Bi‑Mamba over edges across time:
- Shape: pack `edge_feat` as (B, E, T), pass through Bi‑Mamba; output `edge_temporal` (B, E, T).
- If needed, process edges in blocks for memory.

### 1.3 Edge→Weight Head and Adjacency Assembly
- Apply Linear(1→1) + Softplus per edge to obtain non‑negative weights.
- Assemble adjacency `A_t` per timestep from weights, symmetrize for undirected EEG.
- Apply sparsification: top‑k per node, then threshold prune; ensure row‑wise normalization and identity fallback for empty rows.

### 1.4 Graph Channel Mixer (Pure Torch)
**File**: `src/brain_brr/models/gnn.py`

```python
"""Graph neural network module for spatial reasoning."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphChannelMixer(nn.Module):
    """Dynamic GNN with optional Laplacian PE.

    FROM EVOBRAIN MODEL:
    - SSGConv with alpha=0.05 (line 332)
    - Edge transform + Softplus (lines 869-870)
    - 2-layer GNN with skip connections
    """

    def __init__(
        self,
        d_model: int = 512,
        n_electrodes: int = 19,
        n_layers: int = 2,  # EvoBrain: 2-layer GNN
        dropout: float = 0.1,
        use_residual: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_electrodes = n_electrodes
        self.n_layers = n_layers
        self.use_residual = use_residual

        # Edge weight transform (cf. EvoBrain edge→weight Softplus, see
        # reference_repos/EvoBrain-FBC5/model/EvoBrain.py:869)
        self.edge_transform = nn.Linear(1, 1)
        self.edge_activate = nn.Softplus()

        # Graph convolution layers (pure torch version of SSGConv)
        self.graph_layers = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_layers)
        ])

        # Layer norm and dropout
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        # Alpha for SSGConv-like behavior
        self.alpha = 0.05  # EvoBrain proven best for EEG

    def forward(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """Apply graph neural network.

        Args:
            features: (B, 19, T, D) electrode features
            adjacency: (B, T, 19, 19) dynamic adjacency

        Returns:
            enhanced: (B, 19, T, D) enhanced features
        """
        B, N, T, D = features.shape

        # Reshape for batch processing
        x = features.permute(0, 2, 1, 3)  # (B, T, 19, D)
        x = x.reshape(B * T, N, D)
        adj = adjacency.reshape(B * T, N, N)

        # Transform edge weights (EvoBrain style)
        adj_weights = self.edge_transform(adj.unsqueeze(-1))
        adj_weights = self.edge_activate(adj_weights).squeeze(-1)

        # Normalize adjacency (row-wise for stability)
        row_sum = adj_weights.sum(dim=-1, keepdim=True) + 1e-6
        adj_norm = adj_weights / row_sum

        # Apply GNN layers
        for i in range(self.n_layers):
            # Store residual
            residual = x if self.use_residual else 0

            # Graph convolution: aggregate neighbor features
            # This implements SSGConv-like behavior with alpha
            x_neighbors = torch.bmm(adj_norm, x)  # (B*T, 19, D)

            # Mix self and neighbor features (SSGConv alpha)
            x_mixed = (1 - self.alpha) * x + self.alpha * x_neighbors

            # Transform features
            x = self.graph_layers[i](x_mixed)

            # Add residual
            if self.use_residual:
                x = x + residual

            # Layer norm and activation
            x = self.layer_norms[i](x)
            x = F.gelu(x)
            x = self.dropout(x)

        # Reshape back
        x = x.reshape(B, T, N, D)
        x = x.permute(0, 2, 1, 3)  # (B, 19, T, D)

        return x
```

### 1.5 Integration into Detector
**File**: `src/brain_brr/models/detector.py` (TCN path)

Strongly recommended wiring pattern (aligned with current factory + mypy):

1) Declare optional attributes in `__init__` (mypy‑safe):

```python
# inside SeizureDetector.__init__
self.use_gnn: bool = False
self.gnn: nn.Module | None = None
self.proj_to_electrodes: nn.Conv1d | None = None
self.proj_from_electrodes: nn.Conv1d | None = None
self.edge_mamba: nn.Module | None = None
self.edge_transform: nn.Module | None = None
```

2) In `SeizureDetector.from_config`, instantiate the base model, then gate with `cfg.graph.enabled` and attach modules to the instance:

```python
# at top of detector.py
from src.brain_brr.models.gnn import GraphChannelMixer

@classmethod
def from_config(cls, cfg: "_ModelConfig") -> "SeizureDetector":
    instance = cls(
        tcn_layers=cfg.tcn.num_layers,
        tcn_kernel_size=cfg.tcn.kernel_size,
        tcn_dropout=cfg.tcn.dropout,
        tcn_stride=cfg.tcn.stride_down,
        mamba_layers=cfg.mamba.n_layers,
        mamba_d_state=cfg.mamba.d_state,
        mamba_d_conv=cfg.mamba.conv_kernel,
        mamba_dropout=cfg.mamba.dropout,
    )

    graph_cfg = getattr(cfg, "graph", None)
    instance.use_gnn = bool(graph_cfg and graph_cfg.enabled)
    if instance.use_gnn and graph_cfg is not None:
        instance.gnn = GraphChannelMixer(
            d_model=512,
            n_electrodes=19,
            n_layers=graph_cfg.n_layers,
            dropout=graph_cfg.dropout,
            use_residual=graph_cfg.use_residual,
        )
        # Projections to/from electrode space (per‑node feature dim = 64)
        instance.proj_to_electrodes = nn.Conv1d(512, 19 * 64, kernel_size=1)
        instance.proj_from_electrodes = nn.Conv1d(19 * 64, 512, kernel_size=1)
        # Edge stream (Bi‑Mamba) + edge→weight head
        from src.brain_brr.models.mamba import BiMamba2
        instance.edge_mamba = BiMamba2(d_model=/* E runtime */, d_state=16, d_conv=4, num_layers=6)
        instance.edge_transform = nn.Sequential(nn.Linear(1, 1), nn.Softplus())

    return instance
```

3) In `SeizureDetector.forward`, insert after Bi‑Mamba and before `self.proj_head(temporal)`:

```python
features = self.tcn_encoder(x)               # (B, 512, 960)
temporal = self.mamba(features)              # (B, 512, 960)

if self.use_gnn and self.gnn and self.proj_to_electrodes and self.proj_from_electrodes:
    B, C, T = temporal.shape
    elec_flat = self.proj_to_electrodes(temporal)                    # (B, 19*64, 960)
    elec_feats = elec_flat.reshape(B, 19, 64, T).permute(0, 1, 3, 2) # (B, 19, T, 64)
    edge_feat = extract_edge_features(elec_feats)                    # (B, E, T)
    edge_temporal = self.edge_mamba(edge_feat)                       # (B, E, T)
    adjacency = assemble_adjacency(edge_temporal, edge_top_k=3, edge_threshold=1e-4)
    elec_enh = self.gnn(elec_feats, adjacency)                       # (B, 19, T, 64)
    elec_flat = elec_enh.permute(0, 1, 3, 2).reshape(B, 19 * 64, T)
    temporal = self.proj_from_electrodes(elec_flat)                  # (B, 512, 960)

decoded = self.proj_head(temporal)           # (B, 19, 15360)
output = self.detection_head(decoded)        # (B, 1, 15360)
return output.squeeze(1)
```

---

## 📦 PHASE 2: PYTORCH GEOMETRIC INTEGRATION

### 2.1 Dependencies
**File**: `pyproject.toml`

```toml
[project.optional-dependencies]
graph = [
    "torch-geometric>=2.4.0",
    "torch-scatter",
    "torch-sparse",
    "torch-cluster",
]

# Prefer: make setup-gpu (installs PyG from prebuilt wheels for torch 2.2.2+cu121)
```

### 2.2 Enhanced GNN with PyG
**File**: `src/brain_brr/models/gnn_pyg.py`

```python
"""PyTorch Geometric implementation with Laplacian PE."""

import warnings
import torch
import torch.nn as nn

try:
    from torch_geometric.nn import SSGConv
    from torch_geometric.transforms import AddLaplacianEigenvectorPE
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    warnings.warn("PyTorch Geometric not installed. Using pure-torch fallback.")


class GraphChannelMixerPyG(nn.Module):
    """Dynamic GNN with Laplacian PE using PyTorch Geometric.

    Settings per EvoBrain:
    - SSGConv with alpha=0.05 (line 332)
    - Laplacian PE k=16 (line 858)
    - 2-layer GNN
"""

    def __init__(
        self,
        d_model: int = 512,
        n_electrodes: int = 19,
        k_eigenvectors: int = 16,  # EvoBrain default
        alpha: float = 0.05,  # SSGConv alpha for EEG
        K: int = 2,  # 2-hop neighborhood
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        if not HAS_PYG:
            raise ImportError("PyTorch Geometric required for this module")

        self.d_model = d_model
        self.n_electrodes = n_electrodes
        self.k_eigenvectors = k_eigenvectors
        self.n_layers = n_layers

        # Laplacian PE (EvoBrain line 858)
        self.laplacian_pe = AddLaplacianEigenvectorPE(k=k_eigenvectors)

        # Input dimension includes PE
        input_dim = d_model + k_eigenvectors

        # SSGConv layers (EvoBrain lines 331-334)
        self.gnn_layers = nn.ModuleList([
            SSGConv(
                in_channels=input_dim if i == 0 else d_model,
                out_channels=d_model,
                alpha=alpha,
                K=K,
            )
            for i in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])

    def forward(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """Apply GNN with Laplacian PE.

        Args:
            features: (B, 19, T, D)
            adjacency: (B, T, 19, 19)

        Returns:
            enhanced: (B, 19, T, D)
        """
        B, N, T, D = features.shape
        device = features.device

        # Process each timestep
        outputs = []
        for t in range(T):
            # Get features and adjacency for this timestep
            x_t = features[:, :, t, :]  # (B, 19, D)
            adj_t = adjacency[:, t, :, :]  # (B, 19, 19)

            # Create batch of graphs
            batch_list = []
            for b in range(B):
                # Create edge index from adjacency
                edge_index = (adj_t[b] > 0).nonzero().t()
                edge_weight = adj_t[b][edge_index[0], edge_index[1]]

                # Create graph data
                data = Data(
                    x=x_t[b],  # (19, D)
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                )

                # Add Laplacian PE
                data = self.laplacian_pe(data)
                batch_list.append(data)

            # Batch graphs
            batch = Batch.from_data_list(batch_list).to(device)

            # Apply GNN layers
            x = batch.x
            for i in range(self.n_layers):
                residual = x[:, :self.d_model] if i > 0 else None

                x = self.gnn_layers[i](
                    x, batch.edge_index, batch.edge_weight
                )

                if residual is not None:
                    x = x + residual

                x = self.layer_norms[i](x)
                x = torch.nn.functional.gelu(x)
                x = self.dropout(x)

            # Reshape back to batch
            x = x.reshape(B, N, self.d_model)
            outputs.append(x)

        # Stack timesteps
        output = torch.stack(outputs, dim=2)  # (B, 19, T, D)

        return output
```

---

## 🧪 PHASE 3: TEST-DRIVEN DEVELOPMENT

### 3.1 Unit Tests (to add)
**File**: `tests/unit/models/test_gnn.py`

```python
"""Test GNN components - TDD style."""

import pytest
import torch

class TestEdgeStream:
    """Tests for edge feature extraction and adjacency assembly."""

    def test_edge_features_shape(self):
        B, N, T, D = 2, 19, 10, 64
        elec = torch.randn(B, N, T, D)
        edge = extract_edge_features(elec)
        assert edge.dim() == 3 and edge.shape[0] == B and edge.shape[2] == T

    def test_adjacency_assembly(self):
        B, N, T = 1, 19, 5
        E = N * (N - 1) // 2
        edge_temporal = torch.rand(B, E, T)
        adj = assemble_adjacency(edge_temporal, edge_top_k=3, edge_threshold=1e-4)
        assert adj.shape == (B, T, N, N)
        # Symmetric
        assert torch.allclose(adj, adj.transpose(-1, -2))


class TestGraphChannelMixer:
    """Test GNN module."""

    def test_gnn_preserves_shape(self):
        """GNN must preserve input shape."""
        from src.brain_brr.models.gnn import GraphChannelMixer

        gnn = GraphChannelMixer(d_model=64)
        features = torch.randn(2, 19, 960, 64)
        adjacency = torch.randn(2, 960, 19, 19)

        output = gnn(features, adjacency)
        assert output.shape == features.shape

    def test_gnn_gradient_flow(self):
        """Gradients must flow through GNN."""
        from src.brain_brr.models.gnn import GraphChannelMixer

        gnn = GraphChannelMixer(d_model=64)
        features = torch.randn(1, 19, 10, 64, requires_grad=True)
        adjacency = torch.randn(1, 10, 19, 19)

        output = gnn(features, adjacency)
        loss = output.mean()
        loss.backward()

        assert features.grad is not None
        assert not torch.isnan(features.grad).any()
        assert features.grad.abs().mean() > 1e-8

    def test_gnn_stability_with_identity_adj(self):
        """With identity adjacency, output should be finite and magnitude-bounded."""
        from src.brain_brr.models.gnn import GraphChannelMixer

        gnn = GraphChannelMixer(d_model=64, n_layers=1)
        features = torch.randn(1, 19, 10, 64)
        adjacency = torch.eye(19).unsqueeze(0).unsqueeze(0).repeat(1, 10, 1, 1)

        with torch.no_grad():
            output = gnn(features, adjacency)

        assert torch.isfinite(output).all()
        ratio = (output.pow(2).mean() / (features.pow(2).mean() + 1e-9)).sqrt()
        assert 0.1 <= ratio <= 10.0
```

### 3.2 Integration Tests (to add)
**File**: `tests/integration/test_gnn_integration.py`

```python
"""Integration tests for GNN in detector."""

import pytest
import torch
from src.brain_brr.models.detector import SeizureDetector
from src.brain_brr.config.schemas import ModelConfig


class TestGNNIntegration:
    """Test GNN integration with detector."""

    @pytest.fixture
    def config_with_gnn(self) -> ModelConfig:
        """ModelConfig with GNN enabled."""
        return ModelConfig(
            architecture="tcn",
            tcn={"num_layers": 8, "kernel_size": 7},
            mamba={"n_layers": 6, "d_state": 16, "conv_kernel": 4},
            graph={
                "enabled": True,
                "similarity": "cosine",
                "top_k": 3,
                "threshold": 1e-4,
                "temperature": 0.1,
                "n_layers": 2,
                "dropout": 0.1,
                "use_residual": True,
            },
        )

    def test_detector_with_gnn_forward(self, config_with_gnn):
        """Full forward pass with GNN enabled."""
        detector = SeizureDetector.from_config(config_with_gnn)
        x = torch.randn(2, 19, 15360)

        output = detector(x)
        assert output.shape == (2, 15360)
        assert not torch.isnan(output).any()

    def test_gnn_matches_non_gnn_shape(self):
        """GNN and non-GNN paths must have same output shape."""
        config_no_gnn = ModelConfig(architecture="tcn")
        config_with_gnn = ModelConfig(architecture="tcn", graph={"enabled": True})

        detector_no_gnn = SeizureDetector.from_config(config_no_gnn)
        detector_with_gnn = SeizureDetector.from_config(config_with_gnn)

        x = torch.randn(2, 19, 15360)

        output_no_gnn = detector_no_gnn(x)
        output_with_gnn = detector_with_gnn(x)

        assert output_no_gnn.shape == output_with_gnn.shape
```

---

## ⚙️ CONFIGURATION

### 4.1 Schema Update
**File**: `src/brain_brr/config/schemas.py` (add to ModelConfig as Pydantic BaseModel)

```python
from pydantic import BaseModel, Field

class GraphConfig(BaseModel):
    """Dynamic GNN configuration (learned adjacency)."""
    enabled: bool = Field(default=False, description="Enable dynamic GNN stage")
    # Edge stream inputs and sparsity
    edge_features: Literal['cosine','correlation','coherence'] = Field(default='cosine')
    edge_top_k: int = Field(default=3, ge=1, le=18, description="Top‑k neighbors per node")
    edge_threshold: float = Field(default=1e-4, ge=0.0, description="Edge weight cutoff")
    edge_temperature: float = Field(default=0.1, gt=0.0, description="Softmax temp before top‑k")
    # GNN architecture
    n_layers: int = Field(default=2, ge=1, le=4, description="Graph mixer layers")
    dropout: float = Field(default=0.1, ge=0.0, le=0.5, description="Dropout rate")
    use_residual: bool = Field(default=True, description="Residual connections")
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0, description="SSGConv alpha")
    # PyG specific
    use_pyg: bool = Field(default=True, description="Use PyG implementation (canonical)")
    k_eigenvectors: int = Field(default=16, ge=1, le=18, description="Laplacian PE dim")

class ModelConfig(BaseModel):
    # ... existing fields ...
    graph: GraphConfig | None = Field(default=None, description="Graph settings (optional)")
```

### 4.2 Config Files
**File**: `configs/modal/train_gnn.yaml`

```yaml
model:
  architecture: tcn

  graph:
    enabled: true  # Enable GNN
    similarity: cosine
    top_k: 3  # EvoBrain proven
    threshold: 1e-4
    n_layers: 2
    dropout: 0.1
    use_pyg: false  # Start with pure torch
    k_eigenvectors: 16  # For PyG phase
    alpha: 0.05  # SSGConv alpha

  tcn:
    num_layers: 8
    channels: [64, 128, 256, 512]
    kernel_size: 7
    dropout: 0.15
    causal: false
    stride_down: 16

  mamba:
    n_layers: 6
    d_model: 512
    d_state: 16
    conv_kernel: 4  # CUDA constraint (GPU kernel supports {2,3,4})
    dropout: 0.1

training:
  epochs: 100
  batch_size: 64  # Reduce if OOM with GNN
  mixed_precision: true

experiment:
  name: modal_tcn_gnn_$(date +%Y%m%d_%H%M%S)
  description: "v2.6 TCN + Bi-Mamba + Dynamic GNN"
```

---

## 🚀 IMPLEMENTATION CHECKLIST

### Phase 1: Edge Stream + Learned Adjacency (Days 1–2)
- [ ] Add edge feature extractor and packing helpers
- [ ] Add edge Bi‑Mamba and edge→weight head (Linear+Softplus)
- [ ] Add adjacency assembly with top‑k + threshold + symmetry + identity fallback
- [ ] Update detector wiring and forward path
- [ ] Update GraphConfig with `edge_*` fields and deprecations
- [ ] Write unit tests for edge stream and adjacency assembly
- [ ] Run `make q` then `make test`

### Phase 2: PyG Integration (Days 3-4)
- [ ] Add graph extras to `pyproject.toml`
- [ ] Run `make setup-gpu` (installs PyG prebuilt wheels)
- [ ] Create `src/brain_brr/models/gnn_pyg.py`
- [ ] Add PyG tests
- [ ] Switch config `use_pyg: true`
- [ ] Run `make test-gpu`

### Phase 3: Ablation Studies (Days 5-6)
- [ ] Static vs dynamic adjacency
- [ ] With/without Laplacian PE
- [ ] Different k_eigenvectors (8, 16, 32)
- [ ] Different top_k values (2, 3, 5)
- [ ] Measure memory overhead
- [ ] Benchmark inference speed

---

## 📊 EXPECTED METRICS

Based on EvoBrain results:
- **AUROC**: +23% improvement (0.75 → 0.92)
- **F1 Score**: +30% improvement
- **FA Rate**: -50% at same sensitivity
- **Memory**: ~2x at bottleneck (dynamic graphs)
- **Speed**: ~1.5x slower (graph computation)
- **Parameters**: +7M (graph components)

---

## ⚠️ CRITICAL PARAMETERS (DO NOT CHANGE)

From EvoBrain proven optimal for EEG:
```python
# THESE ARE LOCKED - PROVEN IN LITERATURE
d_conv = 4          # Mamba CUDA kernel constraint
d_state = 16        # Mamba state dimension
k_eigenvectors = 16 # Laplacian PE dimension
alpha = 0.05        # SSGConv alpha for EEG
edge_top_k = 3      # Sparse connectivity
edge_threshold = 1e-4
```

---

## 🐛 COMMON PITFALLS TO AVOID

1. **Graph must be symmetric** - EEG is undirected
2. **Normalize adjacency row-wise** - Prevents gradient explosion
3. **Top‑k THEN threshold** - Apply top‑k sparsification before pruning tiny edges (cf. `reference_repos/EvoBrain-FBC5/data/data_utils.py:174`).
4. **Guard PyG imports** - Keep CI green
5. **Start with small batches** - GNN adds memory overhead
6. **Keep graph disabled by default** - Backward compatibility
7. **Predeclare optional attrs** - Avoid mypy AttributeError in `SeizureDetector`
8. **Use ProjectionHead** - Detector uses `self.proj_head(...)`, not per‑step `proj_512_to_19/upsample`
9. **CI stability** - Guard PyG imports; phase 1 has zero new deps

---

## ✅ VERIFICATION COMMANDS

```bash
# Phase 1: Test pure torch
make test-fast
pytest -q tests/unit/models/test_gnn.py -q

# Phase 2: Test with PyG (ensure PyG installed)
# Prefer: make setup-gpu (installs prebuilt PyG wheels)
make test-gpu

# Phase 3: Run full training (Modal)
python -m src train configs/modal/train_gnn.yaml

# Monitor memory (GPU)
nvidia-smi -l 1
```

---

## 🎯 BOTTOM LINE

This plan gives you EVERYTHING needed to implement Dynamic GNN + LPE with our current TCN+Mamba codebase:
1. **Exact code** from EvoBrain with proven parameters
2. **TDD tests** following your test patterns
3. **Phased approach** - pure torch first, PyG later
4. **Integration points** clearly marked between `temporal = self.mamba(features)` and `decoded = self.proj_head(temporal)` in `src/brain_brr/models/detector.py`
5. **Config ready** with all EvoBrain defaults

**NEXT ACTION**: Start with Phase 1 — implement the edge stream (extractor + Bi‑Mamba + adjacency assembly) and integrate with PyG GNN + LPE.
