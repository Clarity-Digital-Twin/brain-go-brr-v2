# 🧠 v2.6 Dynamic GNN Implementation Plan

## Executive Summary
Adding Dynamic GNN + Laplacian PE after Bi-Mamba in our TCN architecture, based on EvoBrain paper.

## Current Architecture (v2.3)
```
EEG (19ch) → TCN Encoder → Bi-Mamba → Projection → TCN Decoder → Detection
```

## Target Architecture (v2.6)
```
EEG → TCN Encoder → Bi-Mamba → [Dynamic GNN + LPE] → Projection → TCN Decoder
                                        ↑
                                 INSERT HERE
```

## Implementation Requirements

### 1. Core Components from EvoBrain

#### A. Dynamic Graph Builder
**Location**: `src/brain_brr/models/graph_builder.py`
```python
class DynamicGraphBuilder:
    """Build time-evolving adjacency matrices from features"""

    def __init__(self, similarity='cosine', top_k=3, threshold=1e-4):
        # EvoBrain params: top_k=3, threshold=1e-4
        pass

    def build_adjacency(self, features: Tensor) -> Tensor:
        # Input: (B, 19, T, D) - features per electrode
        # Output: (B, T, 19, 19) - dynamic adjacency per timestep
        # 1. Compute similarity (cosine/xcorr)
        # 2. Apply top-k sparsification
        # 3. Apply threshold pruning
        pass
```

#### B. Laplacian Positional Encoding
**From EvoBrain**: Uses PyG's `AddLaplacianEigenvectorPE(k=16)`
```python
# EvoBrain code (line 857-859):
from torch_geometric.transforms import AddLaplacianEigenvectorPE
self.laplacian_pe = AddLaplacianEigenvectorPE(k=16)
```

#### C. Graph Neural Network Module
**Location**: `src/brain_brr/models/gnn.py`
```python
class GraphChannelMixer(nn.Module):
    """Dynamic GNN with Laplacian PE"""

    def __init__(self,
                 d_model=512,
                 n_electrodes=19,
                 k_eigenvectors=16,  # EvoBrain default
                 gnn_type='SSGConv',  # EvoBrain uses SSGConv
                 alpha=0.05):  # SSGConv alpha

        # From EvoBrain model/EvoBrain.py:331-348
        self.gnn = SSGConv(
            in_channels=d_model,
            out_channels=d_model,
            alpha=alpha,  # 0.05 proven best for EEG
            K=2,  # 2-hop neighborhood
        )

        # Edge transform (EvoBrain lines 272-273)
        self.edge_transform = nn.Linear(d_model, 1)
        self.edge_activate = nn.Softplus()
```

### 2. Integration Points

#### A. Detector Modification
**File**: `src/brain_brr/models/detector.py`
**Location**: Line 135 (after Mamba, before projection)

```python
# Current code (line 134-136):
features = self.tcn_encoder(x)  # (B, 512, 960)
temporal = self.mamba(features)  # (B, 512, 960)
chan19 = self.proj_512_to_19(temporal)  # (B, 19, 960)

# Modified with GNN:
features = self.tcn_encoder(x)  # (B, 512, 960)
temporal = self.mamba(features)  # (B, 512, 960)

if self.use_gnn:  # NEW
    # Project to electrode space
    electrode_features = self.proj_to_electrodes(temporal)  # (B, 19, 960, D)

    # Build dynamic graph
    adjacency = self.graph_builder(electrode_features)  # (B, 960, 19, 19)

    # Apply GNN with LPE
    electrode_features = self.gnn(electrode_features, adjacency)  # (B, 19, 960, D)

    # Project back
    temporal = self.proj_from_electrodes(electrode_features)  # (B, 512, 960)

chan19 = self.proj_512_to_19(temporal)  # Continue as normal
```

### 3. Configuration Schema

#### A. Update schemas.py
```python
@dataclass
class GraphConfig:
    """Dynamic GNN configuration"""
    enabled: bool = False  # Default OFF
    gnn_type: str = "SSGConv"  # From EvoBrain
    k_eigenvectors: int = 16  # Laplacian PE dimension
    top_k: int = 3  # Edges per node
    threshold: float = 1e-4  # Edge weight threshold
    similarity: str = "cosine"  # Edge computation
    alpha: float = 0.05  # SSGConv alpha
    edge_dim: int = 1  # Edge feature dimension
    dynamic: bool = True  # Time-evolving adjacency
    use_lpe: bool = True  # Laplacian positional encoding
```

### 4. Dependencies

#### A. Optional PyTorch Geometric
**File**: `pyproject.toml`
```toml
[project.optional-dependencies]
graph = [
    "torch-geometric>=2.4.0",
    "torch-scatter",
    "torch-sparse",
    "torch-cluster",
]
```

#### B. Import Guards
```python
# In gnn.py
try:
    import torch_geometric
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    warnings.warn("PyTorch Geometric not installed. Using pure-torch GNN fallback.")
```

### 5. Implementation Phases

#### Phase 1: Pure-Torch MVP (2-3 days)
- [ ] Implement DynamicGraphBuilder with cosine similarity
- [ ] Create basic GNN with matrix multiplication (no PyG)
- [ ] Wire into detector with feature flag
- [ ] Add unit tests for shapes

#### Phase 2: PyG Integration (2-3 days)
- [ ] Add PyG optional dependency
- [ ] Implement SSGConv from EvoBrain
- [ ] Add Laplacian PE (AddLaplacianEigenvectorPE)
- [ ] Benchmark vs pure-torch

#### Phase 3: Testing & Optimization (2-3 days)
- [ ] Integration tests with full pipeline
- [ ] Memory profiling (dynamic graphs add overhead)
- [ ] Ablation: static vs dynamic graph
- [ ] Ablation: with/without LPE

### 6. Key EvoBrain Insights

#### Critical Parameters (DO NOT CHANGE)
```python
# From EvoBrain experiments:
d_conv = 4  # Mamba CUDA constraint (we already have this)
d_state = 16  # Mamba state dimension (we already have this)
k_eigenvectors = 16  # Laplacian PE dimension
top_k = 3  # Sparse graph connectivity
alpha = 0.05  # SSGConv alpha for EEG
```

#### Architecture Decisions
1. **Time-then-Graph**: EvoBrain proves this order is optimal (Mamba → GNN, not GNN → Mamba)
2. **Dynamic Adjacency**: Must evolve per timestep (not static)
3. **Edge Pruning**: Both top-k AND threshold needed
4. **Dual-Stream**: Future work - separate node/edge Mambas

### 7. Testing Strategy

#### Unit Tests
```python
# tests/unit/models/test_gnn.py
def test_graph_builder_shapes():
    # (B=2, N=19, T=960, D=64) → (B, T, N, N)

def test_gnn_identity_init():
    # Verify skip connections preserve input

def test_laplacian_pe_dims():
    # k_eigenvectors ≤ N-1 (max 18 for 19 nodes)
```

#### Integration Tests
```python
# tests/integration/test_gnn_integration.py
def test_detector_with_gnn():
    # Full forward pass with graph enabled
    # Verify shapes match non-graph path
```

### 8. Expected Impact

Based on EvoBrain results:
- **AUROC**: +23% improvement over baseline
- **F1**: +30% improvement
- **Memory**: ~2x at bottleneck (dynamic graphs)
- **Speed**: ~1.5x slower (graph computation)

### 9. Files to Create/Modify

**New Files**:
- `src/brain_brr/models/gnn.py` - GraphChannelMixer
- `src/brain_brr/models/graph_builder.py` - DynamicGraphBuilder
- `tests/unit/models/test_gnn.py`
- `tests/integration/test_gnn_integration.py`

**Modified Files**:
- `src/brain_brr/models/detector.py` - Add GNN hook
- `src/brain_brr/config/schemas.py` - Add GraphConfig
- `pyproject.toml` - Optional graph dependencies
- `configs/modal/*.yaml` - Add graph section (disabled by default)

### 10. Command Sequence

```bash
# 1. Create feature branch
git checkout -b feature/v2.6-dynamic-gnn

# 2. Implement pure-torch MVP
# ... code implementation ...

# 3. Test without PyG
make test

# 4. Add PyG dependencies
uv sync -E graph

# 5. Test with PyG
make test-gpu

# 6. Run ablation
python -m src train configs/modal/smoke.yaml --graph.enabled=true
```

## Bottom Line

We're adding EvoBrain's Dynamic GNN after Bi-Mamba in the TCN path. Start with pure-torch, add PyG later. Keep it gated with `graph.enabled=false` by default.

**Next Step**: Implement DynamicGraphBuilder and wire it into detector.py at line 135.