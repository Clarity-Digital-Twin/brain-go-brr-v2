# Dynamic Laplacian Positional Encoding Implementation Plan

## Executive Summary

**CRITICAL FINDING**: EvoBrain computes Laplacian PE **dynamically per timestep** based on the evolving adjacency matrix from the edge stream. Our current V3 uses **static PE** computed once from the structural 10-20 montage. This is the **biggest architectural gap** preventing us from capturing temporal evolution of brain network topology during seizures.

## Current Implementation (STATIC)

### Location: `src/brain_brr/models/gnn_pyg.py`

```python
class GraphChannelMixerPyG(nn.Module):
    def __init__(self, ..., use_dynamic_pe: bool = False):
        # Line 70-71: Static PE computed once at initialization
        if not use_dynamic_pe:
            self.register_buffer("static_pe", self._compute_static_pe())

    def _compute_static_pe(self) -> torch.Tensor:
        # Line 94-120: Computes PE from fixed structural adjacency
        adj = get_structural_adjacency(19)  # Fixed 10-20 montage
        data = Data(x=dummy, edge_index=edges)
        data = self.laplacian_pe(data)
        return data.laplacian_eigenvector_pe  # (19, k)

    def forward_vectorized(self, features, adjacency):
        # Line 84-91: Uses static PE for all timesteps
        pe = self.static_pe.expand(batch_size * seq_len, -1, -1)
```

## Proposed Dynamic Implementation

### Step 1: Add Configuration Flag

**File**: `src/brain_brr/config/schemas.py`
```python
class GraphConfig(BaseModel):
    # ... existing fields ...
    use_dynamic_pe: bool = False  # Default False for backward compat
    dynamic_pe_cache_size: int = 100  # Cache recent PE computations
```

### Step 2: Modify GNN Module

**File**: `src/brain_brr/models/gnn_pyg.py`

```python
class GraphChannelMixerPyG(nn.Module):
    def __init__(self, ..., use_dynamic_pe: bool = False):
        super().__init__()
        self.use_dynamic_pe = use_dynamic_pe

        # Laplacian PE transform
        self.laplacian_pe = AddLaplacianEigenvectorPE(k=k_eigenvectors)

        if not use_dynamic_pe:
            # Current static approach
            self.register_buffer("static_pe", self._compute_static_pe())
        else:
            # Dynamic PE cache to avoid recomputing identical adjacencies
            self.pe_cache = {}  # Will store hash(adj) -> pe
            self.cache_hits = 0
            self.cache_misses = 0

    def _compute_dynamic_pe_batched(
        self,
        adjacency: torch.Tensor,  # (B, 19, 19)
    ) -> torch.Tensor:  # (B, 19, k)
        """Compute Laplacian PE for a batch of adjacency matrices."""
        batch_size = adjacency.shape[0]
        device = adjacency.device
        pe_list = []

        for b in range(batch_size):
            adj_b = adjacency[b]  # (19, 19)

            # Create hash for caching (optional optimization)
            adj_key = hash(adj_b.cpu().numpy().tobytes())

            if hasattr(self, 'pe_cache') and adj_key in self.pe_cache:
                # Cache hit
                pe_b = self.pe_cache[adj_key].to(device)
                self.cache_hits += 1
            else:
                # Compute PE for this adjacency
                self.cache_misses += 1

                # Extract edges from adjacency
                edge_indices = (adj_b > 0).nonzero(as_tuple=False)

                if len(edge_indices) == 0:
                    # Disconnected graph - use zeros
                    pe_b = torch.zeros(19, self.k_eigenvectors, device=device)
                else:
                    edge_index = edge_indices.t()  # (2, E)
                    edge_weight = adj_b[edge_indices[:, 0], edge_indices[:, 1]]

                    # Create PyG data object
                    data = Data(
                        x=torch.randn(19, 1, device=device),  # Dummy features
                        edge_index=edge_index,
                        edge_weight=edge_weight
                    )

                    # Compute Laplacian PE
                    with torch.no_grad():  # PE computation doesn't need gradients
                        data = self.laplacian_pe(data)

                    if hasattr(data, 'laplacian_eigenvector_pe'):
                        pe_b = data.laplacian_eigenvector_pe
                    else:
                        # Fallback if PE fails
                        pe_b = torch.zeros(19, self.k_eigenvectors, device=device)

                # Cache the result (limit cache size)
                if hasattr(self, 'pe_cache'):
                    if len(self.pe_cache) > 100:  # Limit cache size
                        # Remove oldest entries (simple FIFO)
                        self.pe_cache = {}
                    self.pe_cache[adj_key] = pe_b.cpu()

            pe_list.append(pe_b)

        return torch.stack(pe_list, dim=0)  # (B, 19, k)

    def forward_vectorized(self, features, adjacency):
        """Process with dynamic or static PE."""
        batch_size, n_nodes, seq_len, feat_dim = features.shape
        device = features.device

        # Flatten for batch processing
        x = features.permute(0, 2, 1, 3).reshape(-1, n_nodes, feat_dim)
        adj = adjacency.reshape(-1, n_nodes, n_nodes)

        # ... build edge lists (same as current) ...

        # Compute PE
        if self.use_dynamic_pe:
            # DYNAMIC: Compute PE per timestep based on learned adjacency
            pe_all = []

            for t in range(seq_len):
                # Get adjacency for this timestep across batch
                adj_t = adjacency[:, t, :, :]  # (B, 19, 19)

                # Compute PE for this timestep
                pe_t = self._compute_dynamic_pe_batched(adj_t)  # (B, 19, k)
                pe_all.append(pe_t)

            # Stack and reshape
            pe = torch.stack(pe_all, dim=1)  # (B, T, 19, k)
            pe_flat = pe.reshape(-1, self.k_eigenvectors)  # (B*T*19, k)

            # Log cache statistics periodically
            if self.training and random.random() < 0.001:  # Log 0.1% of the time
                print(f"[Dynamic PE] Cache hits: {self.cache_hits}, misses: {self.cache_misses}")
        else:
            # STATIC: Use precomputed PE (current approach)
            pe = self.static_pe.unsqueeze(0).expand(batch_size * seq_len, -1, -1)
            pe_flat = pe.reshape(-1, self.k_eigenvectors)

        # Concatenate features with PE
        x_with_pe = torch.cat([x.reshape(-1, feat_dim), pe_flat], dim=-1)

        # ... rest of GNN processing (same as current) ...
```

### Step 3: Update Detector to Pass Dynamic Flag

**File**: `src/brain_brr/models/detector.py`

```python
@classmethod
def from_config(cls, cfg: "_ModelConfig") -> "SeizureDetector":
    # ... existing code ...

    if instance.use_gnn and graph_cfg is not None:
        # Line 369-382: Pass dynamic PE flag
        is_v3 = cfg.architecture == "v3"
        use_dynamic_pe = graph_cfg.use_dynamic_pe if hasattr(graph_cfg, 'use_dynamic_pe') else False

        instance.gnn = GraphChannelMixerPyG(
            d_model=64,
            n_electrodes=19,
            k_eigenvectors=graph_cfg.k_eigenvectors,
            alpha=graph_cfg.alpha,
            k_hops=2,
            n_layers=graph_cfg.n_layers,
            dropout=graph_cfg.dropout,
            use_residual=graph_cfg.use_residual,
            use_vectorized=is_v3,
            use_dynamic_pe=use_dynamic_pe,  # NEW: Pass dynamic flag
            bypass_edge_transform=is_v3,
        )
```

### Step 4: Update Configuration Files

**File**: `configs/local/train.yaml`
```yaml
model:
  graph:
    # ... existing fields ...
    use_dynamic_pe: false  # Start with false for testing
    # use_dynamic_pe: true  # Enable after validation
```

**File**: `configs/modal/train.yaml`
```yaml
model:
  graph:
    # ... existing fields ...
    use_dynamic_pe: true  # Modal has more compute
```

## Performance Considerations

### Computational Cost
- **Static PE**: Computed once at initialization, O(1) during forward
- **Dynamic PE**: Computed per timestep, O(T) during forward
- **For 960 timesteps**: ~960x slower PE computation

### Memory Cost
- **Static PE**: 19 × k × 4 bytes (single buffer)
- **Dynamic PE**: B × T × 19 × k × 4 bytes (per-batch storage)
- **For B=8, T=960, k=16**: ~4.7 MB additional memory

### Optimization Strategies

1. **Caching**: Cache PE computations for identical adjacencies (included above)
2. **Approximation**: Use lower k (8 instead of 16) for dynamic PE
3. **Hybrid**: Use dynamic PE only for edge-heavy timesteps
4. **Parallel**: Compute PE in parallel across timesteps (GPU)

## Testing Plan

### Phase 1: Validation (No Training)
```python
# Test script: test_dynamic_pe.py
import torch
from src.brain_brr.models.gnn_pyg import GraphChannelMixerPyG

# Create module with dynamic PE
gnn_dynamic = GraphChannelMixerPyG(use_dynamic_pe=True)
gnn_static = GraphChannelMixerPyG(use_dynamic_pe=False)

# Random inputs
features = torch.randn(2, 19, 960, 64)
adjacency = torch.rand(2, 960, 19, 19)
adjacency = (adjacency > 0.7).float()  # Sparsify

# Forward pass
out_dynamic = gnn_dynamic(features, adjacency)
out_static = gnn_static(features, adjacency)

print(f"Dynamic output shape: {out_dynamic.shape}")
print(f"Static output shape: {out_static.shape}")
print(f"Outputs differ: {not torch.allclose(out_dynamic, out_static)}")
```

### Phase 2: Smoke Test
```bash
# Test with dynamic PE on smoke dataset
BGB_LIMIT_FILES=3 python -m src train configs/local/smoke.yaml \
    --model.graph.use_dynamic_pe true
```

### Phase 3: A/B Testing
```bash
# Train two models in parallel
# Model A: Static PE (current)
python -m src train configs/local/train.yaml --experiment.name static_pe

# Model B: Dynamic PE (proposed)
python -m src train configs/local/train.yaml \
    --model.graph.use_dynamic_pe true \
    --experiment.name dynamic_pe
```

## Rollback Plan

If dynamic PE causes issues:

1. **Immediate**: Set `use_dynamic_pe: false` in configs
2. **Code**: Dynamic PE code paths are fully gated by the flag
3. **Models**: Checkpoints include the PE mode in metadata

## Expected Impact

### Positive
- **Better expressivity**: Captures evolving brain network topology
- **Matches EvoBrain**: Proven architecture from NeurIPS paper
- **Theoretically sound**: PE should reflect current connectivity

### Negative
- **Slower training**: ~2-3x overall slowdown expected
- **More memory**: Additional B×T×19×k storage
- **Numerical stability**: More PE computations = more potential for issues

## Migration Timeline

1. **Week 1**: Implement and test locally
2. **Week 2**: Run A/B experiments on Modal
3. **Week 3**: Analyze results and decide on default
4. **Week 4**: Update all configs and documentation

## Key Implementation Notes

1. **Gradient Flow**: PE computation uses `torch.no_grad()` - no gradients through eigendecomposition
2. **Numerical Stability**: Handle disconnected graphs (return zero PE)
3. **Caching**: Critical for performance - many adjacencies repeat
4. **Backward Compatibility**: Flag defaults to False, preserving current behavior

## Questions for Team Review

1. Should we make dynamic PE the default for V3?
2. What cache size is optimal for PE computations?
3. Should we add a "semi-dynamic" mode that updates PE every N timesteps?
4. Can we use a differentiable PE computation for end-to-end learning?

---

**RECOMMENDATION**: Implement with flag defaulting to False, run A/B tests on Modal, then make informed decision based on empirical results.