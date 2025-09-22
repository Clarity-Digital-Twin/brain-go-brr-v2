# 🔥 EVOBRAIN IMPLEMENTATION EXTRACTION

Status: reference-only. This document extracts patterns from the EvoBrain open-source code for future integration. Nothing here is implemented yet in `src/` and some items require PyTorch Geometric.

Primary differences vs our stack:
- EvoBrain evaluates early seizure prediction as well as detection; we focus on detection (per‑timestep logits), so keep graph as a feature refiner (no graph classification pooling required).
- EvoBrain default uses FFT/STFT features (`use_fft=True`); our pipeline can add STFT as a separate incremental change.
- EvoBrain Mamba uses `d_conv=4`; our CUDA path already coerces 5→4, so align configs.

## Critical Code Patterns for Brain-Go-Brr Integration

### 1. DUAL-STREAM MAMBA ARCHITECTURE

**File**: `model/EvoBrain.py` (lines 863-864, 906-928)

```python
# TWO SEPARATE MAMBA STREAMS
self.snn_edge = sequentialize("mamba", feat_input_size_edge, embed_inside_size)
self.snn_node = sequentialize("mamba", feat_input_size_node, embed_inside_size)

# Mamba config (line 812-817)
Mamba(
    d_model=feat_target_size,  # 512 in their experiments
    d_state=16,                 # SSM state expansion factor
    d_conv=4,                   # Local convolution width
    expand=2,                   # Block expansion factor
)

# Processing (lines 906-928)
# Node stream processes EEG channels
if self.reduce_node == "mamba":
    inputs = inputs.permute(1, 0, 2)  # (batch*node, timestep, dim)
    node_embeds = self.snn_node.forward(inputs)
    node_embeds = node_embeds.permute(1, 0, 2)  # back to (timestep, batch*node, dim)

# Edge stream processes adjacency features
edge_embeds, _ = self.snn_edge.forward(edge_features)
```

### 2. LAPLACIAN POSITIONAL ENCODING

**File**: `model/EvoBrain.py` (lines 857-859, 943-957)

```python
from torch_geometric.transforms import AddLaplacianEigenvectorPE

# Initialize
if num_eigenvectors > 0:
    self.laplacian_pe = AddLaplacianEigenvectorPE(k=num_eigenvectors)  # k=16

# Apply AFTER Mamba, BEFORE GNN (lines 943-957)
if self.num_eigenvectors > 0:
    data = Data(
        x=current_node_embeds[-1].detach(),
        edge_index=edge_tuples.detach(),
        edge_attr=current_edge_embeds[-1].detach(),
        edge_weight=edge_weights.detach()  # For Laplacian computation
    )
    data = self.laplacian_pe(data)

    # Concatenate PE with node features
    pe_features = data.laplacian_eigenvector_pe
    node_features_with_pe = torch.cat([
        current_node_embeds[-1],
        pe_features
    ], dim=-1)
```

### 3. DYNAMIC GRAPH CONSTRUCTION

**File**: `model/EvoBrain.py` (lines 970-981)

```python
def create_edge_tuples_and_features(self, adj):
    """
    adj shape: (batch_size, timesteps, num_nodes, num_nodes)
    Returns dynamic edges per timestep
    """
    batch_size, timesteps, num_nodes, _ = adj.shape

    # Create full connectivity pattern
    node_indices = torch.arange(num_nodes)
    edge_tuples = torch.stack(
        torch.meshgrid(node_indices, node_indices)
    ).reshape(2, -1)

    # Extract time-varying edge features
    edge_features = adj.reshape(timesteps, batch_size, -1).unsqueeze(-1)

    # Sparsify: only keep significant edges
    non_zero_indices = torch.nonzero(
        (edge_features.sum(dim=(0, 1)) > 0.0001) |
        (edge_features.sum(dim=(0, 1)) < -0.0001)
    )

    # Return sparse edge representation
    non_zero_edge_tuples = edge_tuples[:, non_zero_indices[:,0]]
    non_zero_edge_features = edge_features[:, :, non_zero_indices[:, 0], :]

    return non_zero_edge_tuples, non_zero_edge_features
```

### 4. GNN ARCHITECTURE

**File**: `model/EvoBrain.py` (lines 230-436)

```python
class GNNx2(Model):
    """2-layer GNN with skip connections"""

    def __init__(self, ...):
        # Two GNN layers
        self.gnn1 = self.graphicalize(convolve, ...)  # First layer
        self.gnn2 = self.graphicalize(convolve, ...)  # Second layer

        # Skip connection
        if feat_input_size_node == feat_target_size:
            self.skip = torch.nn.Identity()
        else:
            self.skip = torch.nn.Linear(feat_input_size_node, feat_target_size)

    def graphicalize(self, name, ...):
        """Factory for different GNN types"""
        if name == "ssg":  # Their default
            module = geo_nn.SSGConv(
                feat_input_size_node,
                feat_target_size,
                alpha=0.05
            )
        elif name == "gcn":
            module = geo_nn.GCNConv(
                feat_input_size_node,
                feat_target_size
            )
        # ... other variants

    def forward(self, edge_tuples, edge_feats, node_feats):
        # Transform edges
        edge_embeds = self.edge_activate(
            self.edge_transform.forward(edge_feats)
        )

        # Two-layer GNN with activation
        node_embeds = self.gnn1.forward(node_feats, edge_tuples, edge_embeds)
        node_embeds = self.gnn2.forward(
            self.activate(node_embeds), edge_tuples, edge_embeds
        )

        # Skip connection
        node_residuals = self.skip.forward(node_feats)
        return node_embeds + self.doskip * node_residuals
```

### 5. ADJACENCY MATRIX GENERATION

**Files**: `data/data_utils.py`, `data/dataloader_prediction.py`

```python
def comp_xcorr(x, y, fs, w, n_bands=1):
    """Compute cross-correlation for adjacency matrix"""
    # They use cross-correlation to build dynamic adjacency
    # Top-k edges kept based on correlation strength

def keep_topk(adj, top_k=3):
    """Keep only top-k edges per node (default k=3)"""
    # Sparsification strategy
```

---

## 🎯 INTEGRATION STRATEGY FOR BRAIN-GO-BRR

### ~~Phase 1: Static GNN~~ SKIP STRAIGHT TO DYNAMIC!

### Phase 2: FULL Dynamic GNN + LPE (v2.6) 🔥
```python
# Enhance with positional encoding
from torch_geometric.transforms import AddLaplacianEigenvectorPE

class GraphChannelMixerWithPE(GraphChannelMixer):
    def __init__(self, d_model=512, num_nodes=19, num_eigenvectors=16):
        super().__init__(d_model + num_eigenvectors, num_nodes)
        self.laplacian_pe = AddLaplacianEigenvectorPE(k=num_eigenvectors)

    def forward(self, x):
        # Add PE before GCN
        data = self.build_graph_data(x)
        data = self.laplacian_pe(data)

        # Concatenate PE
        x_with_pe = torch.cat([x, data.laplacian_eigenvector_pe], dim=-1)
        return super().forward(x_with_pe)
```

### Phase 3: Dynamic Graphs (v2.6+)
```python
class DynamicGraphMixer(nn.Module):
    def __init__(self, d_model=512, num_nodes=19):
        super().__init__()
        self.edge_mlp = nn.Linear(d_model * 2, 1)  # Learn edge weights
        self.gcn = SSGConv(d_model, d_model)

    def forward(self, x):
        # x: (batch, seq_len, channels, d_model)

        # Build dynamic adjacency per timestep
        for t in range(x.shape[1]):
            node_feats = x[:, t]

            # Compute pairwise similarities
            adj_t = self.compute_dynamic_adjacency(node_feats)

            # Sparsify
            edge_index, edge_weight = self.sparsify(adj_t, top_k=5)

            # Apply GCN with dynamic graph
            node_embeds = self.gcn(node_feats, edge_index, edge_weight)
```

### Phase 4: Dual-Stream Mamba (v3.2)
```python
class DualStreamMamba(nn.Module):
    def __init__(self, d_model=512, num_nodes=19):
        super().__init__()

        # Node stream - processes EEG channels
        self.node_mamba = nn.Sequential(
            nn.Linear(d_model, d_model),
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        )

        # Edge stream - processes relationships
        self.edge_mamba = nn.Sequential(
            nn.Linear(1, d_model),  # Edge features start as scalars
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        )

    def forward(self, x, adj):
        # Process nodes
        node_embeds = self.node_mamba(x)

        # Process edges
        edge_features = adj.reshape(-1, 1)  # Flatten adjacency
        edge_embeds = self.edge_mamba(edge_features)

        return node_embeds, edge_embeds
```

---

## 📊 EXPERIMENTAL PARAMETERS FROM EVOBRAIN

| Component | Configuration | Notes |
|-----------|--------------|-------|
| Mamba | d_state=16, d_conv=4, expand=2 | Both streams use same |
| GNN | 2-layer SSGConv, alpha=0.05 | With skip connections |
| Laplacian PE | k=16 eigenvectors | Concatenated to features |
| Edge pruning | \|weight\| > 0.0001 | Sparsification threshold |
| Activation | Tanh (GNN), Softplus (edges) | Different per component |
| Aggregation | Mean pooling over nodes | For final classification |
| Graph type | dynamic/individual/combined | Default dynamic with dual_random_walk filter |
| Top‑k | 3 | For correlation graphs |
| Input features | FFT/STFT | `use_fft=True` by default |

---

## 🚀 IMMEDIATE ACTIONS

1. **Create `src/brain_brr/models/graph.py`** with FULL DYNAMIC GNN + LPE (v2.6)
2. **Test integration** after current Bi-Mamba-2
3. **Benchmark** FA rate reduction on validation set
4. **If successful**, proceed to Laplacian PE (v2.6)
5. **Document** performance delta at each step

---

**Key Insight**: EvoBrain's success comes from the COMBINATION of:
- Dual-stream temporal modeling (separate node/edge dynamics)
- Explicit dynamic graphs (not static)
- Laplacian PE for stable grounding
- Time-then-graph ordering (proven optimal)

We can adopt these incrementally, testing each addition's impact.
