# 🚨 CRITICAL GNN IMPLEMENTATION DISCREPANCIES - FIX PLAN

## ⚠️ VERDICT: PARTIAL IMPLEMENTATION - MISSING CRITICAL COMPONENTS

**YES**, our GNN implementation is **INCOMPLETE** compared to EvoBrain. We have:
- ✅ SSGConv with alpha=0.05
- ✅ Laplacian PE with k=16
- ❌ **MISSING: Mamba for EDGE stream processing**
- ❌ **MISSING: Time-evolving edge embeddings**
- ❌ **MISSING: Proper sequential-then-graph architecture**

## 🔴 CRITICAL MISSING: MAMBA FOR EDGE STREAM

EvoBrain uses **TWO MAMBA STREAMS** (lines 1010-1011):
```python
reduce_edge="mamba",  # EDGE features processed by Mamba
reduce_node="mamba",  # NODE features processed by Mamba
```

**OUR IMPLEMENTATION**: Only uses Mamba for node features, then builds static adjacency per timestep.

**EVOBRAIN**:
1. Edge features go through Mamba to learn temporal edge dynamics
2. Node features go through Mamba to learn temporal node dynamics
3. BOTH streams feed into GNN

## 🔴 MISSING: SEQUENTIAL NEURAL NETWORK (SNN) MODULE

EvoBrain has dedicated `sequentialize()` function that creates:
- Mamba/MinGRU/LSTM/GRU for temporal processing
- Separate SNNs for edges AND nodes (lines 863-864):
```python
self.snn_edge = sequentialize(reduce_edge, feat_input_size_edge, embed_inside_size)
self.snn_node = sequentialize(reduce_node, feat_input_size_node, embed_inside_size)
```

## 🔴 MISSING: EDGE EMBEDDING TRANSFORMATION

EvoBrain processes edge embeddings through:
1. SNN (Mamba) to get temporal edge embeddings (line 927)
2. Linear transformation + Softplus activation (lines 869-870, 940)
3. These become edge weights for GNN

**OUR IMPLEMENTATION**: Just computes cosine similarity - no learned edge dynamics!

## 🔴 ARCHITECTURE MISMATCH

### EvoBrain Flow:
```
1. Input features → SNN_node (Mamba) → node_embeds
2. Adjacency → SNN_edge (Mamba) → edge_embeds
3. edge_embeds → Linear → Softplus → edge_weights
4. Combine node_embeds + Laplacian PE
5. GNN(node_embeds_with_PE, edge_weights) → output
```

### Our Current Flow:
```
1. Mamba(features) → temporal
2. Build adjacency from cosine similarity (NO LEARNING!)
3. GNN(temporal, static_adjacency) → output
```

## 📋 FIX IMPLEMENTATION PLAN

### Step 1: Add Edge Stream Mamba
```python
# In detector.py from_config():
if instance.use_gnn:
    # Add edge stream Mamba
    instance.edge_mamba = BidirectionalMamba(
        d_model=1,  # Edge features are scalar
        d_state=16,
        d_conv=4,
        n_layers=cfg.mamba.n_layers,
    )
```

### Step 2: Create Edge Feature Extraction
```python
# New method in detector.py:
def extract_edge_features(self, x: torch.Tensor) -> torch.Tensor:
    """Extract time-varying edge features from EEG.

    Returns: (B, T, 19*19) edge features
    """
    # Compute cross-channel correlations/coherence
    # This becomes input to edge_mamba
```

### Step 3: Process Both Streams
```python
def forward(self, x):
    # Node stream (existing)
    node_features = self.tcn_encoder(x)
    node_temporal = self.mamba(node_features)

    # Edge stream (NEW)
    edge_features = self.extract_edge_features(x)
    edge_temporal = self.edge_mamba(edge_features)

    # Transform edges to weights
    edge_weights = self.edge_activate(self.edge_transform(edge_temporal))

    # Apply GNN with learned edge weights
    output = self.gnn(node_temporal, edge_weights)
```

### Step 4: Fix GNN to Accept Edge Weights
```python
# In gnn_pyg.py:
def forward(self, features, edge_weights):
    # Use provided edge_weights instead of computing from similarity
    # These are LEARNED weights from edge Mamba stream
```

## 🟢 LAPLACIAN PE STATUS: CORRECT

The LPE implementation is **CORRECT**:
- ✅ Uses AddLaplacianEigenvectorPE(k=16)
- ✅ Concatenates PE to node features
- ✅ Applied per timestep batch

## 🎯 PRIORITY FIXES

1. **URGENT**: Add edge stream Mamba
2. **URGENT**: Remove static adjacency builder, use learned edge weights
3. **IMPORTANT**: Proper sequential-then-graph flow
4. **NICE**: Support other SNN options (MinGRU, LSTM)

## 💀 WHY THIS MATTERS

Without edge stream Mamba, we're missing:
- **Temporal edge dynamics**: How electrode connectivity evolves over time
- **Learned relationships**: Network is learning what connections matter
- **23% AUROC gain**: This is likely WHERE the gain comes from!

The current "cosine similarity" approach is a **static heuristic**, not learned dynamics!

## 🚀 NEXT STEPS

1. Implement edge stream Mamba immediately
2. Modify GNN to use learned edge weights
3. Add edge feature extraction (cross-channel coherence)
4. Update tests for dual-stream architecture
5. Retrain with proper EvoBrain architecture

**BOTTOM LINE**: We built a fucking Toyota when we needed a Ferrari. The edge stream Mamba is CRITICAL for EvoBrain's performance gains!