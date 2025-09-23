# 🚀 V3.0 FULL TCN-EVOBRAIN HYBRID MIGRATION PLAN

## Executive Decision
**SKIP INCREMENTAL. GO FULL HYBRID.** Every half-implementation breaks. The edge stream isn't optional - it's integral to making GNN work properly.

## Target Architecture (TCN + Full EvoBrain)

```
Raw EEG (B, T, 19)
    ↓
TCN Encoder [KEEP - IT WORKS]
- 8 layers, channels [64,128,256,512]
- Stride-down factor: 16
- Output: (B, 60, 512)
    ↓
Project to Electrodes
- Linear: 512 → 19×64
- Reshape: (B, 60, 19, 64)
    ↓
═══════════════════════════════════
    FULL EVOBRAIN BACKEND
═══════════════════════════════════
    ↓
Compute Edge Features
- Cross-correlation per electrode pair
- Output: (B, 171, 60, 1)  # 19×18/2 edges
    ↓
    ├─────────────────┬─────────────────┐
    ↓                 ↓
Bi-Mamba Node     Bi-Mamba Edge
(B,19,60,64)      (B,171,60,1)
    ↓                 ↓
(B,19,60,512)     (B,171,60,64)
    ↓                 ↓
    └─────────────────┴─────────────────┘
                 ↓
         Build Adjacency
         - Edge weights → 19×19 matrix
         - Top-k=3, threshold=1e-4
                 ↓
         GNN Processing
         - SSGConv (α=0.05)
         - Laplacian PE (k=16)
         - Apply to LAST timestep only
                 ↓
         Final Projection
         - (B, 19, 512) → (B, 19, 2)
```

## Why This Will Work

### 1. **Proven Components**
- TCN: Already working, replaced U-Net successfully
- Bi-Mamba: Proven in both EvoBrain and our tests
- GNN: Works when properly fed (EvoBrain proves this)

### 2. **Complete System**
- Node stream alone = incomplete context
- Node + Edge streams = full spatiotemporal modeling
- Edge stream LEARNS which connections matter when

### 3. **No More Frankenstein**
- Stop mixing incomplete implementations
- Use EvoBrain's proven dual-stream design
- Only modification: TCN frontend instead of raw input

## Implementation Order (Clean Room)

### Step 1: Create New Architecture File
```python
# src/brain_brr/models/detector_v3.py
class SeizureDetectorV3(nn.Module):
    """TCN + Full EvoBrain dual-stream architecture"""

    def __init__(self):
        # Frontend (ours)
        self.tcn_encoder = TCNEncoder(...)  # KEEP AS IS

        # EvoBrain backend
        self.node_mamba = BiMamba2(d_model=64, n_layers=6)
        self.edge_mamba = BiMamba2(d_model=1, n_layers=2)
        self.gnn = SSGConv(alpha=0.05, K=2)
        self.laplacian_pe = AddLaplacianEigenvectorPE(k=16)
```

### Step 2: Edge Feature Extraction
```python
# src/brain_brr/models/edge_features.py
def compute_cross_correlation_features(x):
    """
    Compute edge features via cross-correlation
    Following EvoBrain exactly (data_utils.py:298-301)
    """
    B, T, N, D = x.shape
    E = N * (N - 1) // 2  # Number of edges

    edge_features = torch.zeros(B, E, T, 1)
    edge_idx = 0

    for i in range(N):
        for j in range(i + 1, N):
            # Cross-correlation between electrode pairs
            xcorr = F.conv1d(
                x[:, :, i].unsqueeze(1),
                x[:, :, j].unsqueeze(1).flip(-1),
                padding='same'
            )
            edge_features[:, edge_idx, :, 0] = xcorr.squeeze(1)
            edge_idx += 1

    return edge_features
```

### Step 3: Dual Stream Processing
```python
def forward(self, x):
    # TCN encoding
    x_tcn = self.tcn_encoder(x)  # (B, 60, 512)

    # Project to electrode space
    x_elec = self.to_electrodes(x_tcn)  # (B, 60, 19, 64)
    x_elec = x_elec.permute(0, 2, 1, 3)  # (B, 19, 60, 64)

    # PARALLEL STREAMS
    # Node stream
    node_embeds = self.node_mamba(x_elec)  # (B, 19, 60, 512)

    # Edge stream
    edge_feats = compute_cross_correlation_features(x_elec)  # (B, 171, 60, 1)
    edge_embeds = self.edge_mamba(edge_feats)  # (B, 171, 60, 64)

    # Build learned adjacency from edge embeddings
    adjacency = self.edges_to_adjacency(edge_embeds)  # (B, 60, 19, 19)

    # GNN on LAST timestep only (following EvoBrain)
    x_last = node_embeds[:, :, -1]  # (B, 19, 512)
    adj_last = adjacency[:, -1]  # (B, 19, 19)

    # Compute PE once
    pe = self.compute_laplacian_pe(adj_last)  # (B, 19, 16)
    x_with_pe = torch.cat([x_last, pe], dim=-1)

    # Apply GNN
    out = self.gnn(x_with_pe, adj_last)  # (B, 19, 512)

    # Final classification
    return self.classifier(out)  # (B, 19, 2)
```

### Step 4: Proper Batching (Following EvoBrain)
```python
# Process each batch item separately for GNN
outputs = []
for b in range(B):
    x_b = node_embeds[b]  # (19, 60, 512)
    adj_b = adjacency[b]  # (60, 19, 19)

    # Last timestep
    x_last = x_b[:, -1]  # (19, 512)
    adj_last = adj_b[-1]  # (19, 19)

    # GNN forward
    out_b = self.gnn(x_last, adj_last)
    outputs.append(out_b)

return torch.stack(outputs)  # (B, 19, 512)
```

## Configuration (v3.0)

```yaml
# configs/local/train_v3.yaml
model:
  architecture: "tcn_evobrain_hybrid"

  tcn:
    layers: 8
    channels: [64, 128, 256, 512]
    kernel_size: 7
    stride_down: 16

  node_stream:
    type: "mamba"
    d_model: 512
    d_state: 16
    d_conv: 4
    n_layers: 6

  edge_stream:
    type: "mamba"
    d_model: 64
    d_state: 8
    d_conv: 4
    n_layers: 2

  gnn:
    type: "ssgconv"
    alpha: 0.05
    K: 2
    hidden_dim: 256

  laplacian_pe:
    k_eigenvectors: 16

training:
  learning_rate: 5e-5  # Conservative
  gradient_clip: 0.5
  mixed_precision: false  # Until stable
```

## Testing Strategy

### 1. Component Tests
```python
# tests/unit/models/test_v3_components.py

def test_edge_feature_extraction():
    """Verify cross-correlation computation"""
    x = torch.randn(2, 19, 60, 64)
    edge_feats = compute_cross_correlation_features(x)
    assert edge_feats.shape == (2, 171, 60, 1)
    assert torch.isfinite(edge_feats).all()

def test_dual_stream_shapes():
    """Verify parallel stream outputs"""
    model = SeizureDetectorV3()
    x = torch.randn(2, 960, 19)
    out = model(x)
    assert out.shape == (2, 19, 2)
```

### 2. Integration Test
```python
def test_full_v3_forward():
    """End-to-end v3 forward pass"""
    model = SeizureDetectorV3.from_config(config)
    x = torch.randn(4, 960, 19)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (4, 19, 2)
    assert torch.isfinite(out).all()
```

### 3. Performance Test
```python
@pytest.mark.gpu
def test_v3_performance():
    """V3 must be <5s/batch"""
    model = SeizureDetectorV3().cuda()
    x = torch.randn(12, 960, 19).cuda()

    # Warmup
    _ = model(x)
    torch.cuda.synchronize()

    # Time
    start = time.perf_counter()
    _ = model(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    assert elapsed < 5.0, f"Too slow: {elapsed:.1f}s"
```

## Migration Timeline

### Day 1: Clean Implementation
- Morning: Implement detector_v3.py with dual streams
- Afternoon: Add edge feature extraction
- Evening: Wire up GNN with proper PE

### Day 2: Testing & Validation
- Morning: Unit tests for all components
- Afternoon: Integration testing
- Evening: Performance profiling

### Day 3: Training
- Launch v3 training with conservative hyperparams
- Monitor for NaN issues
- Compare with v2.3 baseline

## Expected Outcomes

### Performance
- **Training speed**: 3-5s/batch (vs 30-40s broken v2.6)
- **GPU utilization**: >80% (vs 43% current)
- **Memory usage**: ~18GB (dual streams add overhead)

### Accuracy (Based on EvoBrain)
- **Sensitivity**: 95%+ at 10 FA/24h
- **Improvement over v2.3**: +10-15% from edge stream
- **Stability**: No NaN explosions with proper batching

## Risk Mitigation

### If V3 Fails
1. Fall back to v2.3 (TCN + Bi-Mamba node only)
2. Disable GNN until properly debugged
3. Ship what works, iterate later

### If Partially Works
1. Keep TCN + dual Mamba streams
2. Disable GNN temporarily
3. Still better than current broken state

## The Bottom Line

**Stop half-implementations. Go full EvoBrain backend.**

The edge stream isn't optional - it provides the learned adjacency that makes GNN meaningful. Without it, we're feeding garbage graphs to GNN and wondering why it's slow and ineffective.

**TCN + Complete EvoBrain = The Way Forward**