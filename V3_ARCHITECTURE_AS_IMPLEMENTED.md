# V3 Architecture: As Actually Implemented

## Overview
This document describes the **ACTUAL** V3 architecture as implemented in the codebase, not the plans or proposals. This is the ground truth of what runs when `architecture: v3` is configured.

## High-Level Architecture

```
Input EEG → TCN Encoder → Dual-Stream Processing → GNN → Detection
                          ├─ Node Stream (Mamba)
                          └─ Edge Stream (Mamba)
```

## Detailed Pipeline Flow

### 1. Input Stage
- **Input**: Raw EEG data `(B, 19, 15360)`
  - B = batch size
  - 19 = electrodes (10-20 montage)
  - 15360 = 60 seconds at 256 Hz

### 2. TCN Encoder (`src/brain_brr/models/tcn.py`)
- **Module**: `TCNEncoder` with 8 layers
- **Channels**: [64, 128, 256, 512] progression
- **Kernel size**: 7
- **Stride down**: 16 (temporal compression)
- **Output**: `(B, 512, 960)` - compressed temporal features

### 3. V3 Dual-Stream Processing (`src/brain_brr/models/detector.py`)

#### 3.1 Electrode Projection
```python
proj_to_electrodes: Conv1d(512 → 19*64)
```
- **Input**: `(B, 512, 960)` from TCN
- **Output**: `(B, 19, 960, 64)` - per-electrode features

#### 3.2 Node Stream (Per-Electrode Temporal)
```python
node_mamba: BiMamba2(d_model=64, n_layers=6, d_state=16)
```
- **Reshape**: `(B, 19, 960, 64)` → `(B*19, 64, 960)`
- **Process**: Each electrode independently through bidirectional Mamba
- **Output**: `(B, 19, 960, 64)` - temporally enhanced electrode features

#### 3.3 Edge Stream (Learned Adjacency)
```python
edge_features → edge_mamba → edge_weights → adjacency
```

**Components**:
1. **Edge Feature Extraction** (`src/brain_brr/models/edge_features.py`):
   - Compute pairwise similarity: cosine or correlation
   - Input: `(B, 19, 960, 64)` electrode features
   - Output: `(B, 171, 960, 1)` - scalar edge features for 19*(19-1)/2 = 171 pairs

2. **Edge Mamba Processing**:
   ```python
   edge_in_proj: Conv1d(1 → 16)  # Learned lift for CUDA alignment
   edge_mamba: BiMamba2(d_model=16, n_layers=2, d_state=8)
   edge_out_proj: Conv1d(16 → 1)  # Project back to scalar
   edge_activate: Softplus()      # Non-negative weights
   ```
   - Input: `(B*171, 1, 960)` reshaped edge features
   - Process: Temporal evolution of edge weights
   - Output: `(B, 171, 960)` - learned edge weights

3. **Adjacency Assembly**:
   - Map 171 edges to `(B, 960, 19, 19)` adjacency matrices
   - Apply top-k sparsification (default k=3 per node)
   - Threshold pruning (default 1e-4)
   - Symmetrize and add identity fallback

### 4. Graph Neural Network (`src/brain_brr/models/gnn_pyg.py`)
```python
GraphChannelMixerPyG(use_vectorized=True, static_pe=True)
```

**Key Features**:
- **Vectorized Processing**: All 960 timesteps in one batch
- **Static Laplacian PE**: Computed once from structural 10-20 montage
- **Architecture**: 2-layer SSGConv with α=0.05 mixing
- **Input**: Node features `(B, 19, 960, 64)` + Adjacency `(B, 960, 19, 19)`
- **Output**: `(B, 19, 960, 64)` - graph-enhanced features

**Implementation Details**:
1. Flatten to `(B*960, 19, 64)` for batch processing
2. Build PyG disjoint batch with edge indices
3. Add static PE (16 eigenvectors) to first layer
4. Apply GNN with residual connections
5. Reshape back to `(B, 19, 960, 64)`

### 5. Back-Projection and Detection
```python
proj_from_electrodes: Conv1d(19*64 → 512)
proj_head: ProjectionHead(512 → 19, upsample 960 → 15360)
detection_head: Conv1d(19 → 1)
```
- **Merge electrodes**: `(B, 19, 960, 64)` → `(B, 512, 960)`
- **Upsample**: Restore original temporal resolution `(B, 19, 15360)`
- **Detection**: Final seizure probability `(B, 15360)`

## Tensor Shape Summary

| Stage | Shape | Description |
|-------|-------|-------------|
| Input | `(B, 19, 15360)` | Raw EEG |
| TCN Out | `(B, 512, 960)` | Compressed features |
| Electrode Features | `(B, 19, 960, 64)` | Per-electrode |
| Node Stream | `(B, 19, 960, 64)` | After node Mamba |
| Edge Features | `(B, 171, 960, 1)` | Pairwise similarity |
| Edge Weights | `(B, 171, 960)` | After edge Mamba |
| Adjacency | `(B, 960, 19, 19)` | Learned graphs |
| GNN Out | `(B, 19, 960, 64)` | Graph-enhanced |
| Merged | `(B, 512, 960)` | Back in TCN space |
| Upsampled | `(B, 19, 15360)` | Full resolution |
| Output | `(B, 15360)` | Detection logits |

## Key Implementation Choices

### 1. Mamba Configuration
- **Node Mamba**: Fixed at d_model=64, 6 layers
- **Edge Mamba**: d_model=16 (for CUDA alignment), 2 layers
- **Fallback**: Both use Conv1d fallback if CUDA kernels fail

### 2. Edge Processing
- **Metric**: Cosine similarity (default) or correlation
- **171 edges**: All unique pairs from 19 electrodes
- **Learned projection**: 1→16→1 to satisfy CUDA requirements

### 3. GNN Vectorization
- **Single forward pass**: Process all 960 timesteps together
- **Static PE**: Laplacian eigenvectors computed once
- **NOT EvoBrain style**: We process ALL timesteps, not just last

### 4. Safety Features
- **Assertions**: Edge d_model must be multiple of 8
- **Contiguous tensors**: Enforced before Mamba
- **Identity fallback**: Prevents disconnected nodes

## Configuration Parameters

```yaml
model:
  architecture: v3  # Activates V3 path

  graph:
    enabled: true              # Required for V3
    edge_features: cosine      # Edge metric
    edge_top_k: 3              # Sparsity
    edge_threshold: 1.0e-4     # Pruning
    edge_mamba_layers: 2       # Edge temporal layers
    edge_mamba_d_state: 8      # Edge state dimension
    edge_mamba_d_model: 16     # Must be multiple of 8

    n_layers: 2                # GNN layers
    dropout: 0.1               # GNN dropout
    use_residual: true         # Skip connections
    alpha: 0.05                # SSGConv mixing
    k_eigenvectors: 16         # Laplacian PE dimension
```

## Performance Characteristics

### Memory Usage
- **Batch size 8**: ~12-20GB VRAM on RTX 4090
- **Batch size 48**: ~40-60GB VRAM on A100

### Computational Complexity
- **TCN**: O(N) with respect to sequence length
- **Node Mamba**: O(N) per electrode, O(19*N) total
- **Edge Mamba**: O(N) per edge, O(171*N) total
- **GNN**: O(E + V) per timestep, vectorized over 960 timesteps
- **Overall**: O(N) complexity maintained

### Known Issues
1. **Mamba fallback warnings**: CUDA kernel alignment issues cause fallback to Conv1d
   - Doesn't break training, just less efficient
   - Fixed for edge stream with d_model=16

2. **Memory pressure**: V3 uses more memory than V2.6 due to dual streams
   - Reduced batch sizes recommended

## Validation Status

### Tests Passing
- ✅ `test_v3_detector_from_config` - Components initialize correctly
- ✅ `test_v3_forward_shape` - Tensor shapes correct
- ✅ `test_v3_forward_no_nan` - Numerical stability
- ✅ `test_v3_without_gnn` - Works without GNN
- ✅ `test_v3_edge_config_stored` - Configuration preserved
- ✅ `test_v2_still_works` - Backward compatibility

### Current Training
- Local full training running (100 epochs)
- No crashes or NaN issues observed
- Mamba fallback warnings present but not blocking

## Differences from Original Plans

1. **Edge Mamba d_model**: Changed from 1 to 16 for CUDA alignment
2. **GNN processing**: Vectorized over ALL timesteps, not just last
3. **Static PE**: Default, not dynamic (massive speedup)
4. **Projections**: Using Conv1d instead of Linear for efficiency

## Code Locations

| Component | File | Lines |
|-----------|------|-------|
| V3 Detector | `src/brain_brr/models/detector.py` | 169-234 |
| Edge Features | `src/brain_brr/models/edge_features.py` | 39-169 |
| GNN Vectorized | `src/brain_brr/models/gnn_pyg.py` | 122-203 |
| BiMamba2 | `src/brain_brr/models/mamba.py` | 207-243 |
| TCN Encoder | `src/brain_brr/models/tcn.py` | 178-204 |
| Config Schema | `src/brain_brr/config/schemas.py` | 171-179 |

## Summary

The V3 implementation successfully combines:
1. **TCN** for multi-scale temporal feature extraction
2. **Dual-stream Mamba** for O(N) temporal modeling of both nodes and edges
3. **Learned adjacency** via edge stream processing
4. **Vectorized GNN** for efficient graph processing
5. **Static Laplacian PE** for positional awareness

This creates a complete TCN → EvoBrain-inspired dual-stream → GNN pipeline that maintains O(N) complexity while learning both temporal and spatial dynamics for seizure detection.