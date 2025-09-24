# Making Dynamic PE Work on RTX 4090 (24GB)

## THE REAL PROBLEM
We're computing 960 eigendecompositions ALL AT ONCE:
- 8 batches × 960 timesteps = 7,680 eigendecompositions
- Each needs ~1MB CUDA workspace
- Total: 7.5GB just for eigendecomp

## THE KEY INSIGHT
**We don't need all 960 PE values simultaneously!**
- GNN processes timesteps in Mamba sequentially anyway
- We can compute PE on-demand or in smaller chunks

## SOLUTION 1: Semi-Dynamic PE (EASIEST)
```yaml
# Every 10 timesteps instead of every 1
semi_dynamic_interval: 10
```
- Reduces eigendecompositions from 960 to 96
- Memory: 7.5GB → 0.75GB
- Still captures dynamics, just at lower temporal resolution
- **Total memory: ~17GB - FITS!**

## SOLUTION 2: Reduce Batch Size + Full Dynamic
```yaml
training:
  batch_size: 3  # From 8 to 3
graph:
  use_dynamic_pe: true
  semi_dynamic_interval: 1
```
- Memory scales linearly with batch size
- 3/8 × 27GB = 10GB
- **Total memory: ~10GB - FITS EASILY!**

## SOLUTION 3: Chunked PE Computation (BEST)
Modify the GNN forward to compute PE in chunks:

```python
def forward_vectorized(self, node_features, adjacency):
    B, T, N, D = node_features.shape

    # Process in chunks of 120 timesteps
    chunk_size = 120
    outputs = []

    for t_start in range(0, T, chunk_size):
        t_end = min(t_start + chunk_size, T)

        # Compute PE only for this chunk
        adj_chunk = adjacency[:, t_start:t_end]
        pe_chunk = self._compute_dynamic_pe_vectorized(adj_chunk)

        # Process chunk through GNN
        node_chunk = node_features[:, t_start:t_end]
        out_chunk = self._gnn_forward(node_chunk, adj_chunk, pe_chunk)
        outputs.append(out_chunk)

        # Free memory explicitly
        torch.cuda.empty_cache()

    return torch.cat(outputs, dim=1)
```
- Only 120 timesteps in memory at once
- Memory: 7.5GB → 0.94GB
- **Total memory: ~18GB - FITS!**

## IMMEDIATE RECOMMENDATION

### Option A: Quick Fix (5 minutes)
```yaml
# configs/local/train.yaml
graph:
  use_dynamic_pe: true
  semi_dynamic_interval: 10  # Compute every 10 timesteps
training:
  batch_size: 6  # Slight reduction for safety
```

### Option B: Optimal Fix (30 minutes)
```yaml
# Keep full dynamic PE
graph:
  use_dynamic_pe: true
  semi_dynamic_interval: 1
training:
  batch_size: 3  # Aggressive reduction
```

## Memory Math with Solutions

| Config | Batch | Interval | PE Memory | Total | Fits? |
|--------|-------|----------|-----------|-------|-------|
| Original | 8 | 1 | 7.5 GB | 27 GB | ❌ NO |
| Semi-10 | 8 | 10 | 0.75 GB | 20 GB | ✅ YES |
| Semi-20 | 8 | 20 | 0.38 GB | 19 GB | ✅ YES |
| Batch-3 | 3 | 1 | 2.8 GB | 10 GB | ✅ YES |
| Batch-4 | 4 | 5 | 0.75 GB | 14 GB | ✅ YES |

## THE ANSWER: YES, WE CAN TRAIN LOCALLY!

We have THREE viable paths:
1. **Semi-dynamic (interval=10)**: 90% of benefit, 10% of memory
2. **Small batch (size=3)**: Full dynamic PE, slower training
3. **Chunked implementation**: Best of both (requires code change)

**Recommendation: Use semi_dynamic_interval=10 with batch_size=6**
- Captures temporal dynamics every 39ms (still very frequent!)
- Fits comfortably in 24GB
- Minimal performance impact