# PR-5: Edge Similarity Clamping at Source

## Overview

PR-5 implements edge similarity clamping at the source of computation to prevent numerical instabilities that occur when cosine similarities reach exact ±1.0 boundaries. This is a critical architectural stability improvement in v3.2.0.

## Problem Statement

### Root Cause
When computing cosine similarities between electrode features:
- Perfect correlation/anti-correlation produces exactly ±1.0
- These boundary values cause Mamba SSM explosions
- Downstream clamping was insufficient (values already corrupted)
- Led to NaN propagation through the network

### Failure Mode
```python
# Before PR-5: Downstream clamping (too late!)
edge_feats = compute_cosine_similarity(x)  # Can be exactly ±1.0
edge_feats = self.edge_lift(edge_feats)     # Explosion happens here
edge_feats = torch.clamp(edge_feats, -0.99, 0.99)  # Too late!
```

## Solution: Single Source of Truth (SSOT)

### Implementation
Move clamping to the source of computation with a configurable safety margin:

```python
# In edge_features.py
def edge_scalar_series(x, metric="cosine", edge_similarity_margin=0.01):
    if metric == "cosine":
        # Compute similarities
        sim = F.cosine_similarity(x_i, x_j, dim=-1)

        # Apply safety margin IMMEDIATELY at source
        max_val = 1.0 - edge_similarity_margin
        sim = torch.clamp(sim, min=-max_val, max=max_val)

    return sim
```

### Configuration
```yaml
model:
  graph:
    edge_similarity_margin: 0.01  # Safety margin from ±1 boundaries
```

## Technical Details

### Why 0.01 Margin?
- Prevents exact ±1.0 values that cause Mamba explosions
- Small enough to preserve signal fidelity (99% of range)
- Large enough for numerical stability
- Empirically validated through extensive testing

### Integration Points
1. **Edge Feature Computation** (`edge_features.py`):
   - Primary clamping location
   - Applied immediately after similarity calculation

2. **SeizureDetector** (`detector.py`):
   - Extracts margin from config
   - Type-safe parameter passing
   - Maintains backward compatibility

3. **Configuration** (all YAML files):
   - Consistent across local/modal configs
   - Default value: 0.01
   - Can be tuned if needed

## Testing

### Unit Tests
- `test_edge_similarity_margin()`: Verifies clamping behavior
- `test_edge_features_bounds()`: Ensures output bounds
- `test_pr5_integration()`: Full integration test

### Smoke Tests
```bash
# Local validation
make s

# Margin is configured in YAML (model.graph.edge_similarity_margin: 0.01)
```

## Migration Guide

### For Existing Models
1. Add to config:
```yaml
model:
  graph:
    edge_similarity_margin: 0.01
```

2. No code changes needed (backward compatible)

### For New Deployments
- All v3.2.0+ configs include this by default
- No action required

## Performance Impact

- **Memory**: No change
- **Compute**: Negligible (one clamp operation)
- **Stability**: Significant improvement
- **Accuracy**: No degradation observed

## Monitoring

### Configuration
Margin is set in YAML config files:
```yaml
model:
  graph:
    edge_similarity_margin: 0.01  # Adjust if needed (default: 0.01)
```

### Logs to Watch
Edge similarity clamping is applied silently at the source. No special logging is emitted.

## Related PRs

- **PR-1**: Boundary normalization
- **PR-2**: Bounded edge projection
- **PR-3**: Adjacency conditioning
- **PR-4**: Clamp retirement
- **PR-5**: Edge similarity clamping (this PR)

## Validation Checklist

- [x] All configs updated with `edge_similarity_margin`
- [x] Type-safe extraction in `detector.py`
- [x] Source clamping in `edge_features.py`
- [x] Unit tests passing
- [x] Smoke tests stable
- [x] No NaN explosions observed
- [x] CI/CD tests properly skip when PyG unavailable

## References

- Issue: #483 (Architectural Instability)
- Commit: d0faef9 (Initial implementation)
- Release: v3.2.0