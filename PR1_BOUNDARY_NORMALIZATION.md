# PR-1: Boundary Normalization Implementation

## Files to Modify

### 1. Create new normalization module: `src/brain_brr/models/norms.py`

```python
"""Normalization layers for V3 architectural stability."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    From "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019).
    More stable than LayerNorm for our use case.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS norm
        norm = x.norm(2, dim=-1, keepdim=True)
        rms = norm * (1.0 / math.sqrt(x.size(-1)))
        x = x / (rms + self.eps)
        return self.scale * x


class LayerScale(nn.Module):
    """Learnable scaling of residual branches.

    From "Going Deeper with Image Transformers" (Touvron et al., 2021).
    Helps prevent feature collapse in deep networks.
    """

    def __init__(self, dim: int, init_value: float = 0.1):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x


def create_norm_layer(
    norm_type: str,
    dim: int,
    eps: float = 1e-5
) -> Optional[nn.Module]:
    """Factory function for normalization layers."""
    if norm_type == "rmsnorm":
        return RMSNorm(dim, eps)
    elif norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    elif norm_type == "none" or norm_type is None:
        return None
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")
```

### 2. Update `src/brain_brr/models/detector.py`

Add these imports:
```python
from .norms import RMSNorm, LayerScale, create_norm_layer
```

Add normalization layers in `__init__`:
```python
# After line ~120 (after self.proj_to_electrodes)
# Boundary norms configuration
norm_config = self.config.get("norms", {})
self.boundary_norm_type = norm_config.get("boundary_norm", "none")
self.boundary_eps = norm_config.get("boundary_eps", 1e-5)
self.layerscale_alpha = norm_config.get("layerscale_alpha", 0.1)

# Create boundary normalization layers
if self.boundary_norm_type != "none":
    # TCN → Streams boundary
    self.norm_after_tcn_proj = create_norm_layer(
        self.boundary_norm_type, 64, self.boundary_eps
    )

    # Node Mamba → GNN boundary
    self.norm_after_node_mamba = create_norm_layer(
        self.boundary_norm_type, 64, self.boundary_eps
    )

    # Edge Mamba → Adjacency boundary
    self.norm_after_edge_mamba = create_norm_layer(
        self.boundary_norm_type, 1, self.boundary_eps  # Edge is 1D after projection
    )

    # GNN → Decoder boundary
    self.norm_after_gnn = create_norm_layer(
        self.boundary_norm_type, 64, self.boundary_eps
    )

    # Before decoder head
    self.norm_before_decoder = create_norm_layer(
        self.boundary_norm_type, 512, self.boundary_eps
    )

    # LayerScale for residuals
    if self.layerscale_alpha > 0:
        self.layerscale_node = LayerScale(64, self.layerscale_alpha)
        self.layerscale_gnn = LayerScale(64, self.layerscale_alpha)
else:
    self.norm_after_tcn_proj = None
    self.norm_after_node_mamba = None
    self.norm_after_edge_mamba = None
    self.norm_after_gnn = None
    self.norm_before_decoder = None
    self.layerscale_node = None
    self.layerscale_gnn = None
```

Update forward pass to add norms at boundaries:
```python
# After line ~240 (after proj_to_electrodes)
if self.norm_after_tcn_proj is not None:
    electrode_feats = self.norm_after_tcn_proj(electrode_feats)

# After line ~265 (after node_mamba)
if self.norm_after_node_mamba is not None:
    node_out = self.norm_after_node_mamba(node_out)
    if self.layerscale_node is not None:
        node_out = electrode_feats + self.layerscale_node(node_out - electrode_feats)

# After line ~280 (after edge_mamba output)
if self.norm_after_edge_mamba is not None:
    edge_weights = self.norm_after_edge_mamba(edge_weights)

# After line ~320 (after GNN)
if self.norm_after_gnn is not None:
    spatial = self.norm_after_gnn(spatial)
    if self.layerscale_gnn is not None:
        spatial = node_out + self.layerscale_gnn(spatial - node_out)

# Before decoder (line ~295)
if self.norm_before_decoder is not None:
    merged = self.norm_before_decoder(merged)
```

### 3. Update config schema: `src/brain_brr/config/model.py`

Add to ModelConfig:
```python
from typing import Optional

class NormConfig(BaseModel):
    """Normalization configuration for architectural stability."""
    boundary_norm: str = Field(
        "none",
        description="Type of norm at component boundaries: rmsnorm|layernorm|none"
    )
    boundary_eps: float = Field(
        1e-5,
        description="Epsilon for normalization layers"
    )
    layerscale_alpha: float = Field(
        0.1,
        description="Initial value for LayerScale (0 to disable)"
    )

class ModelConfig(BaseModel):
    # ... existing fields ...

    norms: Optional[NormConfig] = Field(
        default_factory=NormConfig,
        description="Normalization configuration"
    )
```

### 4. Update configs to enable (start with OFF for safety)

`configs/local/train.yaml`:
```yaml
model:
  # ... existing config ...
  norms:
    boundary_norm: "none"  # Start with none, enable after testing
    boundary_eps: 1.0e-5
    layerscale_alpha: 0.0  # Start at 0, increase to 0.1 after testing
```

`configs/local/smoke.yaml`:
```yaml
model:
  # ... existing config ...
  norms:
    boundary_norm: "rmsnorm"  # Test in smoke first!
    boundary_eps: 1.0e-5
    layerscale_alpha: 0.1
```

## Testing Plan

### 1. Unit test for norms: `tests/unit/models/test_norms.py`

```python
"""Test normalization layers for stability."""

import torch
import pytest
from src.brain_brr.models.norms import RMSNorm, LayerScale, create_norm_layer


class TestRMSNorm:
    def test_rmsnorm_stability(self):
        """Verify RMSNorm prevents explosion."""
        norm = RMSNorm(64)

        # Test with large input
        x = torch.randn(32, 19, 960, 64) * 100
        y = norm(x)

        assert torch.isfinite(y).all()
        assert y.std() < 10  # Bounded variance

    def test_rmsnorm_gradient_flow(self):
        """Verify gradients flow properly."""
        norm = RMSNorm(64)
        x = torch.randn(8, 64, requires_grad=True)
        y = norm(x)
        loss = y.sum()
        loss.backward()

        assert torch.isfinite(x.grad).all()
        assert x.grad.abs().mean() < 10


class TestLayerScale:
    def test_layerscale_init(self):
        """Verify LayerScale starts small."""
        scale = LayerScale(64, init_value=0.1)
        x = torch.randn(32, 64)
        y = scale(x)

        # Output should be ~0.1x input
        assert torch.allclose(y, 0.1 * x, rtol=1e-5)

    def test_layerscale_learnable(self):
        """Verify LayerScale is learnable."""
        scale = LayerScale(64)
        assert scale.gamma.requires_grad
```

### 2. Integration test: `tests/integration/test_pr1_norms.py`

```python
"""Test PR-1 boundary normalization integration."""

import torch
import pytest
from src.brain_brr.models.detector import SeizureDetector


def test_detector_with_norms():
    """Test detector stability with boundary norms."""
    config = {
        "architecture": "v3",
        "norms": {
            "boundary_norm": "rmsnorm",
            "boundary_eps": 1e-5,
            "layerscale_alpha": 0.1
        }
    }

    model = SeizureDetector(config)
    model.eval()

    # Test with extreme inputs
    x = torch.randn(2, 19, 15360) * 100  # Large input

    with torch.no_grad():
        output = model(x)

    # Should not explode
    assert torch.isfinite(output).all()
    assert output.abs().max() < 1000  # Reasonable bounds
```

## Verification Steps

### Phase 1: Local Testing
```bash
# 1. Run unit tests
pytest tests/unit/models/test_norms.py -xvs

# 2. Run smoke test with norms OFF
make s

# 3. Update smoke.yaml to enable norms
# 4. Run smoke test with norms ON
make s

# 5. Check for NaN/Inf
grep -i "nan\|inf" results/*/train.log
```

### Phase 2: Extended Testing
```bash
# Run for 1000 batches with norms enabled
BGB_NAN_DEBUG=1 python -m src train configs/local/train.yaml \
  --model.norms.boundary_norm rmsnorm \
  --model.norms.layerscale_alpha 0.1 \
  --training.num_epochs 1
```

## Expected Improvements

With PR-1 boundary normalization:
- **Before**: NaN explosion at batch 10-20 without clamps
- **After**: Stable for 1000+ batches even with some clamps removed
- **Gradient variance**: Should drop from >100 to <10
- **Activation magnitudes**: Should stay bounded without manual clamps

## Rollback Plan

If issues arise:
1. Set `boundary_norm: "none"` in config
2. Set `layerscale_alpha: 0.0`
3. All existing clamps remain in place
4. Behavior returns to exact current state

## Next Steps After PR-1

Once PR-1 is stable:
1. Begin removing redundant clamps one at a time
2. Start PR-2: Bounded edge stream
3. Enable stronger LayerScale (α = 0.2 or 0.5)
4. Monitor metrics for any degradation