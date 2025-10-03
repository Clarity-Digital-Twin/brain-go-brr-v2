"""Builder helpers for SeizureDetector construction (SRP compliance).

This module contains factory functions for constructing components of the
SeizureDetector V3 dual-stream architecture. Each builder is responsible for
a single architectural concern, following the Single Responsibility Principle.

Builders:
    build_node_stream: Constructs BiMamba2 for per-electrode processing
        - Input: Config, device
        - Output: BiMamba2 instance (6 layers, d_model=64)
        - Purpose: Temporal feature extraction for each EEG electrode

    build_edge_stream: Constructs edge stream components (BiMamba + projections)
        - Input: Config, device, hidden_dim
        - Output: EdgeStreamComponents container
        - Purpose: Process electrode pair relationships (171 edges)
        - Includes: Edge BiMamba, projections, activation/norm layers

    build_fusion_head: Constructs fusion layer for combining node/edge features
        - Input: Config, d_model
        - Output: (fusion_type, fusion_module) tuple
        - Purpose: Gated or multihead fusion of dual streams
        - Backward compatible: Returns ("add", None) if no fusion config

    build_regularizers: Constructs boundary norms and LayerScale modules
        - Input: Config, edge_d_model
        - Output: RegularizationComponents container
        - Purpose: Gradient stabilization via boundary normalization
        - Includes: Norms for 3 boundary points + optional LayerScale

Why Builders?
    - Separation of concerns: Component construction vs. forward logic
    - Testability: Each builder can be unit tested independently
    - Maintainability: Easier to modify component configs without touching model
    - Readability: from_config() reduced from 199→107 lines

Usage:
    from brain_brr.models.builders import (
        build_node_stream,
        build_edge_stream,
        build_fusion_head,
        build_regularizers,
    )

    node_stream = build_node_stream(cfg, device)
    edge_stream = build_edge_stream(cfg, device, hidden_dim=512)
    fusion_type, fusion_module = build_fusion_head(cfg, d_model=64)
    regularizers = build_regularizers(cfg, edge_d_model=32)
"""

from .edge_stream import build_edge_stream
from .fusion import build_fusion_head
from .node_stream import build_node_stream
from .regularization import build_regularizers

__all__ = [
    "build_edge_stream",
    "build_fusion_head",
    "build_node_stream",
    "build_regularizers",
]
