"""Builder helpers for SeizureDetector construction (SRP compliance)."""

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
