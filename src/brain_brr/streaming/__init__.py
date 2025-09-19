"""Streaming and real-time inference.

This module contains:
- streaming.py: Streaming post-processor with state management
- buffer.py: Ring buffer for overlap handling (future)
"""

from .streaming import HysteresisState, StreamingPostProcessor

__all__ = ["HysteresisState", "StreamingPostProcessor"]
