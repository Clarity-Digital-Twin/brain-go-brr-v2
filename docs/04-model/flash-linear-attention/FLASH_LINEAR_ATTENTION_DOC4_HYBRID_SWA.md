# Doc 4: Hybrid GDN-H1 with Sliding Window Attention - Implementation Plan

> **Note (Oct 12, 2025)**: The canonical dual-stack description is in `docs/04-model/mamba.md`. This doc tracks the optional Hybrid SWA experiment backlog.

**Parent Document**: [FLASH_LINEAR_ATTENTION_RESEARCH.md](FLASH_LINEAR_ATTENTION_RESEARCH.md) (Doc 0 - SSOT)
**Phase**: 3 (OPTIONAL Hybrid Enhancement - ONLY if Phase 2 has short-event deficiency)
**Target**: Add Sliding Window Attention to BiGatedDeltaNet for short-seizure improvement
**Date**: October 8, 2025
**Version**: 2.0 (CONDITIONAL ROADMAP - Only Execute If Phase 2 Has Short-Event Gap)
**Status**: 🚧 **CONDITIONAL ROADMAP** - DO NOT EXECUTE UNLESS PHASE 2 SHOWS SHORT-EVENT DEFICIENCY 🚧

---

## 🛑 STOP - READ THIS FIRST 🛑

**THIS DOCUMENT DESCRIBES A FUTURE STATE THAT DOES NOT EXIST YET.**

**THIS IS THE MOST CONDITIONAL DOC - ONLY PROCEED IF:**
1. ✅ Phase 0 complete (infrastructure built)
2. ✅ Phase 1a complete (edge validated, GO decision)
3. ✅ Phase 1b complete (node validated, GO decision)
4. ✅ **Phase 2 complete (both streams validated, GO decision)**
5. ✅ **CRITICAL**: Manual analysis shows **short-duration (<5s) recall deficiency > 5%**

**Current Reality (October 9, 2025)**:
- ❌ SlidingWindowAttention does NOT exist (`src/brain_brr/models/sliding_window_attention.py` missing)
- ❌ HybridNodeStream does NOT exist (no hybrid builder logic in `node_stream.py`)
- ❌ HybridAttentionConfig does NOT exist (`schemas.py` has no hybrid config)
- ❌ Hybrid config file does NOT exist (`configs/local/hybrid_gdn_test.yaml` missing)
- ❌ Analysis scripts do NOT exist (`scripts/analyze_phase3_results.py` missing)
- ✅ Phase 2 COMPLETE (smoke + medium tests passed Oct 8, ready for Modal full training)

**What This Means**:
- 🚫 **DO NOT** attempt to follow these instructions today - they will fail
- 🚫 **DO NOT** proceed with Phase 3 unless Phase 2 shows specific short-event problems
- ✅ **DO** use this document for planning ONLY IF Phase 2 succeeds but has <5s recall gap
- ✅ **DO** complete Phase 0 → 1a → 1b → 2 FIRST, then analyze for short-event deficiency

**Execution Order with CONDITIONAL Gate**:
```
Phase 0 (4-6d)   Phase 1a (2-3d)   Phase 1b (2-3d)   Phase 2 (2-3d)        DECISION GATE           Phase 3 (2-3d)
    BUILD           VALIDATE          VALIDATE          VALIDATE          CONDITIONAL CHECK         OPTIONAL BUILD
      ↓                 ↓                 ↓                 ↓                      ↓                       ↓
   Doc 0 §14         Doc 1             Doc 2             Doc 3          Analyze Phase 2 Results    >>> Doc 4 (YOU ARE HERE)
   ─────────         ──────            ──────            ──────          ───────────────────────        ──────
   • Wrapper         • Edge            • Node            • Both          IF short-event recall       • SWA Layer
   • Schema          • GDN only        • GDN only        • GDN both      < overall - 5%:             • Hybrid Builder
   • Builders        • Risk: LOW       • Risk: MED       • Risk: HIGH        ├─ YES → Doc 4          • Config
   • Tests                                                                    └─ NO → Skip Phase 3     • Tests
                                                                              (Pure GDN sufficient)

CRITICAL: If Phase 2 overall recall is 85% but <5s recall is 75% → DEFICIENCY = 10% → PROCEED
         If Phase 2 overall recall is 85% and <5s recall is 82% → DEFICIENCY = 3% → SKIP PHASE 3
```

---

## ⚠️ BLOCKING PREREQUISITES

**YOU MUST COMPLETE THESE BEFORE EVEN CONSIDERING PHASE 3**:

1. ✅ **Phase 0 Infrastructure Complete** (Doc 0 Section 14 - 4-6 days)
2. ✅ **Phase 1a Complete with GO decision** (Doc 1 - 2-3 days)
3. ✅ **Phase 1b Complete with GO decision** (Doc 2 - 2-3 days)
4. ✅ **Phase 2 Complete with GO decision** (Doc 3 - 2-3 days)
5. 🔬 **Manual Short-Event Analysis** (REQUIRED - see Section 1.1):
   - Export Phase 2 predictions
   - Filter ground truth for <5s seizures
   - Calculate: `short_recall_deficiency = overall_recall - short_recall`
   - **ONLY proceed if deficiency > 5%**

**If Phase 2 overall recall is good AND <5s recall is similar → SKIP PHASE 3 (pure GDN is sufficient)**

---

**Changelog**:
- v2.0 (Oct 8, 2025): Added MASSIVE conditional warning banner (🛑 STOP section)
- v2.0 (Oct 8, 2025): Added execution order diagram with DECISION GATE
- v2.0 (Oct 8, 2025): Changed status to "CONDITIONAL ROADMAP" (was "Ready for Implementation")
- v2.0 (Oct 8, 2025): Listed current reality (❌ nothing exists yet)
- v2.0 (Oct 8, 2025): Emphasized Phase 3 is OPTIONAL (only if short-event gap exists)
- v1.1 (Oct 7, 2025): Fixed missing `import torch.nn as nn` in builder code
- v1.1 (Oct 7, 2025): Removed dependency on non-existent `sensitivity_short_seizures` metric
- v1.1 (Oct 7, 2025): Added manual short-duration analysis workflow for Phase 3 decision
- v1.0 (Oct 7, 2025): Initial version with proper config workflow and robust W&B analysis

---

## Executive Summary

This document provides **surgical implementation details** for adding Sliding Window Attention (SWA) to the BiGatedDeltaNet architecture. This is Phase 3 of the phased validation strategy and is **HIGHLY CONDITIONAL** - only implement if Phase 2 succeeds but shows specific short-duration seizure recall deficiency.

**Scope of Changes** (TO BE BUILT if Phase 3 warranted):
- 🔨 Create `src/brain_brr/models/sliding_window_attention.py` (TO BE BUILT)
- 🔨 Update `src/brain_brr/models/builders/node_stream.py` (TO BE BUILT)
- 🔨 Add hybrid config support in `src/brain_brr/config/schemas.py` (TO BE BUILT)
- 🔨 Write integration test (`tests/integration/test_hybrid_gdn_swa.py`) (TO BE BUILT)
- 🔬 A/B test: Pure GDN vs Hybrid GDN-H1 (TO BE DONE)
- ❌ **DO NOT TOUCH**: Edge stream (keep pure GDN), GNN, TCN, decoder

**Expected Outcome**:
- Architecture: `[GDN → GDN → SWA] × 2` (6 layers total for node stream)
- Hypothesis: **+1-2% sensitivity for short seizures (<5s)** via local attention
- Target: Improve recall on brief ictal events that SSMs might miss
- Risk: **LOW** (only affects node stream, edge stream unchanged)

**Timeline**: 2-3 days development + 6-8 hours integration test + 1 day analysis

---

## 📊 Architecture Analysis

**Motivation** (from Gated DeltaNet paper):
> "Linear transformers have limitations in modeling local shifts and comparisons, and their fixed state size makes it hard for retrieval tasks. Hybrid models combining linear recurrence with sliding window attention (SWA) achieve both improved training efficiency and superior task performance."

### Current Phase 2 Architecture

**Node Stream** (BiGatedDeltaNet only):
```python
# 6 sequential GDN layers
for i in range(6):
    x = BiGatedDeltaNet_Layer(x)
```

**Problem**: GDN excels at long-range dependencies but may miss **local spike patterns** in short seizures (<5s = <1280 samples @ 256Hz).

### Phase 3 Hybrid Architecture (GDN-H1)

**Node Stream** (Interleaved GDN + SWA, Samba-style):
```python
# Pattern: [GDN, GDN, SWA] × 2
layers = [
    BiGatedDeltaNet(),  # Layer 0: Long-range
    BiGatedDeltaNet(),  # Layer 1: Long-range
    SlidingWindowAttention(window_size=256, overlap=128),  # Layer 2: Local
    BiGatedDeltaNet(),  # Layer 3: Long-range
    BiGatedDeltaNet(),  # Layer 4: Long-range
    SlidingWindowAttention(window_size=256, overlap=128),  # Layer 5: Local
]
```

**Why This Works**:
- ✅ **GDN layers**: Capture long-range temporal patterns (seizure evolution over 10-60s)
- ✅ **SWA layers**: Capture local spike patterns (rhythmic discharges within 1-2s)
- ✅ **Throughput boost**: SWA is faster than full attention (proven in GDN-H1 benchmarks)
- ✅ **Complementary**: Linear recurrence + local attention = best of both worlds

### SWA Configuration

**Window Parameters**:
- **Window size**: 256 samples (1 second @ 256Hz)
- **Overlap**: 50% (128 samples = 0.5s)
- **Number of heads**: 8 (match GDN's multi-head design)
- **Head dimension**: 8 (64 / 8 heads)

**Why 1-second windows?**
- Typical spike duration: 50-200ms
- Typical rhythmic discharge: 3-5 Hz (200-333ms period)
- 1s window captures 3-5 cycles → sufficient for local pattern recognition

---

## 1. Prerequisites

### 1.1. Phase 2 Must Succeed AND Show Short-Event Deficiency

**Critical Decision Point**:
```bash
# Review Phase 2 results
cat PHASE2_RESULTS.md

# Analyze short-event performance manually:
# 1. Export predictions from Phase 2 validation run
# 2. Filter TUSZ ground truth for seizures with duration < 5s
# 3. Compute sensitivity on short-duration subset vs all seizures
# 4. Compare: Is short-duration recall significantly worse?

# Example manual analysis:
python -c "
import pandas as pd

# Load predictions and ground truth from Phase 2
# (Assumes you saved predictions during evaluation)
predictions = pd.read_csv('results/phase2_predictions.csv')
ground_truth = pd.read_csv('data_ext4/tusz/seizure_list.csv')

# Filter for short seizures (<5 seconds)
short_seizures = ground_truth[ground_truth['duration'] < 5.0]

# Compute recall on short vs all
short_recall = compute_recall(predictions, short_seizures)
overall_recall = compute_recall(predictions, ground_truth)

print(f'Overall recall: {overall_recall:.2%}')
print(f'Short (<5s) recall: {short_recall:.2%}')
print(f'Deficiency: {overall_recall - short_recall:.2%}')
"

# Decision matrix:
# ✅ Phase 2 success + short-recall deficiency > 5% → Proceed to Phase 3
# ❌ Phase 2 failed → Do NOT proceed (rollback to best phase)
# ❌ Short-recall deficiency < 5% → Skip Phase 3 (pure GDN is sufficient)
```

**Specific Triggers for Phase 3**:
1. Phase 2 (pure GDN) shows **+3-5% overall sensitivity** ✅
2. BUT: Manual analysis reveals **short-duration recall deficit > 5%** ❌
3. Hypothesis: GDN's fixed state size misses local spike patterns
4. Goal: Add SWA to capture local patterns without sacrificing long-range modeling

**NOTE**: The current training pipeline does NOT log a separate `sensitivity_short_seizures` metric. You must perform manual post-hoc analysis of predictions filtered by event duration to determine if Phase 3 is warranted.

### 1.2. Environment Setup

```bash
# Verify Phase 2 completed successfully
git log --oneline | head -10  # Should show Phase 2 commit

# Verify BiGatedDeltaNet wrapper exists
python -c "from src.brain_brr.models.gated_deltanet import BiGatedDeltaNet; print('✅ GDN exists')"

# Create Phase 3 branch
git checkout -b feature/hybrid-gdn-swa
git tag v3.8.3-pre-hybrid-swa
```

---

## 2. Implementation: Sliding Window Attention Layer

**🚨 REMINDER**: This section describes code that **DOES NOT EXIST YET**. Only create this file if you've completed Phase 0-2 AND confirmed short-event deficiency.

### 2.1. File: `src/brain_brr/models/sliding_window_attention.py` (TO BE CREATED)

**Purpose**: Efficient sliding window attention layer using FLA's attention primitive.

**Reference Implementation** (create this file when Phase 3 is warranted):

```python
"""Sliding Window Attention layer for hybrid GDN-H1 architecture.

Provides local attention within fixed windows to complement GDN's long-range recurrence.
"""

import logging
import math
import torch
import torch.nn as nn

try:
    from fla.ops.common import attention as fla_attention
    FLA_AVAILABLE = True
except ImportError:
    FLA_AVAILABLE = False

logger = logging.getLogger(__name__)


class SlidingWindowAttention(nn.Module):
    """Sliding Window Attention layer for local pattern modeling.

    Uses windowed attention to capture local spike patterns that linear recurrence
    may miss. Designed to interleave with BiGatedDeltaNet layers in hybrid architecture.

    Args:
        d_model: Model dimension (64 for node stream)
        num_heads: Number of attention heads (8 default)
        window_size: Attention window size in samples (256 = 1s @ 256Hz)
        overlap: Window overlap in samples (128 = 50% overlap)
        dropout: Dropout rate (0.1 default)
        use_layerscale: Enable LayerScale on output (match GDN)
        layerscale_init: LayerScale initial value (match GDN)

    Raises:
        ImportError: If FLA library not installed
    """

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 8,
        window_size: int = 256,
        overlap: int = 128,
        dropout: float = 0.1,
        use_layerscale: bool = False,
        layerscale_init: float = 0.1,
    ):
        super().__init__()

        if not FLA_AVAILABLE:
            raise ImportError(
                "FLA library not installed. Run: pip install flash-linear-attention"
            )

        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        assert window_size > overlap, f"window_size ({window_size}) must be > overlap ({overlap})"
        assert overlap >= 0, f"overlap ({overlap}) must be >= 0"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        self.overlap = overlap
        self.stride = window_size - overlap
        self.scale = 1.0 / math.sqrt(self.head_dim)

        logger.info(
            f"SlidingWindowAttention init: d_model={d_model}, num_heads={num_heads}, "
            f"window_size={window_size}, overlap={overlap}, stride={self.stride}"
        )

        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        # LayerScale (optional, to match GDN)
        if use_layerscale:
            from src.brain_brr.models.norms import LayerScale
            self.layerscale = LayerScale(d_model, init_value=layerscale_init)
            logger.debug(f"LayerScale enabled (init={layerscale_init})")
        else:
            self.layerscale = None

        # Initialize weights
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.o_proj.weight, gain=0.1)  # Conservative output init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sliding window attention.

        Args:
            x: (B, C, L) where:
               - B = batch size (can be B*19 for node stream)
               - C = d_model (64)
               - L = sequence length (960)

        Returns:
            x: (B, C, L) with local attention applied
        """
        B, C, L = x.shape
        assert C == self.d_model, f"Expected d_model={self.d_model}, got {C}"

        # Transpose to sequence-first: (B, L, C)
        x = x.transpose(1, 2)

        residual = x  # Save for residual connection

        # Project to Q, K, V
        q = self.q_proj(x)  # (B, L, C)
        k = self.k_proj(x)  # (B, L, C)
        v = self.v_proj(x)  # (B, L, C)

        # Reshape to multi-head: (B, L, num_heads, head_dim)
        q = q.view(B, L, self.num_heads, self.head_dim)
        k = k.view(B, L, self.num_heads, self.head_dim)
        v = v.view(B, L, self.num_heads, self.head_dim)

        # Apply windowed attention
        # Strategy: Split sequence into overlapping windows, compute attention per window
        attn_output = self._windowed_attention(q, k, v)  # (B, L, num_heads, head_dim)

        # Reshape back: (B, L, C)
        attn_output = attn_output.view(B, L, self.d_model)

        # Output projection
        output = self.o_proj(attn_output)

        # Apply LayerScale if enabled
        if self.layerscale is not None:
            output = self.layerscale(output)

        # Dropout + residual
        output = residual + self.dropout(output)

        # Transpose back to channel-first: (B, C, L)
        return output.transpose(1, 2)

    def _windowed_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention within sliding windows.

        Args:
            q, k, v: (B, L, num_heads, head_dim)

        Returns:
            output: (B, L, num_heads, head_dim)
        """
        B, L, num_heads, head_dim = q.shape

        # Initialize output buffer
        output = torch.zeros_like(q)
        counts = torch.zeros(B, L, 1, 1, device=q.device)  # Track overlaps for averaging

        # Compute window starts (with stride)
        window_starts = list(range(0, L - self.window_size + 1, self.stride))
        if not window_starts or window_starts[-1] + self.window_size < L:
            # Ensure last window covers end of sequence
            window_starts.append(max(0, L - self.window_size))

        logger.debug(f"Windowed attention: L={L}, windows={len(window_starts)}, window_size={self.window_size}")

        for start in window_starts:
            end = min(start + self.window_size, L)
            window_len = end - start

            # Extract window: (B, window_len, num_heads, head_dim)
            q_win = q[:, start:end, :, :]
            k_win = k[:, start:end, :, :]
            v_win = v[:, start:end, :, :]

            # Compute attention for this window
            # Reshape for attention: (B, num_heads, window_len, head_dim)
            q_win = q_win.transpose(1, 2)  # (B, num_heads, window_len, head_dim)
            k_win = k_win.transpose(1, 2)
            v_win = v_win.transpose(1, 2)

            # Standard scaled dot-product attention
            scores = torch.matmul(q_win, k_win.transpose(-2, -1)) * self.scale  # (B, num_heads, window_len, window_len)
            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_out = torch.matmul(attn_weights, v_win)  # (B, num_heads, window_len, head_dim)

            # Transpose back: (B, window_len, num_heads, head_dim)
            attn_out = attn_out.transpose(1, 2)

            # Accumulate into output (handle overlaps)
            output[:, start:end, :, :] += attn_out
            counts[:, start:end, :, :] += 1.0

        # Average overlapping regions
        output = output / counts.clamp(min=1.0)

        return output

    def __repr__(self) -> str:
        return (
            f"SlidingWindowAttention(d_model={self.d_model}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, overlap={self.overlap})"
        )
```

**Testing the SWA layer**:

```python
# Quick sanity check
python -c "
from src.brain_brr.models.sliding_window_attention import SlidingWindowAttention
import torch

# Node stream config
swa = SlidingWindowAttention(d_model=64, num_heads=8, window_size=256, overlap=128)
print(f'Params: {sum(p.numel() for p in swa.parameters()):,}')

# Test forward pass
x = torch.randn(8*19, 64, 960)  # (B*19, 64, 960) - node stream
y = swa(x)
print(f'Input: {x.shape}, Output: {y.shape}')
assert x.shape == y.shape, 'Shape mismatch!'
print('✅ SWA layer works!')
"
```

---

## 3. Implementation: Hybrid Node Stream Builder

**🚨 REMINDER**: This section describes modifications that **SHOULD NOT BE MADE YET**. Only modify this file if you've completed Phase 0-2 AND confirmed short-event deficiency.

### 3.1. File: `src/brain_brr/models/builders/node_stream.py` (TO BE MODIFIED)

**Changes Required** (only make these if Phase 3 is warranted):

1. Import SlidingWindowAttention
2. Support `hybrid_attention` config option
3. Build interleaved GDN+SWA layer list

**Implementation**:

```python
"""Node stream builder - per-electrode BiMamba/BiGatedDeltaNet/Hybrid component."""

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from src.brain_brr.constants import LAYERSCALE_ALPHA_FALLBACK

from ..mamba import BiMamba2
from ..norms import LayerScale

# Import GDN conditionally (from Phase 1a)
try:
    from ..gated_deltanet import BiGatedDeltaNet
    GDN_AVAILABLE = True
except ImportError:
    GDN_AVAILABLE = False

# Import SWA conditionally (from Phase 3)
try:
    from ..sliding_window_attention import SlidingWindowAttention
    SWA_AVAILABLE = True
except ImportError:
    SWA_AVAILABLE = False

if TYPE_CHECKING:
    from src.brain_brr.config.schemas import ModelConfig

logger = logging.getLogger(__name__)


def build_node_stream(cfg: "ModelConfig") -> BiMamba2 | "BiGatedDeltaNet" | "HybridNodeStream":
    """Build node stream: per-electrode BiMamba, BiGatedDeltaNet, or Hybrid GDN+SWA.

    V3 Architecture: Processes per-electrode temporal features with BiMamba, GDN, or Hybrid.
    This is a SHARED module (or ModuleList) that processes flattened (B*19, d_model, T) tensors.

    Args:
        cfg: Model configuration containing mamba and norms settings

    Returns:
        BiMamba2, BiGatedDeltaNet, or HybridNodeStream module

    Notes:
        - d_model=64 (per-electrode feature dimension)
        - headdim=8 for BiMamba2/GDN
        - num_layers=6 (6 for pure, interleaved for hybrid)
        - Hybrid: [GDN, GDN, SWA] × 2 pattern (Samba-style)
        - LayerScale enabled if boundary_norm != "none"
    """
    norms_cfg = getattr(cfg, "norms", None)
    mamba_cfg = cfg.mamba

    use_layerscale = bool(norms_cfg and norms_cfg.boundary_norm != "none")
    layerscale_init = float(norms_cfg.layerscale_alpha if norms_cfg else LAYERSCALE_ALPHA_FALLBACK)

    # Determine temporal model type
    temporal_type = getattr(mamba_cfg, "temporal_type_node", None)
    if temporal_type is None:
        temporal_type = getattr(mamba_cfg, "temporal_type", "bimamba2")

    # Check for hybrid attention config
    hybrid_cfg = getattr(mamba_cfg, "hybrid_attention", None)
    use_hybrid = hybrid_cfg is not None and getattr(hybrid_cfg, "enabled", False)

    logger.debug(f"Node stream temporal_type: {temporal_type}, hybrid: {use_hybrid}")

    if use_hybrid and temporal_type == "gated_deltanet":
        # Build Hybrid GDN + SWA node stream
        if not GDN_AVAILABLE:
            raise ImportError(
                "Hybrid mode requires Gated DeltaNet. "
                "Install FLA: pip install flash-linear-attention"
            )
        if not SWA_AVAILABLE:
            raise ImportError(
                "Hybrid mode requires SlidingWindowAttention. "
                "Ensure Phase 3 implementation complete."
            )

        node_stream = _build_hybrid_node_stream(
            cfg=cfg,
            use_layerscale=use_layerscale,
            layerscale_init=layerscale_init,
        )
        logger.info("Node stream: Hybrid GDN+SWA (6 layers interleaved)")

    elif temporal_type == "gated_deltanet":
        # Build BiGatedDeltaNet for node stream (Phase 1b/2)
        if not GDN_AVAILABLE:
            raise ImportError(
                "Gated DeltaNet requested but not available. "
                "Ensure Phase 1a completed: pip install flash-linear-attention"
            )

        node_stream = BiGatedDeltaNet(
            d_model=64,
            headdim=8,  # 6 × 8 = 48 = 0.75 × 64 ✅
            num_layers=6,
            dropout=mamba_cfg.dropout,
            fusion_mode=getattr(mamba_cfg, "fusion_mode", "sum"),
            allow_neg_eigval=getattr(mamba_cfg, "allow_neg_eigval", False),
            use_layerscale=use_layerscale,
            layerscale_init=layerscale_init,
        )
        logger.info(
            f"Node stream: BiGatedDeltaNet (d_model=64, headdim=8, layers=6, "
            f"fusion_mode={getattr(mamba_cfg, 'fusion_mode', 'sum')})"
        )
    else:
        # Build BiMamba2 for node stream (default/baseline)
        node_stream = BiMamba2(
            d_model=64,
            d_state=16,
            d_conv=4,
            expand=2,
            headdim=8,
            num_layers=6,
            dropout=mamba_cfg.dropout,
            use_layerscale=use_layerscale,
            layerscale_init=layerscale_init,
        )
        logger.info("Node stream: BiMamba2 (d_model=64, headdim=8, layers=6)")

    return node_stream


def _build_hybrid_node_stream(
    cfg: "ModelConfig",
    use_layerscale: bool,
    layerscale_init: float,
) -> "HybridNodeStream":
    """Build hybrid node stream with interleaved GDN and SWA layers.

    Pattern: [GDN, GDN, SWA] × 2 = 6 layers total

    Args:
        cfg: Model configuration
        use_layerscale: Enable LayerScale
        layerscale_init: LayerScale initial value

    Returns:
        HybridNodeStream module with interleaved layers
    """
    mamba_cfg = cfg.mamba
    hybrid_cfg = getattr(mamba_cfg, "hybrid_attention", None)

    # Extract hybrid config
    swa_layers = getattr(hybrid_cfg, "layers", [2, 5])  # Default: layers 2 and 5
    window_size = getattr(hybrid_cfg, "window_size", 256)  # Default: 1 second
    overlap_ratio = getattr(hybrid_cfg, "overlap_ratio", 0.5)  # Default: 50%
    overlap = int(window_size * overlap_ratio)

    logger.info(
        f"Hybrid node stream config: swa_layers={swa_layers}, "
        f"window_size={window_size}, overlap={overlap}"
    )

    # Build interleaved layers
    layers = []
    for i in range(6):  # Total 6 layers
        if i in swa_layers:
            # SWA layer
            layer = SlidingWindowAttention(
                d_model=64,
                num_heads=8,
                window_size=window_size,
                overlap=overlap,
                dropout=mamba_cfg.dropout,
                use_layerscale=use_layerscale,
                layerscale_init=layerscale_init,
            )
            logger.debug(f"Layer {i}: SlidingWindowAttention")
        else:
            # GDN layer
            layer = BiGatedDeltaNet(
                d_model=64,
                headdim=8,
                num_layers=1,  # Single layer (will be stacked in ModuleList)
                dropout=mamba_cfg.dropout,
                fusion_mode=getattr(mamba_cfg, "fusion_mode", "sum"),
                allow_neg_eigval=getattr(mamba_cfg, "allow_neg_eigval", False),
                use_layerscale=use_layerscale,
                layerscale_init=layerscale_init,
            )
            logger.debug(f"Layer {i}: BiGatedDeltaNet")

        layers.append(layer)

    return HybridNodeStream(layers)


class HybridNodeStream(nn.Module):
    """Hybrid node stream with interleaved GDN and SWA layers.

    Processes per-electrode temporal features through alternating long-range (GDN)
    and local (SWA) layers for complementary pattern modeling.

    Args:
        layers: List of BiGatedDeltaNet and SlidingWindowAttention layers
    """

    def __init__(self, layers: list):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through interleaved layers.

        Args:
            x: (B*19, 64, 960) - flattened node features

        Returns:
            x: (B*19, 64, 960) - processed features
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self) -> str:
        layer_types = [type(layer).__name__ for layer in self.layers]
        return f"HybridNodeStream(layers={layer_types})"
```

---

## 4. Implementation: Config Schema Update

**🚨 REMINDER**: This section describes schema additions that **DO NOT EXIST YET**. Only add these if you've completed Phase 0-2 AND confirmed short-event deficiency.

### 4.1. File: `src/brain_brr/config/schemas.py` (TO BE MODIFIED)

**Add to MambaConfig** (only if Phase 3 is warranted):

```python
class HybridAttentionConfig(BaseModel):
    """Hybrid attention configuration for GDN-H1."""
    enabled: bool = Field(
        default=False,
        description="Enable hybrid GDN + SWA architecture"
    )
    layers: list[int] = Field(
        default=[2, 5],
        description="Layer indices to use SWA (0-indexed, e.g., [2, 5] for 6-layer arch)"
    )
    window_size: int = Field(
        default=256,
        ge=64,
        le=512,
        description="SWA window size in samples (256 = 1s @ 256Hz)"
    )
    overlap_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=0.9,
        description="SWA window overlap ratio (0.5 = 50% overlap)"
    )

    @field_validator("layers")
    @classmethod
    def validate_layers(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("layers must not be empty")
        if any(layer < 0 or layer >= 6 for layer in v):
            raise ValueError("layer indices must be in range [0, 5] for 6-layer architecture")
        if len(v) != len(set(v)):
            raise ValueError("layer indices must be unique")
        return sorted(v)


class MambaConfig(BaseModel):
    """Mamba/GDN configuration."""
    n_layers: int = Field(default=6, ge=1, le=12)
    d_model: int = Field(default=512, ge=64, le=2048)
    d_state: int = Field(default=16, ge=8, le=64)
    conv_kernel: int = Field(default=4, ge=2, le=4)
    dropout: float = Field(default=0.1, ge=0.0, le=0.5)

    # Stream-specific temporal types (for Phase 1a/1b/2 isolation)
    temporal_type_node: str | None = Field(
        default=None,
        description="Node stream temporal type: 'bimamba2' or 'gated_deltanet' (overrides temporal_type)"
    )
    temporal_type_edge: str | None = Field(
        default=None,
        description="Edge stream temporal type: 'bimamba2' or 'gated_deltanet' (overrides temporal_type)"
    )

    # Fallback for both streams (Phase 2 convenience)
    temporal_type: str = Field(
        default="bimamba2",
        description="Temporal model type (fallback if stream-specific not set): 'bimamba2' or 'gated_deltanet'"
    )

    # GDN-specific settings
    fusion_mode: str = Field(
        default="sum",
        description="Bidirectional fusion: 'sum' or 'concat'"
    )
    allow_neg_eigval: bool = Field(
        default=False,
        description="Allow negative eigenvalues (β_t ∈ (0,2))"
    )

    # Hybrid attention (Phase 3)
    hybrid_attention: HybridAttentionConfig | None = Field(
        default=None,
        description="Hybrid GDN+SWA configuration (optional, Phase 3)"
    )

    @field_validator("temporal_type", "temporal_type_node", "temporal_type_edge")
    @classmethod
    def validate_temporal_type(cls, v: str | None) -> str | None:
        if v is not None and v not in ["bimamba2", "gated_deltanet"]:
            raise ValueError(f"temporal_type must be 'bimamba2' or 'gated_deltanet', got {v}")
        return v

    @field_validator("fusion_mode")
    @classmethod
    def validate_fusion_mode(cls, v: str) -> str:
        if v not in ["sum", "concat"]:
            raise ValueError(f"fusion_mode must be 'sum' or 'concat', got {v}")
        return v
```

---

## 5. Implementation: Config File Update

**🚨 REMINDER**: This config file **DOES NOT EXIST YET**. Only create it if you've completed Phase 0-2 AND confirmed short-event deficiency.

### 5.1. File: `configs/local/hybrid_gdn_test.yaml` (TO BE CREATED)

**Reference config for Phase 3 testing** (create this when Phase 3 is warranted):

```yaml
# Hybrid GDN-H1 Test Config
# Phase 3: Add Sliding Window Attention to BiGatedDeltaNet (node stream only)

experiment:
  name: hybrid_gdn_h1_test
  description: "Phase 3 - Hybrid GDN+SWA for short-seizure improvement"
  seed: 42
  output_dir: results/hybrid_gdn_test
  cache_dir: cache/tusz_mmap
  device: cuda
  log_level: INFO
  save_model: true
  save_best_only: true

  wandb:
    enabled: true
    project: seizure-v3-hybrid-gdn
    entity: null

model:
  architecture: v3

  # PR-1: Boundary Normalization (ENABLED)
  norms:
    boundary_norm: layernorm
    boundary_eps: 1.0e-5
    layerscale_alpha: 0.1
    after_tcn_proj: true
    after_node_mamba: true
    after_edge_mamba: true
    after_gnn: true
    before_decoder: true

  tcn:
    num_layers: 8
    kernel_size: 7
    dropout: 0.15
    causal: false
    stride_down: 16
    use_cuda_optimizations: true

  mamba:
    n_layers: 6
    d_model: 512
    d_state: 16
    conv_kernel: 4
    dropout: 0.1

    # PHASE 3: Enable GDN for both streams + Hybrid SWA for node stream
    temporal_type: gated_deltanet  # Both streams use GDN (from Phase 2)
    fusion_mode: sum
    allow_neg_eigval: false

    # Hybrid attention config (Phase 3 NEW)
    hybrid_attention:
      enabled: true            # Enable hybrid architecture
      layers: [2, 5]           # SWA at layers 2 and 5 (pattern: [GDN, GDN, SWA] × 2)
      window_size: 256         # 1 second @ 256Hz
      overlap_ratio: 0.5       # 50% overlap (128 samples)

  # Graph configuration (V3)
  graph:
    enabled: true

    # PR-2: Bounded Edge Stream (ENABLED)
    edge_lift_activation: tanh
    edge_lift_norm: layernorm
    edge_lift_init_gain: 0.1

    # V3: Edge stream config (pure GDN, NO hybrid)
    edge_features: cosine
    edge_top_k: 3
    edge_threshold: 1.0e-4
    edge_mamba_layers: 2       # Edge GDN layers (pure, no SWA)
    edge_mamba_d_state: 8
    edge_mamba_d_model: 32     # FLA hardware requirement (not 16!)
    edge_similarity_margin: 0.01

    # PR-3: Adjacency Conditioning (ENABLED)
    adj_row_softmax: true
    adj_softmax_tau: 1.0
    adj_ema_beta: 0.95
    adj_force_symmetric: true
    laplacian_eps: 1.0e-3
    laplacian_normalize: true

    # GNN architecture
    n_layers: 2
    dropout: 0.1
    use_residual: true
    alpha: 0.05
    k_eigenvectors: 16

    # Dynamic PE config (v3) - OPTIMIZED FOR RTX 4090
    use_dynamic_pe: true
    semi_dynamic_interval: 5
    pe_sign_consistency: true

data:
  dataset: tuh_eeg
  data_dir: data_ext4/tusz/edf
  cache_dir: cache/tusz_mmap
  sampling_rate: 256
  n_channels: 19
  window_size: 60
  stride: 10
  use_balanced_sampling: true
  num_workers: 0
  pin_memory: true
  persistent_workers: false
  prefetch_factor: 2

preprocessing:
  montage: "10-20"
  bandpass: [0.5, 120.0]
  notch_freq: 60
  normalize: true

training:
  epochs: 10  # Short test run

  batch_size: 8

  learning_rate: 1.0e-4
  weight_decay: 0.01
  optimizer: adamw

  gradient_clip: 0.5

  mixed_precision: false

  loss: focal
  focal_alpha: 0.5
  focal_gamma: 2.0

  scheduler:
    type: cosine
    warmup_ratio: 0.03

  warmup_schedule:
    enabled: true
    warmup_steps: 1000
    adj_temperature_enabled: true
    adj_temperature_start: 2.0
    adj_temperature_end: 1.0
    focal_gamma_enabled: true
    focal_gamma_start: 1.0
    focal_gamma_end: 2.0

  early_stopping:
    patience: 5
    metric: sensitivity_at_10fa

  checkpoint_interval: 1
  mid_checkpoint_interval_s: 1800
  mid_epoch_keep: 3
  gradient_accumulation_steps: 1

postprocessing:
  hysteresis:
    tau_on: 0.86
    tau_off: 0.78
  morphology:
    opening_kernel: 11
    closing_kernel: 31
  duration:
    min_duration_s: 3.0
    max_duration_s: 600.0
  events:
    tau_merge: 2.0
    confidence_method: mean

evaluation:
  fa_rates: [10, 5, 2.5, 1]
  save_predictions: false
  save_plots: false

logging:
  log_every_n_steps: 50
  log_gradients: false
  log_weights: false
```

---

## 6. Testing Strategy

### 6.1. Integration Test: `tests/integration/test_hybrid_gdn_swa.py`

```python
"""Integration test for hybrid GDN+SWA architecture."""

import pytest
import torch

from src.brain_brr.models.builders.node_stream import build_node_stream
from src.brain_brr.config.schemas import ModelConfig, MambaConfig, HybridAttentionConfig

# Skip if FLA not installed
pytest.importorskip("fla")


class TestHybridGDNSWA:
    """Test hybrid GDN+SWA architecture (Phase 3)."""

    @pytest.fixture
    def base_config(self):
        """Base config with hybrid attention enabled."""
        return ModelConfig(
            mamba=MambaConfig(
                n_layers=6,
                d_model=512,
                d_state=16,
                conv_kernel=4,
                dropout=0.1,
                temporal_type="gated_deltanet",
                hybrid_attention=HybridAttentionConfig(
                    enabled=True,
                    layers=[2, 5],
                    window_size=256,
                    overlap_ratio=0.5,
                ),
            ),
        )

    def test_build_hybrid_node_stream(self, base_config):
        """Test building hybrid node stream."""
        from src.brain_brr.models.builders.node_stream import HybridNodeStream

        node_stream = build_node_stream(base_config)

        assert isinstance(node_stream, HybridNodeStream)
        assert len(node_stream.layers) == 6

        # Verify layer pattern: [GDN, GDN, SWA, GDN, GDN, SWA]
        from src.brain_brr.models.gated_deltanet import BiGatedDeltaNet
        from src.brain_brr.models.sliding_window_attention import SlidingWindowAttention

        expected_pattern = [
            BiGatedDeltaNet,  # 0
            BiGatedDeltaNet,  # 1
            SlidingWindowAttention,  # 2
            BiGatedDeltaNet,  # 3
            BiGatedDeltaNet,  # 4
            SlidingWindowAttention,  # 5
        ]

        for i, (layer, expected_type) in enumerate(zip(node_stream.layers, expected_pattern)):
            assert isinstance(layer, expected_type), f"Layer {i} should be {expected_type.__name__}"

    def test_hybrid_forward_pass(self, base_config):
        """Test forward pass through hybrid architecture."""
        node_stream = build_node_stream(base_config)

        batch_size = 2
        num_electrodes = 19
        seq_len = 960

        x = torch.randn(batch_size * num_electrodes, 64, seq_len)
        y = node_stream(x)

        assert y.shape == x.shape
        assert not torch.isnan(y).any()

    def test_hybrid_parameter_count(self, base_config):
        """Test parameter count for hybrid architecture.

        Hybrid should have MORE params than pure GDN due to SWA layers.
        """
        # Pure GDN (Phase 2)
        base_config.mamba.hybrid_attention = None
        pure_gdn = build_node_stream(base_config)
        pure_params = sum(p.numel() for p in pure_gdn.parameters())

        # Hybrid GDN+SWA (Phase 3)
        base_config.mamba.hybrid_attention = HybridAttentionConfig(
            enabled=True,
            layers=[2, 5],
            window_size=256,
            overlap_ratio=0.5,
        )
        hybrid = build_node_stream(base_config)
        hybrid_params = sum(p.numel() for p in hybrid.parameters())

        print(f"Pure GDN params: {pure_params:,}")
        print(f"Hybrid GDN+SWA params: {hybrid_params:,}")
        print(f"Increase: {hybrid_params - pure_params:,} (+{(hybrid_params/pure_params - 1)*100:.1f}%)")

        # Hybrid should have MORE params (SWA adds Q/K/V/O projections)
        assert hybrid_params > pure_params, "Hybrid should have more params than pure GDN"

    def test_swa_layer_standalone(self):
        """Test SWA layer in isolation."""
        from src.brain_brr.models.sliding_window_attention import SlidingWindowAttention

        swa = SlidingWindowAttention(
            d_model=64,
            num_heads=8,
            window_size=256,
            overlap=128,
            dropout=0.0,
        )

        x = torch.randn(8 * 19, 64, 960)
        y = swa(x)

        assert y.shape == x.shape
        assert not torch.isnan(y).any()

    def test_gradient_flow_hybrid(self, base_config):
        """Test gradients flow through hybrid architecture."""
        node_stream = build_node_stream(base_config)

        x = torch.randn(8 * 19, 64, 960, requires_grad=True)
        y = node_stream(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        for name, param in node_stream.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_cuda_compatibility_hybrid(self, base_config):
        """Test CUDA compatibility."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        node_stream = build_node_stream(base_config).cuda()

        x = torch.randn(8 * 19, 64, 960).cuda()
        y = node_stream(x)

        assert y.is_cuda
        assert not torch.isnan(y).any()

    def test_hybrid_config_validation(self):
        """Test hybrid config validation."""
        # Valid config
        valid_cfg = HybridAttentionConfig(
            enabled=True,
            layers=[2, 5],
            window_size=256,
            overlap_ratio=0.5,
        )
        assert valid_cfg.layers == [2, 5]

        # Invalid: empty layers
        with pytest.raises(ValueError, match="layers must not be empty"):
            HybridAttentionConfig(enabled=True, layers=[])

        # Invalid: layer index out of range
        with pytest.raises(ValueError, match="layer indices must be in range"):
            HybridAttentionConfig(enabled=True, layers=[0, 10])

        # Invalid: duplicate layers
        with pytest.raises(ValueError, match="layer indices must be unique"):
            HybridAttentionConfig(enabled=True, layers=[2, 2, 5])
```

**Run integration tests**:

```bash
pytest tests/integration/test_hybrid_gdn_swa.py -xvs
```

---

## 7. Validation & Benchmarking

### 7.1. Smoke Test (3 files, 1 epoch)

```bash
export BGB_SMOKE_TEST=1
export BGB_NAN_DEBUG=1
python -m src train configs/local/hybrid_gdn_test.yaml

# Expected:
# - Loads 3 files
# - 1 epoch completes
# - Node stream logs "HybridNodeStream(layers=[...])"
# - No crashes, no NaNs
```

### 7.2. Integration Test (50 files, 10 epochs)

```bash
export BGB_LIMIT_FILES=50
python -m src train configs/local/hybrid_gdn_test.yaml

# Monitor:
# - Loss curve (should decrease)
# - Gradient norms (should be stable)
# - Memory usage (~20-22GB on RTX 4090, slight increase from SWA)
# - Throughput (5-10% FASTER than pure GDN due to efficient SWA)
```

### 7.3. A/B Comparison

**CRITICAL**: CLI does NOT support `--experiment.name` overrides. Create separate configs for each experiment.

```bash
# Set limited file count for all experiments
export BGB_LIMIT_FILES=50

# 1. Phase 2 Baseline: Pure GDN (both streams)
cp configs/local/full_gdn_test.yaml configs/local/phase2_pure_gdn.yaml
# Edit configs/local/phase2_pure_gdn.yaml:
#   experiment.name: "phase2_pure_gdn"
#   training.epochs: 10
python -m src train configs/local/phase2_pure_gdn.yaml

# 2. Phase 3 Hybrid: GDN+SWA (node stream only)
cp configs/local/hybrid_gdn_test.yaml configs/local/phase3_hybrid_gdn.yaml
# Edit configs/local/phase3_hybrid_gdn.yaml:
#   experiment.name: "phase3_hybrid_gdn"
#   training.epochs: 10
python -m src train configs/local/phase3_hybrid_gdn.yaml
```

**Compare** (using robust W&B analysis):

```python
"""Compare Phase 3 results against Phase 2 baseline.

USAGE:
    python scripts/analyze_phase3_results.py --project seizure-v3-hybrid-gdn
"""
import argparse
import sys
from typing import Optional

import wandb


def find_run_by_experiment_name(runs: list, experiment_name: str) -> Optional[wandb.apis.public.Run]:
    """Find run by experiment.name config field (robust to W&B name suffixes).

    Args:
        runs: List of W&B runs
        experiment_name: Expected value of config.experiment.name

    Returns:
        Matching run or None if not found
    """
    matches = [
        r for r in runs
        if r.config.get('experiment', {}).get('name') == experiment_name
    ]

    if not matches:
        return None
    if len(matches) > 1:
        print(f"Warning: Found {len(matches)} runs with experiment.name='{experiment_name}', using first")
        print(f"  Run IDs: {[r.id for r in matches]}")

    return matches[0]


def get_metric(run: wandb.apis.public.Run, metric: str, default: float = 0.0) -> float:
    """Safely extract metric from run summary.

    Args:
        run: W&B run object
        metric: Metric name (e.g., 'sensitivity_at_10fa')
        default: Default value if metric missing

    Returns:
        Metric value or default
    """
    value = run.summary.get(metric)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main():
    parser = argparse.ArgumentParser(description="Analyze Phase 3 Hybrid GDN+SWA results")
    parser.add_argument('--project', required=True, help='W&B project name')
    parser.add_argument('--entity', default=None, help='W&B entity (optional)')
    args = parser.parse_args()

    # Initialize API
    api = wandb.Api()
    project_path = f"{args.entity}/{args.project}" if args.entity else args.project
    runs = api.runs(project_path)

    # Find runs by experiment.name config field (robust to W&B suffixes)
    phase2_pure = find_run_by_experiment_name(runs, "phase2_pure_gdn")
    phase3_hybrid = find_run_by_experiment_name(runs, "phase3_hybrid_gdn")

    # Verify all runs found
    missing = []
    if not phase2_pure: missing.append("phase2_pure_gdn")
    if not phase3_hybrid: missing.append("phase3_hybrid_gdn")

    if missing:
        print(f"ERROR: Missing runs: {missing}")
        print("\nAvailable runs:")
        for r in runs:
            exp_name = r.config.get('experiment', {}).get('name', 'UNKNOWN')
            print(f"  - {r.id}: {r.name} (experiment.name={exp_name})")
        sys.exit(1)

    # Extract metrics at different FA rates
    phase2_sens_10fa = get_metric(phase2_pure, 'sensitivity_at_10fa')
    phase2_sens_5fa = get_metric(phase2_pure, 'sensitivity_at_5fa')
    phase2_sens_1fa = get_metric(phase2_pure, 'sensitivity_at_1fa')

    phase3_sens_10fa = get_metric(phase3_hybrid, 'sensitivity_at_10fa')
    phase3_sens_5fa = get_metric(phase3_hybrid, 'sensitivity_at_5fa')
    phase3_sens_1fa = get_metric(phase3_hybrid, 'sensitivity_at_1fa')

    # Calculate gains
    if phase2_sens_10fa > 0:
        gain_10fa = (phase3_sens_10fa - phase2_sens_10fa) / phase2_sens_10fa
        gain_5fa = (phase3_sens_5fa - phase2_sens_5fa) / phase2_sens_5fa if phase2_sens_5fa > 0 else 0
        gain_1fa = (phase3_sens_1fa - phase2_sens_1fa) / phase2_sens_1fa if phase2_sens_1fa > 0 else 0
    else:
        print(f"ERROR: Phase 2 baseline sensitivity is zero or missing")
        sys.exit(1)

    # Print results
    print("=" * 80)
    print("Phase 3 Hybrid GDN+SWA Results")
    print("=" * 80)
    print(f"\nPhase 2 (Pure GDN):")
    print(f"  Sensitivity@10FA: {phase2_sens_10fa:.2%}")
    print(f"  Sensitivity@5FA:  {phase2_sens_5fa:.2%}")
    print(f"  Sensitivity@1FA:  {phase2_sens_1fa:.2%}")

    print(f"\nPhase 3 (Hybrid GDN+SWA):")
    print(f"  Sensitivity@10FA: {phase3_sens_10fa:.2%} ({gain_10fa:+.2%})")
    print(f"  Sensitivity@5FA:  {phase3_sens_5fa:.2%} ({gain_5fa:+.2%})")
    print(f"  Sensitivity@1FA:  {phase3_sens_1fa:.2%} ({gain_1fa:+.2%})")

    # Decision logic
    print("\n" + "=" * 80)
    print("Recommendations")
    print("=" * 80)
    print("\nNOTE: Short-duration seizure analysis requires manual post-processing.")
    print("Export predictions and filter by event duration to assess <5s seizure recall.")

    if gain_10fa >= 0.01:
        print(f"\n✅ PROCEED with Phase 3 (Hybrid): {gain_10fa:+.2%} overall gain")
        print(f"   - Improvement across all FA rates")
        print(f"   - SWA successfully captures local patterns")
        print(f"   - Recommend manual short-duration analysis to confirm benefit")
    else:
        print(f"\n❌ REVERT to Phase 2 (Pure GDN): No significant improvement")
        print(f"   - Overall gain: {gain_10fa:+.2%} (below +1% target)")
        print(f"   - Pure GDN is sufficient")

    print("=" * 80)


if __name__ == '__main__':
    main()
```

**Setup analysis script**:

```bash
# Create the analysis script
cat > scripts/analyze_phase3_results.py << 'EOF'
# (Copy the Python script from above)
EOF

chmod +x scripts/analyze_phase3_results.py
```

**Run analysis**:

```bash
python scripts/analyze_phase3_results.py --project seizure-v3-hybrid-gdn
```

---

## 8. Success Criteria

### 8.1. Technical Criteria

✅ **Integration tests pass**: All tests in `test_hybrid_gdn_swa.py` pass
✅ **Smoke test completes**: 3 files, 1 epoch, no crashes
✅ **No NaNs**: Forward/backward passes produce finite values
✅ **Hybrid architecture**: Logs show "HybridNodeStream(layers=[...])"
✅ **SWA layers functional**: Windowed attention computes correctly

### 8.2. Performance Metrics to Monitor

Track these measurements relative to Phase 2. treat thresholds as hypotheses, not hard gates.

✅ **Convergence**: Loss decreases over 10 epochs  
✅ **Val loss delta**: val_loss ≤ Phase 2 + 0.05  
✅ **Sensitivity delta**: sensitivity_at_10fa − Phase 2 (target ≥ +1%, but record actual)  
✅ **Cross-FA behaviour**: Compare 10FA, 5FA, 1FA trajectories  
✅ **Throughput**: ≤ Phase 2 + 10%  
✅ **Memory usage**: ≤ Phase 2 + 2GB  

**NOTE**: Short-duration seizure improvement must be assessed via **manual post-hoc analysis** by filtering predictions by event duration (<5s). The training pipeline does NOT log a separate `sensitivity_short_seizures` metric.

### 8.3. Result Interpretation

After collecting metrics:
- Summarize gains/regressions vs Phase 2 (overall + short-duration)
- Document qualitative observations (training stability, anomalies)
- Decide on next experiments (e.g., different SWA placements, revert to Phase 2 as documented baseline, or pause hybrid exploration)
- Capture conclusions in `PHASE3_RESULTS.md` and update Doc 0 roadmap accordingly

---

## 9. Recovery Procedures & Troubleshooting

If training fails or you need to recover from technical issues:

```bash
# Revert to Phase 2 (Pure GDN)
git checkout v3.8.3-pre-hybrid-swa

# Restore configs
git checkout HEAD~1 configs/local/hybrid_gdn_test.yaml

# Clean up
rm src/brain_brr/models/sliding_window_attention.py
git restore src/brain_brr/models/builders/node_stream.py
git restore src/brain_brr/config/schemas.py

# Verify Phase 2 restored
pytest tests/integration/test_full_gdn_migration.py -xvs
make smoke-test
```

---

## 10. Timeline & Checklist

### Day 1: Implementation
- [ ] Verify Phase 2 completed successfully
- [ ] Review PHASE2_RESULTS.md (check for short-event deficiency)
- [ ] Create `sliding_window_attention.py` layer
- [ ] Update `node_stream.py` builder
- [ ] Update `schemas.py` config
- [ ] Create `hybrid_gdn_test.yaml` config
- [ ] Write integration tests
- [ ] All tests pass locally

### Day 2: Validation
- [ ] Smoke test (3 files, 1 epoch) - 10 min
- [ ] Integration test (50 files, 10 epochs) - 6-8 hours
- [ ] A/B comparison: Phase 2 vs Phase 3
- [ ] Analyze results (overall + short-seizure metrics)

### Day 3: Synthesis
- [ ] Review metrics (overall, short-seizure, throughput)
- [ ] Summarize comparison vs Phase 2
- [ ] Document findings in results + postmortem templates
- [ ] Outline recommended follow-up experiments (if any)
- [ ] Confirm Phase 2 configuration remains available via config toggle

**Total**: 2-3 days

---

## 11. Risk Analysis

### 11.1. Risk Comparison: Phase 2 vs Phase 3

| Risk Factor | Phase 2 (Pure GDN) | Phase 3 (Hybrid GDN+SWA) |
|-------------|-------------------|--------------------------|
| **Complexity** | Medium | **High** (interleaved layers) |
| **Parameters** | ~291K | **~350K** (SWA adds projections) |
| **Memory** | ~20GB | **~22GB** (modest increase) |
| **Throughput** | Baseline | **Same or faster** (SWA efficient) |
| **Risk level** | Medium | **LOW** (only node stream, edge unchanged) |

**Key Insight**: Phase 3 has **LOW RISK** because:
- Only affects node stream (edge stream unchanged)
- SWA is proven in production (Samba, GDN-H1)
- Easy rollback to Phase 2 if needed

### 11.2. What Could Go Wrong

1. **No short-seizure improvement**: If SWA doesn't help, revert to Phase 2
2. **Memory increase**: +2GB is acceptable, but monitor on RTX 4090
3. **Implementation complexity**: Interleaved layers more complex than pure GDN
4. **Overfitting risk**: More parameters may overfit on small datasets

**Mitigation**: 10-epoch validation gate before committing to 100-epoch full run

---

## 12. Next Steps

### If Phase 3 Succeeds:

1. **Document results**: Capture metrics, training notes, and qualitative observations in `PHASE3_RESULTS.md`
2. **Compare against Phase 2**: Quantify gains/losses relative to pure GDN stack
3. **Decide follow-up research**: e.g., longer runs, alternative SWA placements, or preparing publication material

### If Phase 3 Fails:

1. **Document findings**: Add details to `PHASE3_POSTMORTEM.md`
2. **Root cause analysis**: Why didn't SWA help?
   - Insufficient window size (try 512 samples = 2 seconds)?
   - Wrong layer positions (try [1, 3, 5] instead of [2, 5])?
   - Short-seizure hypothesis incorrect?
3. **Alternative strategies**:
   - Try different SWA positions
   - Try larger windows (512 samples)
   - Try more SWA layers (3 instead of 2)
   - Accept Phase 2 as the documented result if no improvement surfaces

---

## 13. References

- **Doc 0 (SSOT)**: [FLASH_LINEAR_ATTENTION_RESEARCH.md](FLASH_LINEAR_ATTENTION_RESEARCH.md)
- **Doc 1 (Edge Stream)**: [FLASH_LINEAR_ATTENTION_DOC1_EDGE_MIGRATION.md](FLASH_LINEAR_ATTENTION_DOC1_EDGE_MIGRATION.md)
- **Doc 2 (Node Stream)**: [FLASH_LINEAR_ATTENTION_DOC2_NODE_MIGRATION.md](FLASH_LINEAR_ATTENTION_DOC2_NODE_MIGRATION.md)
- **Doc 3 (Full Migration)**: [FLASH_LINEAR_ATTENTION_DOC3_FULL_MIGRATION.md](FLASH_LINEAR_ATTENTION_DOC3_FULL_MIGRATION.md)
- **FLA Library**: https://github.com/fla-org/flash-linear-attention
- **Gated DeltaNet Paper**: https://arxiv.org/abs/2412.06464 (Section 4.4: Hybrid GDN-H1)
- **Samba Paper**: Samba hybrid architecture (interleaved attention + SSM)
- **Current v3.8.3 Baseline**: RELEASE_NOTES.md

---

**Document Status**: ✅ Ready for Implementation (pending Phase 2 success AND short-event deficiency)
**Prerequisites**: Phase 2 must succeed AND show poor recall on <5s seizures
**Optional**: Phase 3 is OPTIONAL - only implement if needed
