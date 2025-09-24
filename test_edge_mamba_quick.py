#!/usr/bin/env python3
"""Quick test of edge Mamba configuration."""

import torch
from mamba_ssm import Mamba2

print("Testing Edge Mamba2 directly...")

# Create edge Mamba with fixed headdim
edge_mamba = Mamba2(
    d_model=16,
    d_state=8,
    d_conv=4,
    expand=2,
    headdim=4  # Critical: (16*2)/4 = 8
).cuda()

# Test input
x = torch.randn(1368, 960, 16).cuda()  # (B*171, L, D)

try:
    output = edge_mamba(x)
    print(f"✅ SUCCESS! No errors. Output shape: {output.shape}")
except Exception as e:
    if "causal_conv1d" in str(e) and "strides" in str(e):
        print(f"❌ FAILED - Still getting stride error: {e}")
    else:
        print(f"❌ FAILED with different error: {e}")