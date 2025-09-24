#!/usr/bin/env python3
"""Test that V3 Mamba configuration no longer has fallback issues."""

import torch
import sys
import os

# Ensure we can import from src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing V3 Mamba fix...")
print("=" * 80)

# Test node stream configuration
print("\n1. Testing Node Stream BiMamba2...")
from src.brain_brr.models.mamba import BiMamba2

try:
    node_mamba = BiMamba2(
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2,
        headdim=8,  # Fixed: (64*2)/8 = 16
        num_layers=6
    ).cuda()

    # Test with B*19 = 152 (batch_size=8, 19 electrodes)
    x_node = torch.randn(152, 64, 960).cuda()  # (B*19, C, L)

    # Capture any warnings
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        output = node_mamba(x_node)

    if any("[MAMBA]" in str(warning.message) for warning in w):
        print("  ❌ FAILED - Still getting fallback warnings")
        for warning in w:
            if "[MAMBA]" in str(warning.message):
                print(f"     {warning.message}")
    else:
        print(f"  ✅ SUCCESS - No fallback! Output shape: {output.shape}")

except Exception as e:
    print(f"  ❌ FAILED: {e}")

# Test edge stream configuration
print("\n2. Testing Edge Stream BiMamba2...")

try:
    edge_mamba = BiMamba2(
        d_model=16,
        d_state=8,
        d_conv=4,
        expand=2,
        headdim=4,  # Fixed: (16*2)/4 = 8
        num_layers=2
    ).cuda()

    # Test with B*171 = 1368 (batch_size=8, 171 edges)
    x_edge = torch.randn(1368, 16, 960).cuda()  # (B*171, C, L)

    # Capture any warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        output = edge_mamba(x_edge)

    if any("[MAMBA]" in str(warning.message) for warning in w):
        print("  ❌ FAILED - Still getting fallback warnings")
        for warning in w:
            if "[MAMBA]" in str(warning.message):
                print(f"     {warning.message}")
    else:
        print(f"  ✅ SUCCESS - No fallback! Output shape: {output.shape}")

except Exception as e:
    print(f"  ❌ FAILED: {e}")

# Test full V3 detector
print("\n3. Testing Full V3 Detector...")

from src.brain_brr.config.schemas import Config
from src.brain_brr.models.detector import SeizureDetector

try:
    # Create V3 config
    cfg = Config(
        model=Config.ModelConfig(
            architecture="v3",
            graph=Config.GraphConfig(
                enabled=True,
                edge_mamba_d_model=16,
                edge_mamba_d_state=8,
                edge_mamba_layers=2
            )
        )
    )

    # Create detector
    detector = SeizureDetector.from_config(cfg).cuda()

    # Test input (B=8, C=19, T=15360)
    x = torch.randn(8, 19, 15360).cuda()

    # Capture output and warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        output = detector(x)

    fallback_warnings = [warning for warning in w if "[MAMBA]" in str(warning.message)]

    if fallback_warnings:
        print("  ❌ FAILED - Still getting fallback warnings:")
        for warning in fallback_warnings[:3]:  # Show first 3
            print(f"     {warning.message}")
    else:
        print(f"  ✅ SUCCESS - Full V3 working! Output shape: {output.shape}")

except Exception as e:
    print(f"  ❌ FAILED: {e}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
If all tests pass with ✅:
- The headdim fix is working correctly
- No more Conv1d fallbacks
- Full Mamba2 state-space modeling is active

If any test fails with ❌:
- Check the error message for details
- Ensure CUDA is available and mamba-ssm is installed
""")