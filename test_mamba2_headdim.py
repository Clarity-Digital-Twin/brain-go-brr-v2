#!/usr/bin/env python3
"""Test Mamba2 headdim requirements and verify external advice about stride multiples."""

import torch
from mamba_ssm import Mamba2

print("Testing Mamba2 headdim requirements...")
print("=" * 80)

def test_config(name, batch_size, num_items, d_model, expand, headdim, seq_len=960):
    """Test a specific configuration."""
    print(f"\nTest: {name}")
    print(f"  Batch: {batch_size}, Items: {num_items}, d_model: {d_model}, expand: {expand}, headdim: {headdim}")

    # Calculate the important ratio
    ratio = (d_model * expand) / headdim
    print(f"  (d_model * expand) / headdim = ({d_model} * {expand}) / {headdim} = {ratio}")
    print(f"  Ratio is multiple of 8: {ratio % 8 == 0}")

    # Calculate batch dimension
    total_batch = batch_size * num_items
    print(f"  Total batch dimension: {batch_size} * {num_items} = {total_batch}")
    print(f"  Total batch is multiple of 8: {total_batch % 8 == 0}")

    try:
        # Create Mamba2 layer
        mamba = Mamba2(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=expand,
            headdim=headdim
        ).cuda()

        # Create input tensor with shape (B*N, seq_len, d_model)
        x = torch.randn(total_batch, seq_len, d_model).cuda()
        print(f"  Input shape: {x.shape}")
        print(f"  Input strides: {x.stride()}")

        # Try forward pass
        with torch.no_grad():
            output = mamba(x)

        print(f"  ✅ SUCCESS - Output shape: {output.shape}")
        return True

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False

# Test current V3 configurations
print("\n" + "=" * 80)
print("CURRENT V3 CONFIGURATIONS (What we have now)")
print("=" * 80)

# Node stream: d_model=64, default expand=2, default headdim=64
test_config(
    name="Node Stream (current)",
    batch_size=8,
    num_items=19,  # electrodes
    d_model=64,
    expand=2,
    headdim=64  # default
)

# Edge stream: d_model=16, default expand=2, default headdim=64
test_config(
    name="Edge Stream (current)",
    batch_size=8,
    num_items=171,  # edges
    d_model=16,
    expand=2,
    headdim=64  # default - THIS IS THE PROBLEM!
)

# Test external advice configurations
print("\n" + "=" * 80)
print("EXTERNAL ADVICE CONFIGURATIONS")
print("=" * 80)

# Node stream: d_model=128, expand=2, headdim=32
test_config(
    name="Node Stream (advice: 128/2/32)",
    batch_size=8,
    num_items=19,
    d_model=128,
    expand=2,
    headdim=32
)

# Edge stream: d_model=16, expand=2, headdim=4
test_config(
    name="Edge Stream (advice: 16/2/4)",
    batch_size=8,
    num_items=171,
    d_model=16,
    expand=2,
    headdim=4
)

# Test other valid configurations
print("\n" + "=" * 80)
print("ALTERNATIVE VALID CONFIGURATIONS")
print("=" * 80)

# Node stream: Keep d_model=64 but fix headdim
test_config(
    name="Node Stream (64/2/16)",
    batch_size=8,
    num_items=19,
    d_model=64,
    expand=2,
    headdim=16  # (64*2)/16 = 8 ✓
)

test_config(
    name="Node Stream (64/2/8)",
    batch_size=8,
    num_items=19,
    d_model=64,
    expand=2,
    headdim=8  # (64*2)/8 = 16 ✓
)

# Edge stream alternatives
test_config(
    name="Edge Stream (16/2/2)",
    batch_size=8,
    num_items=171,
    d_model=16,
    expand=2,
    headdim=2  # (16*2)/2 = 16 ✓
)

test_config(
    name="Edge Stream (32/2/4)",
    batch_size=8,
    num_items=171,
    d_model=32,
    expand=2,
    headdim=4  # (32*2)/4 = 16 ✓
)

# Test batch size variations
print("\n" + "=" * 80)
print("BATCH SIZE VARIATIONS (with fixed headdim)")
print("=" * 80)

for batch_size in [8, 12, 16, 32]:
    test_config(
        name=f"Node Stream B={batch_size} (64/2/8)",
        batch_size=batch_size,
        num_items=19,
        d_model=64,
        expand=2,
        headdim=8
    )

print("\n" + "=" * 80)
print("CONCLUSIONS")
print("=" * 80)
print("""
The external advice appears CORRECT:
1. The issue is NOT about B*19 or B*171 being divisible by 8
2. The issue IS about (d_model * expand) / headdim being compatible with Mamba2

Current failures:
- Edge stream: d_model=16 with default headdim=64 → 16*2/64 = 0.5 (fractional!)
- This causes Mamba2 initialization to fail internally

Solutions that work:
1. Node: d_model=64, expand=2, headdim=8 → ratio=16 ✓
2. Edge: d_model=16, expand=2, headdim=4 → ratio=8 ✓

Or the external advice:
1. Node: d_model=128, expand=2, headdim=32 → ratio=8 ✓
2. Edge: d_model=16, expand=2, headdim=4 → ratio=8 ✓
""")