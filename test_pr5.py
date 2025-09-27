#!/usr/bin/env python
"""Test PR-5 changes with minimal model."""

import torch
import torch.nn as nn
from src.brain_brr.config.schemas import ModelConfig
from src.brain_brr.models.detector import SeizureDetector

# Create minimal config
cfg = ModelConfig(
    architecture="v3",
    tcn={
        "num_layers": 8,
        "kernel_size": 7,
        "stride_down": 16,
        "dropout": 0.15,
    },
    mamba={
        "n_layers": 6,
        "d_model": 512,
        "d_state": 16,
        "conv_kernel": 4,
        "dropout": 0.1,
    },
    graph={
        "enabled": True,
        "edge_features": "cosine",
        "edge_top_k": 3,
        "edge_threshold": 1e-4,
        "edge_mamba_layers": 2,
        "edge_mamba_d_state": 8,
        "edge_mamba_d_model": 16,
        # PR-2 enabled
        "edge_lift_activation": "tanh",
        "edge_lift_norm": "layernorm",
        "edge_lift_init_gain": 0.1,
        # PR-3 enabled
        "adj_row_softmax": True,
        "adj_softmax_tau": 1.0,
        "adj_ema_beta": 0.9,
        "adj_force_symmetric": True,
        "laplacian_eps": 1e-3,
        "laplacian_normalize": True,
        # Standard graph config
        "n_layers": 2,
        "dropout": 0.1,
        "use_residual": True,
        "alpha": 0.05,
        "k_eigenvectors": 16,
        "use_dynamic_pe": True,
        "semi_dynamic_interval": 10,
        "pe_sign_consistency": True,
    },
    norms={
        # PR-1 enabled
        "boundary_norm": "layernorm",
        "boundary_eps": 1e-5,
        "layerscale_alpha": 0.1,
        "after_tcn_proj": True,
        "after_node_mamba": True,
        "after_edge_mamba": True,
        "after_gnn": True,
        "before_decoder": True,
    },
    fusion={
        # PR-4 enabled
        "fusion_type": "gated",
        "fusion_heads": 4,
        "fusion_dropout": 0.1,
    },
)

print("Creating model...")
model = SeizureDetector.from_config(cfg)

if torch.cuda.is_available():
    print("Moving to GPU...")
    model = model.cuda()

    # Test with larger batch
    print("Testing forward pass with batch_size=8...")
    x = torch.randn(8, 19, 15360, device="cuda")

    with torch.no_grad():
        try:
            output = model(x)
            print(f"✅ Forward pass succeeded! Output shape: {output.shape}")
            print(f"   Output range: [{output.min().item():.2f}, {output.max().item():.2f}]")

            # Check for NaN/Inf
            if torch.isnan(output).any():
                print("❌ WARNING: Output contains NaN!")
            elif torch.isinf(output).any():
                print("❌ WARNING: Output contains Inf!")
            else:
                print("✅ Output is finite")

        except Exception as e:
            print(f"❌ Forward pass failed: {e}")

    # Check memory usage
    print(f"\nGPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
else:
    print("No GPU available")