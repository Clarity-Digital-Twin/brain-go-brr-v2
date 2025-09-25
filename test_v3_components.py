#!/usr/bin/env python
"""Test V3 components in isolation to find where NaNs originate."""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.brain_brr.config.schemas import load_config
from src.brain_brr.models.detector import SeizureDetector


def test_component(name: str, module: nn.Module, input_tensor: torch.Tensor):
    """Test a single component for NaN/Inf."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Input stats: mean={input_tensor.mean():.3f}, std={input_tensor.std():.3f}")

    with torch.no_grad():
        try:
            output = module(input_tensor)

            has_nan = torch.isnan(output).any().item()
            has_inf = torch.isinf(output).any().item()

            print(f"Output shape: {output.shape}")
            print(f"Output stats: mean={output.mean():.3f}, std={output.std():.3f}")
            print(f"Max value: {output.max():.3f}, Min value: {output.min():.3f}")

            if has_nan:
                print(f"❌ FAILURE: Output contains NaN!")
                nan_count = torch.isnan(output).sum().item()
                print(f"   NaN count: {nan_count}/{output.numel()} ({100*nan_count/output.numel():.1f}%)")
            elif has_inf:
                print(f"❌ FAILURE: Output contains Inf!")
                inf_count = torch.isinf(output).sum().item()
                print(f"   Inf count: {inf_count}/{output.numel()}")
            else:
                print(f"✅ PASS: No NaN/Inf detected")

            return output, not (has_nan or has_inf)

        except Exception as e:
            print(f"❌ EXCEPTION: {e}")
            return None, False


def main():
    """Test V3 components step by step."""

    # Load config
    cfg = load_config("configs/local/train.yaml")

    # Force V3
    cfg.model.architecture = "v3"

    print("="*60)
    print("V3 COMPONENT TESTING")
    print("="*60)

    # Create model
    print("\n1. Creating V3 model...")
    model = SeizureDetector.from_config(cfg.model)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        device = "cuda"
    else:
        device = "cpu"

    print(f"Model created successfully on {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test input
    batch_size = 2  # Small batch for testing
    input_tensor = torch.randn(batch_size, 19, 15360, device=device)

    # Test TCN
    tcn_out, tcn_ok = test_component(
        "TCN Encoder",
        model.tcn_encoder,
        input_tensor
    )

    if not tcn_ok:
        print("\n⚠️  TCN FAILED - Cannot continue")
        return

    # Test projection to electrodes
    if hasattr(model, 'proj_to_electrodes'):
        proj_out, proj_ok = test_component(
            "Projection to Electrodes",
            model.proj_to_electrodes,
            tcn_out
        )

        if not proj_ok:
            print("\n⚠️  Projection FAILED")
            return

        # Reshape for node/edge streams
        B, C, T = proj_out.shape
        elec_feats = proj_out.reshape(B, 19, 64, T).permute(0, 1, 3, 2)

        # Test Node Mamba
        if hasattr(model, 'node_mamba'):
            node_input = elec_feats.permute(0, 1, 3, 2).reshape(B * 19, 64, T)
            node_out, node_ok = test_component(
                "Node Mamba Stream",
                model.node_mamba,
                node_input
            )

            if not node_ok:
                print("\n⚠️  Node Mamba FAILED")

        # Test Edge features
        if hasattr(model, 'edge_in_proj'):
            from src.brain_brr.models.edge_features import edge_scalar_series

            edge_feats = edge_scalar_series(elec_feats, metric="cosine")
            print(f"\nEdge features shape: {edge_feats.shape}")
            print(f"Edge features stats: mean={edge_feats.mean():.3f}, std={edge_feats.std():.3f}")

            # Test edge projection
            edge_flat = edge_feats.squeeze(-1).permute(0, 2, 1).reshape(
                B * T, 171, 1
            )
            edge_lifted, edge_proj_ok = test_component(
                "Edge Input Projection",
                model.edge_in_proj,
                edge_flat
            )

            if not edge_proj_ok:
                print("\n⚠️  Edge Projection FAILED - THIS IS LIKELY THE PROBLEM!")

            # Test Edge Mamba
            if hasattr(model, 'edge_mamba') and edge_proj_ok:
                edge_out, edge_ok = test_component(
                    "Edge Mamba Stream",
                    model.edge_mamba,
                    edge_lifted.transpose(1, 2)
                )

                if not edge_ok:
                    print("\n⚠️  Edge Mamba FAILED")

    # Test full forward pass
    print("\n" + "="*60)
    print("FULL FORWARD PASS TEST")
    full_input = torch.randn(batch_size, 19, 15360, device=device)

    with torch.no_grad():
        try:
            output = model(full_input)

            has_nan = torch.isnan(output).any().item()
            has_inf = torch.isinf(output).any().item()

            print(f"Final output shape: {output.shape}")
            print(f"Final output stats: mean={output.mean():.3f}, std={output.std():.3f}")

            if has_nan or has_inf:
                print(f"\n❌ V3 FORWARD PASS FAILED")
                if has_nan:
                    print(f"   Contains {torch.isnan(output).sum().item()} NaN values")
                if has_inf:
                    print(f"   Contains {torch.isinf(output).sum().item()} Inf values")
            else:
                print(f"\n✅ V3 FORWARD PASS SUCCESSFUL!")

        except Exception as e:
            print(f"\n❌ V3 FORWARD PASS EXCEPTION: {e}")

    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()