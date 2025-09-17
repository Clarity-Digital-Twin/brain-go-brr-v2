#!/usr/bin/env python
"""Manual test of full model forward pass and basic stats.

Run: python scripts/test_model.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch


def main() -> None:
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.experiment.models import SeizureDetector  # late import

    print("=" * 60)
    print("SeizureDetector Full Model Test")
    print("=" * 60)

    model = SeizureDetector()
    model.eval()

    info = model.get_layer_info()
    print("\nModel Statistics:")
    print(f"  Encoder params:  {info['encoder_params']:,}")
    print(f"  ResCNN params:   {info['rescnn_params']:,}")
    print(f"  Mamba params:    {info['mamba_params']:,}")
    print(f"  Decoder params:  {info['decoder_params']:,}")
    print(f"  Head params:     {info['head_params']:,}")
    print(f"  Total params:    {info['total_params']:,}")

    for bs in [1, 8, 16, 32]:
        mem = model.get_memory_usage(bs)
        print(f"\nBatch size {bs}:")
        print(f"  Model:      {mem['model_size_mb']:.1f} MB")
        print(f"  Activation: {mem['activation_size_mb']:.1f} MB")
        print(f"  Total:      {mem['total_size_mb']:.1f} MB")

    print("\nTesting forward pass...")
    for bs in [1, 4, 8, 16]:
        x = torch.randn(bs, 19, 15360)
        start = time.perf_counter()
        with torch.no_grad():
            y = model(x)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        print(f"  Batch {bs:2d}: {tuple(x.shape)} -> {tuple(y.shape)}  ({elapsed_ms:.1f} ms)")
        assert y.shape == (bs, 15360)
        assert torch.all(y >= 0) and torch.all(y <= 1)
        assert not torch.isnan(y).any()

    if torch.cuda.is_available():
        print("\nGPU sanity test...")
        model = model.cuda()
        x = torch.randn(16, 19, 15360, device="cuda")
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        print(f"  GPU forward (batch 16): {elapsed_ms:.1f} ms")

    print("\nDone.")


if __name__ == "__main__":
    main()

