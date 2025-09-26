#!/usr/bin/env python3
"""Debug script to test model forward pass and find where infinities arise."""

import torch
import torch.nn as nn
from src.brain_brr.models import SeizureDetector
from src.brain_brr.config.schemas import ModelConfig

# Set up model
model = SeizureDetector.from_config(ModelConfig())
model.eval()

# Create input
batch_size = 4
n_channels = 19
window_samples = 15360
x = torch.randn(batch_size, n_channels, window_samples)

print(f"Input shape: {x.shape}")
print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")

# Hook to capture intermediate values
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
        print(f"{name}: shape={output.shape}, range=[{output.min():.3f}, {output.max():.3f}], has_inf={torch.isinf(output).any()}, has_nan={torch.isnan(output).any()}")
    return hook

# Register hooks
model.tcn_encoder.register_forward_hook(get_activation('tcn_encoder'))
if model.proj_to_electrodes is not None:
    model.proj_to_electrodes.register_forward_hook(get_activation('proj_to_electrodes'))
if model.node_mamba is not None:
    model.node_mamba.register_forward_hook(get_activation('node_mamba'))
if model.edge_mamba is not None:
    model.edge_mamba.register_forward_hook(get_activation('edge_mamba'))
if model.gnn is not None:
    model.gnn.register_forward_hook(get_activation('gnn'))
if model.proj_from_electrodes is not None:
    model.proj_from_electrodes.register_forward_hook(get_activation('proj_from_electrodes'))
model.proj_head.register_forward_hook(get_activation('proj_head'))
model.detection_head.register_forward_hook(get_activation('detection_head'))

# Forward pass
with torch.no_grad():
    output = model(x)

print(f"\nFinal output: shape={output.shape}, range=[{output.min():.3f}, {output.max():.3f}]")
print(f"Has inf: {torch.isinf(output).any()}")
print(f"Has nan: {torch.isnan(output).any()}")
print(f"Num inf: {torch.isinf(output).sum()}")
print(f"Num nan: {torch.isnan(output).sum()}")

# Check detection_head weights
print(f"\nDetection head weight range: [{model.detection_head.weight.min():.6f}, {model.detection_head.weight.max():.6f}]")
print(f"Detection head bias: {model.detection_head.bias}")