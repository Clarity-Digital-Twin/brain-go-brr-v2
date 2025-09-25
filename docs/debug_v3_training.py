#!/usr/bin/env python
"""Debug script to identify where V3 produces NaNs during training."""

import torch
import yaml
import sys
from pathlib import Path
from torch.utils.data import DataLoader

# Add hooks to track activations
activation_stats = {}

def register_hooks(model):
    """Register forward hooks to track activations."""
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                has_nan = torch.isnan(output).any().item()
                has_inf = torch.isinf(output).any().item()
                if has_nan or has_inf:
                    print(f"  ❌ NaN/Inf detected in {name}: NaN={has_nan}, Inf={has_inf}")
                    activation_stats[name] = {
                        'has_nan': has_nan,
                        'has_inf': has_inf,
                        'shape': output.shape,
                        'mean': output[~torch.isnan(output)].mean().item() if not output.isnan().all() else float('nan'),
                        'std': output[~torch.isnan(output)].std().item() if not output.isnan().all() else float('nan')
                    }
        return hook

    # Register hooks on key V3 components
    if hasattr(model, 'tcn_encoder'):
        model.tcn_encoder.register_forward_hook(make_hook('tcn_encoder'))
    if hasattr(model, 'node_mamba'):
        model.node_mamba.register_forward_hook(make_hook('node_mamba'))
    if hasattr(model, 'edge_in_proj'):
        model.edge_in_proj.register_forward_hook(make_hook('edge_in_proj'))
    if hasattr(model, 'edge_mamba'):
        model.edge_mamba.register_forward_hook(make_hook('edge_mamba'))
    if hasattr(model, 'gnn'):
        model.gnn.register_forward_hook(make_hook('gnn'))
    if hasattr(model, 'proj_from_electrodes'):
        model.proj_from_electrodes.register_forward_hook(make_hook('proj_from_electrodes'))
    if hasattr(model, 'decoder'):
        model.decoder.register_forward_hook(make_hook('decoder'))

def main():
    print("="*60)
    print("V3 DIAGNOSTIC TRAINING TEST")
    print("="*60)

    # Load config
    with open('configs/local/train.yaml') as f:
        cfg_dict = yaml.safe_load(f)

    from src.brain_brr.config.schemas import Config
    config = Config(**cfg_dict)

    print(f"Architecture: {config.model.architecture}")
    print(f"Dynamic PE: {config.model.graph.use_dynamic_pe}")
    print(f"Edge Mamba layers: {config.model.graph.edge_mamba_layers}")

    # Create model
    print("\nCreating model...")
    from src.brain_brr.models.detector import SeizureDetector
    model = SeizureDetector.from_config(config.model)
    model.train()

    if torch.cuda.is_available():
        model = model.cuda()
        device = 'cuda'
    else:
        device = 'cpu'

    print(f"Model on {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Register hooks
    register_hooks(model)
    print("Registered forward hooks for debugging")

    # Create dataset
    print("\nCreating dataset...")
    from src.brain_brr.data.datasets import BalancedSeizureDataset
    train_dataset = BalancedSeizureDataset(
        cache_dir=Path('cache/tusz/train'),
        full_ratio=0.3,
        background_ratio=2.5,
        seed=42
    )

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=config.data.pin_memory,
        drop_last=True
    )

    print(f"Dataset: {len(train_dataset)} windows")
    print(f"DataLoader: {len(train_loader)} batches")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )

    # Loss function
    loss_fn = torch.nn.BCEWithLogitsLoss()

    print("\nStarting training loop...")
    print("="*60)

    # Train for a few batches
    for batch_idx, (windows, labels) in enumerate(train_loader):
        if batch_idx >= 5:
            break

        windows = windows.to(device)
        labels = labels.to(device)

        if labels.dim() == 3:
            labels = labels.max(dim=1)[0]

        print(f"\nBatch {batch_idx}:")
        print(f"  Input: shape={windows.shape}, mean={windows.mean():.3f}, std={windows.std():.3f}")

        # Clear activation stats
        activation_stats.clear()

        # Forward pass
        optimizer.zero_grad()
        logits = model(windows)

        # Check output
        has_nan = torch.isnan(logits).any().item()
        has_inf = torch.isinf(logits).any().item()

        if has_nan or has_inf:
            print(f"  ❌ FAILURE: Output has NaN={has_nan}, Inf={has_inf}")
            print(f"     NaN locations: {torch.isnan(logits).sum().item()}/{logits.numel()}")

            # Print activation stats
            if activation_stats:
                print("\n  Activation statistics at failure:")
                for name, stats in activation_stats.items():
                    print(f"    {name}: {stats}")

            # Save the problematic batch and model state
            torch.save({
                'batch_idx': batch_idx,
                'windows': windows.cpu(),
                'labels': labels.cpu(),
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'activation_stats': activation_stats
            }, f'debug_failure_batch_{batch_idx}.pt')

            print(f"\n  Saved debug info to debug_failure_batch_{batch_idx}.pt")
            break
        else:
            print(f"  ✅ Forward OK: mean={logits.mean():.3f}, std={logits.std():.3f}")

        # Loss and backward
        loss = loss_fn(logits, labels)

        if torch.isnan(loss):
            print(f"  ❌ Loss is NaN!")
            break

        print(f"  Loss: {loss.item():.4f}")

        loss.backward()

        # Check gradients
        max_grad = 0
        nan_grad_params = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_grad_params.append(name)
                else:
                    max_grad = max(max_grad, param.grad.abs().max().item())

        if nan_grad_params:
            print(f"  ❌ NaN gradients in: {nan_grad_params[:3]}")
            break

        print(f"  Max gradient: {max_grad:.4f}")

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)

        # Optimizer step
        optimizer.step()

        # Check parameters
        nan_params = []
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                nan_params.append(name)

        if nan_params:
            print(f"  ❌ NaN parameters after update: {nan_params[:3]}")
            break

    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")

    if batch_idx == 4:
        print("✅ All 5 batches completed successfully!")
    else:
        print(f"❌ Failed at batch {batch_idx}")

if __name__ == "__main__":
    main()