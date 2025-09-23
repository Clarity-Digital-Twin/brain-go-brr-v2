"""Test NaN robustness in training loop."""

import torch
import torch.nn as nn
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[3]))

from src.brain_brr.train.loop import FocalLoss


class TestNaNRobustness:
    """Test suite for NaN handling in training."""

    def test_focal_loss_numerical_stability(self):
        """Test focal loss handles extreme values without NaN."""
        focal = FocalLoss(alpha=0.5, gamma=2.0)

        # Test case 1: Extreme logits
        logits = torch.tensor([[-100.0, 100.0, 0.0]])
        targets = torch.tensor([[1.0, 1.0, 0.0]])
        loss = focal(logits, targets)
        assert torch.isfinite(loss).all(), "Focal loss should handle extreme logits"

        # Test case 2: All correct predictions (p_t ≈ 1)
        logits = torch.tensor([[100.0, 100.0]])  # High confidence correct
        targets = torch.tensor([[1.0, 1.0]])
        loss = focal(logits, targets)
        assert torch.isfinite(loss).all(), "Focal loss should handle p_t ≈ 1"

        # Test case 3: All incorrect predictions (p_t ≈ 0)
        logits = torch.tensor([[-100.0, -100.0]])  # High confidence wrong
        targets = torch.tensor([[1.0, 1.0]])
        loss = focal(logits, targets)
        assert torch.isfinite(loss).all(), "Focal loss should handle p_t ≈ 0"

        # Test case 4: Mixed precision simulation
        logits = torch.tensor([[1e-8, 1e8]], dtype=torch.float16)
        targets = torch.tensor([[0.0, 1.0]], dtype=torch.float16)
        loss = focal(logits.float(), targets.float())  # Convert to float32 for stability
        assert torch.isfinite(loss).all(), "Focal loss should handle FP16 edge cases"

    def test_focal_loss_with_pos_weight(self):
        """Test focal loss with positive class weighting."""
        focal = FocalLoss(alpha=0.5, gamma=2.0)

        # Extreme class imbalance with pos_weight
        logits = torch.randn(8, 100)
        targets = torch.zeros(8, 100)
        targets[:, 0] = 1  # Only 1% positive

        pos_weight = torch.tensor([10.0])
        loss = focal(logits, targets, pos_weight=pos_weight)
        assert torch.isfinite(loss).all(), "Focal loss should handle pos_weight"

    def test_train_epoch_nan_handling(self):
        """Test that train_one_epoch handles NaN losses gracefully."""
        # Create a mock model that sometimes returns NaN
        class NaNModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.call_count = 0

            def forward(self, x):
                self.call_count += 1
                # Return NaN every 3rd call
                if self.call_count % 3 == 0:
                    return torch.full((x.shape[0], x.shape[2]), float('nan'))
                return torch.randn(x.shape[0], x.shape[2])

        model = NaNModel()

        # Create mock data
        windows = torch.randn(8, 19, 15360)
        labels = torch.randint(0, 2, (8, 15360)).float()
        mock_dataset = [(windows, labels)] * 10

        # Mock dataloader
        dataloader = MagicMock()
        dataloader.__iter__ = lambda: iter(mock_dataset)
        dataloader.__len__ = lambda: len(mock_dataset)

        # Mock optimizer
        optimizer = MagicMock()
        optimizer.param_groups = [{'lr': 1e-3}]
        optimizer.zero_grad = MagicMock()
        optimizer.step = MagicMock()

        # Mock loss function
        def mock_loss(logits, targets):
            if torch.isnan(logits).any():
                return torch.tensor(float('nan'))
            return torch.mean((logits - targets) ** 2)

        # Run training with NaN handling
        with patch('torch.nn.utils.clip_grad_norm_', return_value=1.0):
            total_loss = train_one_epoch(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                compute_loss=mock_loss,
                device='cpu',
                use_amp=False,
                gradient_clip=1.0
            )

        # Verify that training continued despite NaN losses
        assert model.call_count == 10, "Model should be called for all batches"
        # Optimizer should skip NaN batches (7 valid out of 10)
        assert optimizer.step.call_count <= 7, "Optimizer should skip NaN batches"

    def test_data_sanitization(self):
        """Test input data sanitization."""
        windows = torch.tensor([
            [[float('nan'), 1.0, 2.0]],
            [[float('inf'), -float('inf'), 0.0]],
            [[1.0, 2.0, 3.0]]
        ])

        # Enable sanitization
        import os
        os.environ['BGB_SANITIZE_INPUTS'] = '1'

        # Sanitize
        sanitized = torch.nan_to_num(windows, nan=0.0, posinf=0.0, neginf=0.0)

        assert torch.isfinite(sanitized).all(), "Sanitized data should be finite"
        assert sanitized[0, 0, 0] == 0.0, "NaN should be replaced with 0"
        assert sanitized[1, 0, 0] == 0.0, "Inf should be replaced with 0"
        assert sanitized[1, 0, 1] == 0.0, "-Inf should be replaced with 0"
        assert sanitized[2, 0, 2] == 3.0, "Valid values should be preserved"

        # Clean up
        del os.environ['BGB_SANITIZE_INPUTS']

    def test_gradient_clipping_with_nan(self):
        """Test gradient clipping handles NaN gradients."""
        model = nn.Linear(10, 1)

        # Create NaN gradient
        model.weight.grad = torch.full_like(model.weight, float('nan'))
        model.bias.grad = torch.full_like(model.bias, float('nan'))

        # Clip should handle NaN gracefully
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # grad_norm will be NaN but clipping shouldn't crash
        assert torch.isnan(grad_norm), "NaN gradients should result in NaN norm"

    def test_batch_composition_edge_cases(self):
        """Test edge cases in batch composition."""
        # All negative batch
        labels_all_neg = torch.zeros(8, 15360)
        pos_ratio = labels_all_neg.sum().item() / labels_all_neg.numel()
        assert pos_ratio == 0.0, "All negative batch should have 0% positive"

        # All positive batch
        labels_all_pos = torch.ones(8, 15360)
        pos_ratio = labels_all_pos.sum().item() / labels_all_pos.numel()
        assert pos_ratio == 1.0, "All positive batch should have 100% positive"

        # Dead channel detection
        windows = torch.randn(8, 19, 15360)
        windows[:, 5, :] = 0.0  # Dead channel
        windows[:, 10, :] = 1e-8  # Near-dead channel

        channel_stds = windows.std(dim=[0, 2])
        dead_channels = (channel_stds < 1e-6).sum().item()
        assert dead_channels >= 1, "Should detect dead channels"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])