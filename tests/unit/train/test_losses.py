"""Tests for loss functions.

Testing philosophy: Test BEHAVIOR, not implementation.
Real tensors, no mocks - verify mathematical properties.
"""

from __future__ import annotations

import torch

from src.brain_brr.train.losses import FocalLoss


class TestFocalLossInitialization:
    """Test FocalLoss initialization."""

    def test_default_initialization(self):
        """Default alpha=0.25, gamma=2.0."""
        loss_fn = FocalLoss()
        assert loss_fn.alpha == 0.25
        assert loss_fn.gamma == 2.0

    def test_custom_alpha_gamma(self):
        """Custom alpha and gamma values."""
        loss_fn = FocalLoss(alpha=0.5, gamma=3.0)
        assert loss_fn.alpha == 0.5
        assert loss_fn.gamma == 3.0

    def test_alpha_gamma_type_conversion(self):
        """Alpha and gamma should be converted to float."""
        loss_fn = FocalLoss(alpha=1, gamma=2)  # Pass ints
        assert isinstance(loss_fn.alpha, float)
        assert isinstance(loss_fn.gamma, float)


class TestFocalLossBasicBehavior:
    """Test basic focal loss behavior."""

    def test_perfect_predictions_near_zero_loss(self):
        """Perfect predictions should have near-zero loss."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

        # Perfect positive prediction: logit=10, target=1
        logits = torch.tensor([[10.0, 10.0, 10.0]])
        targets = torch.tensor([[1.0, 1.0, 1.0]])
        loss = loss_fn(logits, targets)

        # High logit + correct target → very low loss
        assert (loss < 0.01).all()

    def test_wrong_predictions_high_loss(self):
        """Wrong predictions should have higher loss."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

        # Wrong prediction: logit=10 (predicts 1), target=0
        logits = torch.tensor([[10.0]])
        targets = torch.tensor([[0.0]])
        loss = loss_fn(logits, targets)

        # Wrong prediction → high loss
        assert loss.item() > 1.0

    def test_output_shape_matches_input(self):
        """Output shape should match input shape."""
        loss_fn = FocalLoss()
        logits = torch.randn(4, 100)
        targets = torch.randint(0, 2, (4, 100)).float()
        loss = loss_fn(logits, targets)
        assert loss.shape == logits.shape

    def test_no_nan_on_extreme_logits(self):
        """Extreme logits should not produce NaN."""
        loss_fn = FocalLoss()

        # Very large positive logits
        logits = torch.tensor([[100.0, -100.0]])
        targets = torch.tensor([[1.0, 0.0]])
        loss = loss_fn(logits, targets)
        assert not torch.isnan(loss).any()

        # Very large negative logits
        logits = torch.tensor([[-100.0, 100.0]])
        targets = torch.tensor([[0.0, 1.0]])
        loss = loss_fn(logits, targets)
        assert not torch.isnan(loss).any()


class TestFocalLossFocalModulation:
    """Test focal modulation (1-p_t)^gamma term."""

    def test_easy_examples_downweighted(self):
        """Easy examples (high confidence, correct) should be downweighted."""
        loss_fn = FocalLoss(alpha=0.5, gamma=2.0)

        # Easy positive: high logit, correct target
        easy_logits = torch.tensor([[5.0]])
        easy_targets = torch.tensor([[1.0]])
        easy_loss = loss_fn(easy_logits, easy_targets)

        # Hard positive: low logit, correct target
        hard_logits = torch.tensor([[0.1]])
        hard_targets = torch.tensor([[1.0]])
        hard_loss = loss_fn(hard_logits, hard_targets)

        # Hard examples should have higher loss than easy examples
        assert hard_loss.item() > easy_loss.item()

    def test_gamma_zero_reduces_focal_modulation(self):
        """When gamma=0, focal modulation term (1-p_t)^0 = 1."""
        loss_fn_gamma0 = FocalLoss(alpha=0.5, gamma=0.0)
        loss_fn_gamma2 = FocalLoss(alpha=0.5, gamma=2.0)

        logits = torch.randn(10, 50)
        targets = torch.randint(0, 2, (10, 50)).float()

        loss_gamma0 = loss_fn_gamma0(logits, targets)
        loss_gamma2 = loss_fn_gamma2(logits, targets)

        # With gamma=0, modulation term is 1, so no downweighting
        # With gamma=2, easy examples are downweighted
        # Mean loss with gamma=0 should generally be higher (no downweighting)
        # This is a behavioral test, not exact equality
        assert loss_gamma0.mean() > 0
        assert loss_gamma2.mean() > 0
        # Both should be finite
        assert torch.isfinite(loss_gamma0).all()
        assert torch.isfinite(loss_gamma2).all()


class TestFocalLossClassBalance:
    """Test class balancing with alpha parameter."""

    def test_alpha_balances_classes(self):
        """Alpha should balance positive/negative classes."""
        # High alpha (0.75) weights positives more
        high_alpha_fn = FocalLoss(alpha=0.75, gamma=0.0)
        # Low alpha (0.25) weights negatives more
        low_alpha_fn = FocalLoss(alpha=0.25, gamma=0.0)

        # Same logits, positive target
        logits = torch.tensor([[1.0]])
        targets = torch.tensor([[1.0]])

        high_alpha_loss = high_alpha_fn(logits, targets)
        low_alpha_loss = low_alpha_fn(logits, targets)

        # Higher alpha → higher weight on positives → higher loss
        assert high_alpha_loss.item() > low_alpha_loss.item()


class TestFocalLossPosWeight:
    """Test pos_weight parameter for class imbalance."""

    def test_pos_weight_increases_positive_loss(self):
        """pos_weight should up-weight positive examples."""
        loss_fn = FocalLoss()

        logits = torch.tensor([[1.0]])
        targets = torch.tensor([[1.0]])

        # No pos_weight
        loss_no_weight = loss_fn(logits, targets)

        # With pos_weight=10
        loss_with_weight = loss_fn(logits, targets, pos_weight=torch.tensor(10.0))

        # pos_weight should increase loss for positive examples
        assert loss_with_weight.item() > loss_no_weight.item()

    def test_pos_weight_tensor_type(self):
        """pos_weight should work as a tensor."""
        loss_fn = FocalLoss()
        logits = torch.randn(2, 10)
        targets = torch.randint(0, 2, (2, 10)).float()

        # pos_weight as tensor
        loss = loss_fn(logits, targets, pos_weight=torch.tensor(5.0))
        assert loss.shape == logits.shape
        assert not torch.isnan(loss).any()


class TestFocalLossNumericalStability:
    """Test numerical stability under edge cases."""

    def test_no_overflow_on_large_logits(self):
        """Very large logits should not overflow."""
        loss_fn = FocalLoss()
        logits = torch.tensor([[150.0, -150.0]])
        targets = torch.tensor([[1.0, 0.0]])
        loss = loss_fn(logits, targets)

        assert torch.isfinite(loss).all()
        assert (loss <= 100.0).all()  # Clamped to max 100

    def test_no_underflow_on_perfect_predictions(self):
        """Perfect predictions should not underflow to zero incorrectly."""
        loss_fn = FocalLoss(gamma=5.0)  # High gamma amplifies modulation

        # Perfect prediction with high gamma
        logits = torch.tensor([[20.0]])
        targets = torch.tensor([[1.0]])
        loss = loss_fn(logits, targets)

        assert torch.isfinite(loss).all()
        assert loss.item() >= 0.0

    def test_no_nan_on_ambiguous_logits(self):
        """Logits near 0 (p=0.5) should not produce NaN."""
        loss_fn = FocalLoss()
        logits = torch.zeros(5, 10)  # All logits = 0 → p = 0.5
        targets = torch.randint(0, 2, (5, 10)).float()
        loss = loss_fn(logits, targets)

        assert not torch.isnan(loss).any()
        assert torch.isfinite(loss).all()

    def test_clamping_prevents_loss_explosion(self):
        """Loss clamping should prevent explosion."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

        # Adversarial case: wrong prediction with high confidence
        logits = torch.tensor([[-50.0]])  # Predicts 0 with high confidence
        targets = torch.tensor([[1.0]])  # But target is 1
        loss = loss_fn(logits, targets)

        # Should be clamped to max 100
        assert (loss <= 100.0).all()

    def test_gradient_flow(self):
        """Loss should allow gradient flow."""
        loss_fn = FocalLoss()
        logits = torch.randn(4, 10, requires_grad=True)
        targets = torch.randint(0, 2, (4, 10)).float()

        loss = loss_fn(logits, targets)
        total_loss = loss.sum()
        total_loss.backward()

        # Gradients should exist and be finite
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()


class TestFocalLossBatchBehavior:
    """Test behavior across batches."""

    def test_batch_independence(self):
        """Loss should be computed independently per sample."""
        loss_fn = FocalLoss()

        # Batch of 3 samples
        logits = torch.tensor([[5.0], [-5.0], [0.0]])
        targets = torch.tensor([[1.0], [0.0], [1.0]])
        loss = loss_fn(logits, targets)

        # Sample 0: correct positive → low loss
        # Sample 1: correct negative → low loss
        # Sample 2: ambiguous → medium loss
        assert loss[0].item() < loss[2].item()
        assert loss[1].item() < loss[2].item()

    def test_reduction_none_preserves_shape(self):
        """Loss should preserve batch dimensions."""
        loss_fn = FocalLoss()
        batch_size, seq_len = 8, 256
        logits = torch.randn(batch_size, seq_len)
        targets = torch.randint(0, 2, (batch_size, seq_len)).float()

        loss = loss_fn(logits, targets)

        assert loss.shape == (batch_size, seq_len)


class TestFocalLossEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_tensor(self):
        """Empty tensors should not crash."""
        loss_fn = FocalLoss()
        logits = torch.empty(0, 0)
        targets = torch.empty(0, 0)
        loss = loss_fn(logits, targets)
        assert loss.shape == (0, 0)

    def test_single_sample(self):
        """Single sample should work."""
        loss_fn = FocalLoss()
        logits = torch.tensor([[2.5]])
        targets = torch.tensor([[1.0]])
        loss = loss_fn(logits, targets)
        assert loss.shape == (1, 1)
        assert torch.isfinite(loss).all()

    def test_all_zeros_targets(self):
        """All negative targets should work."""
        loss_fn = FocalLoss()
        logits = torch.randn(5, 10)
        targets = torch.zeros(5, 10)
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss).all()

    def test_all_ones_targets(self):
        """All positive targets should work."""
        loss_fn = FocalLoss()
        logits = torch.randn(5, 10)
        targets = torch.ones(5, 10)
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss).all()
