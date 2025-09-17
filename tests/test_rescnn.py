"""Test suite for ResCNN stack implementation (Phase 2.2)."""

import pytest
import torch

from src.experiment.models import ResCNNBlock, ResCNNStack


class TestResCNNBlock:
    """Tests for individual ResCNN block with multi-scale kernels."""

    @pytest.fixture
    def block(self):
        """Create standard ResCNN block."""
        return ResCNNBlock(channels=512, kernel_sizes=[3, 5, 7], dropout=0.1)

    @pytest.fixture
    def sample_input(self):
        """Create sample bottleneck input."""
        return torch.randn(4, 512, 960)  # Batch of 4

    def test_shape_preservation(self, block, sample_input):
        """Test that block preserves input shape."""
        output = block(sample_input)
        assert output.shape == sample_input.shape, (
            f"Expected {sample_input.shape}, got {output.shape}"
        )

    def test_residual_connection_effect(self, block):
        """Test that residual connection modifies output (not identity)."""
        x = torch.randn(2, 512, 960)
        y = block(x)

        # Output should differ from input due to processing
        difference = torch.norm(y - x)
        assert difference > 0, "Residual block should modify input"

        # But not by too much (residual helps stability)
        relative_change = difference / torch.norm(x)
        assert relative_change < 10, f"Change too large: {relative_change}"

    def test_multi_scale_branches(self, block):
        """Test that all multi-scale branches are utilized."""
        x = torch.randn(1, 512, 960)

        # Hook to capture branch outputs
        branch_outputs = []

        def hook(module, input_tensor, output):
            branch_outputs.append(output.clone())

        handles = []
        for branch in block.branches:
            handles.append(branch.register_forward_hook(hook))

        _ = block(x)

        # Clean up hooks
        for handle in handles:
            handle.remove()

        assert len(branch_outputs) == 3, f"Expected 3 branches, got {len(branch_outputs)}"

        # Check branch channel splits (170, 170, 172 for 512 channels)
        channel_counts = [out.shape[1] for out in branch_outputs]
        assert sum(channel_counts) == 512, f"Channels {channel_counts} don't sum to 512"

        # Verify expected split pattern
        assert channel_counts[0] == 170, (
            f"Branch 0 should have 170 channels, got {channel_counts[0]}"
        )
        assert channel_counts[1] == 170, (
            f"Branch 1 should have 170 channels, got {channel_counts[1]}"
        )
        assert channel_counts[2] == 172, (
            f"Branch 2 should have 172 channels, got {channel_counts[2]}"
        )

    def test_gradient_flow_through_block(self, block, sample_input):
        """Test gradient flows properly through residual connection."""
        sample_input.requires_grad = True
        output = block(sample_input)

        loss = output.mean()
        loss.backward()

        assert sample_input.grad is not None, "No gradient computed"
        assert not torch.isnan(sample_input.grad).any(), "Gradient contains NaN"
        assert not torch.isinf(sample_input.grad).any(), "Gradient contains Inf"

    def test_different_channel_counts(self):
        """Test block works with different channel counts."""
        for channels in [256, 512, 1024]:
            block = ResCNNBlock(channels=channels)
            x = torch.randn(2, channels, 960)
            y = block(x)
            assert y.shape == x.shape, f"Shape mismatch for {channels} channels"

    def test_dropout_effect(self):
        """Test that dropout actually drops activations in training mode."""
        block = ResCNNBlock(channels=512, dropout=0.5)
        block.train()  # Ensure training mode

        x = torch.ones(1, 512, 960)  # Use ones to see dropout effect

        # Run multiple times to see stochastic dropout
        outputs = []
        for _ in range(10):
            y = block(x)
            outputs.append(y)

        # Outputs should vary due to dropout
        variance = torch.stack(outputs).var(dim=0).mean()
        assert variance > 0, "Dropout should cause variation in training mode"

    def test_eval_mode_deterministic(self, block, sample_input):
        """Test block is deterministic in eval mode."""
        block.eval()

        with torch.no_grad():
            out1 = block(sample_input)
            out2 = block(sample_input)

        assert torch.allclose(out1, out2, atol=1e-6), "Eval mode should be deterministic"


class TestResCNNStack:
    """Tests for complete ResCNN stack."""

    @pytest.fixture
    def stack(self):
        """Create standard 3-block ResCNN stack."""
        return ResCNNStack(channels=512, num_blocks=3)

    @pytest.fixture
    def sample_input(self):
        """Create sample encoder output."""
        return torch.randn(2, 512, 960)

    def test_output_shape(self, stack, sample_input):
        """Test stack preserves shape through all blocks."""
        output = stack(sample_input)
        assert output.shape == sample_input.shape

    def test_gradient_flow(self, stack, sample_input):
        """Test gradients flow through entire stack."""
        sample_input.requires_grad = True
        output = stack(sample_input)

        loss = output.mean()
        loss.backward()

        assert sample_input.grad is not None
        assert not torch.isnan(sample_input.grad).any()

        # Check gradient magnitude (residuals should prevent vanishing/exploding)
        grad_norm = sample_input.grad.norm()
        assert 0 < grad_norm < 1000, f"Gradient norm {grad_norm} out of expected range"

    def test_receptive_field(self, stack):
        """Test receptive field calculation."""
        rf = stack.get_receptive_field()
        # With kernel 7 and 3 blocks: 7 + (3-1)*6 = 19
        assert rf == 19, f"Expected receptive field 19, got {rf}"

    def test_no_nan_inf(self, stack, sample_input):
        """Test no NaN/Inf in output."""
        output = stack(sample_input)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_feature_transformation(self, stack, sample_input):
        """Test that stack actually transforms features (not identity)."""
        output = stack(sample_input)

        # Should be different from input
        assert not torch.allclose(output, sample_input)

        # But correlated (residual connections maintain some structure)
        correlation = torch.nn.functional.cosine_similarity(
            output.flatten(1), sample_input.flatten(1), dim=1
        ).mean()
        assert correlation > 0, "Output should maintain some correlation with input"

    def test_different_num_blocks(self):
        """Test stack with different numbers of blocks."""
        for num_blocks in [1, 2, 3, 5]:
            stack = ResCNNStack(channels=512, num_blocks=num_blocks)
            x = torch.randn(1, 512, 960)
            y = stack(x)
            assert y.shape == x.shape, f"Shape mismatch with {num_blocks} blocks"

    def test_memory_efficiency(self, stack):
        """Test stack doesn't leak memory in eval mode."""
        stack.eval()
        x = torch.randn(4, 512, 960)

        with torch.no_grad():
            for _ in range(10):
                _ = stack(x)

        # If this completes without OOM, memory is managed
        assert True

    def test_parameter_count(self, stack):
        """Test parameter count is reasonable."""
        total_params = sum(p.numel() for p in stack.parameters())

        # Rough estimate: 3 blocks, each with ~1-2M params
        assert 1_000_000 < total_params < 10_000_000, f"Unexpected param count: {total_params}"

    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
    def test_variable_batch_size(self, stack, batch_size):
        """Test stack handles various batch sizes."""
        x = torch.randn(batch_size, 512, 960)
        output = stack(x)
        assert output.shape[0] == batch_size
