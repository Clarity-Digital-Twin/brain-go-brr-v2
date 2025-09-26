"""Comprehensive NaN robustness tests for all model components."""

import pytest
import torch

from src.brain_brr.models.detector import SeizureDetector
from src.brain_brr.models.edge_features import edge_scalar_series
from src.brain_brr.models.mamba import BiMamba2
from src.brain_brr.models.tcn import TCNEncoder
from src.brain_brr.train.loop import FocalLoss

# Check if PyTorch Geometric is available
try:
    import torch_geometric

    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class TestNaNRobustness:
    """Test suite to ensure 100% NaN robustness across all components."""

    @pytest.fixture
    def device(self):
        """Get appropriate device for testing."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========== Edge Case Input Tests ==========

    def test_tcn_with_nan_input(self, device):
        """Test TCN handles NaN inputs gracefully."""
        tcn = TCNEncoder(init_gain=0.2).to(device)

        # Create input with NaN values
        x = torch.randn(2, 19, 15360, device=device)
        x[0, 5, 1000:2000] = float("nan")
        x[1, :, 5000] = float("nan")

        output = tcn(x)

        # Should sanitize NaN and produce finite output
        assert torch.isfinite(output).all(), "TCN output contains NaN/Inf with NaN input"
        assert output.shape == (2, 512, 960)

    def test_tcn_with_inf_input(self, device):
        """Test TCN handles infinite inputs gracefully."""
        tcn = TCNEncoder(init_gain=0.2).to(device)

        # Create input with infinite values
        x = torch.randn(2, 19, 15360, device=device)
        x[0, 3, 500:600] = float("inf")
        x[1, 10, :100] = float("-inf")

        output = tcn(x)

        assert torch.isfinite(output).all(), "TCN output contains NaN/Inf with Inf input"

    def test_tcn_with_extreme_values(self, device):
        """Test TCN handles extreme values without overflow."""
        tcn = TCNEncoder(init_gain=0.2).to(device)

        # Create input with extreme values
        x = torch.randn(2, 19, 15360, device=device) * 1e6  # Very large
        output = tcn(x)
        assert torch.isfinite(output).all(), "TCN fails with large inputs"

        x = torch.randn(2, 19, 15360, device=device) * 1e-6  # Very small
        output = tcn(x)
        assert torch.isfinite(output).all(), "TCN fails with small inputs"

    def test_mamba_with_nan_input(self, device):
        """Test Mamba handles NaN inputs gracefully."""
        mamba = BiMamba2(init_gain=0.2).to(device)

        # Input with NaN
        x = torch.randn(2, 512, 960, device=device)
        x[0, :100, :] = float("nan")

        output = mamba(x)
        assert torch.isfinite(output).all(), "Mamba output contains NaN/Inf with NaN input"

    def test_mamba_with_zero_input(self, device):
        """Test Mamba handles all-zero inputs."""
        mamba = BiMamba2(init_gain=0.2).to(device)

        x = torch.zeros(2, 512, 960, device=device)
        output = mamba(x)
        assert torch.isfinite(output).all(), "Mamba fails with zero input"

    # ========== Edge Feature Tests ==========

    def test_edge_features_cosine_with_zero_norm(self, device):
        """Test cosine similarity with zero-norm vectors."""
        # Create input where some nodes have zero norm
        x = torch.randn(2, 10, 19, 64, device=device)
        x[:, :, 5, :] = 0  # Node 5 has zero features
        x[:, :, 10, :] = 1e-10  # Node 10 has near-zero features

        edge_feats = edge_scalar_series(x, metric="cosine")

        assert torch.isfinite(edge_feats).all(), "Edge features contain NaN with zero-norm vectors"
        assert (edge_feats >= -1.0).all(), "Cosine similarity below -1.0"
        assert (edge_feats <= 1.0).all(), "Cosine similarity above 1.0"

    def test_edge_features_correlation_with_constant(self, device):
        """Test correlation with constant vectors."""
        # Create input where some nodes are constant
        x = torch.randn(2, 10, 19, 64, device=device)
        x[:, :, 3, :] = 5.0  # Node 3 is constant

        edge_feats = edge_scalar_series(x, metric="correlation")

        assert torch.isfinite(edge_feats).all(), "Edge features contain NaN with constant vectors"
        assert (edge_feats >= -1.0).all(), "Correlation below -1.0"
        assert (edge_feats <= 1.0).all(), "Correlation above 1.0"

    # ========== GNN Tests ==========

    @pytest.mark.skipif(not HAS_PYG, reason="PyTorch Geometric not installed")
    def test_gnn_with_disconnected_graph(self, device):
        """Test GNN with disconnected graph (zero adjacency)."""
        if not HAS_PYG:
            pytest.skip("PyTorch Geometric not installed")

        from src.brain_brr.models.gnn_pyg import GraphChannelMixerPyG

        gnn = GraphChannelMixerPyG(
            d_model=512, n_electrodes=19, use_dynamic_pe=True, k_eigenvectors=8
        ).to(device)

        # Zero adjacency matrix (disconnected graph)
        x = torch.randn(2, 19, 5, 512, device=device)  # B=2, N=19, T=5, C=512
        adj = torch.zeros(2, 5, 19, 19, device=device)

        output = gnn(x, adj)
        assert torch.isfinite(output).all(), "GNN fails with disconnected graph"

    @pytest.mark.skipif(not HAS_PYG, reason="PyTorch Geometric not installed")
    def test_gnn_with_singular_laplacian(self, device):
        """Test GNN with singular Laplacian matrix."""
        if not HAS_PYG:
            pytest.skip("PyTorch Geometric not installed")

        from src.brain_brr.models.gnn_pyg import GraphChannelMixerPyG

        gnn = GraphChannelMixerPyG(
            d_model=512, n_electrodes=19, use_dynamic_pe=True, k_eigenvectors=8
        ).to(device)

        # Create adjacency that leads to singular Laplacian
        x = torch.randn(2, 19, 5, 512, device=device)  # B=2, N=19, T=5, C=512
        adj = torch.ones(2, 5, 19, 19, device=device)  # Complete graph

        output = gnn(x, adj)
        assert torch.isfinite(output).all(), "GNN fails with singular Laplacian"

    @pytest.mark.skipif(not HAS_PYG, reason="PyTorch Geometric not installed")
    def test_gnn_with_ill_conditioned_matrix(self, device):
        """Test GNN with ill-conditioned adjacency matrix."""
        if not HAS_PYG:
            pytest.skip("PyTorch Geometric not installed")

        from src.brain_brr.models.gnn_pyg import GraphChannelMixerPyG

        gnn = GraphChannelMixerPyG(
            d_model=512, n_electrodes=19, use_dynamic_pe=True, k_eigenvectors=8
        ).to(device)

        x = torch.randn(2, 19, 5, 512, device=device)  # B=2, N=19, T=5, C=512
        # Create ill-conditioned adjacency
        adj = torch.eye(19, device=device).unsqueeze(0).unsqueeze(0)
        adj = adj.expand(2, 5, -1, -1)
        adj = adj + torch.randn_like(adj) * 1e-8  # Add tiny noise

        output = gnn(x, adj)
        assert torch.isfinite(output).all(), "GNN fails with ill-conditioned matrix"

    # ========== Loss Function Tests ==========

    def test_focal_loss_with_extreme_logits(self, device):
        """Test focal loss with extreme logit values."""
        loss_fn = FocalLoss(alpha=0.5, gamma=2.0)

        # Extreme positive logits
        logits = torch.tensor([[1000.0, -1000.0]], device=device)
        targets = torch.tensor([[1.0, 0.0]], device=device)
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss).all(), "Focal loss fails with extreme positive logits"

        # Extreme negative logits
        logits = torch.tensor([[-1000.0, 1000.0]], device=device)
        targets = torch.tensor([[1.0, 0.0]], device=device)
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss).all(), "Focal loss fails with extreme negative logits"

    def test_focal_loss_with_all_ones_targets(self, device):
        """Test focal loss with all positive targets."""
        loss_fn = FocalLoss(alpha=0.999, gamma=2.0)  # Extreme alpha for imbalance

        logits = torch.randn(10, 100, device=device)
        targets = torch.ones(10, 100, device=device)

        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss).all(), "Focal loss fails with all positive targets"

    def test_focal_loss_with_all_zeros_targets(self, device):
        """Test focal loss with all negative targets."""
        loss_fn = FocalLoss(alpha=0.001, gamma=2.0)  # Extreme alpha for imbalance

        logits = torch.randn(10, 100, device=device)
        targets = torch.zeros(10, 100, device=device)

        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss).all(), "Focal loss fails with all negative targets"

    # ========== End-to-End Tests ==========

    def test_detector_with_pathological_input(self, device):
        """Test full detector with pathological inputs."""
        from src.brain_brr.config.schemas import MambaConfig, ModelConfig, TCNConfig

        config = ModelConfig(
            tcn=TCNConfig(num_layers=4, kernel_size=3, dropout=0.0, stride_down=16),
            mamba=MambaConfig(n_layers=1, d_model=512, d_state=16, conv_kernel=4, dropout=0.0),
        )

        model = SeizureDetector.from_config(config).to(device)
        model.eval()

        # Test various pathological inputs
        test_cases = [
            # Case 1: Mix of NaN, Inf, and normal values
            lambda: torch.randn(1, 19, 15360, device=device).masked_fill_(
                torch.rand(1, 19, 15360, device=device) < 0.1, float("nan")
            ),
            # Case 2: All zeros
            lambda: torch.zeros(1, 19, 15360, device=device),
            # Case 3: Extreme values
            lambda: torch.randn(1, 19, 15360, device=device) * 1e6,
            # Case 4: Very small values
            lambda: torch.randn(1, 19, 15360, device=device) * 1e-6,
            # Case 5: Single spike
            lambda: torch.zeros(1, 19, 15360, device=device).index_fill_(
                2, torch.tensor([1000], device=device), 1e6
            ),
        ]

        with torch.no_grad():
            for i, test_input_fn in enumerate(test_cases):
                x = test_input_fn()
                output = model(x)
                assert torch.isfinite(output).all(), f"Detector fails on test case {i + 1}"
                assert output.shape == (1, 15360)

    def test_detector_gradient_flow_with_nan(self, device):
        """Test gradient flow through detector with NaN loss."""
        from src.brain_brr.config.schemas import MambaConfig, ModelConfig, TCNConfig

        config = ModelConfig(
            tcn=TCNConfig(num_layers=4, kernel_size=3, dropout=0.0, stride_down=16),
            mamba=MambaConfig(n_layers=1, d_model=512, d_state=16, conv_kernel=4, dropout=0.0),
        )

        model = SeizureDetector.from_config(config).to(device)

        # Normal input
        x = torch.randn(2, 19, 15360, device=device, requires_grad=True)
        output = model(x)

        # Create a loss that would normally produce NaN gradients
        loss = output.sum() / 0.0  # Division by zero

        # This should not crash
        try:
            loss.backward()
            # Gradients might be NaN but shouldn't crash
            has_nan = any(
                torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None
            )
            # This is expected with div by zero
            assert has_nan or torch.isinf(loss), "Expected NaN/Inf gradients with div by zero"
        except RuntimeError:
            # Acceptable if autograd detects the issue
            pass

    # ========== Stress Tests ==========

    @pytest.mark.slow
    def test_repeated_forward_passes(self, device):
        """Test model stability over many forward passes."""
        from src.brain_brr.config.schemas import MambaConfig, ModelConfig, TCNConfig

        config = ModelConfig(
            tcn=TCNConfig(num_layers=4, kernel_size=3, dropout=0.0, stride_down=16),
            mamba=MambaConfig(n_layers=1, d_model=512, d_state=16, conv_kernel=4, dropout=0.0),
        )

        model = SeizureDetector.from_config(config).to(device)
        model.eval()

        # Run many forward passes with random inputs
        with torch.no_grad():
            for i in range(100):
                x = torch.randn(1, 19, 15360, device=device)
                # Occasionally inject extreme values
                if i % 10 == 0:
                    x *= 100
                if i % 20 == 0:
                    x[:, :, ::100] = 1e6

                output = model(x)
                assert torch.isfinite(output).all(), f"Model became unstable at iteration {i + 1}"

    def test_config_consistency_local_vs_modal(self):
        """Verify local and modal configs maintain identical NaN safeguards."""

        import yaml

        # Load configs
        with open("configs/local/train.yaml") as f:
            local_config = yaml.safe_load(f)
        with open("configs/modal/train.yaml") as f:
            modal_config = yaml.safe_load(f)

        # Check critical NaN-related settings
        # Both should have focal loss for imbalanced data
        assert local_config["training"]["loss"] == "focal"
        assert modal_config["training"]["loss"] == "focal"

        # Gradient clipping should be present
        assert "gradient_clip" in local_config["training"]
        assert "gradient_clip" in modal_config["training"]

        # Mixed precision differs but is intentional (RTX 4090 vs A100)
        assert not local_config["training"]["mixed_precision"]  # RTX 4090 needs this
        assert modal_config["training"]["mixed_precision"]  # A100 can handle this

        # Both should use balanced sampling for seizure detection (in data section, not training)
        assert local_config["data"].get("use_balanced_sampling", False)
        assert modal_config["data"].get("use_balanced_sampling", False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
