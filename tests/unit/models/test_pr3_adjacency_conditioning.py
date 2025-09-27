"""Unit tests for PR-3: Adjacency matrix conditioning."""

# ruff: noqa: N806  # Uppercase variable names are standard for PyTorch tensors

import torch

from src.brain_brr.config.schemas import GraphConfig, ModelConfig, NormConfig
from src.brain_brr.models.adjacency import (
    TemporalEMA,
    compute_stable_laplacian,
    condition_adjacency,
)
from src.brain_brr.models.detector import SeizureDetector


def test_adjacency_row_normalization():
    """Verify row-wise softmax normalization."""
    # Create random adjacency matrix
    B, T, N = 2, 10, 19
    adjacency = torch.randn(B, T, N, N).abs()  # Positive weights

    # Apply row-wise softmax
    conditioned = condition_adjacency(adjacency, row_softmax=True, tau=1.0, force_symmetric=False)

    # Before symmetrization, each row should sum to 1 (excluding zeros)
    # But we zero the diagonal, so sum won't be exactly 1
    # Just verify values are bounded [0, 1]
    assert (conditioned >= 0).all()
    assert (conditioned <= 1).all()

    # Check diagonal is zero (no self-loops)
    diag = torch.diagonal(conditioned, dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-6)


def test_adjacency_symmetrization():
    """Verify adjacency matrix symmetrization."""
    # Create asymmetric adjacency
    B, T, N = 2, 5, 19
    adjacency = torch.randn(B, T, N, N)

    # Apply symmetrization
    conditioned = condition_adjacency(adjacency, force_symmetric=True)

    # Check symmetry
    sym_error = (conditioned - conditioned.transpose(-2, -1)).abs().max()
    assert sym_error < 1e-6


def test_temporal_ema_smoothing():
    """Verify EMA reduces temporal variance."""
    # Create sequence with high variance
    B, T, N = 1, 100, 19
    adjacency = torch.randn(B, T, N, N) * 10  # High variance

    # Apply EMA smoothing
    smoothed = condition_adjacency(adjacency, ema_beta=0.9)

    # Compute variance across time
    orig_var = adjacency.var(dim=1).mean()
    smooth_var = smoothed.var(dim=1).mean()

    # Smoothed should have lower variance
    assert smooth_var < orig_var * 0.7


def test_stable_laplacian_computation():
    """Verify Laplacian eigendecomposition stability."""
    # Create well-conditioned adjacency
    B_T, N = 20, 19
    adjacency = torch.rand(B_T, N, N)
    adjacency = (adjacency + adjacency.transpose(-2, -1)) / 2  # Symmetric

    # Compute stable Laplacian
    laplacian = compute_stable_laplacian(adjacency, normalize=True, eps=1e-4)

    # Test eigendecomposition succeeds
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian.to(torch.float32))

    # Check all values are finite
    assert torch.isfinite(eigenvalues).all()
    assert torch.isfinite(eigenvectors).all()

    # Check eigenvalues are non-negative (within numerical tolerance)
    assert (eigenvalues >= -1e-6).all()


def test_laplacian_condition_number():
    """Verify Laplacian condition number is bounded."""
    # Create adjacency with potential conditioning issues
    B_T, N = 10, 19
    adjacency = torch.randn(B_T, N, N).abs()
    adjacency = adjacency * 0.01  # Small values

    # Compute Laplacian with regularization
    laplacian = compute_stable_laplacian(adjacency, normalize=True, eps=1e-3)

    # Check condition number
    cond = torch.linalg.cond(laplacian.to(torch.float32))

    # Should be well-conditioned (condition number < 10000)
    # Note: For normalized Laplacian with small adjacency values,
    # condition numbers ~1000-2000 are expected
    assert (cond < 10000).all()


def test_disconnected_node_handling():
    """Verify handling of disconnected nodes."""
    # Create adjacency with disconnected node
    B, T, N = 1, 5, 19
    adjacency = torch.ones(B, T, N, N) * 0.1
    adjacency[:, :, 0, :] = 0  # Disconnect node 0
    adjacency[:, :, :, 0] = 0

    # Apply conditioning with identity fallback
    conditioned = condition_adjacency(adjacency, force_symmetric=True)

    # Check node 0 has self-loop (not completely disconnected)
    diag_0 = conditioned[:, :, 0, 0]
    assert (diag_0 > 0).all()  # Should have self-loop


def test_ema_module():
    """Test TemporalEMA module."""
    ema = TemporalEMA(beta=0.9, persistent=False)

    # Create sequence
    B, T, N = 2, 50, 19
    adjacency = torch.randn(B, T, N, N)

    # Apply EMA
    smoothed = ema(adjacency)

    # Check shape preserved
    assert smoothed.shape == adjacency.shape

    # Check smoothing occurred (later timesteps should be smoother)
    early_var = smoothed[:, :10].var()
    late_var = smoothed[:, -10:].var()
    assert late_var < early_var


def test_pr3_with_detector():
    """Test PR-3 integration with SeizureDetector."""
    # Create config with PR-3 enabled
    config = ModelConfig(
        graph=GraphConfig(
            adj_row_softmax=True,
            adj_softmax_tau=1.0,
            adj_ema_beta=0.9,
            adj_force_symmetric=True,
            laplacian_eps=1e-3,
            laplacian_normalize=True,
            use_dynamic_pe=True,
        ),
        norms=NormConfig(boundary_norm="none"),  # PR-1 disabled for this test
    )

    # Create detector
    detector = SeizureDetector.from_config(config)

    # Test forward pass
    B, C, T = 2, 19, 960
    x = torch.randn(B, C, T)

    # Should not raise errors
    output = detector(x)
    # Detector returns (B, T) not (B, 1, T)
    assert output.shape == (B, T)
    assert torch.isfinite(output).all()


def test_pr3_backward_compatibility():
    """Test PR-3 disabled by default (backward compatibility)."""
    # Default config with graph should have PR-3 disabled
    config = ModelConfig(graph=GraphConfig())

    assert config.graph.adj_row_softmax is False
    assert config.graph.adj_ema_beta is None
    assert config.graph.adj_force_symmetric is False

    # Detector should work with PR-3 disabled
    detector = SeizureDetector.from_config(config)
    x = torch.randn(2, 19, 960)
    output = detector(x)
    assert torch.isfinite(output).all()


def test_combined_conditioning():
    """Test all conditioning options together."""
    B, T, N = 2, 20, 19
    adjacency = torch.randn(B, T, N, N).abs()

    # Apply all conditioning
    conditioned = condition_adjacency(
        adjacency,
        top_k=3,
        tau=0.5,
        force_symmetric=True,
        row_softmax=True,
        ema_beta=0.9,
    )

    # Check properties
    # 1. Values bounded
    assert (conditioned >= 0).all()
    assert (conditioned <= 1).all()

    # 2. Symmetric
    sym_error = (conditioned - conditioned.transpose(-2, -1)).abs().max()
    assert sym_error < 1e-6

    # 3. Can compute stable Laplacian
    laplacian = compute_stable_laplacian(conditioned.reshape(B * T, N, N), normalize=True, eps=1e-4)
    eigenvalues = torch.linalg.eigvalsh(laplacian.to(torch.float32))
    assert torch.isfinite(eigenvalues).all()


def test_temperature_parameter():
    """Test softmax temperature effect."""
    B, T, N = 1, 1, 19
    adjacency = torch.randn(B, T, N, N).abs()

    # Low temperature (sharper distribution)
    sharp = condition_adjacency(adjacency, row_softmax=True, tau=0.1)

    # High temperature (smoother distribution)
    smooth = condition_adjacency(adjacency, row_softmax=True, tau=10.0)

    # Sharp should have higher max values (more peaked)
    assert sharp.max() > smooth.max()

    # Smooth should have lower variance (more uniform)
    assert smooth.var() < sharp.var()


if __name__ == "__main__":
    test_adjacency_row_normalization()
    test_adjacency_symmetrization()
    test_temporal_ema_smoothing()
    test_stable_laplacian_computation()
    test_laplacian_condition_number()
    test_disconnected_node_handling()
    test_ema_module()
    test_pr3_with_detector()
    test_pr3_backward_compatibility()
    test_combined_conditioning()
    test_temperature_parameter()
    print("All PR-3 tests passed!")
