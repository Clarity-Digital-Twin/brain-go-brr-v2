import pytest
import torch

torch_geometric = pytest.importorskip("torch_geometric", reason="PyG required for GNN tests")


@pytest.mark.unit
def test_gnn_pyg_preserves_shape_small():
    from src.brain_brr.models.gnn_pyg import GraphChannelMixerPyG

    batch_size, n_nodes, seq_len, feat_dim = 2, 19, 5, 64
    x = torch.randn(batch_size, n_nodes, seq_len, feat_dim)

    # Simple symmetric adjacency with small random weights
    a = torch.rand(batch_size, seq_len, n_nodes, n_nodes)
    a = (a + a.transpose(-1, -2)) / 2
    a = torch.where(a > 0.8, a, torch.zeros_like(a))  # sparsify a bit

    gnn = GraphChannelMixerPyG(d_model=feat_dim, n_electrodes=n_nodes, k_eigenvectors=8)
    y = gnn(x, a)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.unit
def test_gnn_pyg_has_vectorized_flags():
    from src.brain_brr.models.gnn_pyg import GraphChannelMixerPyG

    gnn = GraphChannelMixerPyG(d_model=64)
    assert hasattr(gnn, "static_pe")
    assert hasattr(gnn, "use_vectorized")
    assert hasattr(gnn, "use_dynamic_pe")


@pytest.mark.unit
def test_jitter_prevents_eigenvalue_degeneracy():
    """Test that diagonal jitter breaks eigenvalue degeneracy.

    This test validates the fix for the epoch 13 crash caused by
    degenerate eigenvalues crashing cuSOLVER.

    Root cause: Symmetric matrices with identical eigenvalues
    (e.g., identity matrix) cause numerical instability in eigh.
    Solution: Random diagonal jitter breaks symmetry.
    """
    from src.brain_brr.models.gnn_pyg import GraphChannelMixerPyG

    N = 19  # noqa: N806
    gnn = GraphChannelMixerPyG(d_model=64, n_electrodes=N, k_eigenvectors=8)

    # Create perfectly degenerate matrix (identity: all eigenvalues = 1)
    degenerate = torch.eye(N, dtype=torch.float32)

    # Apply jitter (test single matrix case)
    jittered = degenerate.clone()
    gnn._add_jitter_for_stability(jittered, jitter_scale=1e-5)

    # Compute eigenvalues
    evals_jittered = torch.linalg.eigvalsh(jittered)

    # Test 1: Eigenvalue gaps should be non-zero (most should be > 1e-8)
    # Note: With 1e-5 jitter, gaps vary from ~1e-7 to ~1e-5 due to eigenvalue solver numerics
    # We require at least 75% of gaps to be distinct (random jitter may create near-collisions)
    gaps = evals_jittered[1:] - evals_jittered[:-1]
    num_distinct = (gaps.abs() > 1e-8).sum()
    min_required = int(0.75 * (N - 1))  # At least 75% distinct
    assert num_distinct >= min_required, f"Only {num_distinct}/{N-1} gaps are distinct (need ≥{min_required})"

    # Test 2: Eigenvalues should be close to original (only perturbed slightly)
    assert torch.allclose(evals_jittered, torch.ones(N), atol=1e-4)

    # Test 3: Jitter should work on batched matrices
    batch_degenerate = torch.eye(N, dtype=torch.float32).unsqueeze(0).repeat(4, 1, 1)
    batch_jittered = batch_degenerate.clone()
    gnn._add_jitter_for_stability(batch_jittered, jitter_scale=1e-5)

    for b in range(4):
        evals_b = torch.linalg.eigvalsh(batch_jittered[b])
        gaps_b = evals_b[1:] - evals_b[:-1]
        num_distinct_b = (gaps_b.abs() > 1e-8).sum()
        assert num_distinct_b >= min_required, f"Batch {b}: Only {num_distinct_b}/{N-1} gaps are distinct (need ≥{min_required})"


@pytest.mark.unit
def test_validate_and_process_eigendecomp():
    """Test eigendecomposition validation helper."""
    from src.brain_brr.models.gnn_pyg import GraphChannelMixerPyG

    gnn = GraphChannelMixerPyG(d_model=64, n_electrodes=19, k_eigenvectors=8)

    # Test 1: Valid eigendecomposition
    valid_evals = torch.tensor([0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    valid_evecs = torch.randn(8, 8)
    pe, clamped_evals = gnn._validate_and_process_eigendecomp(valid_evals, valid_evecs)
    assert pe.shape == (8, 8), "PE shape incorrect"
    assert torch.isfinite(clamped_evals).all(), "Eigenvalues should be finite"

    # Test 2: NaN eigenvalues should raise
    nan_evals = valid_evals.clone()
    nan_evals[3] = float("nan")
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        gnn._validate_and_process_eigendecomp(nan_evals, valid_evecs)

    # Test 3: Inf eigenvectors should raise
    inf_evecs = valid_evecs.clone()
    inf_evecs[2, 5] = float("inf")
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        gnn._validate_and_process_eigendecomp(valid_evals, inf_evecs)
