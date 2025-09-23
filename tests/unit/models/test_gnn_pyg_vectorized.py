import os
import pytest
import torch


torch_geometric = pytest.importorskip("torch_geometric", reason="PyG required for GNN tests")


@pytest.mark.unit
def test_gnn_pyg_preserves_shape_small():
    from src.brain_brr.models.gnn_pyg import GraphChannelMixerPyG

    B, N, T, D = 2, 19, 5, 64
    x = torch.randn(B, N, T, D)

    # Simple symmetric adjacency with small random weights
    a = torch.rand(B, T, N, N)
    a = (a + a.transpose(-1, -2)) / 2
    a = torch.where(a > 0.8, a, torch.zeros_like(a))  # sparsify a bit

    gnn = GraphChannelMixerPyG(d_model=D, n_electrodes=N, k_eigenvectors=8)
    y = gnn(x, a)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.unit
@pytest.mark.xfail(reason="Vectorized path and static PE buffer not yet implemented", strict=False)
def test_gnn_pyg_has_vectorized_flags():
    from src.brain_brr.models.gnn_pyg import GraphChannelMixerPyG

    gnn = GraphChannelMixerPyG(d_model=64)
    assert hasattr(gnn, "static_pe")
    assert hasattr(gnn, "use_vectorized")
    assert hasattr(gnn, "use_dynamic_pe")

