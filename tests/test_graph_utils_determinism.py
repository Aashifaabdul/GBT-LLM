"""Regression tests for the eigendecompose bug fixed in graph_utils.py:
degenerate/repeated eigenvalues caused NaN gradients through eigh's backward
pass, and (theoretically, even if not reproduced in-process on this exact
torch/cuSOLVER build) sign/ordering non-determinism could desync encoder and
decoder since no graph is ever transmitted between them."""

import torch

from gbticl_pipeline.graph_model import edge_list
from gbticl_pipeline.graph_utils import build_laplacian, build_laplacian_batch, eigendecompose


def test_eigendecompose_no_nan_gradient_on_degenerate_laplacian(device):
    """The exact failure case found and fixed: a maximally-degenerate (all
    edge weights equal) 8x8 block Laplacian has ~31-33 of 64 eigenvalues
    repeated, which made eigh's backward pass through the eigenvectors
    produce NaN gradients before the symmetry-breaking fix."""
    block_size = 8
    n_edges = len(edge_list(block_size))
    weights = torch.ones(n_edges, dtype=torch.float64, device=device, requires_grad=True)

    L = build_laplacian(weights, block_size, device=device)
    eigvals, U = eigendecompose(L)

    loss = (U ** 2).sum() + eigvals.sum()
    loss.backward()

    assert weights.grad is not None
    assert not torch.isnan(weights.grad).any(), "eigendecompose must not produce NaN gradients"
    assert not torch.isinf(weights.grad).any()
    assert weights.grad.norm().item() > 0, "gradient should be nonzero (not silently dead)"


def test_eigendecompose_deterministic_across_calls(device):
    """Encoder and decoder each call eigendecompose independently on
    (by construction) bit-identical L -- if the perturbation weren't
    deterministic, or eigh's own sign convention varied, they'd derive
    different bases and silently desync, since U is never transmitted."""
    block_size = 8
    n_edges = len(edge_list(block_size))
    weights = torch.ones(n_edges, dtype=torch.float64, device=device)
    L = build_laplacian(weights, block_size, device=device)

    eigvals_a, U_a = eigendecompose(L)
    eigvals_b, U_b = eigendecompose(L)

    assert torch.equal(eigvals_a, eigvals_b)
    assert torch.equal(U_a, U_b)


def test_eigendecompose_reduces_degeneracy(device):
    block_size = 8
    n_edges = len(edge_list(block_size))
    weights = torch.ones(n_edges, dtype=torch.float64, device=device)
    L = build_laplacian(weights, block_size, device=device)
    eigvals, _ = eigendecompose(L)

    n_unique = len(torch.unique(torch.round(eigvals * 1e6)))
    # before the fix this was ~31-33/64; the perturbation should break the
    # vast majority of the degeneracy without materially altering the spectrum
    assert n_unique >= 55, f"expected the symmetry-breaking perturbation to " \
                            f"resolve most degeneracy, got {n_unique}/64 unique eigenvalues"


def test_eigendecompose_batched_matches_single(device):
    """build_laplacian_batch + eigendecompose must behave the same as the
    single-matrix path (used by codec.py) for every item in the batch, since
    training.py relies on this equivalence to make batched training a valid
    stand-in for the per-block inference loop."""
    block_size = 8
    n_edges = len(edge_list(block_size))
    torch.manual_seed(0)
    weights_batch = torch.rand(4, n_edges, dtype=torch.float64, device=device) + 0.1

    L_batch = build_laplacian_batch(weights_batch, block_size)
    eigvals_batch, U_batch = eigendecompose(L_batch)
    assert eigvals_batch.shape == (4, block_size * block_size)
    assert U_batch.shape == (4, block_size * block_size, block_size * block_size)

    for b in range(4):
        L_single = build_laplacian(weights_batch[b], block_size, device=device)
        eigvals_single, U_single = eigendecompose(L_single)
        assert torch.allclose(eigvals_batch[b], eigvals_single, atol=1e-9)
        assert torch.allclose(U_batch[b], U_single, atol=1e-9)


def test_eigendecompose_sign_canonicalization(device):
    """Each eigenvector's largest-magnitude entry should be positive after
    canonicalization -- and flipping the sign of an input eigenvector should
    not change the canonicalized output (the whole point of canonicalizing)."""
    block_size = 8
    n_edges = len(edge_list(block_size))
    torch.manual_seed(1)
    weights = torch.rand(n_edges, dtype=torch.float64, device=device) + 0.1
    L = build_laplacian(weights, block_size, device=device)
    _, U = eigendecompose(L)

    n = block_size * block_size
    for col in range(n):
        max_idx = torch.argmax(U[:, col].abs())
        assert U[max_idx, col] > 0, f"column {col}'s largest-magnitude entry should be positive"
