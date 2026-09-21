"""Graph Laplacian construction, eigendecomposition and the fixed DCT basis.

Everything runs on the device of the input tensors.
"""

import numpy as np
import torch

from .graph_model import edge_list


def build_laplacian(edge_weights, block_size, device=None):
    """L = D - W for a 4-connected grid graph with the given per-edge weights.

    Args:
        edge_weights: (n_edges,) float tensor, matching edge_list(block_size) order.
        block_size: int
        device: torch.device to build L on; defaults to edge_weights' own device.

    Returns:
        L: (n, n) float64 tensor on `device`. Symmetric, positive semi-definite.
    """
    device = device or edge_weights.device
    edges = edge_list(block_size)
    n = block_size * block_size

    idx_i = torch.tensor([e[0] for e in edges], device=device, dtype=torch.long)
    idx_j = torch.tensor([e[1] for e in edges], device=device, dtype=torch.long)
    w = edge_weights.to(device=device, dtype=torch.float64)

    # Symmetric weighted adjacency matrix
    W = torch.zeros((n, n), dtype=torch.float64, device=device)
    W[idx_i, idx_j] = w
    W[idx_j, idx_i] = w
    D = torch.diag(W.sum(dim=1))
    return D - W


def build_laplacian_batch(edge_weights, block_size):
    """Batched build_laplacian: (B, n_edges) weights -> (B, n, n) Laplacians (used in training).

    Args:
        edge_weights: (B, n_edges) float tensor, matching edge_list(block_size) order
        block_size: int

    Returns:
        L: (B, n, n) float64 tensor, same device as edge_weights
    """
    device = edge_weights.device
    edges = edge_list(block_size)
    n = block_size * block_size
    b = edge_weights.shape[0]

    idx_i = torch.tensor([e[0] for e in edges], device=device, dtype=torch.long)
    idx_j = torch.tensor([e[1] for e in edges], device=device, dtype=torch.long)
    w = edge_weights.to(dtype=torch.float64)  # (B, n_edges)

    W = torch.zeros((b, n, n), dtype=torch.float64, device=device)
    W[:, idx_i, idx_j] = w
    W[:, idx_j, idx_i] = w
    D = torch.diag_embed(W.sum(dim=2))
    return D - W


def eigendecompose(L, eps=1e-4, canonicalize_sign=True):
    """
    Eigendecomposition L = U Λ Uᵀ of a symmetric Laplacian (single or batched).

    Eigenvalues are returned in ascending order (eigvals[0] is the DC term).
    Two measures make the result well-defined and identical for encoder and
    decoder, both of which must derive the same U without transmitting it:

      1. A small deterministic diagonal perturbation, L + diag(eps * [0..n-1]),
         removes exactly repeated eigenvalues. For repeated eigenvalues (as in
         the uniform-weight grid graph) the eigenvectors are not unique and the
         backward pass of torch.linalg.eigh returns NaN gradients.
      2. Sign canonicalisation: every eigenvector is flipped so that its
         largest-magnitude entry is positive, removing the arbitrary sign of
         eigh's output.

    Args:
        L: (n, n) or (B, n, n) symmetric matrix
        eps: strength of the perturbation (small relative to the Laplacian entries)
        canonicalize_sign: apply the sign convention above

    Returns:
        eigvals: (n,) or (B, n) float64, ascending
        U: (n, n) or (B, n, n) float64, columns are the orthonormal eigenvectors
    """
    n = L.shape[-1]
    perturb = torch.diag(eps * torch.arange(n, device=L.device, dtype=L.dtype))
    eigvals, U = torch.linalg.eigh(L + perturb)  # broadcasts perturb over any leading batch dim
    eigvals = torch.clamp(eigvals, min=0.0)  # guard tiny negative numerical noise

    if canonicalize_sign:
        # flip each eigenvector so its largest-magnitude entry is positive
        idx = torch.argmax(U.abs(), dim=-2, keepdim=True)  # (..., 1, n)
        sign = torch.sign(torch.gather(U, -2, idx))  # (..., 1, n)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)  # never zero out a column
        U = U * sign  # broadcasts (..., 1, n) against (..., n, n)

    return eigvals, U


_dct_cache = {}


def dct_basis_and_eigvals(block_size, device=None):
    """
    Fixed 2D DCT-II basis for the DCT baseline (content-independent transform).

    The DCT-II basis is the eigenbasis of the path-graph Laplacian with
    reflecting boundaries; the 2D basis is its Kronecker product. The result has
    the same contract as eigendecompose (ascending eigenvalues, orthonormal
    columns), so it can replace the predicted basis in the codec. It is computed
    once per block size and cached.

    Returns:
        eigvals: (n,) float64, ascending, eigvals[0] == 0
        U: (n, n) float64, orthonormal columns in row-major pixel order
    """
    key = block_size
    if key not in _dct_cache:
        n = block_size
        idx = np.arange(n)
        r = idx.reshape(-1, 1)
        basis_1d = np.cos(np.pi * (2 * r + 1) * idx / (2 * n)) * np.sqrt(2.0 / n)
        basis_1d[:, 0] *= 1.0 / np.sqrt(2.0)  # DC-term normalization for orthonormality
        eigval_1d = 2.0 - 2.0 * np.cos(np.pi * idx / n)  # path-graph Neumann-Laplacian spectrum

        U_2d = np.kron(basis_1d, basis_1d)  # (n^2, n^2), row-major over (row, col) -> matches edge_list()
        eigval_2d = (eigval_1d.reshape(-1, 1) + eigval_1d.reshape(1, -1)).reshape(-1)  # same kron ordering

        order = np.argsort(eigval_2d)  # ascending, matching eigendecompose()'s contract
        U_2d = U_2d[:, order]
        eigval_2d = eigval_2d[order]
        _dct_cache[key] = (
            torch.tensor(eigval_2d, dtype=torch.float64),
            torch.tensor(U_2d, dtype=torch.float64),
        )

    eigvals, U = _dct_cache[key]
    if device is not None:
        eigvals, U = eigvals.to(device), U.to(device)
    return eigvals, U
