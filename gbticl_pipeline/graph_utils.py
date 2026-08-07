"""Build the graph Laplacian from predicted edge weights, and eigendecompose it.

Both operations run on whatever device the input tensors live on --
torch.linalg.eigh is GPU-accelerated automatically when its input is a CUDA
tensor, no separate GPU code path needed.
"""

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

    W = torch.zeros((n, n), dtype=torch.float64, device=device)
    W[idx_i, idx_j] = w
    W[idx_j, idx_i] = w
    D = torch.diag(W.sum(dim=1))
    return D - W


def build_laplacian_batch(edge_weights, block_size):
    """Batched version of build_laplacian -- builds (B, n, n) Laplacians from
    (B, n_edges) predicted weights in one shot, so eigendecompose() below can
    run torch.linalg.eigh on the whole batch at once (it natively supports a
    batched (..., n, n) input) instead of looping per block. Used by
    training.py, where many sampled blocks are processed per optimizer step.

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


def eigendecompose(L):
    """L = U Λ Uᵀ. L is symmetric, so torch.linalg.eigh is exact and returns
    eigenvalues already sorted ascending -- eigvals[0] == 0 always (constant
    eigenvector), matching the graph-Laplacian "DC term" property. Runs on
    whatever device L is already on (CPU or GPU) -- nothing else to configure.
    Works identically for a single (n, n) L or a batched (B, n, n) L (used by
    training.py); torch.linalg.eigh supports both natively.

    Returns:
        eigvals: (n,) or (B, n) float64 tensor, ascending, same device as L
        U: (n, n) or (B, n, n) float64 tensor, columns are the orthonormal
           eigenvectors, same device as L

    NOTE ON GRADIENTS (relevant to training.py): eigh's backward pass is
    numerically unstable when eigenvalues are repeated/degenerate -- exactly
    the situation found and debugged earlier for the uniform 8x8 grid
    Laplacian (~31 of 64 eigenvalues repeated). Gradients through eigenVALUES
    stay well-behaved even then; gradients through eigenVECTORS (U) can spike.
    training.py clips gradient norms for this reason -- see its docstring.
    """
    eigvals, U = torch.linalg.eigh(L)
    eigvals = torch.clamp(eigvals, min=0.0)  # guard tiny negative numerical noise
    return eigvals, U
