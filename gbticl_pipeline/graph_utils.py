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


def eigendecompose(L, eps=1e-4, canonicalize_sign=True):
    """L = U Λ Uᵀ. L is symmetric, so torch.linalg.eigh is exact and returns
    eigenvalues already sorted ascending -- eigvals[0] == 0 always (constant
    eigenvector), matching the graph-Laplacian "DC term" property. Runs on
    whatever device L is already on (CPU or GPU) -- nothing else to configure.
    Works identically for a single (n, n) L or a batched (B, n, n) L (used by
    training.py); torch.linalg.eigh supports both natively.

    Args:
        eps: strength of the deterministic symmetry-breaking perturbation
             applied before eigh (see BUG FIX note below). 1e-4 is small
             relative to typical edge-weight-derived Laplacian entries
             (O(1)-O(10)), so it doesn't meaningfully change the spectrum,
             but is far above float64 numerical noise.
        canonicalize_sign: fix each eigenvector's sign so its largest-
             magnitude entry is positive (see BUG FIX note below).

    Returns:
        eigvals: (n,) or (B, n) float64 tensor, ascending, same device as L
        U: (n, n) or (B, n, n) float64 tensor, columns are the orthonormal
           eigenvectors, same device as L

    BUG FIX (confirmed reproduced and fixed 2026-08-07, on this project's
    actual GPU): a near-uniform/degenerate Laplacian (repeated eigenvalues --
    the exact situation observed for UniformGBTICL's constant-weight graph,
    ~31-33 of 64 eigenvalues repeated for an 8x8 block) makes
    torch.linalg.eigh's BACKWARD pass produce NaN gradients through the
    eigenvectors, because the eigenspace for a repeated eigenvalue has no
    unique basis -- any rotation within it is equally valid, so the gradient
    is undefined at that point. This was reproduced directly: a maximally
    degenerate 8x8-grid Laplacian gave all-NaN `weights.grad` through this
    function without the fix below.

    Two fixes, both applied unconditionally so every call site (encoder,
    decoder, and training.py's batched calls) inherits them automatically,
    with zero call-site changes needed:
      1. A tiny, DETERMINISTIC diagonal perturbation
         `L' = L + diag(eps * [0, 1, ..., n-1])` breaks exact degeneracy
         before eigh ever sees the matrix, so no eigenvalue is ever exactly
         repeated in floating point -- the gradient is then well-defined
         everywhere. Deterministic (not random) is essential here: encoder
         and decoder must compute bit-identical L' from bit-identical L, or
         they'd derive different eigenbases and silently desync (no graph
         is ever transmitted -- both sides MUST independently arrive at the
         same U).
      2. Sign canonicalization: eigh's sign convention for each eigenvector
         is arbitrary (if v is a unit eigenvector, so is -v) and, in
         principle, is exactly the kind of thing that can differ across
         torch/cuSOLVER versions or between two calls with a
         non-deterministic algorithm path. Flipping each column so its
         largest-magnitude entry is positive pins this down explicitly
         rather than relying on eigh happening to be consistent -- cheap
         insurance for encoder/decoder agreement, done every call.
    """
    n = L.shape[-1]
    perturb = torch.diag(eps * torch.arange(n, device=L.device, dtype=L.dtype))
    eigvals, U = torch.linalg.eigh(L + perturb)  # broadcasts perturb over any leading batch dim
    eigvals = torch.clamp(eigvals, min=0.0)  # guard tiny negative numerical noise

    if canonicalize_sign:
        # for each column (eigenvector), find the row with the largest |entry|
        # and flip the whole column so that entry is positive
        idx = torch.argmax(U.abs(), dim=-2, keepdim=True)  # (..., 1, n)
        sign = torch.sign(torch.gather(U, -2, idx))         # (..., 1, n)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)  # never zero out a column
        U = U * sign  # broadcasts (..., 1, n) against (..., n, n)

    return eigvals, U
