"""Graph Fourier Transform: forward (pixels -> coefficients) and inverse.

Both are matrix products with the eigenvector matrix U and run on whichever
device the tensors live on.
"""

import torch


def forward_gft(block, U):
    """x̂ = Uᵀx, applied independently per colour channel.

    Args:
        block: (block_size, block_size, 3) tensor (any numeric dtype), same
               device as U
        U: (n, n) eigenvector matrix from graph_utils.eigendecompose, n = block_size**2

    Returns:
        coeffs: (n, 3) float64 tensor, same device as U
    """
    bs = block.shape[0]
    n = bs * bs
    x = block.reshape(n, 3).to(dtype=torch.float64, device=U.device)
    return U.T @ x


def inverse_gft(coeffs, U, block_size):
    """x = U x̂, applied independently per colour channel.

    Args:
        coeffs: (n, 3) float64 tensor, same device as U
        U: (n, n) eigenvector matrix
        block_size: int

    Returns:
        block: (block_size, block_size, 3) float64 tensor, same device as U
    """
    x = U @ coeffs
    return x.reshape(block_size, block_size, 3)


def forward_gft_batch(blocks, U):
    """Batched forward GFT: many blocks, each with its own eigenbasis (used in training).

    Args:
        blocks: (B, block_size, block_size, 3) tensor, same device as U
        U: (B, n, n) eigenvector matrix, n = block_size**2

    Returns:
        coeffs: (B, n, 3) float64 tensor, same device as U
    """
    b, bs = blocks.shape[0], blocks.shape[1]
    n = bs * bs
    x = blocks.reshape(b, n, 3).to(dtype=torch.float64, device=U.device)
    return U.transpose(-1, -2) @ x


def inverse_gft_batch(coeffs, U, block_size):
    """Batched inverse GFT, mirror of forward_gft_batch.

    Args:
        coeffs: (B, n, 3) float64 tensor, same device as U
        U: (B, n, n) eigenvector matrix
        block_size: int

    Returns:
        blocks: (B, block_size, block_size, 3) float64 tensor, same device as U
    """
    b = coeffs.shape[0]
    x = U @ coeffs
    return x.reshape(b, block_size, block_size, 3)
