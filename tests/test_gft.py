"""GFT forward/inverse should be an exact round-trip (U is orthonormal by
construction from eigh), for both the single-block and batched variants."""

import torch

from gbticl_pipeline.graph_model import edge_list
from gbticl_pipeline.graph_utils import build_laplacian, build_laplacian_batch, eigendecompose
from gbticl_pipeline.gft import forward_gft, inverse_gft, forward_gft_batch, inverse_gft_batch


def test_gft_roundtrip_single(device):
    block_size = 8
    n_edges = len(edge_list(block_size))
    torch.manual_seed(0)
    weights = torch.rand(n_edges, dtype=torch.float64, device=device) + 0.1
    L = build_laplacian(weights, block_size, device=device)
    _, U = eigendecompose(L)

    block = torch.randint(0, 256, (block_size, block_size, 3), dtype=torch.uint8, device=device)
    coeffs = forward_gft(block, U)
    recon = inverse_gft(coeffs, U, block_size)

    assert torch.allclose(recon, block.to(torch.float64), atol=1e-6)


def test_gft_roundtrip_batched(device):
    block_size = 8
    n_edges = len(edge_list(block_size))
    torch.manual_seed(1)
    weights = torch.rand(6, n_edges, dtype=torch.float64, device=device) + 0.1
    L = build_laplacian_batch(weights, block_size)
    _, U = eigendecompose(L)

    blocks = torch.randint(0, 256, (6, block_size, block_size, 3), dtype=torch.uint8, device=device)
    coeffs = forward_gft_batch(blocks, U)
    recon = inverse_gft_batch(coeffs, U, block_size)

    assert torch.allclose(recon, blocks.to(torch.float64), atol=1e-6)
