"""DCT baseline: graph_utils.dct_basis_and_eigvals and the fixed_basis=True
path of codec.encode_image/decode_image."""

import torch

from gbticl_pipeline.graph_utils import dct_basis_and_eigvals
from gbticl_pipeline.codec import encode_image, decode_image
from gbticl_pipeline.evaluate import psnr


def test_dct_basis_orthonormal(device):
    _, U = dct_basis_and_eigvals(8, device=device)
    n = 64
    assert torch.allclose(U.T @ U, torch.eye(n, dtype=torch.float64, device=device), atol=1e-9)


def test_dct_basis_eigvals_ascending_from_zero(device):
    eigvals, _ = dct_basis_and_eigvals(8, device=device)
    assert eigvals[0].item() == 0.0
    assert torch.all(eigvals[1:] >= eigvals[:-1])


def test_dct_basis_cached_same_object_reused(device):
    e1, u1 = dct_basis_and_eigvals(8, device=device)
    e2, u2 = dct_basis_and_eigvals(8, device=device)
    assert torch.equal(e1, e2) and torch.equal(u1, u2)


def test_codec_fixed_basis_roundtrip(device):
    import numpy as np
    rng = np.random.default_rng(0)
    h = w = 16
    yy, xx = np.mgrid[0:h, 0:w]
    base = 128 + 50 * np.sin(xx / 4.0) + 30 * np.cos(yy / 5.0)
    img = np.clip(np.stack([base] * 3, axis=-1) + rng.normal(0, 5, size=(h, w, 3)), 0, 255).astype(np.uint8)

    payload, meta = encode_image(img, block_size=8, quant_step=8.0, symbol_range=(-2200, 2200),
                                  device=device, fixed_basis=True)
    assert meta["fixed_basis"] is True
    recon = decode_image(payload, meta, device=device)
    p = psnr(img, recon)
    assert p > 20, f"DCT baseline PSNR unexpectedly low: {p:.1f}dB"


def test_codec_fixed_basis_ignores_gbticl_model():
    """With fixed_basis=True the graph model is unused: passing an untrained
    model must give the same payload as passing none."""
    import numpy as np
    from gbticl_pipeline.graph_model import GBTICLNet
    device = torch.device("cpu")
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)

    payload1, meta1 = encode_image(img, block_size=8, quant_step=8.0, device=device, fixed_basis=True)
    payload2, meta2 = encode_image(img, block_size=8, quant_step=8.0, device=device, fixed_basis=True,
                                    gbticl_model=GBTICLNet(block_size=8))
    assert payload1 == payload2
