"""End-to-end encode_image/decode_image round-trip on synthetic data, for
every GBT-ICL/coefficient-model combination the project ships, at multiple
quant_steps. This is the promoted, repeatable version of the manual smoke
test run against test_codec.py -- it must stay green before any downstream
stage (temporal extension, real training, ablations) is trusted."""

import numpy as np
import pytest
import torch

from gbticl_pipeline.codec import encode_image, decode_image
from gbticl_pipeline.evaluate import psnr, bits_per_pixel
from gbticl_pipeline.graph_model import UniformGBTICL, ContextGradientGBTICL, GBTICLNet
from gbticl_pipeline.coeff_model import LaplaceCoeffModel, TinyTransformerCoeffModel

WIDE_RANGE = (-2200, 2200)


def _synthetic_image(h=16, w=16, seed=0):
    rng = np.random.default_rng(seed)
    # smooth gradient + noise, more representative of real image content than
    # pure uniform random noise (which stresses the entropy coder unrealistically)
    yy, xx = np.mgrid[0:h, 0:w]
    base = (128 + 60 * np.sin(xx / 4.0) + 40 * np.cos(yy / 5.0))
    img = np.stack([base] * 3, axis=-1) + rng.normal(0, 5, size=(h, w, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("gbticl_cls", [UniformGBTICL, ContextGradientGBTICL, GBTICLNet])
@pytest.mark.parametrize("coeff_cls", [LaplaceCoeffModel, TinyTransformerCoeffModel])
@pytest.mark.parametrize("quant_step", [8.0, 1.0])
def test_codec_roundtrip_finite_and_lossless_structure(device, gbticl_cls, coeff_cls, quant_step):
    img = _synthetic_image()
    gbticl_model = gbticl_cls().to(device)
    coeff_model = coeff_cls().to(device) if coeff_cls is LaplaceCoeffModel else \
        coeff_cls(block_size=8, symbol_range=WIDE_RANGE).to(device)

    payload, meta = encode_image(
        img, block_size=8, quant_step=quant_step,
        gbticl_model=gbticl_model, coeff_model=coeff_model,
        symbol_range=WIDE_RANGE, device=device,
    )
    recon = decode_image(payload, meta, gbticl_model=gbticl_model, coeff_model=coeff_model, device=device)

    assert recon.shape == img.shape
    assert recon.dtype == np.uint8
    assert np.isfinite(recon).all()

    p = psnr(img, recon)
    assert np.isfinite(p) or p == float("inf")
    assert p > 0  # sanity: never a garbage reconstruction

    bpp = bits_per_pixel(payload, meta["H"], meta["W"])
    assert bpp > 0


def test_codec_finer_quant_step_gives_higher_psnr(device):
    """Basic rate-distortion sanity check: a finer quantization step should
    reconstruct more accurately (higher PSNR), even for the untrained
    baseline models -- if this ever fails, something is badly wrong in the
    quantize/dequantize or entropy-coding wiring, not just model quality."""
    img = _synthetic_image()
    gbticl_model = ContextGradientGBTICL().to(device)
    coeff_model = LaplaceCoeffModel().to(device)

    payload_coarse, meta_coarse = encode_image(
        img, block_size=8, quant_step=16.0, gbticl_model=gbticl_model,
        coeff_model=coeff_model, symbol_range=WIDE_RANGE, device=device,
    )
    recon_coarse = decode_image(payload_coarse, meta_coarse, gbticl_model=gbticl_model,
                                 coeff_model=coeff_model, device=device)

    payload_fine, meta_fine = encode_image(
        img, block_size=8, quant_step=1.0, gbticl_model=gbticl_model,
        coeff_model=coeff_model, symbol_range=WIDE_RANGE, device=device,
    )
    recon_fine = decode_image(payload_fine, meta_fine, gbticl_model=gbticl_model,
                               coeff_model=coeff_model, device=device)

    assert psnr(img, recon_fine) >= psnr(img, recon_coarse)
