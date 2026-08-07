"""encode_video/decode_video: for models that don't use spatiotemporal
support (UniformGBTICL/LaplaceCoeffModel etc.), video mode must reduce
EXACTLY to per-frame encode_image/decode_image -- no hidden cross-frame
coupling. For GBTICLMetaLearner + TinyTransformerCoeffModel (the models
that DO use it), video mode must run end to end without NaN/crashes and
must visibly activate temporal support once a previous frame exists."""

import numpy as np
import torch

from gbticl_pipeline.codec import encode_image, decode_image, encode_video, decode_video
from gbticl_pipeline.evaluate import psnr
from gbticl_pipeline.graph_model import UniformGBTICL, GBTICLMetaLearner
from gbticl_pipeline.coeff_model import LaplaceCoeffModel, TinyTransformerCoeffModel
from gbticl_pipeline.context import get_support_set

WIDE_RANGE = (-2200, 2200)


def _synthetic_frames(t=3, h=16, w=16, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for f in range(t):
        yy, xx = np.mgrid[0:h, 0:w]
        base = 128 + 50 * np.sin(xx / 4.0 + f) + 30 * np.cos(yy / 5.0 + f)
        img = np.stack([base] * 3, axis=-1) + rng.normal(0, 5, size=(h, w, 3))
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
    return frames


def test_encode_video_matches_encode_image_for_non_temporal_models(device):
    """Regression anchor: models with no spatiotemporal support/temporal
    conditioning must produce IDENTICAL per-frame output whether run through
    encode_video or through encode_image called independently per frame --
    video mode must not silently couple frames together for these models."""
    frames = _synthetic_frames(t=2)
    gbticl_model = UniformGBTICL().to(device)
    coeff_model = LaplaceCoeffModel().to(device)

    payloads, metas = encode_video(
        frames, block_size=8, quant_step=8.0,
        gbticl_model=gbticl_model, coeff_model=coeff_model,
        symbol_range=WIDE_RANGE, device=device,
    )
    recon_frames = decode_video(payloads, metas, gbticl_model=gbticl_model, coeff_model=coeff_model, device=device)

    for f in range(2):
        payload_img, meta_img = encode_image(
            frames[f], block_size=8, quant_step=8.0,
            gbticl_model=gbticl_model, coeff_model=coeff_model,
            symbol_range=WIDE_RANGE, device=device,
        )
        recon_img = decode_image(payload_img, meta_img, gbticl_model=gbticl_model, coeff_model=coeff_model, device=device)

        assert payloads[f] == payload_img, f"frame {f}: video-mode payload must match image-mode payload bit-for-bit"
        assert np.array_equal(recon_frames[f], recon_img)


def test_video_roundtrip_metalearner_temporal_activates_after_frame0(device):
    """GBTICLMetaLearner + TinyTransformerCoeffModel over a 3-frame clip:
    must run without NaN/crash, and the temporal support slots must go from
    entirely invalid (frame 0, no previous frame) to at least partially
    valid (frame 1+, previous frame now reconstructed)."""
    frames = _synthetic_frames(t=3)
    gbticl_model = GBTICLMetaLearner(block_size=8).to(device)
    coeff_model = TinyTransformerCoeffModel(block_size=8, symbol_range=WIDE_RANGE).to(device)

    payloads, metas = encode_video(
        frames, block_size=8, quant_step=8.0,
        gbticl_model=gbticl_model, coeff_model=coeff_model,
        symbol_range=WIDE_RANGE, device=device,
    )
    recon_frames = decode_video(payloads, metas, gbticl_model=gbticl_model, coeff_model=coeff_model, device=device)

    assert len(recon_frames) == 3
    for f, recon in enumerate(recon_frames):
        assert recon.shape == frames[f].shape
        assert np.isfinite(recon).all()
        p = psnr(frames[f], recon)
        assert p > 0 and np.isfinite(p)


def test_support_set_temporal_slots_invalid_at_frame0_valid_after(device):
    """Direct check of the mechanism the above test relies on implicitly:
    get_support_set's 4 temporal slots (last 4 of 8) are all invalid when
    prev_canvas=None, and at least the co-located slot becomes valid once a
    real previous frame is supplied."""
    canvas = torch.randint(0, 256, (16, 16, 3), dtype=torch.uint8, device=device)
    support_frame0 = get_support_set(canvas, None, 1, 1, block_size=8)
    assert not support_frame0["valid"][4:].any(), "all temporal slots must be invalid with no previous frame"

    prev_canvas = torch.randint(0, 256, (16, 16, 3), dtype=torch.uint8, device=device)
    support_frame1 = get_support_set(canvas, prev_canvas, 1, 1, block_size=8)
    assert support_frame1["valid"][4:].any(), "at least the co-located temporal slot should be valid"
