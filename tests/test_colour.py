import numpy as np

from gbticl_pipeline.colour import rgb_to_ycbcr, ycbcr_to_rgb


def test_ycbcr_roundtrip_near_lossless():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    recon = ycbcr_to_rgb(rgb_to_ycbcr(rgb))
    # rounding through the forward+inverse matrix isn't bit-exact, but
    # should be within a couple of levels
    assert np.max(np.abs(recon.astype(int) - rgb.astype(int))) <= 2


def test_ycbcr_grey_is_128_128():
    grey = np.full((4, 4, 3), 128, dtype=np.uint8)
    ycbcr = rgb_to_ycbcr(grey)
    assert np.allclose(ycbcr[..., 1], 128, atol=1)
    assert np.allclose(ycbcr[..., 2], 128, atol=1)
