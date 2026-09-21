"""RGB <-> YCbCr conversion (BT.601, full range).

The codec treats the three channels identically, so frames are converted to
YCbCr before encoding and back to RGB after decoding. Chroma is coded at full
resolution (no 4:2:0 subsampling).
"""

import numpy as np


def rgb_to_ycbcr(rgb_u8):
    """(H, W, 3) uint8 RGB -> (H, W, 3) uint8 YCbCr."""
    rgb = rgb_u8.astype(np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
    ycbcr = np.stack([y, cb, cr], axis=-1)
    return np.clip(np.round(ycbcr), 0, 255).astype(np.uint8)


def ycbcr_to_rgb(ycbcr_u8):
    """Inverse of rgb_to_ycbcr."""
    ycbcr = ycbcr_u8.astype(np.float64)
    y, cb, cr = ycbcr[..., 0], ycbcr[..., 1] - 128.0, ycbcr[..., 2] - 128.0
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)
