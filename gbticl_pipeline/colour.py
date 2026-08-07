"""RGB <-> YCbCr colour conversion, BT.601 full-range -- the same matrix
gbticl_frame_prep.py already uses for its raw-YUV-to-RGB extraction, so
round-tripping a frame through rgb_to_ycbcr -> (codec) -> ycbcr_to_rgb uses
one consistent colour convention throughout the whole pipeline.

Why this exists: the codec's internals (gft.py, quantization.py, the range
coder) are already channel-agnostic -- they loop `for ch in range(3)` and
treat every channel identically, with no assumption about what the 3
channels represent. That means converting to YCbCr before encoding needs no
changes to the codec itself, just a thin colourspace wrapper called
immediately before encode_image/encode_video and immediately after
decode_image/decode_video. This gives standard, literature-comparable
Y-PSNR/Y-SSIM (the usual headline metric in the compression literature)
while still producing full-colour RGB reconstructions for display.

SCOPE NOTE: no 4:2:0 chroma subsampling here -- Cb/Cr are coded at full
resolution, same as Y. Chroma subsampling is a real, standard technique
(and free bitrate savings) but is explicitly out of scope for v1; coding
Cb/Cr at full resolution is simpler and doesn't compromise evaluation of
the core GBT-ICL / LLM-coefficient-predictor contribution, which is
per-channel-identical regardless of subsampling.
"""

import numpy as np


def rgb_to_ycbcr(rgb_u8):
    """rgb_u8: (H, W, 3) uint8 -> (H, W, 3) uint8, channel order (Y, Cb, Cr)."""
    rgb = rgb_u8.astype(np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
    ycbcr = np.stack([y, cb, cr], axis=-1)
    return np.clip(np.round(ycbcr), 0, 255).astype(np.uint8)


def ycbcr_to_rgb(ycbcr_u8):
    """Inverse of rgb_to_ycbcr -- same BT.601 full-range matrix
    gbticl_frame_prep.py's read_frame_i420() already uses for YUV->RGB, so
    both conversions in this project agree on one convention."""
    ycbcr = ycbcr_u8.astype(np.float64)
    y, cb, cr = ycbcr[..., 0], ycbcr[..., 1] - 128.0, ycbcr[..., 2] - 128.0
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)
