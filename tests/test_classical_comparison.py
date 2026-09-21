"""comparison/classical_comparison.py: size functions of the lossless codecs."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "comparison"))

import classical_comparison as cc  # noqa: E402


def _noise_image(seed=0, size=64):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8))


def test_flat_image_compresses_far_below_raw_size():
    img = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))
    raw = np.array(img).tobytes()
    for size in (cc.gzip_size(raw), cc.lzma_size(raw), cc.png_size(img), cc.webp_size(img)):
        assert 0 < size < len(raw) // 20


def test_noise_is_nearly_incompressible():
    img = _noise_image()
    raw = np.array(img).tobytes()
    for size in (cc.gzip_size(raw), cc.lzma_size(raw), cc.png_size(img), cc.webp_size(img)):
        assert size > 0.9 * len(raw)


def test_sizes_are_deterministic():
    img = _noise_image(seed=1)
    raw = np.array(img).tobytes()
    assert cc.gzip_size(raw) == cc.gzip_size(raw)
    assert cc.png_size(img) == cc.png_size(img)
    assert cc.webp_size(img) == cc.webp_size(img)
