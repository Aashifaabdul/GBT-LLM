"""Border-padding logic for context extraction: blocks on the top row / left
column of an image have no real neighbour, and must return PAD_VALUE with
valid_top/valid_left=False -- both encoder and decoder rely on this exact
convention being identical on both sides."""

import torch

from gbticl_pipeline.context import get_context, get_block, set_block, PAD_VALUE


def _make_canvas(h=32, w=32, device="cpu"):
    canvas = torch.zeros((h, w, 3), dtype=torch.uint8, device=device)
    # fill with a recognisable ramp so we can check the right pixels come back
    for r in range(h):
        canvas[r, :, 0] = r % 256
    for c in range(w):
        canvas[:, c, 1] = c % 256
    return canvas


def test_top_left_block_has_no_valid_context(device):
    canvas = _make_canvas(device=device)
    top, left, valid_top, valid_left = get_context(canvas, 0, 0, block_size=8)
    assert valid_top is False
    assert valid_left is False
    assert torch.all(top == PAD_VALUE)
    assert torch.all(left == PAD_VALUE)


def test_interior_block_has_valid_context(device):
    canvas = _make_canvas(device=device)
    top, left, valid_top, valid_left = get_context(canvas, 1, 1, block_size=8)
    assert valid_top is True
    assert valid_left is True
    # top context = bottom row of the block above -> row index (1-1)*8+7 = 7
    expected_top = canvas[7, 8:16, :]
    assert torch.equal(top, expected_top)
    # left context = right column of the block to the left -> col index (1-1)*8+7 = 7
    expected_left = canvas[8:16, 7, :]
    assert torch.equal(left, expected_left)


def test_get_set_block_roundtrip(device):
    canvas = _make_canvas(device=device)
    block = get_block(canvas, 1, 2, block_size=8)
    assert block.shape == (8, 8, 3)

    new_block = torch.full((8, 8, 3), 42, dtype=torch.uint8, device=device)
    set_block(canvas, 1, 2, 8, new_block)
    assert torch.all(get_block(canvas, 1, 2, block_size=8) == 42)
    # untouched neighbour block must be unaffected
    assert not torch.all(get_block(canvas, 1, 1, block_size=8) == 42)
