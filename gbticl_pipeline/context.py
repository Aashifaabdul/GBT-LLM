"""
Context extraction for the encode/decode loop.

Unlike gbticl_context_extraction.py (which precomputes context for every
block of an already-fully-decoded frame, for dataset-prep / inspection),
this version reads context from a `canvas` tensor that is being filled in
block by block as the raster-order loop runs -- i.e. it only ever sees the
same "already decoded" pixels a real decoder would have at that point. This
is the version the actual codec loop uses. `canvas` lives on whatever device
codec.py put it on (CPU or GPU); every tensor returned here stays on that
same device, no implicit transfers.
"""

import torch

PAD_VALUE = 128


def get_context(canvas, i, j, block_size):
    """
    Args:
        canvas: (H, W, 3) uint8 tensor, on some device. Blocks already written
                are real pixels; blocks not yet processed are never read by a
                correct caller (raster order guarantees this).
        i, j: block row/col index
        block_size: int

    Returns:
        top, left: (block_size, 3) uint8 tensors, same device as canvas
        valid_top, valid_left: bool
    """
    device = canvas.device

    if i > 0:
        r0 = (i - 1) * block_size
        c0 = j * block_size
        top = canvas[r0 + block_size - 1, c0:c0 + block_size, :]
        valid_top = True
    else:
        top = torch.full((block_size, 3), PAD_VALUE, dtype=torch.uint8, device=device)
        valid_top = False

    if j > 0:
        r0 = i * block_size
        c0 = (j - 1) * block_size
        left = canvas[r0:r0 + block_size, c0 + block_size - 1, :]
        valid_left = True
    else:
        left = torch.full((block_size, 3), PAD_VALUE, dtype=torch.uint8, device=device)
        valid_left = False

    return top, left, valid_top, valid_left


def get_block(image, i, j, block_size):
    r0, c0 = i * block_size, j * block_size
    return image[r0:r0 + block_size, c0:c0 + block_size, :]


def set_block(canvas, i, j, block_size, block):
    r0, c0 = i * block_size, j * block_size
    canvas[r0:r0 + block_size, c0:c0 + block_size, :] = block
