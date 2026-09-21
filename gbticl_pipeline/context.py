"""
Causal context and support-set extraction for the encode/decode loop.

All functions read from a `canvas` tensor that is filled in block by block in
raster order, so they only ever see pixels that the decoder also has at that
point. Returned tensors stay on the device of the canvas.
"""

import torch

PAD_VALUE = 128


def get_context(canvas, i, j, block_size):
    """
    Causal context of block (i, j): the last row of the block above and the last
    column of the block to its left. Missing context (image border) is padded
    with PAD_VALUE and flagged invalid.

    Args:
        canvas: (H, W, 3) uint8 tensor holding the blocks decoded so far
        i, j: block row/column index
        block_size: int

    Returns:
        top, left: (block_size, 3) uint8 tensors on the canvas device
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
    """View of block (i, j) of `image`."""
    r0, c0 = i * block_size, j * block_size
    return image[r0:r0 + block_size, c0:c0 + block_size, :]


def set_block(canvas, i, j, block_size, block):
    """Write `block` into position (i, j) of `canvas`."""
    r0, c0 = i * block_size, j * block_size
    canvas[r0:r0 + block_size, c0:c0 + block_size, :] = block


# Support set for GBTICLMetaLearner: other, already-decoded blocks (never the
# query block itself). Offsets are (di, dj) relative to the query block.
#
# Spatial: causal neighbours in the current frame (left, top, top-left, top-right).
_SPATIAL_OFFSETS = [(0, -1), (-1, 0), (-1, -1), (-1, 1)]

# Temporal: neighbours in the previous, fully reconstructed frame (co-located,
# above, below, left); any position is available there, not only causal ones.
_TEMPORAL_OFFSETS = [(0, 0), (-1, 0), (1, 0), (0, -1)]

N_SUPPORT = len(_SPATIAL_OFFSETS) + len(_TEMPORAL_OFFSETS)  # 8


def get_support_set(canvas, prev_canvas, i, j, block_size):
    """
    Assemble the support set for query block (i, j): up to 4 spatial neighbours
    from `canvas` and up to 4 temporal neighbours from `prev_canvas` (None for
    the first frame, in which case the temporal slots are marked invalid).

    Returns a dict of tensors stacked over N_SUPPORT = 8 slots, on the canvas device:
        top, left:             (N_SUPPORT, block_size, 3) uint8 context of each support block
        valid_top, valid_left: (N_SUPPORT,) bool
        block:                 (N_SUPPORT, block_size, block_size, 3) uint8 pixels of each
                               support block (the caller derives reference graphs from them)
        valid:                 (N_SUPPORT,) bool, False for slots that fall outside the
                               frame or have no previous frame
    """
    device = canvas.device
    H, W, _ = canvas.shape
    n_bh, n_bw = H // block_size, W // block_size

    tops, lefts, valid_tops, valid_lefts, blocks, valids = [], [], [], [], [], []

    def _pad_ctx():
        return torch.full((block_size, 3), PAD_VALUE, dtype=torch.uint8, device=device)

    def _pad_block():
        return torch.full((block_size, block_size, 3), PAD_VALUE, dtype=torch.uint8, device=device)

    for di, dj in _SPATIAL_OFFSETS:
        si, sj = i + di, j + dj
        if 0 <= si < n_bh and 0 <= sj < n_bw:
            top, left, vt, vl = get_context(canvas, si, sj, block_size)
            block = get_block(canvas, si, sj, block_size)
            tops.append(top); lefts.append(left)
            valid_tops.append(vt); valid_lefts.append(vl)
            blocks.append(block); valids.append(True)
        else:
            tops.append(_pad_ctx()); lefts.append(_pad_ctx())
            valid_tops.append(False); valid_lefts.append(False)
            blocks.append(_pad_block()); valids.append(False)

    for di, dj in _TEMPORAL_OFFSETS:
        si, sj = i + di, j + dj
        if prev_canvas is not None and 0 <= si < n_bh and 0 <= sj < n_bw:
            top, left, vt, vl = get_context(prev_canvas, si, sj, block_size)
            block = get_block(prev_canvas, si, sj, block_size)
            tops.append(top); lefts.append(left)
            valid_tops.append(vt); valid_lefts.append(vl)
            blocks.append(block); valids.append(True)
        else:
            tops.append(_pad_ctx()); lefts.append(_pad_ctx())
            valid_tops.append(False); valid_lefts.append(False)
            blocks.append(_pad_block()); valids.append(False)

    return dict(
        top=torch.stack(tops), left=torch.stack(lefts),
        valid_top=torch.tensor(valid_tops, device=device, dtype=torch.bool),
        valid_left=torch.tensor(valid_lefts, device=device, dtype=torch.bool),
        block=torch.stack(blocks),
        valid=torch.tensor(valids, device=device, dtype=torch.bool),
    )
