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


# ---------------------------------------------------------------------------
# Support-set assembly for GBTICLMetaLearner (graph_model.py) -- the few-shot
# in-context conditioning set for a query block, built from OTHER already-
# decoded blocks (never the query block itself, which isn't decoded yet).
# ---------------------------------------------------------------------------

# Fixed candidate offsets, in (di, dj) relative to the query block (i, j).
# Spatial: causal neighbours in the CURRENT (in-progress) frame -- all
# guaranteed already-decoded in raster order (di<0, or di==0 and dj<0).
_SPATIAL_OFFSETS = [(0, -1), (-1, 0), (-1, -1), (-1, 1)]  # left, top, top-left, top-right
# Temporal: neighbours in the PREVIOUS (fully reconstructed) frame -- any
# position is valid there since the whole frame is already known, not just
# a causal subset.
_TEMPORAL_OFFSETS = [(0, 0), (-1, 0), (1, 0), (0, -1)]  # co-located, up, down, left

N_SUPPORT = len(_SPATIAL_OFFSETS) + len(_TEMPORAL_OFFSETS)  # 8


def get_support_set(canvas, prev_canvas, i, j, block_size):
    """
    Assemble the few-shot in-context support set for query block (i, j):
    up to 4 already-decoded spatial neighbours from `canvas` (the frame
    currently being encoded/decoded) plus up to 4 neighbours from
    `prev_canvas` (the previous frame, fully reconstructed already, or None
    for the first frame of a sequence -- in which case all 4 temporal slots
    are simply marked invalid and GBTICLMetaLearner falls back to its
    learned "missing support" embedding for them, degrading gracefully to
    spatial-only support).

    Each support item's reference edge weights are computed directly from
    ITS OWN true (already-decoded) pixels via
    graph_model.reference_edge_weights() -- never transmitted, since encoder
    and decoder each independently reconstruct the same pixels for any
    already-decoded block, at the point they need them.

    Returns a dict of stacked tensors, all length N_SUPPORT=8, on the same
    device as `canvas`:
        top, left:            (N_SUPPORT, block_size, 3) uint8
        valid_top, valid_left: (N_SUPPORT,) bool
        block:                (N_SUPPORT, block_size, block_size, 3) uint8
                               -- the support block's own true pixels, for
                               the caller to derive reference weights from
                               (kept separate from graph_model to avoid a
                               circular import between context.py and
                               graph_model.py)
        valid: (N_SUPPORT,) bool -- False for entirely-missing slots (off
               top of the current frame, or no previous frame at all)
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
