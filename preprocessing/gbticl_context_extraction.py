"""Neighbour-context extraction for every 8x8 block of the sampled frames.

Reads the block tensors data/<sequence>/blocks/frameNNNN_blocks_8x8.npy written
by gbticl_frame_prep.py and, for every block, extracts the pixels of the two
causal neighbours (raster order) that the graph predictor conditions on:
  top  = bottom row of the block above         -> blocks[i-1, j, -1, :, :]
  left = right column of the block to the left -> blocks[i, j-1, :, -1, :]
Where a neighbour does not exist (first block row / column) the side is filled
with PAD_VALUE (128, mid-grey). The data root data/ can be changed with
$GBTICL_OUT_ROOT.

The context is taken from the original pixels, not from reconstructed ones. The
codec (gbticl_pipeline.codec) builds its context from its own reconstructed
canvas, so encoder and decoder see identical context; this script only dumps
the idealised (unquantised) context for inspection, and nothing else in the
repository reads its output.

Output data/<sequence>/context/frameNNNN_context.npz (existing files are skipped):
  top        (n_bh, n_bw, block_size, 3)  uint8  top-neighbour border pixels or padding
  left       (n_bh, n_bw, block_size, 3)  uint8  left-neighbour border pixels or padding
  valid_top  (n_bh, n_bw)                 bool   True if a real top neighbour exists
  valid_left (n_bh, n_bw)                 bool   True if a real left neighbour exists

Usage:
    python preprocessing/gbticl_context_extraction.py [Beauty|HoneyBee]
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


import os
import glob
from pathlib import Path

import numpy as np


# ----------------------------- CONFIG ---------------------------------------
BLOCK_SIZE = 8
PAD_VALUE = 128  # mid-grey, also the chroma zero point


DATASET_ROOT = os.environ.get("GBTICL_OUT_ROOT", str(ROOT_DIR / "data"))
SEQUENCES = ["Beauty", "HoneyBee"]


def extract_context_for_frame(blocks, block_size, pad_value):
    """Build top/left context arrays for every block in one frame's block tensor.

    Args:
        blocks: array (n_bh, n_bw, block_size, block_size, 3) from block_partition() in gbticl_frame_prep.py

    Returns:
        top, left: (n_bh, n_bw, block_size, 3) uint8
        valid_top, valid_left: (n_bh, n_bw) bool
    """
    n_bh, n_bw = blocks.shape[0], blocks.shape[1]

    top = np.full((n_bh, n_bw, block_size, 3), pad_value, dtype=np.uint8)
    left = np.full((n_bh, n_bw, block_size, 3), pad_value, dtype=np.uint8)
    valid_top = np.zeros((n_bh, n_bw), dtype=bool)
    valid_left = np.zeros((n_bh, n_bw), dtype=bool)

    for i in range(n_bh):
        for j in range(n_bw):
            if i > 0:
                # bottom row of the block above, shape (block_size, 3)
                top[i, j] = blocks[i - 1, j, -1, :, :]
                valid_top[i, j] = True
            if j > 0:
                # right column of the block to the left, shape (block_size, 3)
                left[i, j] = blocks[i, j - 1, :, -1, :]
                valid_left[i, j] = True

    return top, left, valid_top, valid_left


def main(only_seq=None):
    """Write a context file for every block file of each sequence (or only `only_seq`)."""
    seqs = SEQUENCES if only_seq is None else [only_seq]

    for seq_name in seqs:
        blocks_dir = os.path.join(DATASET_ROOT, seq_name, "blocks")
        context_dir = os.path.join(DATASET_ROOT, seq_name, "context")
        os.makedirs(context_dir, exist_ok=True)

        block_files = sorted(glob.glob(os.path.join(blocks_dir, "*_blocks_*.npy")))
        print(f"[{seq_name}] found {len(block_files)} block files")

        for bf in block_files:
            frame_tag = os.path.basename(bf).split("_blocks_")[0]  # e.g. "frame0020"
            out_path = os.path.join(context_dir, f"{frame_tag}_context.npz")

            if os.path.exists(out_path):
                continue  # resumable, as in gbticl_frame_prep.py

            blocks = np.load(bf)
            top, left, valid_top, valid_left = extract_context_for_frame(
                blocks, BLOCK_SIZE, PAD_VALUE
            )
            np.savez(out_path, top=top, left=left, valid_top=valid_top, valid_left=valid_left)
            print(f"  [{seq_name}] {frame_tag} -> context saved "
                  f"(top {top.shape}, left {left.shape})")

        print(f"[{seq_name}] done -> {len(block_files)} context files in {context_dir}\n")


if __name__ == "__main__":
    import sys
    main(only_seq=sys.argv[1] if len(sys.argv) > 1 else None)
