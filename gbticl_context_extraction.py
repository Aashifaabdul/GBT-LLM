"""
GBT-ICL pipeline — Stage 2: Context Extraction.

Reads the block tensors produced by gbticl_frame_prep.py (Stage 1: Block
Partition) and, for every block in the frame, extracts the "already decoded"
neighbor pixels that GBT-ICL will condition on to predict that block's graph
Laplacian.

IMPORTANT SIMPLIFICATION (documented, not hidden): a real encoder builds this
context from its own RECONSTRUCTED (post-quantization) neighbor pixels, so
the encoder and decoder see byte-identical context with zero signaling.
Quantization / GFT haven't been implemented yet in this pipeline, so this
script uses the original neighbor pixels as a stand-in "reconstruction". Once
quantization + inverse-GFT exist, swap in the real reconstructed buffer here
instead of the raw block tensor -- the raster-order traversal and edge-padding
logic below stay the same either way.

Neighbor rule (raster order: left-to-right, top-to-bottom):
  - "top" context  = the bottom row of the block directly above   -> blocks[i-1, j, -1, :, :]
  - "left" context = the right column of the block directly left  -> blocks[i, j-1, :, -1, :]
  - if a neighbor doesn't exist (top row / left column of the image),
    that side is filled with a constant PAD_VALUE instead.

Output per frame: a single .npz with:
  top       (n_bh, n_bw, block_size, 3)  uint8 -- top-neighbor border pixels (or padding)
  left      (n_bh, n_bw, block_size, 3)  uint8 -- left-neighbor border pixels (or padding)
  valid_top (n_bh, n_bw)                 bool  -- True if a real top neighbor existed
  valid_left(n_bh, n_bw)                 bool  -- True if a real left neighbor existed

Usage:
    python3 gbticl_context_extraction.py [SequenceName]
"""

import os
import glob
from pathlib import Path

import numpy as np

# ----------------------------- CONFIG ---------------------------------------

BLOCK_SIZE = 8
PAD_VALUE = 128  # neutral gray -- same convention used for chroma zero-point

# Resolved relative to this script's own location (same root gbticl_frame_prep.py
# writes to). Override with GBTICL_OUT_ROOT if needed.
DATASET_ROOT = os.environ.get("GBTICL_OUT_ROOT", str(Path(__file__).resolve().parent))
SEQUENCES = ["Beauty", "HoneyBee"]

# ------------------------------------------------------------------------------


def extract_context_for_frame(blocks, block_size, pad_value):
    """Build top/left context arrays for every block in one frame's block tensor.

    Args:
        blocks: array (n_bh, n_bw, block_size, block_size, 3) from block_partition()

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
                # bottom row of the block above: shape (block_size, 3)
                top[i, j] = blocks[i - 1, j, -1, :, :]
                valid_top[i, j] = True
            if j > 0:
                # right column of the block to the left: shape (block_size, 3)
                left[i, j] = blocks[i, j - 1, :, -1, :]
                valid_left[i, j] = True

    return top, left, valid_top, valid_left


def main(only_seq=None):
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
                continue  # resumable, same pattern as Stage 1

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
