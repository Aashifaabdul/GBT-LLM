"""Frame extraction and block partitioning of the raw 1080p test sequences.

Reads the raw planar 4:2:0 (I420) 8-bit YUV files of Beauty and HoneyBee,
samples NUM_FRAMES uniformly spaced frames per sequence (so the still-image set
covers the variation of the 120 fps clip rather than near-duplicate frames),
converts them to RGB (BT.601) and, for each frame, writes under
data/<sequence>/ (or $GBTICL_OUT_ROOT/<sequence>/):
  frames/frameNNNN.png                  RGB frame
  blocks/frameNNNN_blocks_8x8.npy       block tensor (n_bh, n_bw, 8, 8, 3), uint8
  block_overlay/frameNNNN_grid.png      frame with the block grid drawn in red
Existing outputs are skipped, so an interrupted run can be resumed.

The raw YUV files are looked for in the parent of the repository folder, or in
$GBTICL_DATASET_ROOT (layout in the CONFIG block). SEQUENCES, NUM_FRAMES and
BLOCK_SIZE are set in the CONFIG block rather than on the command line.

Usage:
    python preprocessing/gbticl_frame_prep.py [Beauty|HoneyBee]
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


import os
from pathlib import Path

import numpy as np
from PIL import Image


# ----------------------------- CONFIG ---------------------------------------
WIDTH, HEIGHT = 1920, 1080
NUM_FRAMES = 30  # frames sampled per sequence
BLOCK_SIZE = 8  # N x N block partition size (matches GFT baseline granularity)


# Paths are resolved relative to the repository. Expected layout of the raw input:
#   <DATASET_ROOT>/Beauty_1920x1080_120fps_420_8bit_YUV_RAW (1)/Beauty_..._YUV.yuv
#   <DATASET_ROOT>/HoneyBee_1920x1080_120fps_420_8bit_YUV_RAW/HoneyBee_..._YUV.yuv
# DATASET_ROOT defaults to the parent of the repository; override it with
# GBTICL_DATASET_ROOT, and the output root (default data/) with GBTICL_OUT_ROOT.
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = Path(os.environ.get("GBTICL_DATASET_ROOT", ROOT_DIR.parent))
OUT_ROOT_PATH = Path(os.environ.get("GBTICL_OUT_ROOT", ROOT_DIR / "data"))

SEQUENCES = {
    "Beauty": str(DATASET_ROOT / "Beauty_1920x1080_120fps_420_8bit_YUV_RAW (1)"
                  / "Beauty_1920x1080_120fps_420_8bit_YUV.yuv"),
    "HoneyBee": str(DATASET_ROOT / "HoneyBee_1920x1080_120fps_420_8bit_YUV_RAW"
                     / "HoneyBee_1920x1080_120fps_420_8bit_YUV.yuv"),
}

OUT_ROOT = str(OUT_ROOT_PATH)


# ------------------------------------------------------------------------------
FRAME_SIZE = WIDTH * HEIGHT + 2 * (WIDTH // 2) * (HEIGHT // 2)  # I420: Y + U/2 + V/2


def read_frame_i420(f, width, height):
    """Read one I420 (planar 4:2:0) frame from an open file and return an HxWx3 uint8 RGB array."""
    y_size = width * height
    c_w, c_h = width // 2, height // 2
    c_size = c_w * c_h

    y = np.frombuffer(f.read(y_size), dtype=np.uint8).reshape(height, width)
    u = np.frombuffer(f.read(c_size), dtype=np.uint8).reshape(c_h, c_w)
    v = np.frombuffer(f.read(c_size), dtype=np.uint8).reshape(c_h, c_w)

    # Nearest-neighbour chroma upsampling (ignores 4:2:0 chroma siting; adequate for still frames).
    u_full = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)
    v_full = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)

    yf = y.astype(np.float32)
    uf = u_full.astype(np.float32) - 128.0
    vf = v_full.astype(np.float32) - 128.0

    # BT.601 full-range YCbCr -> RGB.
    r = yf + 1.402 * vf
    g = yf - 0.344136 * uf - 0.714136 * vf
    b = yf + 1.772 * uf

    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def sample_indices(total_frames, num_samples):
    """Return up to num_samples uniformly spaced frame indices over the whole clip."""
    if num_samples >= total_frames:
        return list(range(total_frames))
    return sorted(set(np.linspace(0, total_frames - 1, num_samples, dtype=int).tolist()))


def block_partition(img, block_size):
    """Split an HxWx3 image into non-overlapping block_size x block_size blocks.

    Returns:
        blocks: array of shape (n_blocks_h, n_blocks_w, block_size, block_size, 3)
        h_crop, w_crop: height and width actually used (trailing rows/columns are
            dropped if the frame is not a multiple of block_size)
    """
    h, w = img.shape[:2]
    h_crop = (h // block_size) * block_size
    w_crop = (w // block_size) * block_size
    img_c = img[:h_crop, :w_crop]

    n_bh = h_crop // block_size
    n_bw = w_crop // block_size

    blocks = img_c.reshape(n_bh, block_size, n_bw, block_size, 3).swapaxes(1, 2)
    return blocks, h_crop, w_crop


def save_block_grid_overlay(img, block_size, out_path):
    """Save a copy of the frame with the block boundaries drawn in red."""
    overlay = img.copy()
    overlay[::block_size, :, :] = [255, 0, 0]
    overlay[:, ::block_size, :] = [255, 0, 0]
    Image.fromarray(overlay).save(out_path)


def main(only_seq=None):
    """Extract and partition the sampled frames of every sequence (or only `only_seq`)."""
    os.makedirs(OUT_ROOT, exist_ok=True)

    seqs = SEQUENCES if only_seq is None else {only_seq: SEQUENCES[only_seq]}

    for seq_name, path in seqs.items():
        file_bytes = os.path.getsize(path)
        total_frames = file_bytes // FRAME_SIZE
        assert file_bytes % FRAME_SIZE == 0, f"{seq_name}: file size not a multiple of frame size"

        idxs = sample_indices(total_frames, NUM_FRAMES)

        frames_dir = os.path.join(OUT_ROOT, seq_name, "frames")
        blocks_dir = os.path.join(OUT_ROOT, seq_name, "blocks")
        overlay_dir = os.path.join(OUT_ROOT, seq_name, "block_overlay")
        for d in (frames_dir, blocks_dir, overlay_dir):
            os.makedirs(d, exist_ok=True)

        print(f"[{seq_name}] total_frames={total_frames}, sampling {len(idxs)} frames at indices {idxs[:5]}...{idxs[-3:]}")

        with open(path, "rb") as f:
            for k, idx in enumerate(idxs):
                # Frames are fixed-size, so seek directly to the sampled index.
                f.seek(idx * FRAME_SIZE)
                rgb = read_frame_i420(f, WIDTH, HEIGHT)

                frame_tag = f"frame{idx:04d}"
                png_path = os.path.join(frames_dir, f"{frame_tag}.png")
                blocks_path = os.path.join(blocks_dir, f"{frame_tag}_blocks_{BLOCK_SIZE}x{BLOCK_SIZE}.npy")
                overlay_path = os.path.join(overlay_dir, f"{frame_tag}_grid.png")

                # Resumable: skip frames whose outputs already exist.
                if os.path.exists(png_path) and os.path.exists(blocks_path) and os.path.exists(overlay_path):
                    print(f"  [{seq_name}] {frame_tag} already done, skipping ({k+1}/{len(idxs)})")
                    continue

                if not os.path.exists(png_path):
                    Image.fromarray(rgb).save(png_path)

                if not os.path.exists(blocks_path):
                    blocks, h_crop, w_crop = block_partition(rgb, BLOCK_SIZE)
                    np.save(blocks_path, blocks)

                if not os.path.exists(overlay_path):
                    save_block_grid_overlay(rgb, BLOCK_SIZE, overlay_path)

                print(f"  [{seq_name}] {frame_tag} done ({k+1}/{len(idxs)})")

        print(f"[{seq_name}] done -> {len(idxs)} PNG frames, {len(idxs)} block tensors "
              f"(shape ~ ({HEIGHT // BLOCK_SIZE}, {WIDTH // BLOCK_SIZE}, {BLOCK_SIZE}, {BLOCK_SIZE}, 3)), "
              f"{len(idxs)} grid overlays\n")


if __name__ == "__main__":
    import sys
    main(only_seq=sys.argv[1] if len(sys.argv) > 1 else None)
