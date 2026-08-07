"""
GBT-ICL pipeline — Stage 0/1 data prep: frame extraction + block partitioning.

Stage 0 (this script, part A): read raw planar 4:2:0 8-bit YUV video, convert
to RGB, and uniformly sample N frames per sequence so the still-image test set
covers the sequence's visual variation (rather than N near-duplicate frames
from a 120fps slow-motion clip).

Stage 1 (this script, part B): partition each sampled frame into non-overlapping
N x N pixel blocks (the "Block Partition" stage in gbticl_architecture.svg),
saving both the block tensor (for the downstream Context Extraction / GBT-ICL
model) and a visual grid overlay for sanity-checking.

Usage:
    python3 gbticl_frame_prep.py

Config is set in the CONFIG block below — edit SEQUENCES / NUM_FRAMES /
BLOCK_SIZE as needed rather than passing CLI flags, to keep this reproducible
as a single documented artifact for the dissertation.
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image

# ----------------------------- CONFIG ---------------------------------------

WIDTH, HEIGHT = 1920, 1080
NUM_FRAMES = 30          # frames sampled per sequence (see rationale in README note below)
BLOCK_SIZE = 8           # N x N block partition size (matches GFT baseline granularity)

# Resolved relative to this script's own location, so it works on any machine
# without editing paths by hand. Layout expected:
#   <DATASET_ROOT>/Beauty_1920x1080_120fps_420_8bit_YUV_RAW (1)/Beauty_..._YUV.yuv
#   <DATASET_ROOT>/HoneyBee_1920x1080_120fps_420_8bit_YUV_RAW/HoneyBee_..._YUV.yuv
# Override with env vars GBTICL_DATASET_ROOT / GBTICL_OUT_ROOT if your raw YUV
# files live somewhere else.
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = Path(os.environ.get("GBTICL_DATASET_ROOT", SCRIPT_DIR.parent))
OUT_ROOT_PATH = Path(os.environ.get("GBTICL_OUT_ROOT", SCRIPT_DIR))

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
    """Read one I420 (planar 4:2:0) frame and return an HxWx3 uint8 RGB array."""
    y_size = width * height
    c_w, c_h = width // 2, height // 2
    c_size = c_w * c_h

    y = np.frombuffer(f.read(y_size), dtype=np.uint8).reshape(height, width)
    u = np.frombuffer(f.read(c_size), dtype=np.uint8).reshape(c_h, c_w)
    v = np.frombuffer(f.read(c_size), dtype=np.uint8).reshape(c_h, c_w)

    # Upsample chroma to full resolution (nearest-neighbor, matches 4:2:0 co-siting closely enough for still-frame extraction)
    u_full = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)
    v_full = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)

    yf = y.astype(np.float32)
    uf = u_full.astype(np.float32) - 128.0
    vf = v_full.astype(np.float32) - 128.0

    # BT.601 YUV -> RGB (matches typical camera-source test sequences of this era)
    r = yf + 1.402 * vf
    g = yf - 0.344136 * uf - 0.714136 * vf
    b = yf + 1.772 * uf

    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def sample_indices(total_frames, num_samples):
    """Uniformly spaced frame indices across the whole clip (covers motion/content variation)."""
    if num_samples >= total_frames:
        return list(range(total_frames))
    return sorted(set(np.linspace(0, total_frames - 1, num_samples, dtype=int).tolist()))


def block_partition(img, block_size):
    """Split an HxWx3 image into non-overlapping block_size x block_size blocks.

    Returns:
        blocks: array of shape (n_blocks_h, n_blocks_w, block_size, block_size, 3)
        (cropped H, cropped W actually used, in case dims aren't multiples of block_size)
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
    """Draw block boundaries on the frame for visual QA."""
    overlay = img.copy()
    overlay[::block_size, :, :] = [255, 0, 0]
    overlay[:, ::block_size, :] = [255, 0, 0]
    Image.fromarray(overlay).save(out_path)


def main(only_seq=None):
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
            last_pos = 0
            for k, idx in enumerate(idxs):
                # seek forward to the target frame (frames are read sequentially/monotonically since idxs sorted)
                f.seek(idx * FRAME_SIZE)
                rgb = read_frame_i420(f, WIDTH, HEIGHT)

                frame_tag = f"frame{idx:04d}"
                png_path = os.path.join(frames_dir, f"{frame_tag}.png")
                blocks_path = os.path.join(blocks_dir, f"{frame_tag}_blocks_{BLOCK_SIZE}x{BLOCK_SIZE}.npy")
                overlay_path = os.path.join(overlay_dir, f"{frame_tag}_grid.png")

                # resumable: skip work already done (this script is called repeatedly
                # across short-lived shell invocations, so idempotent skipping matters)
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
