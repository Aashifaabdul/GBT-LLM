"""Extract the small test patches used by neural_compression_benchmark.py.

For each sequence, 5 frames spread evenly over data/<Sequence>/frames/ are
centre-cropped to a 64x64 RGB patch and a 128x128 greyscale patch, saved as PNG
in results/mlic/patches/<Sequence>/.

Usage:
    python comparison/mlic/data_prep.py
"""

import glob
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PATCH_DIR = ROOT / "results" / "mlic" / "patches"

SEQUENCES = ("Beauty", "HoneyBee")
NUM_FRAMES = 5


def centre_crop(img, size):
    """Return the centred size x size crop of a PIL image."""
    w, h = img.size
    left = max(0, (w - size) // 2)
    top = max(0, (h - size) // 2)
    return img.crop((left, top, left + size, top + size))


def prepare_patches():
    """Write the patches and return the list of files written."""
    written = []
    for seq in SEQUENCES:
        frame_dir = DATA_DIR / seq / "frames"
        if not frame_dir.exists() and seq == "Beauty":
            frame_dir = DATA_DIR / seq / "beauty_crop"
        if not frame_dir.exists():
            print(f"Directory not found: {frame_dir}")
            continue

        out_dir = PATCH_DIR / seq
        out_dir.mkdir(parents=True, exist_ok=True)

        frames = sorted(glob.glob(str(frame_dir / "*.png")))
        print(f"{seq}: {len(frames)} frames in {frame_dir}")
        picks = np.linspace(0, len(frames) - 1, min(NUM_FRAMES, len(frames)), dtype=int)

        for idx in picks:
            name = Path(frames[idx]).stem
            with Image.open(frames[idx]) as img:
                patch_rgb = centre_crop(img.convert("RGB"), 64)
                patch_gray = centre_crop(img.convert("L"), 128)
            for suffix, patch in (("patch64_rgb", patch_rgb), ("patch128_gray", patch_gray)):
                path = out_dir / f"{name}_{suffix}.png"
                patch.save(path, format="PNG")
                written.append(path)

    print(f"Wrote {len(written)} patches to {PATCH_DIR}")
    return written


if __name__ == "__main__":
    prepare_patches()
