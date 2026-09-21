"""Centre-crop the extracted frames of a sequence to a block-aligned square.

Reads data/<sequence>/frames/*.png (see gbticl_frame_prep.py) and writes a
--crop x --crop centre crop of every frame to data/<sequence>/<out-name>/, where
<out-name> defaults to <sequence in lower case>_crop. The crop size is rounded
down to a multiple of BLOCK_SIZE (8) and limited to the frame size, and the
crop origin is snapped to the 8x8 block grid.

Usage:
    python preprocessing/crop_frames.py --sequence Beauty --crop 256
    python preprocessing/crop_frames.py --sequence HoneyBee --crop 256
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
from pathlib import Path
import numpy as np
from PIL import Image

BLOCK_SIZE = 8


def center_crop(img_np, size, block_size=BLOCK_SIZE):
    """Return the centred size x size crop of an HxWxC array, aligned to the block grid."""
    h, w = img_np.shape[:2]
    size = (size // block_size) * block_size
    size = min(size, (h // block_size) * block_size, (w // block_size) * block_size)
    r0 = (h - size) // 2
    c0 = (w - size) // 2
    # Snap the origin to the block grid so blocks in the crop match those of the full frame.
    r0 -= r0 % block_size
    c0 -= c0 % block_size
    return img_np[r0:r0 + size, c0:c0 + size, :]


def main():
    """Crop all PNG frames of the chosen sequence."""
    parser = argparse.ArgumentParser(description="Crop sequence frames to centered block-aligned square crop.")
    parser.add_argument("--sequence", type=str, default="Beauty", help="Sequence name (e.g. Beauty, HoneyBee)")
    parser.add_argument("--crop", type=int, default=256, help="Crop size in pixels (default: 256)")
    parser.add_argument("--out-name", type=str, default=None, help="Output folder name inside sequence directory (default: <sequence_lowercase>_crop)")
    args = parser.parse_args()

    script_dir = ROOT_DIR / "data"
    src_dir = script_dir / args.sequence / "frames"

    out_folder_name = args.out_name or f"{args.sequence.lower()}_crop"
    dst_dir = script_dir / args.sequence / out_folder_name
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_files = sorted(src_dir.glob("*.png"))
    if not src_files:
        print(f"No PNG frames found in {src_dir}")
        return

    print(f"Cropping {len(src_files)} frames from {src_dir} to {dst_dir} ({args.crop}x{args.crop} center crop)...")
    for p in src_files:
        img = np.array(Image.open(p).convert("RGB"))
        cropped = center_crop(img, args.crop, BLOCK_SIZE)
        Image.fromarray(cropped).save(dst_dir / p.name)

    print(f"Done! Successfully saved {len(src_files)} cropped frames into {dst_dir}")


if __name__ == "__main__":
    main()
