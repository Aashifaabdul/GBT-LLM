"""Downscale the extracted frames of a sequence to a smaller full frame.

Reads data/<sequence>/frames/*.png (produced by gbticl_frame_prep.py) and
writes every frame, resized with LANCZOS to --width x --height (both multiples
of 8), to data/<sequence>/frames_<W>x<H>/. Unlike the centre crop of
run_dataset_pipeline.py --crop, the whole picture is kept at a lower
resolution. This gives a reduced-size input for the codec, whose per-symbol
range coder makes the run time scale with the block count (W/8)*(H/8). The
native frames are never overwritten; the output folder is passed to
run_dataset_pipeline.py with --frames-dir and --full-frame.

Usage:
    python preprocessing/resize_frames.py --sequence Beauty --width 128 --height 72
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


import argparse
from pathlib import Path

from PIL import Image

BLOCK_SIZE = 8


def main():
    """Resize every PNG frame of the chosen sequence and print the follow-up command."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", choices=["Beauty", "HoneyBee"], required=True)
    ap.add_argument("--width", type=int, required=True, help="Target width, must be a multiple of 8.")
    ap.add_argument("--height", type=int, required=True, help="Target height, must be a multiple of 8.")
    args = ap.parse_args()

    if args.width % BLOCK_SIZE or args.height % BLOCK_SIZE:
        raise SystemExit(f"--width/--height must be multiples of {BLOCK_SIZE} "
                          f"(got {args.width}x{args.height})")

    script_dir = ROOT_DIR / "data"
    src_dir = script_dir / args.sequence / "frames"
    dst_dir = script_dir / args.sequence / f"frames_{args.width}x{args.height}"
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_paths = sorted(src_dir.glob("*.png"))
    if not src_paths:
        raise SystemExit(f"no frames found in {src_dir} -- run preprocessing/gbticl_frame_prep.py first")

    print(f"downscaling {len(src_paths)} frames: {src_dir} -> {dst_dir} "
          f"(whole frame, {args.width}x{args.height}, LANCZOS)")
    for p in src_paths:
        img = Image.open(p).convert("RGB")
        img = img.resize((args.width, args.height), Image.LANCZOS)
        img.save(dst_dir / p.name)

    print(f"done. Run with:\n"
          f"  python run_dataset_pipeline.py --sequence {args.sequence} --ablation full "
          f"--gbticl-checkpoint checkpoints/stageA.pt --coeff-checkpoint checkpoints/stageC.pt "
          f"--frames-dir {dst_dir} --full-frame")


if __name__ == "__main__":
    main()
