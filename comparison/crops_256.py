"""Test images shared by the 256x256 comparisons (classical_comparison.py,
mlic/, deepmind/).

Beauty:   the first NUM_IMAGES crops in data/Beauty/beauty_crop/
          (python preprocessing/crop_frames.py --sequence Beauty --crop 256).
HoneyBee: the exact centre 256x256 crop of the first NUM_IMAGES frames in
          data/HoneyBee/frames/, saved to data/HoneyBee/center_crop_256/.
"""

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BEAUTY_CROP_DIR = ROOT / "data" / "Beauty" / "beauty_crop"
HONEYBEE_FRAME_DIR = ROOT / "data" / "HoneyBee" / "frames"
HONEYBEE_CROP_DIR = ROOT / "data" / "HoneyBee" / "center_crop_256"

CROP = 256
NUM_IMAGES = 5


def collect_samples():
    """Return a list of {'sequence', 'path'} dicts, Beauty first, then HoneyBee."""
    HONEYBEE_CROP_DIR.mkdir(parents=True, exist_ok=True)

    beauty = sorted(BEAUTY_CROP_DIR.glob("*.png"))[:NUM_IMAGES]
    if not beauty:
        raise FileNotFoundError(
            f"No Beauty crops in {BEAUTY_CROP_DIR}; run "
            "`python preprocessing/crop_frames.py --sequence Beauty --crop 256`."
        )
    honeybee = sorted(HONEYBEE_FRAME_DIR.glob("*.png"))[:NUM_IMAGES]
    if not honeybee:
        raise FileNotFoundError(
            f"No HoneyBee frames in {HONEYBEE_FRAME_DIR}; run "
            "`python preprocessing/gbticl_frame_prep.py`."
        )

    samples = [{"sequence": "Beauty", "path": p} for p in beauty]

    for path in honeybee:
        im = Image.open(path).convert("RGB")
        w, h = im.size
        left, top = (w - CROP) // 2, (h - CROP) // 2
        crop_path = HONEYBEE_CROP_DIR / f"honey_{path.name}"
        im.crop((left, top, left + CROP, top + CROP)).save(crop_path)
        samples.append({"sequence": "HoneyBee", "path": crop_path})
    return samples
