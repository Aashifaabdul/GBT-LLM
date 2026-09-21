"""Classical lossless codecs (WebP, PNG, LZMA, Gzip) on the 256x256 test crops.

Per image, the compressed size in bits per byte of the raw RGB data (multiply by
3 for bits per pixel). Codec settings: Gzip level 9, LZMA preset 9 (extreme),
PNG with optimisation, WebP lossless method 6.

Test images: see crops_256.py.
Output: results/classical_comparison/classical_256x256_results.json

Usage:
    python comparison/classical_comparison.py
"""

import gzip
import io
import json
import lzma
from pathlib import Path

import numpy as np
from PIL import Image

from crops_256 import collect_samples

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "classical_comparison"


def gzip_size(data_bytes):
    """Compressed size in bytes: Gzip, level 9."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9) as f:
        f.write(data_bytes)
    return len(buf.getvalue())


def lzma_size(data_bytes):
    """Compressed size in bytes: LZMA, preset 9 with the extreme flag."""
    return len(lzma.compress(data_bytes, preset=9 | lzma.PRESET_EXTREME))


def png_size(image):
    """Compressed size in bytes: lossless PNG with optimisation."""
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return len(buf.getvalue())


def webp_size(image):
    """Compressed size in bytes: lossless WebP, method 6."""
    buf = io.BytesIO()
    image.save(buf, format="WEBP", lossless=True, quality=100, method=6)
    return len(buf.getvalue())


def print_summary(results):
    """Print the mean bits per byte and bits per pixel of each codec per sequence."""
    codecs = [("WebP", "webp_bpb"), ("PNG", "png_bpb"), ("LZMA", "lzma_bpb"), ("Gzip", "gzip_bpb")]
    print(f"{'Codec':<6} {'Sequence':<9} {'bpb':>7} {'bpp':>7}")
    for name, key in codecs:
        for seq in ("Beauty", "HoneyBee"):
            bpb = float(np.mean([r[key] for r in results if r["sequence"] == seq]))
            print(f"{name:<6} {seq:<9} {bpb:7.2f} {bpb * 3:7.2f}")


def main():
    """Compress every test image and write classical_256x256_results.json."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = collect_samples()
    print(f"Test images: {len(samples)}")

    results = []
    for i, sample in enumerate(samples, 1):
        img = Image.open(sample["path"]).convert("RGB")
        w, h = img.size
        raw_bytes = np.array(img, dtype=np.uint8).tobytes()
        raw_size = len(raw_bytes)
        print(f"[{i}/{len(samples)}] {sample['sequence']} {sample['path'].name}")

        results.append({
            "sequence": sample["sequence"],
            "file": sample["path"].name,
            "width": w,
            "height": h,
            "gzip_bpb": gzip_size(raw_bytes) * 8.0 / raw_size,
            "lzma_bpb": lzma_size(raw_bytes) * 8.0 / raw_size,
            "png_bpb": png_size(img) * 8.0 / raw_size,
            "webp_bpb": webp_size(img) * 8.0 / raw_size,
        })

    out_json = OUT_DIR / "classical_256x256_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_json}")
    print_summary(results)
    return results


if __name__ == "__main__":
    main()
