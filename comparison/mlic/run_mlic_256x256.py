"""MLIC baseline on the 256x256 test crops.

Baseline model: CompressAI `cheng2020_attn` (Cheng et al., CVPR 2020), pretrained,
quality 3, reported in the dissertation as "MLIC (Cheng Attention)". The rate is
the ideal code length from the model likelihoods (y and z latents) divided by
the number of pixels; no bitstream is written. PSNR is computed on RGB in [0, 1].

Test images: see comparison/crops_256.py.
Output: results/mlic/mlic_256x256_results.json

Usage:
    python comparison/mlic/run_mlic_256x256.py
"""

import json
import math
import sys
from pathlib import Path

import compressai.zoo as zoo
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crops_256 import collect_samples  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "mlic"

QUALITY = 3


def main():
    """Evaluate every test image and write mlic_256x256_results.json."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cpu"

    model = zoo.cheng2020_attn(quality=QUALITY, pretrained=True).eval().to(device)
    model.update(force=True)

    samples = collect_samples()
    print(f"Test images: {len(samples)}")

    to_tensor = transforms.ToTensor()
    results = []

    for sample in tqdm(samples, desc="MLIC 256x256"):
        img = Image.open(sample["path"]).convert("RGB")
        w, h = img.size
        x = to_tensor(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)
            x_hat = out["x_hat"].clamp(0, 1)
            total_bits = float(-torch.log2(out["likelihoods"]["y"]).sum().item())
            if "z" in out["likelihoods"]:
                total_bits += float(-torch.log2(out["likelihoods"]["z"]).sum().item())
            mse = torch.mean((x - x_hat) ** 2).item()

        results.append({
            "sequence": sample["sequence"],
            "file": sample["path"].name,
            "width": w,
            "height": h,
            "model": "cheng2020_attn",
            "quality": QUALITY,
            "mlic_bpp": total_bits / (w * h),
            "mlic_psnr": 10.0 * math.log10(1.0 / max(1e-10, mse)),
        })

    out_json = OUT_DIR / "mlic_256x256_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_json}")
    return results


if __name__ == "__main__":
    main()
