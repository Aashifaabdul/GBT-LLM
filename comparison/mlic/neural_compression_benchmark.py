"""Learned-codec baseline on the Beauty and HoneyBee patches.

Baseline model: CompressAI `cheng2020_attn` (Cheng et al., CVPR 2020), pretrained,
at quality levels 3 and 6. The dissertation reports it as "MLIC (Cheng Attention)".
If it cannot be loaded, `bmshj2018_hyperprior` is used for that quality level
instead and stored under the key `Neural_LIC_Hyperprior_q<q>`.

Rate is the ideal code length from the model likelihoods, -sum(log2 p) over the
y and z latents divided by the number of pixels; no bitstream is written.
PSNR is computed on RGB in [0, 1].

Input:  results/mlic/patches/  (run data_prep.py first)
Output: results/mlic/neural_lic_results.json

Usage:
    python comparison/mlic/neural_compression_benchmark.py
"""

import json
import math
from pathlib import Path

import compressai.zoo as zoo
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "mlic"
PATCH_DIR = OUT_DIR / "patches"

QUALITIES = (3, 6)
SEQUENCES = ("Beauty", "HoneyBee")
PATCH_SUFFIXES = ("_patch64_rgb.png", "_patch128_gray.png")


def load_models(device):
    """Load the pretrained models, one per quality level."""
    models = {}
    for q in QUALITIES:
        try:
            net = zoo.cheng2020_attn(quality=q, pretrained=True).eval().to(device)
            net.update(force=True)
            models[f"Neural_LIC_Attn_q{q}"] = net
            print(f"Loaded cheng2020_attn quality={q}")
        except Exception as e:
            print(f"Falling back to bmshj2018_hyperprior for quality {q}: {e}")
            net = zoo.bmshj2018_hyperprior(quality=q, pretrained=True).eval().to(device)
            net.update(force=True)
            models[f"Neural_LIC_Hyperprior_q{q}"] = net
    return models


def evaluate_neural_compression():
    """Run every model on every patch and write neural_lic_results.json."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cpu"
    models = load_models(device)
    to_tensor = transforms.ToTensor()
    results = []

    for seq in SEQUENCES:
        seq_dir = PATCH_DIR / seq
        if not seq_dir.exists():
            print(f"Patch directory not found: {seq_dir} (run data_prep.py first)")
            continue

        files = sorted(p for p in seq_dir.iterdir() if p.name.endswith(PATCH_SUFFIXES))
        for path in tqdm(files, desc=f"Evaluating {seq}"):
            img = Image.open(path).convert("RGB")
            w, h = img.size
            num_pixels = w * h
            x = to_tensor(img).unsqueeze(0).to(device)

            # The models need side lengths divisible by 64.
            pad_h = (64 - (h % 64)) % 64
            pad_w = (64 - (w % 64)) % 64
            if pad_h > 0 or pad_w > 0:
                x_padded = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
            else:
                x_padded = x

            entry = {"sequence": seq, "file": path.name, "width": w, "height": h, "models": {}}

            for name, net in models.items():
                with torch.no_grad():
                    out = net(x_padded)
                    x_hat = out["x_hat"][:, :, :h, :w].clamp(0, 1)

                    total_bits = 0.0
                    for key in ("y", "z"):
                        if key in out["likelihoods"]:
                            total_bits += float(-torch.log2(out["likelihoods"][key]).sum().item())

                    mse = torch.mean((x - x_hat) ** 2).item()
                    psnr = 10.0 * math.log10(1.0 / max(1e-10, mse))
                    comp_bytes = total_bits / 8.0
                    raw_bytes = num_pixels * 3

                    entry["models"][name] = {
                        "bpp": round(total_bits / num_pixels, 4),
                        "bpb": round(total_bits / (num_pixels * 3), 4),
                        "psnr_db": round(psnr, 2),
                        "compression_ratio": round(raw_bytes / max(1.0, comp_bytes), 2),
                        "space_saving_pct": round((1.0 - comp_bytes / raw_bytes) * 100.0, 2),
                    }
            results.append(entry)

    out_json = OUT_DIR / "neural_lic_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_json}")
    return results


if __name__ == "__main__":
    evaluate_neural_compression()
