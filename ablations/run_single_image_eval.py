"""Encode and decode one image with the full model (GBT-ICL meta-learner + DistilGPT-2/LoRA coefficient model).

Loads the Stage A (--gbticl-checkpoint) and Stage B (--coeff-checkpoint) checkpoints, encodes the
image to a bitstream, decodes it, and prints payload size, bpp, Y-PSNR and Y-SSIM. The original
(cropped to a multiple of 8) and reconstructed images are saved to --out-dir (default results/single_image).

Usage:
    python ablations/run_single_image_eval.py --image data/Beauty/frames/frame0000.png --quant-step 2.0
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from gbticl_pipeline.device_utils import get_device, load_state_dict_relaxed
from gbticl_pipeline.codec import encode_image, decode_image
from gbticl_pipeline.graph_model import GBTICLMetaLearner
from gbticl_pipeline.coeff_model import HFLoRACoeffModel
from gbticl_pipeline.evaluate import psnr, ssim, bits_per_pixel


def main():
    parser = argparse.ArgumentParser(description="Encode and decode a single image with the full GBT-ICL + LLM model.")
    parser.add_argument("--image", type=str, default="data/Beauty/frames/frame0000.png", help="Path to input RGB image")
    parser.add_argument("--gbticl-checkpoint", type=str, default="checkpoints/stageA.pt", help="Stage A GBT-ICL checkpoint")
    parser.add_argument("--coeff-checkpoint", type=str, default="checkpoints/stageB.pt", help="Stage B LLM Coeff checkpoint")
    parser.add_argument("--quant-step", type=float, default=2.0, help="Quantization step (smaller = higher quality, e.g. 2.0 or 4.0)")
    parser.add_argument("--out-dir", type=str, default="results/single_image", help="Directory to save reconstructed image")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Loading image: {args.image}")

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found at {args.image}")

    img = np.array(Image.open(img_path).convert("RGB"))

    # Crop to a multiple of the 8x8 transform block size.
    h, w = img.shape[:2]
    h_snap = (h // 8) * 8
    w_snap = (w // 8) * 8
    img = img[:h_snap, :w_snap, :]

    gbticl = GBTICLMetaLearner(block_size=8).to(device)
    coeff = HFLoRACoeffModel(base_model_name="distilgpt2", block_size=8).to(device)

    ckptA = torch.load(args.gbticl_checkpoint, map_location=device, weights_only=False)
    ckptB = torch.load(args.coeff_checkpoint, map_location=device, weights_only=False)

    load_state_dict_relaxed(gbticl, ckptA["gbticl_net"], "metalearner")
    load_state_dict_relaxed(coeff, ckptB["coeff_net"], "hf_lora")

    gbticl.eval()
    coeff.eval()

    print(f"Encoding image with quant_step={args.quant_step}...")
    payload, meta = encode_image(
        img,
        block_size=8,
        quant_step=args.quant_step,
        gbticl_model=gbticl,
        coeff_model=coeff,
        device=device
    )

    print(f"Decoding image from bitstream payload ({len(payload)} bytes)...")
    recon = decode_image(
        payload,
        meta,
        gbticl_model=gbticl,
        coeff_model=coeff,
        device=device
    )

    p_y = psnr(img, recon)
    s_y = ssim(img, recon)
    bpp = bits_per_pixel(payload, img.shape[0], img.shape[1])

    print("\n=================== Results ===================")
    print(f"Payload Size : {len(payload):,} bytes")
    print(f"Bitrate (bpp): {bpp:.4f} bpp")
    print(f"Y-PSNR       : {p_y:.2f} dB")
    print(f"Y-SSIM       : {s_y:.4f}")
    print("===============================================")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_out_path = out_dir / f"{img_path.stem}_original.png"
    recon_out_path = out_dir / f"{img_path.stem}_reconstructed_q{int(args.quant_step)}.png"

    Image.fromarray(img).save(orig_out_path)
    Image.fromarray(recon).save(recon_out_path)

    print(f"Saved original image to      : {orig_out_path}")
    print(f"Saved reconstructed image to: {recon_out_path}")


if __name__ == "__main__":
    main()
