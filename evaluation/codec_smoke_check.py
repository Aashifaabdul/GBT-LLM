"""End-to-end round-trip check of the GBT codec on a small crop.

Encodes and decodes a 64x64 crop of data/Beauty/frames/frame0000.png with
gbticl_pipeline.codec (ContextGradientGBTICL graph model, LaplaceCoeffModel
coefficient model, 8x8 blocks) at two quantisation steps: 1 (near-lossless) and
8 (lossy). Prints encode/decode time, payload size, bpp, PSNR and, for the
lossy run, the compression ratio against raw 24-bit RGB. The original crop and
the q=8 reconstruction are saved to results/codec_smoke_check/. Takes no
command-line arguments and runs on import.

Usage:
    python evaluation/codec_smoke_check.py
"""
import time
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gbticl_pipeline.codec import encode_image, decode_image
from gbticl_pipeline.evaluate import psnr, bits_per_pixel
from gbticl_pipeline.graph_model import ContextGradientGBTICL
from gbticl_pipeline.coeff_model import LaplaceCoeffModel
from gbticl_pipeline.device_utils import get_device

SCRIPT_DIR = ROOT_DIR / "results" / "codec_smoke_check"
SCRIPT_DIR.mkdir(parents=True, exist_ok=True)

device = get_device()
print(f"running on device: {device}  (cuda available: {torch.cuda.is_available()})")

img_path = ROOT_DIR / "data" / "Beauty" / "frames" / "frame0000.png"
full = np.array(Image.open(img_path).convert("RGB"))
print("full frame shape:", full.shape)


# A small crop keeps the run short: range coding is a sequential per-symbol
# loop on the CPU (see the gbticl_pipeline.codec docstring).
crop = full[200:264, 300:364, :]  # 64x64 pixels = 8x8 grid of 8x8 blocks
print("crop shape:", crop.shape)
Image.fromarray(crop).save(SCRIPT_DIR / "test_crop_original.png")

gbticl_model = ContextGradientGBTICL().to(device)
coeff_model = LaplaceCoeffModel().to(device)


# Low-frequency (DC-like) coefficients reach about +-255*sqrt(64) = 2040 before
# quantisation, so at fine steps the coder's symbol range must cover them.
WIDE_RANGE = (-2200, 2200)

print("\n=== TEST 1: near-lossless path (fine quant step) ===")
t0 = time.time()
payload, meta = encode_image(crop, block_size=8, quant_step=1.0,
                              gbticl_model=gbticl_model, coeff_model=coeff_model,
                              symbol_range=WIDE_RANGE, device=device)
t1 = time.time()
recon = decode_image(payload, meta, gbticl_model=gbticl_model, coeff_model=coeff_model,
                      device=device)
t2 = time.time()

p = psnr(crop, recon)
bpp = bits_per_pixel(payload, meta["H"], meta["W"])
print(f"encode time: {t1-t0:.2f}s, decode time: {t2-t1:.2f}s")
print(f"symbols coded: {meta['n_symbols_coded']}")
print(f"payload size: {len(payload)} bytes, bpp: {bpp:.3f}")
print(f"PSNR (should be very high, near-lossless): {p:.2f} dB")
print(f"max abs pixel diff: {np.max(np.abs(crop.astype(int) - recon.astype(int)))}")

print("\n=== TEST 2: realistic lossy compression (quant_step=8) ===")
t0 = time.time()
payload2, meta2 = encode_image(crop, block_size=8, quant_step=8.0,
                                gbticl_model=gbticl_model, coeff_model=coeff_model,
                                symbol_range=WIDE_RANGE, device=device)
t1 = time.time()
recon2 = decode_image(payload2, meta2, gbticl_model=gbticl_model, coeff_model=coeff_model,
                       device=device)
t2 = time.time()

p2 = psnr(crop, recon2)
bpp2 = bits_per_pixel(payload2, meta2["H"], meta2["W"])
raw_bpp = 24  # uncompressed 8-bit RGB
print(f"encode time: {t1-t0:.2f}s, decode time: {t2-t1:.2f}s")
print(f"payload size: {len(payload2)} bytes, bpp: {bpp2:.3f} (raw would be {raw_bpp} bpp)")
print(f"compression ratio vs raw: {raw_bpp/bpp2:.2f}x")
print(f"PSNR: {p2:.2f} dB")

Image.fromarray(recon2).save(SCRIPT_DIR / "test_crop_reconstructed_q8.png")
print("\nsaved original + reconstructed crops for visual check")
