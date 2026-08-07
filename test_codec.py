import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gbticl_pipeline.codec import encode_image, decode_image
from gbticl_pipeline.evaluate import psnr, bits_per_pixel
from gbticl_pipeline.graph_model import ContextGradientGBTICL
from gbticl_pipeline.coeff_model import LaplaceCoeffModel
from gbticl_pipeline.device_utils import get_device

SCRIPT_DIR = Path(__file__).resolve().parent

device = get_device()
print(f"running on device: {device}  (cuda available: {torch.cuda.is_available()})")

img_path = SCRIPT_DIR / "Beauty" / "frames" / "frame0000.png"
full = np.array(Image.open(img_path).convert("RGB"))
print("full frame shape:", full.shape)

# small crop for a fast, honest correctness test (full 1920x1080 in a per-symbol
# Python entropy-coding loop is far too slow for this environment's runtime
# limits regardless of device -- see the codec.py docstring on the CPU-bound
# range-coding step)
crop = full[200:264, 300:364, :]  # 64x64 -> 8x8 blocks of 8x8 = 64 blocks
print("crop shape:", crop.shape)
Image.fromarray(crop).save(SCRIPT_DIR / "test_crop_original.png")

gbticl_model = ContextGradientGBTICL().to(device)
coeff_model = LaplaceCoeffModel().to(device)

# DC-like (low-eigenvalue) coefficients can reach roughly +-255*sqrt(64)=2040
# before quantization, so the symbol range needs to cover that at fine quant
# steps -- widen it here rather than at the tighter default.
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
