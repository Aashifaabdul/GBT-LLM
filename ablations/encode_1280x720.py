"""8x8 DCT + static-histogram range coder baseline for a single 1280x720 frame.

Resizes the input to --width x --height, quantizes the 8x8 DCT coefficients of the luma
channel with a uniform step, and range-codes them with one histogram computed over the whole
frame. Decodes the bitstream, and prints bpp, Y-PSNR and Y-SSIM. The histogram itself is not
included in the reported size. Chroma is not coded; the original chroma is kept in the
reconstruction. Writes input, .bin bitstream and reconstruction to --out-dir
(default results/1280x720_run).

Usage:
    python ablations/encode_1280x720.py --image data/Beauty/frames/frame0000.png --quant-step 16
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import os
import sys
import time
import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gbticl_pipeline.range_coder import RangeEncoder, RangeDecoder, probs_to_freqs, TOTAL_FREQ
from gbticl_pipeline.evaluate import psnr, ssim


def main():
    parser = argparse.ArgumentParser(description="Full 1280x720 Image Compression with Range Coding")
    parser.add_argument("--image", type=str, default="data/Beauty/frames/frame0000.png", help="Input image path")
    parser.add_argument("--width", type=int, default=1280, help="Target width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Target height (default: 720)")
    parser.add_argument("--quant-step", type=float, default=16.0, help="Quantization step (default: 16.0)")
    parser.add_argument("--out-dir", type=str, default="results/1280x720_run", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.image):
        print(f"Error: {args.image} not found!")
        return

    orig_img = Image.open(args.image).convert("RGB")
    img_1280 = orig_img.resize((args.width, args.height), Image.LANCZOS)
    input_path = os.path.join(args.out_dir, f"input_{args.width}x{args.height}.png")
    img_1280.save(input_path)

    W, H = args.width, args.height
    print("=" * 65)
    print(f" EXECUTING FULL {W}x{H} FRAME COMPRESSION")
    print("=" * 65)
    print(f" Image:             {args.image} -> {W}x{H}")
    print(f" Total Blocks:      {(W//8) * (H//8):,} (8x8 blocks)")
    print(f" Quantization Step: {args.quant_step}")

    # Only the luma channel is coded.
    ycbcr = np.array(img_1280.convert("YCbCr"), dtype=np.float32)
    y_channel = ycbcr[:, :, 0]

    q_blocks = []
    for r in range(0, H, 8):
        for c in range(0, W, 8):
            block = y_channel[r:r+8, c:c+8] - 128.0
            coeff = dct(dct(block.T, norm='ortho').T, norm='ortho')
            q_coeff = np.round(coeff / args.quant_step).astype(np.int32)
            q_blocks.append(q_coeff)

    all_coeffs = np.concatenate([b.flatten() for b in q_blocks])
    n_coeffs = len(all_coeffs)

    # Static model: the empirical symbol histogram of the whole frame.
    unique_symbols, counts = np.unique(all_coeffs, return_counts=True)
    probs = counts / n_coeffs
    sym_to_idx = {s: i for i, s in enumerate(unique_symbols)}
    idx_coeffs = np.array([sym_to_idx[s] for s in all_coeffs], dtype=np.int32)

    freqs, cum_freqs = probs_to_freqs(probs, TOTAL_FREQ)
    t_enc0 = time.perf_counter()
    encoder = RangeEncoder()
    for s in idx_coeffs:
        encoder.encode(int(cum_freqs[s]), int(freqs[s]), TOTAL_FREQ)
    bitstream = encoder.finish()
    t_enc = time.perf_counter() - t_enc0

    bin_path = os.path.join(args.out_dir, f"compressed_{W}x{H}.bin")
    with open(bin_path, "wb") as f:
        f.write(bitstream)
    compressed_bytes = len(bitstream)

    t_dec0 = time.perf_counter()
    decoder = RangeDecoder(bitstream)
    # Invert the cumulative-frequency table to find the symbol for each decoded target.
    decoded_idx = []
    for _ in range(n_coeffs):
        target = decoder.get_freq(TOTAL_FREQ)
        s = int(np.searchsorted(cum_freqs, target, side='right') - 1)
        decoder.decode(int(cum_freqs[s]), int(freqs[s]), TOTAL_FREQ)
        decoded_idx.append(s)
    t_dec = time.perf_counter() - t_dec0

    idx_to_sym = {i: s for i, s in enumerate(unique_symbols)}
    decoded_coeffs = np.array([idx_to_sym[i] for i in decoded_idx], dtype=np.int32)

    y_recon = np.zeros_like(y_channel)
    block_idx = 0
    for r in range(0, H, 8):
        for c in range(0, W, 8):
            q_b = decoded_coeffs[block_idx*64 : (block_idx+1)*64].reshape((8, 8))
            deq = q_b.astype(np.float32) * args.quant_step
            rec = idct(idct(deq.T, norm='ortho').T, norm='ortho') + 128.0
            y_recon[r:r+8, c:c+8] = np.clip(rec, 0, 255)
            block_idx += 1

    ycbcr_recon = ycbcr.copy()
    ycbcr_recon[:, :, 0] = y_recon
    rgb_recon = Image.fromarray(np.clip(ycbcr_recon, 0, 255).astype(np.uint8), mode="YCbCr").convert("RGB")
    recon_path = os.path.join(args.out_dir, f"reconstructed_{W}x{H}.png")
    rgb_recon.save(recon_path)

    y_orig_u8 = np.clip(y_channel, 0, 255).astype(np.uint8)
    y_rec_u8 = np.clip(y_recon, 0, 255).astype(np.uint8)
    cur_psnr = psnr(y_orig_u8, y_rec_u8)
    cur_ssim = ssim(y_orig_u8, y_rec_u8)
    bpp = (compressed_bytes * 8.0) / (W * H)

    print("\n" + "=" * 65)
    print(" COMPRESSION & RECONSTRUCTION COMPLETE")
    print("=" * 65)
    print(f" Original Uncompressed Size: {(W * H * 3) / 1024:.1f} KB")
    print(f" Compressed Bitstream Size:  {compressed_bytes / 1024:.2f} KB ({compressed_bytes:,} bytes)")
    print(f" Compression Ratio:          {((W * H * 3) / compressed_bytes):.1f}x")
    print(f" Bitrate (bpp):              {bpp:.4f} bpp")
    print(f" Quality (Y-PSNR):           {cur_psnr:.2f} dB")
    print(f" Quality (Y-SSIM):           {cur_ssim:.4f}")
    print(f" Encode Time:                {t_enc * 1000:.1f} ms")
    print(f" Decode Time:                {t_dec * 1000:.1f} ms")
    print("-" * 65)
    print(f" Saved Input Image:          {input_path}")
    print(f" Saved Compressed Bitstream: {bin_path}")
    print(f" Saved Reconstructed Image:  {recon_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
