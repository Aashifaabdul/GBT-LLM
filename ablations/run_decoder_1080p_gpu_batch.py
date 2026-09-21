"""Round-trip check of one 1920x1080 frame: encode to a .bin container, decode it from disk, compare.

Uses the GBT-ICL graph transform with a per-frame symbol histogram and constriction's range coder
(no LLM prior). Verifies that the decoded quantized coefficients and the reconstructed pixels are
identical to the encoder's, and prints bpp, Y/RGB-PSNR, Y-SSIM and timings. The container layout is
the one documented in run_full_1080p_60frames_codec.py. Reads data/<sequence>/frames/frame<idx>.png
and checkpoints/stageA.pt (falls back to the untrained ContextGradientGBTICL if absent); writes the
bitstream and both reconstructions to results/decoder_1080p_gpu_verification/.

Usage:
    python ablations/run_decoder_1080p_gpu_batch.py --sequence Beauty --frame 0 --quant-step 8
"""

import sys
import time
import struct
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import constriction

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gbticl_pipeline.device_utils import load_state_dict_relaxed
from gbticl_pipeline.graph_model import GBTICLMetaLearner, ContextGradientGBTICL
from gbticl_pipeline.graph_utils import build_laplacian, eigendecompose
from gbticl_pipeline.gft import forward_gft, inverse_gft
from gbticl_pipeline.quantization import quantize, dequantize
from gbticl_pipeline.context import get_context, get_block, set_block, get_support_set
from gbticl_pipeline.colour import rgb_to_ycbcr, ycbcr_to_rgb
from gbticl_pipeline.evaluate import psnr, ssim

MAGIC_HEADER = b"GBT1080P"


@torch.inference_mode()
def encode_1080p_to_bitstream(ycbcr_np, gbticl_model, quant_step=8.0,
                              symbol_range=(-2200, 2200), block_size=8, device="cuda"):
    """Encode a YCbCr frame into a self-contained container.

    Returns:
        (container_bytes, canvas_enc, enc_q, enc_time): container bytes, the encoder-side reconstruction,
        quantized coefficients of shape (n_bh, n_bw, 64, 3) and the time in seconds.
    """
    device = torch.device(device)
    H, W, _ = ycbcr_np.shape
    n_bh, n_bw = H // block_size, W // block_size
    total_blocks = n_bh * n_bw
    n_alphabet = symbol_range[1] - symbol_range[0] + 1
    total_coeffs = total_blocks * block_size * block_size * 3

    ycbcr_t = torch.as_tensor(ycbcr_np, device=device)
    canvas_enc = torch.zeros((H, W, 3), dtype=torch.uint8, device=device)
    enc_q = torch.zeros((n_bh, n_bw, block_size * block_size, 3), dtype=torch.int64, device=device)

    all_symbols = []
    t0 = time.perf_counter()
    block_cnt = 0

    print(f"  [Encode] Processing {total_blocks:,} blocks ({total_coeffs:,} symbols) on GPU...")
    for i in range(n_bh):
        for j in range(n_bw):
            top, left, valid_top, valid_left = get_context(canvas_enc, i, j, block_size)
            support = get_support_set(canvas_enc, None, i, j, block_size)
            weights = gbticl_model.predict_edge_weights(top, left, valid_top, valid_left, block_size, support=support)
            L = build_laplacian(weights, block_size, device=device)
            eigvals, U = eigendecompose(L)

            block = get_block(ycbcr_t, i, j, block_size)
            coeffs = forward_gft(block, U)
            q = quantize(coeffs, quant_step)
            enc_q[i, j] = q

            # Symbols are collected first; the range coder runs once over the whole frame.
            q_cpu = q.detach().cpu().numpy()
            for k in range(block_size * block_size):
                for ch in range(3):
                    val = int(q_cpu[k, ch])
                    sym_idx = val - symbol_range[0]
                    sym_idx = max(0, min(sym_idx, n_alphabet - 1))
                    all_symbols.append(sym_idx)

            # Reconstruct as the decoder will, so the causal context matches.
            dq = dequantize(q, quant_step)
            recon = inverse_gft(dq, U, block_size)
            recon_u8 = torch.clamp(torch.round(recon), 0, 255).to(torch.uint8)
            set_block(canvas_enc, i, j, block_size, recon_u8)

            block_cnt += 1
            if block_cnt % 5000 == 0 or block_cnt == total_blocks:
                elapsed = time.perf_counter() - t0
                spd = block_cnt / elapsed if elapsed > 0 else 0
                eta_s = (total_blocks - block_cnt) / spd if spd > 0 else 0
                print(f"\r  [Encode] {block_cnt:,}/{total_blocks:,} blocks ({block_cnt/total_blocks*100:.1f}%) "
                      f"| Speed: {spd:.1f} blk/s | ETA: {eta_s:.0f}s", end="", flush=True)

    print("\n  [Encode] Range-encoding bitstream with compiled C++/Rust constriction...")
    symbols_arr = np.array(all_symbols, dtype=np.int32)

    u_syms, counts = np.unique(symbols_arr, return_counts=True)
    # Frame histogram; unused symbols below the maximum get a small floor so no probability is zero.
    max_sym = int(np.max(symbols_arr))
    probs = np.full(max_sym + 1, 1e-7, dtype=np.float64)
    for s, c in zip(u_syms, counts):
        probs[s] = c
    probs = probs / probs.sum()

    cat_model = constriction.stream.model.Categorical(probs, perfect=False)
    encoder = constriction.stream.queue.RangeEncoder()
    encoder.encode(symbols_arr, cat_model)
    compressed_words = encoder.get_compressed()
    compressed_data = compressed_words.tobytes()

    # Header, histogram and payload are concatenated; the histogram counts towards the file size.
    probs_bytes = probs.astype(np.float64).tobytes()
    header_pack = struct.pack(
        ">8sHHBfiiiI",
        MAGIC_HEADER,
        H, W, block_size, quant_step,
        symbol_range[0], symbol_range[1], max_sym, len(probs)
    )
    container_bytes = header_pack + probs_bytes + compressed_data
    enc_time = time.perf_counter() - t0

    return container_bytes, canvas_enc, enc_q, enc_time


@torch.inference_mode()
def decode_1080p_from_bitstream_file(bitstream_path, gbticl_model, device="cuda"):
    """Read a .bin container and reconstruct the frame from it alone.

    Returns:
        (canvas_dec, dec_q, dec_time, meta) with the reconstructed YCbCr canvas, the decoded
        quantized coefficients, the decoding time in seconds and a dict of header fields and sizes.
    """
    device = torch.device(device)
    with open(bitstream_path, "rb") as f:
        container_bytes = f.read()

    t0 = time.perf_counter()
    header_size = struct.calcsize(">8sHHBfiiiI")
    header = struct.unpack(">8sHHBfiiiI", container_bytes[:header_size])
    magic, H, W, block_size, quant_step, sym_min, sym_max, max_sym, n_probs = header

    assert magic == MAGIC_HEADER, f"Invalid bitstream container header: {magic}"

    probs_size = n_probs * 8
    probs_start = header_size
    compressed_start = probs_start + probs_size

    probs = np.frombuffer(container_bytes[probs_start:compressed_start], dtype=np.float64)
    compressed_bytes = container_bytes[compressed_start:]
    compressed_words = np.frombuffer(compressed_bytes, dtype=np.uint32)

    cat_model = constriction.stream.model.Categorical(probs, perfect=False)
    n_bh, n_bw = H // block_size, W // block_size
    total_blocks = n_bh * n_bw
    total_coeffs = total_blocks * block_size * block_size * 3

    print(f"  [Decode] Range-decoding {total_coeffs:,} symbols with compiled C++/Rust constriction...")
    t_range_start = time.perf_counter()
    decoder = constriction.stream.queue.RangeDecoder(compressed_words)
    decoded_symbols = decoder.decode(cat_model, total_coeffs)
    t_range = time.perf_counter() - t_range_start
    print(f"  [Decode] Range-decoded in {t_range*1000:.2f} ms ({total_coeffs/t_range:,.0f} sym/s)")

    print(f"  [Decode] Reconstructing {total_blocks:,} blocks on GPU...")
    canvas_dec = torch.zeros((H, W, 3), dtype=torch.uint8, device=device)
    dec_q = torch.zeros((n_bh, n_bw, block_size * block_size, 3), dtype=torch.int64, device=device)

    sym_ptr = 0
    block_cnt = 0
    t_recon_start = time.perf_counter()

    for i in range(n_bh):
        for j in range(n_bw):
            top, left, valid_top, valid_left = get_context(canvas_dec, i, j, block_size)
            support = get_support_set(canvas_dec, None, i, j, block_size)
            weights = gbticl_model.predict_edge_weights(top, left, valid_top, valid_left, block_size, support=support)
            L = build_laplacian(weights, block_size, device=device)
            eigvals, U = eigendecompose(L)

            q_block = torch.zeros((block_size * block_size, 3), dtype=torch.int64, device=device)
            for k in range(block_size * block_size):
                for ch in range(3):
                    sym_val = int(decoded_symbols[sym_ptr]) + sym_min
                    q_block[k, ch] = sym_val
                    sym_ptr += 1

            dec_q[i, j] = q_block

            dq = dequantize(q_block, quant_step)
            recon = inverse_gft(dq, U, block_size)
            recon_u8 = torch.clamp(torch.round(recon), 0, 255).to(torch.uint8)
            set_block(canvas_dec, i, j, block_size, recon_u8)

            block_cnt += 1
            if block_cnt % 5000 == 0 or block_cnt == total_blocks:
                elapsed = time.perf_counter() - t_recon_start
                spd = block_cnt / elapsed if elapsed > 0 else 0
                eta_s = (total_blocks - block_cnt) / spd if spd > 0 else 0
                print(f"\r  [Decode] {block_cnt:,}/{total_blocks:,} blocks ({block_cnt/total_blocks*100:.1f}%) "
                      f"| Speed: {spd:.1f} blk/s | ETA: {eta_s:.0f}s", end="", flush=True)

    print()
    dec_time = time.perf_counter() - t0
    meta = {
        "H": H, "W": W, "block_size": block_size, "quant_step": quant_step,
        "symbol_range": (sym_min, sym_max), "compressed_bytes": len(container_bytes),
        "payload_bytes": len(compressed_bytes)
    }
    return canvas_dec, dec_q, dec_time, meta


def run_1080p_frame_check(sequence="Beauty", frame_idx=0, quant_step=8.0, device="cuda"):
    """Encode and decode one frame, then compare encoder and decoder outputs and print a report."""
    print("=" * 80)
    print(f" FULL 1080p HD (1920x1080) GPU DECODER BITSTREAM VERIFICATION (FRAME {frame_idx})")
    print("=" * 80)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"  Device:               {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"  Sequence:             {sequence}")
    print("  Resolution:           1920 x 1080 (32,400 blocks of 8x8, 6,220,800 symbols)")
    print(f"  Quantization Step:    {quant_step}")

    img_path = ROOT_DIR / f"data/{sequence}/frames/frame{frame_idx:04d}.png"
    if not img_path.exists():
        raise FileNotFoundError(f"Input image not found: {img_path}")

    raw_img = Image.open(img_path).convert("RGB")
    orig_rgb = np.array(raw_img)
    # Crop to a multiple of the 8x8 block size.
    H, W = orig_rgb.shape[:2]
    H_snap = (H // 8) * 8
    W_snap = (W // 8) * 8
    orig_rgb = orig_rgb[:H_snap, :W_snap, :]
    orig_ycbcr = rgb_to_ycbcr(orig_rgb)

    out_dir = ROOT_DIR / "results/decoder_1080p_gpu_verification"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ROOT_DIR / "checkpoints/stageA.pt"
    if ckpt_path.exists():
        print(f"  Loading model from:   {ckpt_path.name}")
        gbticl_model = GBTICLMetaLearner(block_size=8).to(device)
        ckpt_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        load_state_dict_relaxed(gbticl_model, ckpt_dict["gbticl_net"], "GBTICLMetaLearner")
    else:
        print("  Using baseline ContextGradientGBTICL model")
        gbticl_model = ContextGradientGBTICL().to(device)
    gbticl_model.eval()

    print("\n[Step 1/3] Running GPU 1080p Encoder & Writing .bin bitstream file...")
    container_bytes, enc_canvas, enc_q, enc_time = encode_1080p_to_bitstream(
        orig_ycbcr, gbticl_model, quant_step=quant_step,
        symbol_range=(-2200, 2200), block_size=8, device=device
    )

    bin_path = out_dir / f"bitstream_{sequence}_1080p_frame{frame_idx:04d}_q{int(quant_step)}.bin"
    with open(bin_path, "wb") as f:
        f.write(container_bytes)

    enc_rgb = ycbcr_to_rgb(enc_canvas.detach().cpu().numpy())
    enc_png_path = out_dir / f"encoder_recon_{sequence}_1080p_frame{frame_idx:04d}.png"
    Image.fromarray(enc_rgb).save(enc_png_path)

    n_pixels = H_snap * W_snap
    bpp = (len(container_bytes) * 8.0) / n_pixels
    print(f"  -> Encoded in:        {enc_time:.2f} s ({enc_time/60:.2f} min)")
    print(f"  -> Bitstream Size:    {len(container_bytes):,} bytes ({bpp:.4f} bpp)")
    print(f"  -> Saved Bitstream:   {bin_path.name}")
    print(f"  -> Saved Enc Recon:   {enc_png_path.name}")

    print("\n[Step 2/3] Running Independent GPU 1080p Decoder from .bin file on disk...")
    dec_canvas, dec_q, dec_time, meta = decode_1080p_from_bitstream_file(
        bin_path, gbticl_model, device=device
    )

    dec_rgb = ycbcr_to_rgb(dec_canvas.detach().cpu().numpy())
    dec_png_path = out_dir / f"decoder_recon_{sequence}_1080p_frame{frame_idx:04d}.png"
    Image.fromarray(dec_rgb).save(dec_png_path)

    print(f"  -> Decoded in:        {dec_time:.2f} s ({dec_time/60:.2f} min)")
    print(f"  -> Saved Dec Recon:   {dec_png_path.name}")

    # The decoder rebuilds each graph basis from decoded pixels, so any mismatch means the contexts diverged.
    print("\n[Step 3/3] Cross-Verifying Decoder vs. Encoder...")
    lossless_symbols = torch.equal(enc_q, dec_q)
    coeff_diff_count = int(torch.count_nonzero(enc_q != dec_q).item())
    total_coeffs = enc_q.numel()

    recon_enc_np = enc_rgb.astype(np.float64)
    recon_dec_np = dec_rgb.astype(np.float64)
    pixel_abs_diff = np.abs(recon_enc_np - recon_dec_np)
    max_pixel_diff = float(np.max(pixel_abs_diff))
    pixel_mse = float(np.mean((recon_enc_np - recon_dec_np) ** 2))

    enc_y_psnr = psnr(orig_ycbcr[..., 0], enc_canvas.detach().cpu().numpy()[..., 0])
    dec_y_psnr = psnr(orig_ycbcr[..., 0], dec_canvas.detach().cpu().numpy()[..., 0])
    enc_rgb_psnr = psnr(orig_rgb, enc_rgb)
    dec_rgb_psnr = psnr(orig_rgb, dec_rgb)
    try:
        enc_y_ssim = ssim(orig_ycbcr[..., 0], enc_canvas.detach().cpu().numpy()[..., 0])
        dec_y_ssim = ssim(orig_ycbcr[..., 0], dec_canvas.detach().cpu().numpy()[..., 0])
    except Exception:
        enc_y_ssim, dec_y_ssim = float("nan"), float("nan")

    print("-" * 80)
    print(f" 1. QUANTIZED COEFFICIENTS MATCH:     {'100% BIT-EXACT MATCH [PASS]' if lossless_symbols else 'MISMATCH [FAIL]'}")
    print(f"    - Mismatched Coefficients:        {coeff_diff_count} / {total_coeffs:,}")
    print(f" 2. PIXEL RECONSTRUCTION DIFFERENCE:  {'IDENTICAL 0.0000 DIFF [PASS]' if max_pixel_diff == 0.0 else f'DIFFERENCE: {max_pixel_diff}'}")
    print(f"    - Max Absolute Pixel Error:       {max_pixel_diff:.4f}")
    print(f"    - Encoder vs. Decoder MSE:        {pixel_mse:.8f}")
    print(" 3. RECONSTRUCTED QUALITY (vs Original):")
    print(f"    - Encoder Y-PSNR:                 {enc_y_psnr:.2f} dB  |  Decoder Y-PSNR: {dec_y_psnr:.2f} dB")
    print(f"    - Encoder RGB-PSNR:               {enc_rgb_psnr:.2f} dB  |  Decoder RGB-PSNR: {dec_rgb_psnr:.2f} dB")
    print(f"    - Encoder Y-SSIM:                 {enc_y_ssim:.4f}     |  Decoder Y-SSIM: {dec_y_ssim:.4f}")
    print(" 4. EXECUTION RUNTIME (1080p):")
    print(f"    - Encode Time:                    {enc_time:.2f} s ({enc_time/60:.2f} min)")
    print(f"    - Decode Time:                    {dec_time:.2f} s ({dec_time/60:.2f} min)")
    print(f"    - Total 1080p Frame Time:         {enc_time + dec_time:.2f} s ({(enc_time + dec_time)/60:.2f} min)")
    print("-" * 80)

    if lossless_symbols and max_pixel_diff == 0.0:
        print(f"[SUCCESS] 1080p Frame {frame_idx} decoder reconstruction from .bin is 100% BIT-IDENTICAL to encoder!")
    else:
        print(f"[WARNING] Discrepancy detected in 1080p Frame {frame_idx}!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="1080p single frame GPU decoder verification.")
    parser.add_argument("--sequence", type=str, default="Beauty", choices=["Beauty", "HoneyBee"], help="Sequence name")
    parser.add_argument("--frame", type=int, default=0, help="Frame index (default: 0)")
    parser.add_argument("--quant-step", type=float, default=8.0, help="Quantization step (default: 8.0)")
    parser.add_argument("--device", type=str, default="cuda", help="Hardware device (default: cuda)")
    args = parser.parse_args()

    run_1080p_frame_check(sequence=args.sequence, frame_idx=args.frame, quant_step=args.quant_step, device=args.device)


if __name__ == "__main__":
    main()
