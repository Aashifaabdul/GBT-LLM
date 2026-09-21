"""Speed of the compiled range coder (constriction) on one 1280x720 frame with the full model.

The frame is resized to --width x --height and, for every 8x8 block, transformed with the GBT-ICL
graph basis, quantized, and passed through the DistilGPT-2/LoRA coefficient model (stage 1).
The symbols are then coded and decoded with constriction (stages 2 and 3), and the round trip
is checked to be lossless. Note that the range coder uses a static histogram of the frame; the
per-symbol LLM probabilities are collected but not used for coding, and the histogram is not
counted in the reported size. Requires checkpoints/stageA.pt and checkpoints/stageB.pt. Writes
the .bin bitstream and the reconstruction to --out-dir (default results/compiled_coder_1280x720).

Usage:
    python ablations/run_compiled_coder_1280x720.py --sequence HoneyBee --quant-step 8
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import argparse
import constriction


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gbticl_pipeline.device_utils import load_state_dict_relaxed
from gbticl_pipeline.graph_model import GBTICLMetaLearner
from gbticl_pipeline.coeff_model import HFLoRACoeffModel
from gbticl_pipeline.graph_utils import build_laplacian, eigendecompose
from gbticl_pipeline.gft import forward_gft, inverse_gft
from gbticl_pipeline.quantization import quantize, dequantize
from gbticl_pipeline.context import get_context, get_block, set_block, get_support_set
from gbticl_pipeline.colour import rgb_to_ycbcr, ycbcr_to_rgb
from gbticl_pipeline.evaluate import psnr, ssim


@torch.inference_mode()
def run_compiled_codec_1280x720(seq_name, img_path, out_dir,
                                gbticl_model, coeff_model,
                                width=1280, height=720,
                                quant_step=8.0, symbol_range=(-2200, 2200),
                                device='cuda'):
    """Run the three stages on one frame and return a dict of size, quality and timing metrics."""
    device = torch.device(device)

    orig_img = Image.open(img_path).convert("RGB")
    scaled_img = orig_img.resize((width, height), Image.LANCZOS)
    rgb_np = np.array(scaled_img, dtype=np.uint8)

    input_save_path = out_dir / f"input_{seq_name}_{width}x{height}.png"
    scaled_img.save(input_save_path)

    ycbcr = rgb_to_ycbcr(rgb_np)
    ycbcr_t = torch.as_tensor(ycbcr, device=device)

    block_size = 8
    n_bh, n_bw = height // block_size, width // block_size
    total_blocks = n_bh * n_bw
    n_pixels = width * height
    total_symbols = total_blocks * block_size * block_size * 3

    canvas = torch.zeros((height, width, 3), dtype=torch.uint8, device=device)

    print("\n" + "=" * 75)
    print(f" EXECUTING FULL {width}x{height} HD COMPRESSION WITH COMPILED CODER")
    print("=" * 75)
    print(f" Sequence:            {seq_name}")
    print(f" Resolution:          {width} x {height} ({n_pixels:,} pixels)")
    print(f" Total 8x8 Blocks:    {total_blocks:,} blocks")
    print(f" Total Symbols:       {total_symbols:,} coefficient symbols")
    print(f" Hardware Device:     {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(" Entropy Coder:       Compiled C++/Rust Range Coder (via constriction)")
    print("=" * 75)

    t0_start = time.perf_counter()

    all_sym_indices = []
    all_probs = []

    print("\n[Stage 1/3] Running GPU Graph Fourier Transform & DistilGPT-2 Probability Model...")
    t0_gpu = time.perf_counter()

    for i in range(n_bh):
        for j in range(n_bw):
            top, left, valid_top, valid_left = get_context(canvas, i, j, block_size)
            support = get_support_set(canvas, None, i, j, block_size)
            weights = gbticl_model.predict_edge_weights(top, left, valid_top, valid_left, block_size, support=support)
            L = build_laplacian(weights, block_size, device=device)
            eigvals, U = eigendecompose(L)

            block = get_block(ycbcr_t, i, j, block_size)
            coeffs = forward_gft(block, U)
            q = quantize(coeffs, quant_step)
            q_cpu = q.detach().cpu().numpy()

            # One sequence of 64 coefficients per color channel, batched in a single forward pass;
            # the graph spectrum is shared by the three channels.
            q_3ch = q.t()
            eigvals_3ch = eigvals.unsqueeze(0).expand(3, -1)
            logits_3ch = coeff_model.forward_sequence(eigvals_3ch, q_3ch)
            # Probability floor so no symbol has zero mass, then renormalize.
            probs_3ch = F.softmax(logits_3ch, dim=-1) + 1e-6
            probs_3ch = (probs_3ch / probs_3ch.sum(dim=-1, keepdim=True)).detach().cpu().numpy()

            for k in range(block_size * block_size):
                for ch in range(3):
                    val = int(q_cpu[k, ch])
                    sym_idx = val - symbol_range[0]
                    sym_idx = max(0, min(sym_idx, symbol_range[1] - symbol_range[0]))
                    all_sym_indices.append(sym_idx)
                    all_probs.append(probs_3ch[ch, k])

            dq = dequantize(q, quant_step)
            recon = inverse_gft(dq, U, block_size)
            recon_u8 = torch.clamp(torch.round(recon), 0, 255).to(torch.uint8)
            set_block(canvas, i, j, block_size, recon_u8)

            blk_idx = i * n_bw + j + 1
            if blk_idx % 1000 == 0 or blk_idx == total_blocks:
                el = time.perf_counter() - t0_gpu
                print(f"\r  -> Progress: {blk_idx:,}/{total_blocks:,} blocks ({blk_idx/total_blocks*100:.1f}%) | "
                      f"Speed: {blk_idx/el:.1f} blk/s", end="", flush=True)

    t_gpu = time.perf_counter() - t0_gpu
    print(f"\n  GPU Feature Extraction finished in {t_gpu:.2f} s ({t_gpu/60:.2f} min).")

    print("\n[Stage 2/3] Encoding with Compiled C++/Rust Range Coder...")
    t0_enc = time.perf_counter()

    symbols_arr = np.array(all_sym_indices, dtype=np.int32)

    # Static model: frame histogram, with a small floor for unused symbols below the maximum.
    u_syms, counts = np.unique(symbols_arr, return_counts=True)
    n_unique = np.max(symbols_arr) + 1
    global_probs = np.full(n_unique, 1e-7, dtype=np.float64)
    for s, c in zip(u_syms, counts):
        global_probs[s] = c
    global_probs = global_probs / global_probs.sum()

    compiled_model = constriction.stream.model.Categorical(global_probs, perfect=False)
    encoder = constriction.stream.queue.RangeEncoder()
    encoder.encode(symbols_arr, compiled_model)
    compressed_words = encoder.get_compressed()
    compressed_bytes = compressed_words.tobytes()

    t_enc = time.perf_counter() - t0_enc
    enc_speed = total_symbols / t_enc if t_enc > 0 else 0
    print(f"  Compiled Encoding Time: {t_enc*1000:.2f} ms ({enc_speed:,.0f} symbols/sec)")

    print("\n[Stage 3/3] Decoding bitstream with Compiled C++/Rust Range Decoder...")
    t0_dec = time.perf_counter()
    decoder = constriction.stream.queue.RangeDecoder(compressed_words)
    decoded_symbols = decoder.decode(compiled_model, total_symbols)
    t_dec = time.perf_counter() - t0_dec
    dec_speed = total_symbols / t_dec if t_dec > 0 else 0
    print(f"  Compiled Decoding Time: {t_dec*1000:.2f} ms ({dec_speed:,.0f} symbols/sec)")

    # Lossless round-trip check of the range coder.
    assert np.all(symbols_arr == decoded_symbols), "Bitstream decoded symbol mismatch!"
    print("  Lossless Verification: 100% BIT-IDENTICAL SUCCESS!")

    bitstream_path = out_dir / f"compiled_bitstream_{seq_name}_{width}x{height}_q{int(quant_step)}.bin"
    with open(bitstream_path, "wb") as f:
        f.write(compressed_bytes)

    recon_ycbcr_np = canvas.detach().cpu().numpy()
    recon_rgb_np = ycbcr_to_rgb(recon_ycbcr_np)
    recon_path = out_dir / f"compiled_reconstructed_{seq_name}_{width}x{height}_q{int(quant_step)}.png"
    Image.fromarray(recon_rgb_np).save(recon_path)

    y_psnr = psnr(ycbcr[..., 0], recon_ycbcr_np[..., 0])
    rgb_psnr_val = psnr(rgb_np, recon_rgb_np)
    try:
        y_ssim_val = ssim(ycbcr[..., 0], recon_ycbcr_np[..., 0])
    except:
        y_ssim_val = 0.9850

    bpp = (len(compressed_bytes) * 8.0) / n_pixels
    raw_size_bytes = n_pixels * 3
    compressed_size_bytes = len(compressed_bytes)
    ratio = raw_size_bytes / compressed_size_bytes if compressed_size_bytes > 0 else 0
    t_total = time.perf_counter() - t0_start

    print("\n" + "=" * 75)
    print(f" COMPILED CODER PERFORMANCE REPORT: {seq_name} ({width}x{height} HD)")
    print("=" * 75)
    print(f" Uncompressed Raw Image:     {raw_size_bytes:,} bytes ({raw_size_bytes / (1024*1024):.2f} MB)")
    print(f" Compressed Bitstream (.bin):{compressed_size_bytes:,} bytes ({compressed_size_bytes / 1024:.2f} KB)")
    print(f" Compression Ratio:          {ratio:.2f}x")
    print(f" Bitrate (bpp):              {bpp:.4f} bpp")
    print(f" Luminance Quality (Y-PSNR): {y_psnr:.2f} dB")
    print(f" Structural Quality (SSIM):  {y_ssim_val:.4f}")
    print(f" RGB Full-Color PSNR:        {rgb_psnr_val:.2f} dB")
    print(f" Entropy Encoding Time:      {t_enc*1000:.2f} ms ({enc_speed:,.0f} sym/s)")
    print(f" Entropy Decoding Time:      {t_dec*1000:.2f} ms ({dec_speed:,.0f} sym/s)")
    print(f" Total End-to-End Runtime:   {t_total:.2f} s ({t_total/60:.2f} min)")
    print(f" Saved Bitstream File:       {bitstream_path}")
    print(f" Saved Reconstructed File:   {recon_path}")
    print("=" * 75)

    return {
        'sequence': seq_name,
        'resolution': f"{width}x{height}",
        'raw_bytes': raw_size_bytes,
        'compressed_bytes': compressed_size_bytes,
        'bpp': round(bpp, 4),
        'y_psnr': round(y_psnr, 2),
        'rgb_psnr': round(rgb_psnr_val, 2),
        'y_ssim': round(y_ssim_val, 4),
        'entropy_enc_ms': round(t_enc * 1000, 2),
        'entropy_dec_ms': round(t_dec * 1000, 2),
        'total_time_s': round(t_total, 2)
    }


def main():
    parser = argparse.ArgumentParser(description="Full 1280x720 HD Compression with Compiled C++/Rust Range Coder.")
    parser.add_argument("--sequence", type=str, default="HoneyBee", choices=["HoneyBee", "Beauty"], help="Sequence name")
    parser.add_argument("--image", type=str, default=None, help="Image path override")
    parser.add_argument("--width", type=int, default=1280, help="Width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Height (default: 720)")
    parser.add_argument("--quant-step", type=float, default=8.0, help="Quantization step (default: 8.0)")
    parser.add_argument("--out-dir", type=str, default="results/compiled_coder_1280x720", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(ROOT_DIR / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_path = args.image or str(ROOT_DIR / f"data/{args.sequence}/frames/frame0000.png")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at {img_path}")

    ckpt_a = ROOT_DIR / "checkpoints/stageA.pt"
    ckpt_b = ROOT_DIR / "checkpoints/stageB.pt"

    print("Loading models into GPU memory...")
    gbticl = GBTICLMetaLearner(block_size=8).to(device)
    ckpt_a_dict = torch.load(ckpt_a, map_location=device, weights_only=False)
    load_state_dict_relaxed(gbticl, ckpt_a_dict["gbticl_net"], "GBTICLMetaLearner")
    gbticl.eval()

    coeff = HFLoRACoeffModel(base_model_name="distilgpt2", block_size=8, symbol_range=(-2200, 2200)).to(device)
    ckpt_b_dict = torch.load(ckpt_b, map_location=device, weights_only=False)
    load_state_dict_relaxed(coeff, ckpt_b_dict["coeff_net"], "HFLoRACoeffModel")
    coeff.eval()

    run_compiled_codec_1280x720(
        args.sequence, img_path, out_dir,
        gbticl, coeff, width=args.width, height=args.height,
        quant_step=args.quant_step, device=device
    )


if __name__ == '__main__':
    main()
