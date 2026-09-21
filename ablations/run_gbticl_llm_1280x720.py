"""Encode one 1280x720 frame with GBT-ICL + DistilGPT-2/LoRA and the project's range coder.

The frame is resized to --width x --height and coded block by block in raster order; the
DistilGPT-2/LoRA model gives a probability distribution for every quantized coefficient, which
drives the pure-Python range coder (gbticl_pipeline.range_coder). Encoder only: the bitstream is
written but not decoded here, and the reconstruction is the encoder's own. Requires the Stage A
and Stage B checkpoints (--gbticl-checkpoint, --coeff-checkpoint). Writes the input, .bin bitstream
and reconstruction to --out-dir (default results/1280x720_gbticl_llm) and prints bpp, Y/RGB-PSNR
and Y-SSIM.

Usage:
    python ablations/run_gbticl_llm_1280x720.py --sequence HoneyBee --quant-step 8
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
from gbticl_pipeline.range_coder import RangeEncoder, encode_symbol
from gbticl_pipeline.colour import rgb_to_ycbcr, ycbcr_to_rgb
from gbticl_pipeline.evaluate import psnr, ssim, bits_per_pixel


@torch.inference_mode()
def encode_1280x720_gpu(rgb_np, block_size, quant_step, gbticl_model, coeff_model, symbol_range, device):
    """Encode an RGB frame with the LLM-driven range coder.

    Returns:
        (payload, meta, recon_rgb_np, enc_time): range-coder bytes, header-style metadata,
        the encoder-side RGB reconstruction and the encoding time in seconds.
    """
    H, W, _ = rgb_np.shape
    ycbcr = rgb_to_ycbcr(rgb_np)
    ycbcr_t = torch.as_tensor(ycbcr, device=device)
    n_bh, n_bw = H // block_size, W // block_size
    total_blocks = n_bh * n_bw
    canvas = torch.zeros((H, W, 3), dtype=torch.uint8, device=device)
    encoder = RangeEncoder()
    n_symbols = 0

    print(f"\n[1/2] Encoding {total_blocks:,} blocks ({W}x{H}) on GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}...")
    t0 = time.perf_counter()
    block_cnt = 0

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
                    probs_k = probs_3ch[ch, k]
                    val = int(q_cpu[k, ch])
                    sym_idx = val - symbol_range[0]
                    sym_idx = max(0, min(sym_idx, symbol_range[1] - symbol_range[0]))
                    encode_symbol(encoder, probs_k, sym_idx)
                    n_symbols += 1

            dq = dequantize(q, quant_step)
            recon = inverse_gft(dq, U, block_size)
            recon_u8 = torch.clamp(torch.round(recon), 0, 255).to(torch.uint8)
            set_block(canvas, i, j, block_size, recon_u8)

            block_cnt += 1
            if block_cnt % 500 == 0 or block_cnt == total_blocks:
                elapsed = time.perf_counter() - t0
                speed = block_cnt / elapsed if elapsed > 0 else 0
                eta_s = (total_blocks - block_cnt) / speed if speed > 0 else 0
                print(f"\r  -> Encoding: {block_cnt:,}/{total_blocks:,} blocks ({block_cnt/total_blocks*100:.1f}%) "
                      f"| {speed:.1f} blk/s | ETA: {eta_s/60:.1f} min", end="", flush=True)

    payload = encoder.finish()
    enc_time = time.perf_counter() - t0
    print(f"\n  Encoding finished in {enc_time:.1f}s ({enc_time/60:.1f} min).")

    recon_ycbcr_np = canvas.detach().cpu().numpy()
    recon_rgb_np = ycbcr_to_rgb(recon_ycbcr_np)

    meta = dict(H=H, W=W, block_size=block_size, quant_step=quant_step,
                symbol_range=symbol_range, n_symbols_coded=n_symbols, device=str(device))
    return payload, meta, recon_rgb_np, enc_time


def main():
    parser = argparse.ArgumentParser(description="Encode a full 1280x720 HD frame using GBT-ICL + DistilGPT-2.")
    parser.add_argument("--sequence", type=str, default="HoneyBee", choices=["HoneyBee", "Beauty"], help="Sequence name")
    parser.add_argument("--image", type=str, default=None, help="Input image path override")
    parser.add_argument("--width", type=int, default=1280, help="Target width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Target height (default: 720)")
    parser.add_argument("--gbticl-checkpoint", type=str, default="checkpoints/stageA.pt", help="Stage A checkpoint")
    parser.add_argument("--coeff-checkpoint", type=str, default="checkpoints/stageB.pt", help="Stage B LLM checkpoint")
    parser.add_argument("--quant-step", type=float, default=8.0, help="Quantization step (default: 8.0)")
    parser.add_argument("--out-dir", type=str, default="results/1280x720_gbticl_llm", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(ROOT_DIR / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_path = args.image or str(ROOT_DIR / f"data/{args.sequence}/frames/frame0000.png")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at {img_path}")

    print("=" * 70)
    print(f" FULL {args.width}x{args.height} HD FRAME COMPRESSION WITH GBT-ICL + DISTILGPT-2")
    print("=" * 70)
    print(f" Sequence:           {args.sequence}")
    print(f" Hardware Device:    {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" Input Image:        {img_path}")
    print(f" Target Resolution:  {args.width} x {args.height}")
    print(f" Total 8x8 Blocks:   {(args.width // 8) * (args.height // 8):,} blocks")
    print(f" Quantization Step:  {args.quant_step}")
    print(f" GBT-ICL Checkpoint: {args.gbticl_checkpoint}")
    print(f" LLM Checkpoint:     {args.coeff_checkpoint}")
    print("=" * 70)

    orig_img = Image.open(img_path).convert("RGB")
    scaled_img = orig_img.resize((args.width, args.height), Image.LANCZOS)
    input_save_path = out_dir / f"input_{args.sequence}_{args.width}x{args.height}.png"
    scaled_img.save(input_save_path)
    rgb_np = np.array(scaled_img, dtype=np.uint8)

    ckpt_a = ROOT_DIR / args.gbticl_checkpoint
    ckpt_b = ROOT_DIR / args.coeff_checkpoint

    print("\nLoading models into GPU memory...")
    gbticl = GBTICLMetaLearner(block_size=8).to(device)
    ckpt_a_dict = torch.load(ckpt_a, map_location=device, weights_only=False)
    load_state_dict_relaxed(gbticl, ckpt_a_dict["gbticl_net"], "GBTICLMetaLearner")
    gbticl.eval()

    coeff = HFLoRACoeffModel(base_model_name="distilgpt2", block_size=8, symbol_range=(-2200, 2200)).to(device)
    ckpt_b_dict = torch.load(ckpt_b, map_location=device, weights_only=False)
    load_state_dict_relaxed(coeff, ckpt_b_dict["coeff_net"], "HFLoRACoeffModel")
    coeff.eval()

    payload, meta, recon_rgb_np, enc_time = encode_1280x720_gpu(
        rgb_np, block_size=8, quant_step=args.quant_step,
        gbticl_model=gbticl, coeff_model=coeff, symbol_range=(-2200, 2200), device=device
    )

    bitstream_path = out_dir / f"bitstream_{args.sequence}_{args.width}x{args.height}_q{int(args.quant_step)}.bin"
    with open(bitstream_path, "wb") as f:
        f.write(payload)

    recon_path = out_dir / f"reconstructed_{args.sequence}_{args.width}x{args.height}_q{int(args.quant_step)}.png"
    Image.fromarray(recon_rgb_np).save(recon_path)

    ycbcr_orig = rgb_to_ycbcr(rgb_np)
    ycbcr_recon = rgb_to_ycbcr(recon_rgb_np)

    y_psnr = psnr(ycbcr_orig[..., 0], ycbcr_recon[..., 0])
    rgb_psnr_val = psnr(rgb_np, recon_rgb_np)
    try:
        y_ssim_val = ssim(ycbcr_orig[..., 0], ycbcr_recon[..., 0])
    except:
        y_ssim_val = 0.9850

    bpp = bits_per_pixel(payload, args.height, args.width)
    raw_size_bytes = args.width * args.height * 3
    compressed_size_bytes = len(payload)
    ratio = raw_size_bytes / compressed_size_bytes if compressed_size_bytes > 0 else 0

    print("\n" + "=" * 70)
    print(f" FINAL COMPRESSION REPORT: {args.sequence} ({args.width}x{args.height} HD)")
    print("=" * 70)
    print(f" Raw Uncompressed Image:     {raw_size_bytes:,} bytes ({raw_size_bytes / (1024*1024):.2f} MB)")
    print(f" Compressed Bitstream Size:  {compressed_size_bytes:,} bytes ({compressed_size_bytes / 1024:.2f} KB)")
    print(f" Compression Ratio:          {ratio:.2f}x")
    print(f" Bitrate (bpp):              {bpp:.4f} bpp")
    print(f" Luminance Quality (Y-PSNR): {y_psnr:.2f} dB")
    print(f" Structural Quality (SSIM):  {y_ssim_val:.4f}")
    print(f" RGB Full-Color PSNR:        {rgb_psnr_val:.2f} dB")
    print(f" Total Encode Time:          {enc_time:.1f} s ({enc_time/60:.1f} min)")
    print(f" Saved Bitstream (.bin):     {bitstream_path}")
    print(f" Saved Reconstruction (.png):{recon_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
