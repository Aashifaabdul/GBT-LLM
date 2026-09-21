"""1080p video codec with the full GBT-ICL + DistilGPT-2/LoRA model and the pure-Python range coder.

Encodes and decodes the first --frames frames of --sequence (data/<sequence>/frames), block by
block, with the neural models on the GPU and the range coding in Python
(gbticl_pipeline.range_coder). Each frame is conditioned on the previous frame's reconstruction and
quantized coefficients. Verifies that the decoded coefficients equal the encoded ones and reports
bpp, Y/RGB-PSNR and Y-SSIM per frame. Writes bitstreams, decoded PNG frames, a metrics CSV and an
MP4 (requires ffmpeg) to results/1080p_pure_python_30frames/<sequence>/.

Usage:
    python ablations/run_1080p_pure_python_codec.py --sequence HoneyBee --frames 30 --quant-step 8
    (add --max-blocks 50 for a quick test)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import time
import csv
import argparse
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
import torch

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
from gbticl_pipeline.range_coder import RangeEncoder, RangeDecoder, encode_symbol, decode_symbol
from gbticl_pipeline.colour import rgb_to_ycbcr, ycbcr_to_rgb
from gbticl_pipeline.evaluate import psnr, ssim


@torch.inference_mode()
def encode_frame_pure_python_gpu(ycbcr_np, block_size, quant_step, gbticl_model, coeff_model,
                                 symbol_range, device, prev_canvas=None, prev_q=None, max_blocks=None):
    """Encode one YCbCr frame block by block in raster order.

    Args:
        ycbcr_np: (H, W, 3) uint8 YCbCr frame.
        prev_canvas, prev_q: reconstruction and quantized coefficients of the previous frame, or None.
        max_blocks: stop after this many blocks (testing only).

    Returns:
        (payload, meta, canvas, this_q, enc_time): range-coder bytes, decoder metadata, the encoder-side
        reconstruction, quantized coefficients of shape (n_bh, n_bw, 64, 3) and the encoding time in seconds.
    """
    H, W, _ = ycbcr_np.shape
    ycbcr_t = torch.as_tensor(ycbcr_np, device=device)
    n_bh, n_bw = H // block_size, W // block_size
    n = block_size * block_size
    total_blocks = n_bh * n_bw
    if max_blocks is not None and max_blocks > 0:
        total_blocks = min(total_blocks, max_blocks)

    canvas = torch.zeros((H, W, 3), dtype=torch.uint8, device=device)
    this_q = torch.zeros((n_bh, n_bw, n, 3), dtype=torch.int64, device=device)
    encoder = RangeEncoder()
    n_symbols = 0
    block_cnt = 0
    t0 = time.perf_counter()

    for i in range(n_bh):
        for j in range(n_bw):
            if max_blocks and block_cnt >= max_blocks:
                break

            # Graph basis from the causal context (reconstructed neighbours) and the previous frame.
            top, left, valid_top, valid_left = get_context(canvas, i, j, block_size)
            support = get_support_set(canvas, prev_canvas, i, j, block_size)
            weights = gbticl_model.predict_edge_weights(top, left, valid_top, valid_left, block_size, support=support)
            L = build_laplacian(weights, block_size, device=device)
            eigvals, U = eigendecompose(L)

            block = get_block(ycbcr_t, i, j, block_size)
            coeffs = forward_gft(block, U)
            q = quantize(coeffs, quant_step)
            this_q[i, j] = q

            # Quantized coefficients of the co-located block in the previous frame condition the coefficient model.
            temporal_values = prev_q[i, j].to(torch.float32) if prev_q is not None else None

            # Probabilities for all 64 x 3 symbols in one teacher-forced pass.
            probs_all = coeff_model.precompute_encode_probs(
                eigvals, q, symbol_range,
                **({"temporal_values": temporal_values} if temporal_values is not None else {})
            )

            for k in range(n):
                for ch in range(3):
                    probs = probs_all[k, ch].detach().cpu().numpy()
                    val = int(q[k, ch].item())
                    sym_idx = val - symbol_range[0]
                    # Values outside symbol_range are clipped to the edge symbols.
                    sym_idx = max(0, min(sym_idx, symbol_range[1] - symbol_range[0]))
                    encode_symbol(encoder, probs, sym_idx)
                    n_symbols += 1

            # Reconstruct as the decoder will, so later blocks see identical context.
            dq = dequantize(q, quant_step)
            recon = inverse_gft(dq, U, block_size)
            recon_u8 = torch.clamp(torch.round(recon), 0, 255).to(torch.uint8)
            set_block(canvas, i, j, block_size, recon_u8)

            block_cnt += 1
            if block_cnt % 300 == 0 or block_cnt == total_blocks:
                elapsed = time.perf_counter() - t0
                spd = block_cnt / elapsed if elapsed > 0 else 0
                eta_s = (total_blocks - block_cnt) / spd if spd > 0 else 0
                print(f"\r    [Encode] {block_cnt:,}/{total_blocks:,} blocks ({block_cnt/total_blocks*100:.1f}%) "
                      f"| {spd:.1f} blk/s | ETA: {eta_s/60:.1f} min", end="", flush=True)

        if max_blocks and block_cnt >= max_blocks:
            break

    payload = encoder.finish()
    enc_time = time.perf_counter() - t0
    meta = {
        "H": H, "W": W, "block_size": block_size, "quant_step": quant_step,
        "symbol_range": symbol_range, "n_symbols_coded": n_symbols,
        "total_blocks": block_cnt
    }
    return payload, meta, canvas, this_q, enc_time


@torch.inference_mode()
def decode_frame_pure_python_gpu(payload, meta, gbticl_model, coeff_model, device,
                                 prev_canvas=None, prev_q=None, max_blocks=None):
    """Decode one frame produced by encode_frame_pure_python_gpu.

    Returns:
        (recon_rgb_np, recon_ycbcr_np, canvas_dec, this_q_dec, dec_time).
    """
    H, W = meta["H"], meta["W"]
    block_size = meta["block_size"]
    quant_step = meta["quant_step"]
    symbol_range = meta["symbol_range"]
    n = block_size * block_size
    n_bh, n_bw = H // block_size, W // block_size
    total_blocks = meta.get("total_blocks", n_bh * n_bw)
    if max_blocks is not None and max_blocks > 0:
        total_blocks = min(total_blocks, max_blocks)

    canvas_dec = torch.zeros((H, W, 3), dtype=torch.uint8, device=device)
    this_q_dec = torch.zeros((n_bh, n_bw, n, 3), dtype=torch.int64, device=device)
    decoder = RangeDecoder(payload)
    block_cnt = 0
    t0 = time.perf_counter()

    for i in range(n_bh):
        for j in range(n_bw):
            if max_blocks and block_cnt >= max_blocks:
                break

            # Rebuild the encoder's graph basis from already decoded blocks.
            top, left, valid_top, valid_left = get_context(canvas_dec, i, j, block_size)
            support = get_support_set(canvas_dec, prev_canvas, i, j, block_size)
            weights = gbticl_model.predict_edge_weights(top, left, valid_top, valid_left, block_size, support=support)
            L = build_laplacian(weights, block_size, device=device)
            eigvals, U = eigendecompose(L)

            # Symbols are decoded autoregressively; the coefficient model reuses its KV cache across k.
            temporal_values = prev_q[i, j].to(torch.float32) if prev_q is not None else None
            q_dec = torch.zeros((n, 3), dtype=torch.int64, device=device)
            history = {0: [], 1: [], 2: []}

            for k in range(n):
                for ch in range(3):
                    temp_ch = temporal_values[:, ch] if temporal_values is not None else None
                    kwargs = {"temporal_values": temp_ch} if temp_ch is not None else {}
                    probs = coeff_model.symbol_probs(eigvals, k, history[ch], symbol_range, **kwargs)
                    probs_np = probs.detach().cpu().numpy()
                    sym_idx = decode_symbol(decoder, probs_np)
                    val = sym_idx + symbol_range[0]
                    q_dec[k, ch] = val
                    history[ch].append(val)

            this_q_dec[i, j] = q_dec

            dq = dequantize(q_dec, quant_step)
            recon = inverse_gft(dq, U, block_size)
            recon_u8 = torch.clamp(torch.round(recon), 0, 255).to(torch.uint8)
            set_block(canvas_dec, i, j, block_size, recon_u8)

            block_cnt += 1
            if block_cnt % 300 == 0 or block_cnt == total_blocks:
                elapsed = time.perf_counter() - t0
                spd = block_cnt / elapsed if elapsed > 0 else 0
                eta_s = (total_blocks - block_cnt) / spd if spd > 0 else 0
                print(f"\r    [Decode] {block_cnt:,}/{total_blocks:,} blocks ({block_cnt/total_blocks*100:.1f}%) "
                      f"| {spd:.1f} blk/s | ETA: {eta_s/60:.1f} min", end="", flush=True)

        if max_blocks and block_cnt >= max_blocks:
            break

    dec_time = time.perf_counter() - t0
    recon_ycbcr_np = canvas_dec.detach().cpu().numpy()
    recon_rgb_np = ycbcr_to_rgb(recon_ycbcr_np)
    return recon_rgb_np, recon_ycbcr_np, canvas_dec, this_q_dec, dec_time


def assemble_mp4(frame_dir, out_mp4, fps=30):
    """Assemble decoded_frame*.png in frame_dir into an MP4 with ffmpeg (skipped if ffmpeg is unavailable)."""
    frame_paths = sorted(frame_dir.glob("decoded_frame*.png"))
    if not frame_paths:
        return
    list_path = frame_dir / "_concat_list.txt"
    duration = 1.0 / fps
    with open(list_path, "w") as f:
        for p in frame_paths:
            f.write(f"file '{p.name}'\nduration {duration}\n")
        f.write(f"file '{frame_paths[-1].name}'\n")

    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path),
           "-fps_mode", "vfr", "-pix_fmt", "yuv420p", str(out_mp4)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"\n  -> Reassembled {len(frame_paths)} frames to: {out_mp4}")
        else:
            print(f"\n  (ffmpeg notice: could not write {out_mp4.name})")
    except FileNotFoundError:
        print("\n  (ffmpeg not found on PATH; individual PNG frames are preserved)")


def main():
    parser = argparse.ArgumentParser(description="Full 1080p HD 30-Frame Video Compression & Decompression with Pure Python Range Coder on GPU.")
    parser.add_argument("--sequence", type=str, default="HoneyBee", choices=["HoneyBee", "Beauty"], help="Sequence name")
    parser.add_argument("--frames", type=int, default=30, help="Number of frames to process (default: 30)")
    parser.add_argument("--width", type=int, default=1920, help="Target width (default: 1920 for 1080p)")
    parser.add_argument("--height", type=int, default=1080, help="Target height (default: 1080 for 1080p)")
    parser.add_argument("--gbticl-checkpoint", type=str, default="checkpoints/stageA.pt", help="Stage A checkpoint")
    parser.add_argument("--coeff-checkpoint", type=str, default="checkpoints/stageB.pt", help="Stage B LLM checkpoint")
    parser.add_argument("--quant-step", type=float, default=8.0, help="Quantization step (default: 8.0)")
    parser.add_argument("--symbol-range", type=int, nargs=2, default=[-2200, 2200], help="Quantized alphabet symbol range")
    parser.add_argument("--max-blocks", type=int, default=None, help="Optional test limit on number of blocks per frame (e.g. 50)")
    parser.add_argument("--out-dir", type=str, default="results/1080p_pure_python_30frames", help="Output directory")
    parser.add_argument("--fps", type=int, default=30, help="Framerate for MP4 assembly (default: 30)")
    args = parser.parse_args()

    out_dir = Path(ROOT_DIR / args.out_dir / args.sequence)
    out_dir.mkdir(parents=True, exist_ok=True)
    bitstreams_dir = out_dir / "bitstreams"
    bitstreams_dir.mkdir(parents=True, exist_ok=True)
    recon_dir = out_dir / "decoded_frames"
    recon_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    symbol_range = tuple(args.symbol_range)

    frames_dir = ROOT_DIR / f"data/{args.sequence}/frames"
    frame_files = sorted(frames_dir.glob("*.png"))[:args.frames]
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    total_blocks_per_frame = (args.width // 8) * (args.height // 8)
    if args.max_blocks:
        total_blocks_per_frame = min(total_blocks_per_frame, args.max_blocks)

    print("=" * 80)
    print(" FULL 1080P VIDEO CODEC (30 FRAMES): PURE PYTHON (NO C++) ON GPU")
    print("=" * 80)
    print(f" Sequence:            {args.sequence}")
    print(f" Frames to Process:   {len(frame_files)} frames")
    print(f" Resolution:          {args.width} x {args.height} (1080p Full HD)")
    print(f" Blocks / Frame:      {total_blocks_per_frame:,} blocks")
    print(f" Hardware Device:     {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(" Range Coder:         Pure Python RangeEncoder/RangeDecoder (NO C++/Rust)")
    print(f" Quantization Step:   {args.quant_step}")
    print(f" Checkpoints:         {args.gbticl_checkpoint} & {args.coeff_checkpoint}")
    print(f" Output Directory:    {out_dir}")
    print("=" * 80)

    print("\nLoading models into GPU memory...")
    gbticl = GBTICLMetaLearner(block_size=8).to(device)
    ckpt_a = torch.load(ROOT_DIR / args.gbticl_checkpoint, map_location=device, weights_only=False)
    load_state_dict_relaxed(gbticl, ckpt_a["gbticl_net"], "GBTICLMetaLearner")
    gbticl.eval()

    coeff = HFLoRACoeffModel(base_model_name="distilgpt2", block_size=8, symbol_range=symbol_range).to(device)
    ckpt_b = torch.load(ROOT_DIR / args.coeff_checkpoint, map_location=device, weights_only=False)
    load_state_dict_relaxed(coeff, ckpt_b["coeff_net"], "HFLoRACoeffModel")
    coeff.eval()

    metrics_csv = out_dir / "metrics_30frames.csv"
    csv_file = open(metrics_csv, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_idx", "raw_bytes", "compressed_bytes", "bpp", "y_psnr", "rgb_psnr", "y_ssim", "enc_time_s", "dec_time_s", "verified_lossless"])

    # Previous-frame state for temporal conditioning; encoder and decoder keep separate copies.
    enc_prev_canvas = None
    enc_prev_q = None
    dec_prev_canvas = None
    dec_prev_q = None

    metrics_list = []
    total_pipeline_t0 = time.perf_counter()

    for f_idx, fp in enumerate(frame_files):
        print("\n" + "-" * 70)
        print(f" >>> FRAME {f_idx:04d} / {len(frame_files):04d} [{fp.name}]")
        print("-" * 70)

        orig_img = Image.open(fp).convert("RGB")
        if orig_img.size != (args.width, args.height):
            orig_img = orig_img.resize((args.width, args.height), Image.LANCZOS)
        rgb_np = np.array(orig_img, dtype=np.uint8)
        ycbcr_np = rgb_to_ycbcr(rgb_np)

        payload, meta, enc_canvas, enc_q, enc_time = encode_frame_pure_python_gpu(
            ycbcr_np, block_size=8, quant_step=args.quant_step,
            gbticl_model=gbticl, coeff_model=coeff, symbol_range=symbol_range,
            device=device, prev_canvas=enc_prev_canvas, prev_q=enc_prev_q, max_blocks=args.max_blocks
        )
        enc_prev_canvas = enc_canvas
        enc_prev_q = enc_q

        bin_path = bitstreams_dir / f"bitstream_frame{f_idx:04d}_q{int(args.quant_step)}.bin"
        with open(bin_path, "wb") as f:
            f.write(payload)

        recon_rgb_np, recon_ycbcr_np, dec_canvas, dec_q, dec_time = decode_frame_pure_python_gpu(
            payload, meta, gbticl_model=gbticl, coeff_model=coeff, device=device,
            prev_canvas=dec_prev_canvas, prev_q=dec_prev_q, max_blocks=args.max_blocks
        )
        dec_prev_canvas = dec_canvas
        dec_prev_q = dec_q

        recon_path = recon_dir / f"decoded_frame{f_idx:04d}.png"
        Image.fromarray(recon_rgb_np).save(recon_path)

        # Encoder/decoder consistency check: the quantized coefficients must match exactly.
        lossless_ok = torch.equal(enc_q, dec_q)

        y_psnr_val = psnr(ycbcr_np[..., 0], recon_ycbcr_np[..., 0])
        rgb_psnr_val = psnr(rgb_np, recon_rgb_np)
        try:
            y_ssim_val = ssim(ycbcr_np[..., 0], recon_ycbcr_np[..., 0])
        except Exception:
            y_ssim_val = float("nan")

        total_pixels = (meta["total_blocks"] * 64) if args.max_blocks else (args.width * args.height)
        bpp_val = (len(payload) * 8.0) / total_pixels
        raw_bytes = total_pixels * 3
        comp_bytes = len(payload)

        csv_writer.writerow([f_idx, raw_bytes, comp_bytes, f"{bpp_val:.4f}", f"{y_psnr_val:.2f}",
                             f"{rgb_psnr_val:.2f}", f"{y_ssim_val:.4f}", f"{enc_time:.2f}",
                             f"{dec_time:.2f}", lossless_ok])
        csv_file.flush()

        if device.type == "cuda":
            torch.cuda.empty_cache()

        metrics_list.append({
            "bpp": bpp_val, "y_psnr": y_psnr_val, "rgb_psnr": rgb_psnr_val,
            "y_ssim": y_ssim_val, "enc_time": enc_time, "dec_time": dec_time,
            "bytes": comp_bytes
        })

        print(f"\n  [Frame {f_idx:04d} Summary] Bitstream: {comp_bytes:,} B | {bpp_val:.4f} bpp | "
              f"Y-PSNR: {y_psnr_val:.2f} dB | SSIM: {y_ssim_val:.4f} | "
              f"Time: {enc_time:.1f}s enc, {dec_time:.1f}s dec | Lossless: {lossless_ok}")

    csv_file.close()
    total_time = time.perf_counter() - total_pipeline_t0

    mp4_path = out_dir / f"reconstructed_1080p_{args.sequence}_30frames_q{int(args.quant_step)}.mp4"
    assemble_mp4(recon_dir, mp4_path, fps=args.fps)

    mean_bpp = np.mean([m["bpp"] for m in metrics_list])
    mean_y_psnr = np.mean([m["y_psnr"] for m in metrics_list])
    mean_rgb_psnr = np.mean([m["rgb_psnr"] for m in metrics_list])
    mean_ssim = np.mean([m["y_ssim"] for m in metrics_list])
    total_bytes = sum([m["bytes"] for m in metrics_list])
    mean_enc_time = np.mean([m["enc_time"] for m in metrics_list])
    mean_dec_time = np.mean([m["dec_time"] for m in metrics_list])

    print("\n" + "=" * 80)
    print(f" FINAL 30-FRAME 1080P BENCHMARK REPORT: {args.sequence}")
    print("=" * 80)
    print(f" Processed Frames:            {len(metrics_list)} frames")
    print(f" Total Compressed Payload:    {total_bytes:,} bytes ({total_bytes / (1024*1024):.2f} MB)")
    print(f" Mean Bitrate:                {mean_bpp:.4f} bpp")
    print(f" Mean Luminance Quality:      {mean_y_psnr:.2f} dB Y-PSNR")
    print(f" Mean Structural Quality:     {mean_ssim:.4f} Y-SSIM")
    print(f" Mean RGB Quality:            {mean_rgb_psnr:.2f} dB RGB-PSNR")
    print(f" Mean Encoding Time / Frame:  {mean_enc_time:.1f} s ({mean_enc_time/60:.2f} min)")
    print(f" Mean Decoding Time / Frame:  {mean_dec_time:.1f} s ({mean_dec_time/60:.2f} min)")
    print(f" Total Elapsed Runtime:       {total_time:.1f} s ({total_time/3600:.2f} hours)")
    print(f" Bitstreams Saved In:         {bitstreams_dir}")
    print(f" Decoded Frames Saved In:     {recon_dir}")
    print(f" CSV Metrics Summary:         {metrics_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
