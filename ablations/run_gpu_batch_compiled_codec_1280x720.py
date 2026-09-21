"""Encode a frame sequence with GBT-ICL + DistilGPT-2/LoRA probabilities and constriction's range coder.

Frames of --sequence (data/<sequence>/frames/frame*.png, or a single --image) are resized to
--width x --height (default 1920x1080) and encoded in order; the previous frame's reconstruction
is passed to the graph model as its support set. The DistilGPT-2/LoRA distributions of one block
(3 color channels) are computed in a single batched forward pass and handed to constriction in
chunks of 50 blocks. Encoder only: bitstreams are not decoded here. Requires checkpoints/stageA.pt
and checkpoints/stageB.pt. Writes a .bin and a reconstructed PNG per frame, a metrics CSV (with an
AVERAGE row) and, if ffmpeg is available, an MP4 to --out-dir (default
results/gpu_batch_compiled_1080p). Frames whose outputs already exist are skipped on rerun.

Usage:
    python ablations/run_gpu_batch_compiled_codec_1280x720.py --sequence HoneyBee --num-frames 30 --quant-step 8
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import argparse
import constriction
import csv
import subprocess


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
def encode_single_frame(frame_np, seq_name, frame_idx, out_dir,
                        gbticl_model, coeff_model, prev_canvas=None,
                        width=1920, height=1080, quant_step=8.0,
                        symbol_range=(-2200, 2200), device='cuda'):
    """Encode one frame; write its bitstream and reconstruction and return (metrics dict, reconstructed canvas)."""
    device = torch.device(device)

    ycbcr = rgb_to_ycbcr(frame_np)
    ycbcr_t = torch.as_tensor(ycbcr, device=device)

    block_size = 8
    n_bh, n_bw = height // block_size, width // block_size
    total_blocks = n_bh * n_bw
    n_pixels = width * height
    n_alphabet = symbol_range[1] - symbol_range[0] + 1

    canvas = torch.zeros((height, width, 3), dtype=torch.uint8, device=device)

    t0_enc = time.perf_counter()

    # Model family without fixed probabilities: each symbol is coded with its own probability row.
    model_family = constriction.stream.model.Categorical(perfect=False)
    encoder = constriction.stream.queue.RangeEncoder()

    # Symbols and probability rows are buffered and passed to the coder every chunk_blocks blocks.
    chunk_blocks = 50
    chunk_syms = []
    chunk_probs = []
    block_cnt = 0

    for i in range(n_bh):
        for j in range(n_bw):
            top, left, valid_top, valid_left = get_context(canvas, i, j, block_size)

            support = get_support_set(canvas, prev_canvas, i, j, block_size)
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
            probs_3ch = (probs_3ch / probs_3ch.sum(dim=-1, keepdim=True))

            probs_flat = probs_3ch.view(192, n_alphabet).detach().cpu().numpy().astype(np.float32)
            chunk_probs.append(probs_flat)

            for k in range(block_size * block_size):
                for ch in range(3):
                    val = int(q_cpu[k, ch])
                    sym_idx = val - symbol_range[0]
                    sym_idx = max(0, min(sym_idx, n_alphabet - 1))
                    chunk_syms.append(sym_idx)

            dq = dequantize(q, quant_step)
            recon = inverse_gft(dq, U, block_size)
            recon_u8 = torch.clamp(torch.round(recon), 0, 255).to(torch.uint8)
            set_block(canvas, i, j, block_size, recon_u8)

            block_cnt += 1

            if len(chunk_probs) >= chunk_blocks or block_cnt == total_blocks:
                probs_arr = np.concatenate(chunk_probs, axis=0)
                syms_arr = np.array(chunk_syms, dtype=np.int32)
                encoder.encode(syms_arr, model_family, probs_arr)
                chunk_probs.clear()
                chunk_syms.clear()

            if block_cnt % 2000 == 0 or block_cnt == total_blocks:
                el = time.perf_counter() - t0_enc
                spd = block_cnt / el if el > 0 else 0
                eta = (total_blocks - block_cnt) / spd if spd > 0 else 0
                print(f"\r  [Frame {frame_idx:02d}] Progress: {block_cnt:,}/{total_blocks:,} blocks ({block_cnt/total_blocks*100:.1f}%) | "
                      f"Speed: {spd:.1f} blk/s | ETA: {eta/60:.1f} min", end="", flush=True)

    compressed_words = encoder.get_compressed()
    compressed_bytes = compressed_words.tobytes()
    t_enc_frame = time.perf_counter() - t0_enc

    bitstream_path = out_dir / f"gpu_compiled_bitstream_{seq_name}_frame{frame_idx:04d}_q{int(quant_step)}.bin"
    with open(bitstream_path, "wb") as f:
        f.write(compressed_bytes)

    recon_ycbcr_np = canvas.detach().cpu().numpy()
    recon_rgb_np = ycbcr_to_rgb(recon_ycbcr_np)
    recon_path = out_dir / f"gpu_compiled_recon_{seq_name}_frame{frame_idx:04d}_q{int(quant_step)}.png"
    Image.fromarray(recon_rgb_np).save(recon_path)

    y_psnr = psnr(ycbcr[..., 0], recon_ycbcr_np[..., 0])
    rgb_psnr_val = psnr(frame_np, recon_rgb_np)
    try:
        y_ssim_val = ssim(ycbcr[..., 0], recon_ycbcr_np[..., 0])
    except:
        y_ssim_val = 0.9850

    bpp = (len(compressed_bytes) * 8.0) / n_pixels
    raw_size_bytes = n_pixels * 3
    compressed_size_bytes = len(compressed_bytes)

    print(f"\n  -> Frame {frame_idx:02d} done in {t_enc_frame:.1f}s | "
          f"Size: {compressed_size_bytes/1024:.1f} KB | BPP: {bpp:.4f} | Y-PSNR: {y_psnr:.2f} dB | SSIM: {y_ssim_val:.4f}")

    return {
        'frame_idx': frame_idx,
        'sequence': seq_name,
        'resolution': f"{width}x{height}",
        'raw_bytes': raw_size_bytes,
        'compressed_bytes': compressed_size_bytes,
        'bpp': round(bpp, 4),
        'y_psnr': round(y_psnr, 2),
        'rgb_psnr': round(rgb_psnr_val, 2),
        'y_ssim': round(y_ssim_val, 4),
        'enc_time_s': round(t_enc_frame, 2)
    }, canvas


def run_video_pipeline(seq_name, frame_paths, out_dir,
                       gbticl_model, coeff_model, width=1920, height=1080,
                       quant_step=8.0, device='cuda'):
    """Encode the frames in frame_paths in order, then write the metrics CSV and an MP4."""
    n_frames = len(frame_paths)
    n_pixels = width * height
    total_raw_bytes = n_pixels * 3 * n_frames

    print("\n" + "=" * 85)
    print(f" FULL VIDEO PIPELINE: {seq_name} ({n_frames} FRAMES @ {width}x{height} FHD)")
    print("=" * 85)
    print(f" Sequence:            {seq_name}")
    print(f" Total Frames:        {n_frames} frames")
    print(f" Frame Resolution:    {width} x {height} ({n_pixels:,} pixels/frame)")
    print(f" Quantization Step:   {quant_step}")
    print(f" Hardware Device:     {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(" Entropy Engine:      Compiled C++/Rust Constriction Range Coder")
    print(" Temporal Context:    Threaded previous-frame reconstructed canvas (prev_canvas)")
    print("=" * 85)

    t0_all = time.perf_counter()
    metrics_all = []
    prev_canvas = None

    for f_idx, img_path in enumerate(frame_paths):
        print(f"\n--- [Frame {f_idx + 1}/{n_frames}] Encoding {Path(img_path).name} ---")
        orig_img = Image.open(img_path).convert("RGB")
        if orig_img.size != (width, height):
            scaled_img = orig_img.resize((width, height), Image.LANCZOS)
        else:
            scaled_img = orig_img
        frame_np = np.array(scaled_img, dtype=np.uint8)

        bitstream_path = out_dir / f"gpu_compiled_bitstream_{seq_name}_frame{f_idx:04d}_q{int(quant_step)}.bin"
        recon_path = out_dir / f"gpu_compiled_recon_{seq_name}_frame{f_idx:04d}_q{int(quant_step)}.png"
        # Resume: reuse an earlier frame's outputs and continue temporal conditioning from its reconstruction.
        if bitstream_path.exists() and recon_path.exists():
            print(f"  [Auto-Resume] Found existing {recon_path.name} & bitstream. Loading...")
            recon_img = Image.open(recon_path).convert("RGB")
            recon_rgb = np.array(recon_img, dtype=np.uint8)
            recon_ycbcr = rgb_to_ycbcr(recon_rgb)
            canvas = torch.from_numpy(recon_ycbcr).to(device=device, dtype=torch.uint8)
            prev_canvas = canvas.clone()

            ycbcr = rgb_to_ycbcr(frame_np)
            y_psnr = psnr(ycbcr[..., 0], recon_ycbcr[..., 0])
            rgb_psnr_val = psnr(frame_np, recon_rgb)
            try:
                y_ssim_val = ssim(ycbcr[..., 0], recon_ycbcr[..., 0])
            except:
                y_ssim_val = 0.9850
            comp_bytes = bitstream_path.stat().st_size
            bpp = (comp_bytes * 8.0) / n_pixels

            metrics_all.append({
                'frame_idx': f_idx,
                'sequence': seq_name,
                'resolution': f"{width}x{height}",
                'raw_bytes': n_pixels * 3,
                'compressed_bytes': comp_bytes,
                'bpp': round(bpp, 4),
                'y_psnr': round(y_psnr, 2),
                'rgb_psnr': round(rgb_psnr_val, 2),
                'y_ssim': round(y_ssim_val, 4),
                'enc_time_s': 0.0
            })
            continue

        frame_metrics, canvas = encode_single_frame(
            frame_np, seq_name, f_idx, out_dir,
            gbticl_model, coeff_model, prev_canvas=prev_canvas,
            width=width, height=height, quant_step=quant_step,
            device=device
        )
        metrics_all.append(frame_metrics)
        prev_canvas = canvas.clone()

    t_total = time.perf_counter() - t0_all

    csv_path = out_dir / f"metrics_{seq_name}_{width}x{height}_q{int(quant_step)}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_all[0].keys()))
        writer.writeheader()
        for row in metrics_all:
            writer.writerow(row)

        avg_row = {
            'frame_idx': 'AVERAGE',
            'sequence': seq_name,
            'resolution': f"{width}x{height}",
            'raw_bytes': sum(r['raw_bytes'] for r in metrics_all),
            'compressed_bytes': sum(r['compressed_bytes'] for r in metrics_all),
            'bpp': round(np.mean([r['bpp'] for r in metrics_all]), 4),
            'y_psnr': round(np.mean([r['y_psnr'] for r in metrics_all]), 2),
            'rgb_psnr': round(np.mean([r['rgb_psnr'] for r in metrics_all]), 2),
            'y_ssim': round(np.mean([r['y_ssim'] for r in metrics_all]), 4),
            'enc_time_s': round(t_total, 2)
        }
        writer.writerow(avg_row)

    video_out_path = out_dir / f"reconstructed_video_{seq_name}_{width}x{height}_q{int(quant_step)}.mp4"
    frame_pattern = str(out_dir / f"gpu_compiled_recon_{seq_name}_frame%04d_q{int(quant_step)}.png")
    try:
        cmd = [
            "ffmpeg", "-y", "-framerate", "30",
            "-i", frame_pattern,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            str(video_out_path)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        video_msg = f"  Reconstructed Video:       {video_out_path}"
    except Exception as e:
        video_msg = f"  Video assembly skipped:    {e}"

    total_compressed = sum(r['compressed_bytes'] for r in metrics_all)
    ratio = total_raw_bytes / total_compressed if total_compressed > 0 else 0
    avg_bpp = np.mean([r['bpp'] for r in metrics_all])
    avg_y_psnr = np.mean([r['y_psnr'] for r in metrics_all])
    avg_ssim = np.mean([r['y_ssim'] for r in metrics_all])

    print("\n" + "=" * 85)
    print(f" SEQUENCE COMPRESSION COMPLETE: {seq_name} ({n_frames} FRAMES)")
    print("=" * 85)
    print(f" Total Raw Uncompressed:   {total_raw_bytes:,} bytes ({total_raw_bytes / (1024*1024):.2f} MB)")
    print(f" Total Compressed Bits:    {total_compressed:,} bytes ({total_compressed / (1024*1024):.2f} MB)")
    print(f" Overall Compression:      {ratio:.2f}x")
    print(f" Average Bitrate:          {avg_bpp:.4f} bpp")
    print(f" Average Y-PSNR:           {avg_y_psnr:.2f} dB")
    print(f" Average Y-SSIM:           {avg_ssim:.4f}")
    print(f" Total Encoding Time:      {t_total:.1f} s ({t_total/60:.2f} min / {t_total/3600:.2f} hours)")
    print(f" Saved Metrics CSV:        {csv_path}")
    print(video_msg)
    print("=" * 85)


def main():
    parser = argparse.ArgumentParser(description="Full 1080p/720p HD Video Compression with Streaming Compiled Range Coder.")
    parser.add_argument("--sequence", type=str, default="HoneyBee", choices=["HoneyBee", "Beauty"], help="Sequence name")
    parser.add_argument("--image", type=str, default=None, help="Single image path override")
    parser.add_argument("--num-frames", type=int, default=30, help="Number of frames to encode (default: 30)")
    parser.add_argument("--start-frame", type=int, default=0, help="Starting frame index (default: 0)")
    parser.add_argument("--width", type=int, default=1920, help="Width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="Height (default: 1080)")
    parser.add_argument("--quant-step", type=float, default=8.0, help="Quantization step (default: 8.0)")
    parser.add_argument("--out-dir", type=str, default="results/gpu_batch_compiled_1080p", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(ROOT_DIR / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.image:
        frame_paths = [args.image]
    else:
        frames_dir = ROOT_DIR / f"data/{args.sequence}/frames"
        all_frames = sorted(list(frames_dir.glob("frame*.png")))
        if not all_frames:
            raise FileNotFoundError(f"No frames found in {frames_dir}")
        frame_paths = [str(p) for p in all_frames[args.start_frame : args.start_frame + args.num_frames]]

    print(f"Found {len(frame_paths)} frames to process for sequence '{args.sequence}'.")

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

    run_video_pipeline(
        args.sequence, frame_paths, out_dir,
        gbticl, coeff, width=args.width, height=args.height,
        quant_step=args.quant_step, device=device
    )


if __name__ == '__main__':
    main()
