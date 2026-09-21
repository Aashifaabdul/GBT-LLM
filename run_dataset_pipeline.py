"""
Run the codec over extracted frames and evaluate it.

A whole sequence is encoded as video (temporal support and coefficient
conditioning carried from frame to frame, YCbCr colour) with the real codec
(GBT-ICL graph -> GFT -> quantisation -> range coder), decoded again, and
compared with the original. Per-frame Y-PSNR, Y-SSIM, RGB-PSNR and bits per pixel
are written to metrics.csv, the reconstructed frames are saved and reassembled
into a video, and a predicted-graph figure is saved for models that predict one.

Range coding runs on the CPU, one symbol at a time, so a full 1920x1080 frame
(32,400 blocks x 192 coefficients) takes a long time. By default every frame is
centre-cropped to --crop pixels (256); use --full-frame to disable the crop and
--max-frames to limit the number of frames.

Examples:
  # baselines, no training needed
  python run_dataset_pipeline.py --sequence Beauty --max-frames 3

  # named configurations (see ABLATION_CONFIGS)
  python run_dataset_pipeline.py --sequence Beauty --ablation dct
  python run_dataset_pipeline.py --sequence Beauty --ablation nonadaptive_gbt
  python run_dataset_pipeline.py --sequence Beauty --ablation metalearner_no_llm \
      --gbticl-checkpoint checkpoints/stageA.pt
  python run_dataset_pipeline.py --sequence Beauty --ablation full \
      --gbticl-checkpoint checkpoints/stageA.pt --coeff-checkpoint checkpoints/stageB.pt

  # rate-distortion sweep (one row per quantisation step in ablation_summary.csv)
  python run_dataset_pipeline.py --sequence Beauty --ablation dct --quant-steps 2,4,8,16,32

  # legacy single checkpoint (GBTICLNet + TinyTransformerCoeffModel)
  python run_dataset_pipeline.py --sequence Beauty --checkpoint checkpoints/gbticl_ckpt.pt

  # both sequences at full resolution
  python run_dataset_pipeline.py --sequence both --full-frame
"""

import argparse
import csv
import subprocess
import time
from pathlib import Path

import numpy as np

try:
    import torch
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
except ImportError as e:
    raise SystemExit(f"run_dataset_pipeline.py requires PyTorch and matplotlib (import error: {e})")

from PIL import Image

from gbticl_pipeline.device_utils import get_device, load_state_dict_relaxed
from gbticl_pipeline.codec import encode_image, decode_image, encode_video, decode_video
from gbticl_pipeline.evaluate import psnr, ssim, bits_per_pixel
from gbticl_pipeline.colour import rgb_to_ycbcr, ycbcr_to_rgb
from gbticl_pipeline.graph_model import GBTICLNet, GBTICLMetaLearner, ContextGradientGBTICL, UniformGBTICL, edge_list
from gbticl_pipeline.coeff_model import TinyTransformerCoeffModel, HFLoRACoeffModel, LaplaceCoeffModel

BLOCK_SIZE = 8
DEFAULT_SYMBOL_RANGE = (-2200, 2200)
DEFAULT_QUANT_STEP = 8.0


# Named configurations and the checkpoints each one needs:
#   dct                 fixed DCT basis, Laplace coefficient model
#   nonadaptive_gbt     uniform-weight grid graph, Laplace coefficient model
#   gbticl_no_llm       GBTICLNet checkpoint, Laplace coefficient model
#   metalearner_no_llm  GBTICLMetaLearner checkpoint, Laplace coefficient model
#   full                GBT-ICL checkpoint + coefficient-model checkpoint
ABLATION_CONFIGS = {
    "dct": dict(needs_gbticl_ckpt=False, needs_coeff_ckpt=False),
    "nonadaptive_gbt": dict(needs_gbticl_ckpt=False, needs_coeff_ckpt=False),
    "gbticl_no_llm": dict(needs_gbticl_ckpt=True, needs_coeff_ckpt=False),
    "metalearner_no_llm": dict(needs_gbticl_ckpt=True, needs_coeff_ckpt=False),
    "full": dict(needs_gbticl_ckpt=True, needs_coeff_ckpt=True),
}


def center_crop(img, size):
    """Centre-crop img (H, W, 3) to a size x size region aligned to the block grid."""
    h, w = img.shape[:2]
    size = (size // BLOCK_SIZE) * BLOCK_SIZE
    size = min(size, (h // BLOCK_SIZE) * BLOCK_SIZE, (w // BLOCK_SIZE) * BLOCK_SIZE)
    r0 = (h - size) // 2
    c0 = (w - size) // 2
    r0 -= r0 % BLOCK_SIZE
    c0 -= c0 % BLOCK_SIZE
    return img[r0:r0 + size, c0:c0 + size, :]


def to_block_multiple(img):
    """Trim the bottom/right edge so height and width are multiples of BLOCK_SIZE."""
    h, w = img.shape[:2]
    h2 = (h // BLOCK_SIZE) * BLOCK_SIZE
    w2 = (w // BLOCK_SIZE) * BLOCK_SIZE
    return img[:h2, :w2, :]


def load_models(checkpoint_path, device):
    """
    Load the legacy single-checkpoint pair (GBTICLNet + TinyTransformerCoeffModel).

    Without a checkpoint the non-learned baselines are used. Returns
    (gbticl_model, coeff_model, quant_step, symbol_range, label).
    """
    if checkpoint_path is None:
        print("No --checkpoint given: using the non-learned baselines "
              "(ContextGradientGBTICL + LaplaceCoeffModel)")
        gbticl_model = ContextGradientGBTICL().to(device)
        coeff_model = LaplaceCoeffModel().to(device)
        return gbticl_model, coeff_model, DEFAULT_QUANT_STEP, DEFAULT_SYMBOL_RANGE, "baseline"

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    block_size = ckpt.get("block_size", BLOCK_SIZE)
    symbol_range = tuple(ckpt.get("symbol_range", DEFAULT_SYMBOL_RANGE))
    quant_step = ckpt.get("quant_step", DEFAULT_QUANT_STEP)

    gbticl_model = GBTICLNet(block_size=block_size).to(device)
    load_state_dict_relaxed(gbticl_model, ckpt["gbticl_net"], "GBTICLNet")
    gbticl_model.eval()

    coeff_model = TinyTransformerCoeffModel(block_size=block_size, symbol_range=symbol_range).to(device)
    load_state_dict_relaxed(coeff_model, ckpt["coeff_net"], "TinyTransformerCoeffModel")
    coeff_model.eval()

    print(f"loaded trained checkpoint {checkpoint_path} "
          f"(epoch {ckpt.get('epoch', '?')}, quant_step={quant_step}, symbol_range={symbol_range})")
    return gbticl_model, coeff_model, quant_step, symbol_range, "trained"


def load_gbticl_for_ablation(gbticl_checkpoint, device):
    """Load a GBT-ICL model from a training.py checkpoint; the model class is read from the checkpoint."""
    ckpt = torch.load(gbticl_checkpoint, map_location=device, weights_only=False)
    model_type = ckpt.get("gbticl_model_type", "net")
    block_size = ckpt.get("block_size", BLOCK_SIZE)
    model = (GBTICLMetaLearner if model_type == "metalearner" else GBTICLNet)(block_size=block_size).to(device)
    load_state_dict_relaxed(model, ckpt["gbticl_net"], model_type)
    model.eval()
    print(f"loaded GBT-ICL ({model_type}) from {gbticl_checkpoint} (epoch {ckpt.get('epoch', '?')})")
    return model


def load_coeff_for_ablation(coeff_checkpoint, base_model_name, symbol_range, device):
    """Load a coefficient model from a training.py checkpoint; the model class is read from the checkpoint."""
    ckpt = torch.load(coeff_checkpoint, map_location=device, weights_only=False)
    coeff_type = ckpt.get("coeff_model_type", "tiny")
    block_size = ckpt.get("block_size", BLOCK_SIZE)
    if coeff_type == "hf_lora":
        base_model_name = ckpt.get("base_model_name", base_model_name)
        model = HFLoRACoeffModel(base_model_name=base_model_name, block_size=block_size,
                                  symbol_range=symbol_range).to(device)
    else:
        model = TinyTransformerCoeffModel(block_size=block_size, symbol_range=symbol_range).to(device)
    load_state_dict_relaxed(model, ckpt["coeff_net"], coeff_type)
    model.eval()
    print(f"loaded coefficient model ({coeff_type}) from {coeff_checkpoint}")
    return model


def load_models_for_ablation(ablation, gbticl_checkpoint, coeff_checkpoint, base_model_name,
                              symbol_range, device):
    """
    Build the models of one named configuration.

    Returns (gbticl_model or None, coeff_model, fixed_basis); gbticl_model is None
    only for "dct", where the codec uses the fixed basis instead.
    """
    if ablation not in ABLATION_CONFIGS:
        raise SystemExit(f"unknown --ablation {ablation!r}, choose from {list(ABLATION_CONFIGS)}")
    cfg = ABLATION_CONFIGS[ablation]

    if cfg["needs_gbticl_ckpt"] and not gbticl_checkpoint:
        raise SystemExit(f"--ablation {ablation} requires --gbticl-checkpoint")
    if cfg["needs_coeff_ckpt"] and not coeff_checkpoint:
        raise SystemExit(f"--ablation {ablation} requires --coeff-checkpoint")

    fixed_basis = ablation == "dct"
    gbticl_model = None
    if ablation == "nonadaptive_gbt":
        gbticl_model = UniformGBTICL().to(device)
    elif cfg["needs_gbticl_ckpt"]:
        gbticl_model = load_gbticl_for_ablation(gbticl_checkpoint, device)

    if cfg["needs_coeff_ckpt"]:
        coeff_model = load_coeff_for_ablation(coeff_checkpoint, base_model_name, symbol_range, device)
    else:
        coeff_model = LaplaceCoeffModel().to(device)

    return gbticl_model, coeff_model, fixed_basis


def estimate_runtime(gbticl_model, coeff_model, quant_step, symbol_range, sample_img, device, n_frames,
                      fixed_basis=False):
    """Time one block (encode + decode) and print the extrapolated total runtime."""
    tiny = sample_img[:BLOCK_SIZE, :BLOCK_SIZE, :]
    t0 = time.time()
    payload, meta = encode_image(tiny, block_size=BLOCK_SIZE, quant_step=quant_step,
                                  gbticl_model=gbticl_model, coeff_model=coeff_model,
                                  symbol_range=symbol_range, device=device, fixed_basis=fixed_basis)
    decode_image(payload, meta, gbticl_model=gbticl_model, coeff_model=coeff_model, device=device)
    per_block = time.time() - t0

    h, w = sample_img.shape[:2]
    n_blocks = (h // BLOCK_SIZE) * (w // BLOCK_SIZE)
    est_seconds = per_block * n_blocks * n_frames
    print(f"runtime estimate: ~{per_block*1000:.0f} ms/block x {n_blocks} blocks/frame x "
          f"{n_frames} frames ~= {est_seconds/60:.1f} min total (rough; real runs vary)")
    if est_seconds > 1800:
        print("  -> over 30 minutes; consider a smaller --crop or fewer --max-frames")


def save_graph_figure(gbticl_model, img, out_path, device, title_suffix=""):
    """Save a figure of the graph predicted for the highest-variance block of `img`."""
    bs = BLOCK_SIZE
    h, w = img.shape[:2]
    n_bh, n_bw = h // bs, w // bs

    if n_bh < 2 or n_bw < 2:
        print("skipping predicted-graph figure: image is smaller than a 2x2 block grid")
        return
    best, best_var = (1, 1), -1.0
    for bi in range(1, n_bh):
        for bj in range(1, n_bw):
            r0, c0 = bi * bs, bj * bs
            v = img[r0:r0 + bs, c0:c0 + bs, :].astype(float).var()
            if v > best_var:
                best_var, best = v, (bi, bj)
    bi, bj = best
    r0, c0 = bi * bs, bj * bs
    block = img[r0:r0 + bs, c0:c0 + bs, :]
    top_ctx = img[r0 - 1, c0:c0 + bs, :]
    left_ctx = img[r0:r0 + bs, c0 - 1, :]

    top_t = torch.from_numpy(top_ctx.copy()).to(device)
    left_t = torch.from_numpy(left_ctx.copy()).to(device)
    with torch.no_grad():
        weights = gbticl_model.predict_edge_weights(top_t, left_t, True, True, bs)
    weights = weights.detach().cpu().numpy()
    weights_norm = weights / max(weights.max(), 1e-8)  # scaled for display only

    edges = edge_list(bs)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.imshow(block, interpolation="nearest", extent=(-0.5, bs - 0.5, bs - 0.5, -0.5))
    node_xy = np.array([[c, r] for r in range(bs) for c in range(bs)])
    segs, colors, lws = [], [], []
    for (i, j), w_ in zip(edges, weights_norm):
        p1, p2 = node_xy[i], node_xy[j]
        segs.append([p1, p2])
        colors.append(plt.cm.RdYlGn(min(w_, 1.0)))
        lws.append(0.5 + 3.5 * min(w_, 1.0))
    ax.add_collection(LineCollection(segs, colors=colors, linewidths=lws))
    ax.scatter(node_xy[:, 0], node_xy[:, 1], s=14, color="black", zorder=3)
    ax.set_xlim(-0.7, bs - 0.3); ax.set_ylim(bs - 0.3, -0.7)
    ax.set_title(f"Predicted graph, block ({bi},{bj}){title_suffix}\n"
                  f"raw weight range [{weights.min():.3f}, {weights.max():.3f}]", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def make_concat_file(frame_paths, list_path, fps):
    """Write an ffmpeg concat-demuxer list for the given frames."""
    duration = 1.0 / fps
    with open(list_path, "w") as f:
        for p in frame_paths:
            f.write(f"file '{p.name}'\n")
            f.write(f"duration {duration}\n")

        # The concat demuxer ignores the duration of the last entry unless it is repeated
        if frame_paths:
            f.write(f"file '{frame_paths[-1].name}'\n")


def reassemble_video(frame_dir, out_mp4, fps):
    """Assemble the PNG frames of a directory into an MP4 with ffmpeg (skipped if ffmpeg is missing)."""
    frame_paths = sorted(frame_dir.glob("*.png"))
    if not frame_paths:
        print(f"  (no frames in {frame_dir}, skipping video)")
        return
    list_path = frame_dir / "_concat_list.txt"
    make_concat_file(frame_paths, list_path, fps)
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path),
        "-fps_mode", "vfr", "-pix_fmt", "yuv420p", str(out_mp4),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        print(f"  ffmpeg not found on PATH; skipping {out_mp4} (frames remain in {frame_dir}/)")
        return
    if result.returncode != 0:
        print(f"  ffmpeg failed for {out_mp4}:\n{result.stderr[-2000:]}")
    else:
        print(f"  saved {out_mp4} ({len(frame_paths)} frames @ {fps} fps)")


def process_sequence(seq_name, frames_dir, out_root, gbticl_model, coeff_model,
                      quant_step, symbol_range, device, crop, max_frames, fps, fixed_basis=False):
    """
    Encode, decode and evaluate one sequence as a video.

    Writes metrics.csv, the reconstructed and original (cropped) frames and the
    reassembled videos under out_root/<seq_name>, and returns the list of
    per-frame metric dicts.
    """
    frame_paths = sorted(Path(frames_dir).glob("*.png"))
    if max_frames:
        frame_paths = frame_paths[:max_frames]
    if not frame_paths:
        print(f"[{seq_name}] no frames found in {frames_dir}, skipping")
        return []

    seq_out = out_root / seq_name
    recon_dir = seq_out / "reconstructed"
    orig_dir = seq_out / "original_crop"
    recon_dir.mkdir(parents=True, exist_ok=True)
    orig_dir.mkdir(parents=True, exist_ok=True)

    # Crop and convert to YCbCr
    imgs_rgb = []
    for fp in frame_paths:
        img = np.array(Image.open(fp).convert("RGB"))
        img = center_crop(img, crop) if crop else to_block_multiple(img)
        imgs_rgb.append(img)
    imgs_ycbcr = [rgb_to_ycbcr(img) for img in imgs_rgb]

    estimate_runtime(gbticl_model, coeff_model, quant_step, symbol_range, imgs_ycbcr[0], device,
                      len(frame_paths), fixed_basis=fixed_basis)

    metrics_path = seq_out / "metrics.csv"
    print(f"[{seq_name}] encoding {len(frame_paths)} frames as a video (YCbCr) "
          f"({'crop ' + str(crop) if crop else 'full resolution'}) ...")

    t0 = time.time()
    payloads, metas = encode_video(imgs_ycbcr, block_size=BLOCK_SIZE, quant_step=quant_step,
                                    gbticl_model=gbticl_model, coeff_model=coeff_model,
                                    symbol_range=symbol_range, device=device, fixed_basis=fixed_basis)
    t1 = time.time()
    recon_ycbcr_frames = decode_video(payloads, metas, gbticl_model=gbticl_model, coeff_model=coeff_model,
                                       device=device)
    t2 = time.time()
    print(f"[{seq_name}] encode_video: {t1-t0:.1f}s total, decode_video: {t2-t1:.1f}s total")

    # Per-frame metrics
    rows = []
    for idx, (fp, img_rgb, img_ycbcr, payload, meta, recon_ycbcr) in enumerate(
        zip(frame_paths, imgs_rgb, imgs_ycbcr, payloads, metas, recon_ycbcr_frames)
    ):
        recon_rgb = ycbcr_to_rgb(recon_ycbcr)
        y_psnr = psnr(img_ycbcr[..., 0], recon_ycbcr[..., 0])
        try:
            y_ssim = ssim(img_ycbcr[..., 0], recon_ycbcr[..., 0])
        except ImportError:
            y_ssim = float("nan")  # scikit-image not installed
        rgb_psnr = psnr(img_rgb, recon_rgb)
        bpp = bits_per_pixel(payload, meta["H"], meta["W"])

        Image.fromarray(recon_rgb).save(recon_dir / fp.name)
        Image.fromarray(img_rgb).save(orig_dir / fp.name)

        rows.append(dict(
            frame=fp.name, width=meta["W"], height=meta["H"], payload_bytes=len(payload),
            bpp=bpp, y_psnr_db=y_psnr, y_ssim=y_ssim, rgb_psnr_db=rgb_psnr,
        ))
        print(f"  [{idx+1}/{len(frame_paths)}] {fp.name}: Y-PSNR={y_psnr:.2f}dB Y-SSIM={y_ssim:.4f} "
              f"RGB-PSNR={rgb_psnr:.2f}dB bpp={bpp:.3f}")

    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[{seq_name}] wrote {metrics_path}")

    print(f"[{seq_name}] reassembling video ...")
    reassemble_video(recon_dir, seq_out / "reconstructed_video.mp4", fps)
    reassemble_video(orig_dir, seq_out / "original_video.mp4", fps)

    if gbticl_model is not None:  # the DCT baseline has no predicted graph
        fig_path = out_root / "figures" / f"{seq_name}_predicted_graph.png"
        save_graph_figure(gbticl_model, imgs_rgb[0], fig_path, device, title_suffix=f" -- {seq_name}")

    return rows


def _append_ablation_summary(out_dir, ablation, seq_name, quant_step, rows):
    """Append the mean Y-PSNR / Y-SSIM / bpp of one run to <out_dir>/ablation_summary.csv."""
    if not rows:
        return
    summary_path = Path(out_dir) / "ablation_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    mean_bpp = float(np.mean([r["bpp"] for r in rows]))
    mean_y_psnr = float(np.mean([r["y_psnr_db"] for r in rows]))
    y_ssims = [r["y_ssim"] for r in rows if r["y_ssim"] == r["y_ssim"]]  # drop NaN
    mean_y_ssim = float(np.mean(y_ssims)) if y_ssims else float("nan")
    mean_rgb_psnr = float(np.mean([r["rgb_psnr_db"] for r in rows]))

    row = dict(ablation=ablation, sequence=seq_name, quant_step=quant_step, n_frames=len(rows),
               mean_bpp=mean_bpp, mean_y_psnr_db=mean_y_psnr, mean_y_ssim=mean_y_ssim,
               mean_rgb_psnr_db=mean_rgb_psnr)
    file_exists = summary_path.exists()
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"appended to {summary_path}: {row}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", choices=["Beauty", "HoneyBee", "both"], default="both")
    ap.add_argument("--checkpoint", type=str, default=None,
                     help="Legacy single checkpoint (GBTICLNet + TinyTransformerCoeffModel). "
                          "Omit it and --ablation to use the non-learned baselines.")
    ap.add_argument("--ablation", choices=list(ABLATION_CONFIGS), default=None,
                     help="Named configuration (see ABLATION_CONFIGS); takes precedence over --checkpoint.")
    ap.add_argument("--gbticl-checkpoint", type=str, default=None,
                     help="GBT-ICL checkpoint for --ablation gbticl_no_llm/metalearner_no_llm/full.")
    ap.add_argument("--coeff-checkpoint", type=str, default=None,
                     help="Coefficient-model checkpoint for --ablation full.")
    ap.add_argument("--base-model-name", type=str, default="distilgpt2",
                     help="Base language model, used if the coefficient checkpoint does not record one.")
    ap.add_argument("--out-dir", type=str, default="results")
    ap.add_argument("--crop", type=int, default=256,
                     help="Centre-crop size in pixels (multiple of 8) applied to every frame.")
    ap.add_argument("--full-frame", action="store_true", help="Disable cropping.")
    ap.add_argument("--frames-dir", type=str, default=None,
                     help="Frame directory (default: data/<sequence>/frames). Use it with --full-frame "
                          "to encode downscaled frames from preprocessing/resize_frames.py.")
    ap.add_argument("--max-frames", type=int, default=None, help="Limit frames processed per sequence.")
    ap.add_argument("--fps", type=float, default=5.0, help="Frame rate for the reassembled video.")
    ap.add_argument("--quant-steps", type=str, default=None,
                     help="Comma-separated quantisation steps (e.g. 2,4,8,16,32); one row per step is "
                          "appended to ablation_summary.csv. Overrides --quant-step.")
    ap.add_argument("--quant-step", type=float, default=None,
                     help="Quantisation step (default: the checkpoint's own, else 8).")
    ap.add_argument("--symbol-lo", type=int, default=DEFAULT_SYMBOL_RANGE[0])
    ap.add_argument("--symbol-hi", type=int, default=DEFAULT_SYMBOL_RANGE[1])
    args = ap.parse_args()

    device = get_device()
    print(f"running on device: {device}")

    symbol_range = (args.symbol_lo, args.symbol_hi)
    script_dir = Path(__file__).resolve().parent

    if args.ablation:
        gbticl_model, coeff_model, fixed_basis = load_models_for_ablation(
            args.ablation, args.gbticl_checkpoint, args.coeff_checkpoint, args.base_model_name,
            symbol_range, device,
        )
        label = args.ablation
    else:
        gbticl_model, coeff_model, ckpt_quant_step, ckpt_symbol_range, label = load_models(args.checkpoint, device)
        fixed_basis = False
        if args.quant_step is None and args.quant_steps is None:
            args.quant_step = ckpt_quant_step
        symbol_range = ckpt_symbol_range

    quant_steps = (
        [float(x) for x in args.quant_steps.split(",")] if args.quant_steps
        else [args.quant_step if args.quant_step is not None else DEFAULT_QUANT_STEP]
    )

    crop = None if args.full_frame else args.crop
    out_root = script_dir / args.out_dir / label
    sequences = ["Beauty", "HoneyBee"] if args.sequence == "both" else [args.sequence]
    if args.frames_dir and args.sequence == "both":
        raise SystemExit("--frames-dir only makes sense with a single --sequence (it's a per-sequence dir)")

    for quant_step in quant_steps:
        print(f"\n=== quant_step={quant_step} ===")
        for seq in sequences:
            frames_dir = Path(args.frames_dir) if args.frames_dir else script_dir / "data" / seq / "frames"
            rows = process_sequence(
                seq, frames_dir, out_root,
                gbticl_model, coeff_model, quant_step, symbol_range, device,
                crop, args.max_frames, args.fps, fixed_basis=fixed_basis,
            )
            if args.ablation:
                _append_ablation_summary(script_dir / args.out_dir, args.ablation, seq, quant_step, rows)

    print(f"\nall done. Results under: {out_root}")
    print(f"Next: python visualization/visualize_metrics.py --results-dir {args.out_dir}/{label}")


if __name__ == "__main__":
    main()
