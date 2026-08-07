"""
The full dissertation pipeline over real extracted frames: encode every
frame with the real codec (GBT-ICL graph -> GFT -> quantize -> real range
coder, using either your trained checkpoint from training.py or the
non-learned baselines), decode it straight back, save each reconstructed
frame image one by one, then reassemble the reconstructed frames into a
video, log per-frame metrics, and save a predicted-graph figure using
whichever model actually ran.

REQUIRES PyTorch. Not executed in this environment (no torch here, see the
project README) -- syntax-checked only. Run on your GPU machine.

RUNTIME, READ BEFORE RUNNING ON FULL FRAMES: the range coder (real
arithmetic coding, see range_coder.py) is an inherently sequential,
per-symbol Python loop -- there is no way to GPU-parallelise it, by design
(same reason every learned image/video codec does entropy coding on CPU).
A full 1920x1080 frame is 32,400 8x8 blocks x 192 coefficients/block =
~6.2 million range-coder symbol calls, PER FRAME. That is not fast in pure
Python. By default this script center-crops every frame to --crop pixels
(256 by default) so a run actually finishes in a reasonable time on a
laptop/GPU-box CPU. Pass --full-frame to disable cropping once you've
verified the pipeline works and have time to let it run (consider doing
that as an overnight/background job, and on a subset of frames via
--max-frames first).

USAGE
  # baseline (no training needed), quick sanity run:
  python run_dataset_pipeline.py --sequence Beauty --max-frames 3

  # your trained model:
  python run_dataset_pipeline.py --sequence Beauty --checkpoint checkpoints/gbticl_ckpt.pt

  # both sequences, full resolution, all frames (slow -- see runtime note):
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
    raise SystemExit(
        "run_dataset_pipeline.py needs PyTorch (+ matplotlib, already used "
        f"elsewhere in this project). Not available in this environment: {e}"
    )

from PIL import Image

from gbticl_pipeline.device_utils import get_device
from gbticl_pipeline.codec import encode_image, decode_image
from gbticl_pipeline.evaluate import psnr, bits_per_pixel
from gbticl_pipeline.graph_model import GBTICLNet, ContextGradientGBTICL, edge_list
from gbticl_pipeline.coeff_model import TinyTransformerCoeffModel, LaplaceCoeffModel

BLOCK_SIZE = 8
DEFAULT_SYMBOL_RANGE = (-2200, 2200)
DEFAULT_QUANT_STEP = 8.0


def center_crop(img, size):
    """Crop img (H,W,3) to a centered size x size region, snapped to a
    multiple of BLOCK_SIZE (the codec requires this)."""
    h, w = img.shape[:2]
    size = (size // BLOCK_SIZE) * BLOCK_SIZE
    size = min(size, (h // BLOCK_SIZE) * BLOCK_SIZE, (w // BLOCK_SIZE) * BLOCK_SIZE)
    r0 = (h - size) // 2
    c0 = (w - size) // 2
    r0 -= r0 % BLOCK_SIZE
    c0 -= c0 % BLOCK_SIZE
    return img[r0:r0 + size, c0:c0 + size, :]


def to_block_multiple(img):
    """If not cropping, still need H,W to be multiples of BLOCK_SIZE (codec
    asserts this) -- trim at most BLOCK_SIZE-1 px off the bottom/right."""
    h, w = img.shape[:2]
    h2 = (h // BLOCK_SIZE) * BLOCK_SIZE
    w2 = (w // BLOCK_SIZE) * BLOCK_SIZE
    return img[:h2, :w2, :]


def load_models(checkpoint_path, device):
    """Returns (gbticl_model, coeff_model, quant_step, symbol_range, label)."""
    if checkpoint_path is None:
        print("No --checkpoint given: using the non-learned baselines "
              "(ContextGradientGBTICL + LaplaceCoeffModel). Results reflect the "
              "hand-written heuristics, not a trained model -- pass --checkpoint "
              "once training.py has produced one.")
        gbticl_model = ContextGradientGBTICL().to(device)
        coeff_model = LaplaceCoeffModel().to(device)
        return gbticl_model, coeff_model, DEFAULT_QUANT_STEP, DEFAULT_SYMBOL_RANGE, "baseline"

    ckpt = torch.load(checkpoint_path, map_location=device)
    block_size = ckpt.get("block_size", BLOCK_SIZE)
    symbol_range = tuple(ckpt.get("symbol_range", DEFAULT_SYMBOL_RANGE))
    quant_step = ckpt.get("quant_step", DEFAULT_QUANT_STEP)

    gbticl_model = GBTICLNet(block_size=block_size).to(device)
    gbticl_model.load_state_dict(ckpt["gbticl_net"])
    gbticl_model.eval()

    coeff_model = TinyTransformerCoeffModel(block_size=block_size, symbol_range=symbol_range).to(device)
    coeff_model.load_state_dict(ckpt["coeff_net"])
    coeff_model.eval()

    print(f"loaded trained checkpoint {checkpoint_path} "
          f"(epoch {ckpt.get('epoch', '?')}, quant_step={quant_step}, symbol_range={symbol_range})")
    return gbticl_model, coeff_model, quant_step, symbol_range, "trained"


def estimate_runtime(gbticl_model, coeff_model, quant_step, symbol_range, sample_img, device, n_frames):
    """Time-encode a single sample frame's worth of work isn't cheap either,
    so just time ONE block's full encode+decode via a 1-block crop, then
    extrapolate. Rough estimate only -- printed so you can Ctrl-C before
    committing to an accidentally huge run."""
    tiny = sample_img[:BLOCK_SIZE, :BLOCK_SIZE, :]
    t0 = time.time()
    payload, meta = encode_image(tiny, block_size=BLOCK_SIZE, quant_step=quant_step,
                                  gbticl_model=gbticl_model, coeff_model=coeff_model,
                                  symbol_range=symbol_range, device=device)
    decode_image(payload, meta, gbticl_model=gbticl_model, coeff_model=coeff_model, device=device)
    per_block = time.time() - t0

    h, w = sample_img.shape[:2]
    n_blocks = (h // BLOCK_SIZE) * (w // BLOCK_SIZE)
    est_seconds = per_block * n_blocks * n_frames
    print(f"runtime estimate: ~{per_block*1000:.0f} ms/block x {n_blocks} blocks/frame x "
          f"{n_frames} frames ~= {est_seconds/60:.1f} min total (rough; real runs vary)")
    if est_seconds > 1800:
        print("  -> that's over 30 min. Consider a smaller --crop, fewer --max-frames, "
              "or running this as a background job.")


def save_graph_figure(gbticl_model, img, out_path, device, title_suffix=""):
    """Predicted-graph sanity figure for one representative (high-variance)
    block of `img`, using whatever model actually produced the results in
    this run -- same visual language as visualize_graph.py, but driven by
    the real model instead of the numpy heuristic reimplementation."""
    bs = BLOCK_SIZE
    h, w = img.shape[:2]
    n_bh, n_bw = h // bs, w // bs
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
    weights_norm = weights / max(weights.max(), 1e-8)  # visual scale only

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
    duration = 1.0 / fps
    with open(list_path, "w") as f:
        for p in frame_paths:
            f.write(f"file '{p.name}'\n")
            f.write(f"duration {duration}\n")
        # ffmpeg's concat demuxer needs the last file repeated (its duration is
        # otherwise ignored) -- documented ffmpeg quirk, not a bug here.
        if frame_paths:
            f.write(f"file '{frame_paths[-1].name}'\n")


def reassemble_video(frame_dir, out_mp4, fps):
    frame_paths = sorted(frame_dir.glob("*.png"))
    if not frame_paths:
        print(f"  (no frames in {frame_dir}, skipping video)")
        return
    list_path = frame_dir / "_concat_list.txt"
    make_concat_file(frame_paths, list_path, fps)
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path),
        "-vsync", "vfr", "-pix_fmt", "yuv420p", str(out_mp4),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        print(f"  ffmpeg not found on PATH -- skipping video assembly for {out_mp4}. "
              f"Frames are still saved in {frame_dir}/. Install ffmpeg (Windows: "
              f"`winget install ffmpeg` or download from ffmpeg.org and add its bin/ "
              f"folder to PATH, then re-run) and this step will work without "
              f"re-encoding anything.")
        return
    if result.returncode != 0:
        print(f"  ffmpeg failed for {out_mp4}:\n{result.stderr[-2000:]}")
    else:
        print(f"  saved {out_mp4} ({len(frame_paths)} frames @ {fps} fps)")


def process_sequence(seq_name, frames_dir, out_root, gbticl_model, coeff_model,
                      quant_step, symbol_range, device, crop, max_frames, fps):
    frame_paths = sorted(Path(frames_dir).glob("*.png"))
    if max_frames:
        frame_paths = frame_paths[:max_frames]
    if not frame_paths:
        print(f"[{seq_name}] no frames found in {frames_dir}, skipping")
        return

    seq_out = out_root / seq_name
    recon_dir = seq_out / "reconstructed"
    orig_dir = seq_out / "original_crop"
    recon_dir.mkdir(parents=True, exist_ok=True)
    orig_dir.mkdir(parents=True, exist_ok=True)

    first_img = np.array(Image.open(frame_paths[0]).convert("RGB"))
    first_img = center_crop(first_img, crop) if crop else to_block_multiple(first_img)
    estimate_runtime(gbticl_model, coeff_model, quant_step, symbol_range, first_img, device, len(frame_paths))

    metrics_path = seq_out / "metrics.csv"
    rows = []
    print(f"[{seq_name}] encoding {len(frame_paths)} frames "
          f"({'crop ' + str(crop) if crop else 'full resolution'}) ...")

    for idx, fp in enumerate(frame_paths):
        img = np.array(Image.open(fp).convert("RGB"))
        img = center_crop(img, crop) if crop else to_block_multiple(img)

        t0 = time.time()
        payload, meta = encode_image(img, block_size=BLOCK_SIZE, quant_step=quant_step,
                                      gbticl_model=gbticl_model, coeff_model=coeff_model,
                                      symbol_range=symbol_range, device=device)
        t1 = time.time()
        recon = decode_image(payload, meta, gbticl_model=gbticl_model, coeff_model=coeff_model, device=device)
        t2 = time.time()

        p = psnr(img, recon)
        bpp = bits_per_pixel(payload, meta["H"], meta["W"])

        Image.fromarray(recon).save(recon_dir / fp.name)
        Image.fromarray(img).save(orig_dir / fp.name)

        rows.append(dict(
            frame=fp.name, width=meta["W"], height=meta["H"],
            payload_bytes=len(payload), bpp=bpp, psnr_db=p,
            encode_s=t1 - t0, decode_s=t2 - t1,
        ))
        print(f"  [{idx+1}/{len(frame_paths)}] {fp.name}: PSNR={p:.2f}dB bpp={bpp:.3f} "
              f"enc={t1-t0:.1f}s dec={t2-t1:.1f}s")

    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[{seq_name}] wrote {metrics_path}")

    print(f"[{seq_name}] reassembling video ...")
    reassemble_video(recon_dir, seq_out / "reconstructed_video.mp4", fps)
    reassemble_video(orig_dir, seq_out / "original_video.mp4", fps)

    fig_path = out_root / "figures" / f"{seq_name}_predicted_graph.png"
    save_graph_figure(gbticl_model, first_img, fig_path, device, title_suffix=f" -- {seq_name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", choices=["Beauty", "HoneyBee", "both"], default="both")
    ap.add_argument("--checkpoint", type=str, default=None,
                     help="Path to a training.py checkpoint. Omit to use the non-learned baselines.")
    ap.add_argument("--out-dir", type=str, default="results")
    ap.add_argument("--crop", type=int, default=256,
                     help="Center-crop size (px, multiple of 8) applied to every frame. 0/omit with "
                          "--full-frame to disable.")
    ap.add_argument("--full-frame", action="store_true", help="Disable cropping -- see runtime note above.")
    ap.add_argument("--max-frames", type=int, default=None, help="Limit frames processed per sequence.")
    ap.add_argument("--fps", type=float, default=5.0, help="Frame rate for the reassembled video.")
    args = ap.parse_args()

    device = get_device()
    print(f"running on device: {device}")

    gbticl_model, coeff_model, quant_step, symbol_range, label = load_models(args.checkpoint, device)
    crop = None if args.full_frame else args.crop

    script_dir = Path(__file__).resolve().parent
    out_root = script_dir / args.out_dir / label

    sequences = ["Beauty", "HoneyBee"] if args.sequence == "both" else [args.sequence]
    for seq in sequences:
        process_sequence(
            seq, script_dir / seq / "frames", out_root,
            gbticl_model, coeff_model, quant_step, symbol_range, device,
            crop, args.max_frames, args.fps,
        )

    print(f"\nall done. Results under: {out_root}")
    print("Next: python visualize_metrics.py --results-dir "
          f"{args.out_dir}/{label}")


if __name__ == "__main__":
    main()
