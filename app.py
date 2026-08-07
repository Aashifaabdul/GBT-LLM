"""
Simple Gradio demo: upload an image or a short video clip, run the GBT-ICL
+ LLM Coefficient Predictor codec end to end, see original vs reconstructed
side by side plus Y-PSNR/Y-SSIM/bpp/compression-ratio.

Reuses codec.py's encode_image/decode_image and encode_video/decode_video
directly (no duplicated encode/decode logic), colour.py's YCbCr conversion,
and run_dataset_pipeline.py's checkpoint-loading helpers
(load_gbticl_for_ablation/load_coeff_for_ablation) so model auto-detection
(GBTICLNet vs GBTICLMetaLearner, TinyTransformerCoeffModel vs
HFLoRACoeffModel) works identically here and in the CLI ablation pipeline.

RUNTIME, READ BEFORE INCREASING THE CROP-SIZE SLIDER: the entropy coder is
CPU-only, sequential Python -- see run_dataset_pipeline.py's module
docstring for measured numbers on this project's dev machine (RTX 5050
laptop): an UNTRAINED model at a 256x256 crop (1024 blocks) took ~140s to
encode and ~800-1000s to decode ONE frame (decode is slower without a
KV-cache -- HFLoRACoeffModel has one, TinyTransformerCoeffModel doesn't, see
its class docstring). The sliders below are capped well under that so a
submission can't accidentally turn into an hour-long wait; a full 1920x1080
frame is not realistic for this interactive demo at all.

USAGE
  python app.py
"""

import subprocess
import tempfile
import time
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image

from gbticl_pipeline.codec import encode_image, decode_image, encode_video, decode_video
from gbticl_pipeline.colour import rgb_to_ycbcr, ycbcr_to_rgb
from gbticl_pipeline.evaluate import psnr, ssim, bits_per_pixel
from gbticl_pipeline.graph_model import ContextGradientGBTICL
from gbticl_pipeline.coeff_model import LaplaceCoeffModel
from gbticl_pipeline.device_utils import get_device

from run_dataset_pipeline import load_gbticl_for_ablation, load_coeff_for_ablation, reassemble_video

BLOCK_SIZE = 8
WIDE_RANGE = (-2200, 2200)
SCRIPT_DIR = Path(__file__).resolve().parent
DEVICE = get_device()

# Rough, worst-case (untrained model, no KV-cache) seconds/block, measured
# on this project's dev machine -- see the module docstring above. Used
# only to print a "this will take about N minutes" estimate before running.
SEC_PER_BLOCK_ESTIMATE = 1.3

RAW_BPP = 24.0  # uncompressed 8-bit RGB

BASELINE_LABEL = "Baseline (no training, fast heuristic)"
DCT_LABEL = "DCT baseline (fixed basis, no graph)"
CUSTOM_LABEL = "Custom checkpoint(s)"
PRESET_CONFIGS = [BASELINE_LABEL, DCT_LABEL, CUSTOM_LABEL]


def discover_checkpoints():
    ckpt_dir = SCRIPT_DIR / "checkpoints"
    if not ckpt_dir.exists():
        return []
    return sorted(p.name for p in ckpt_dir.glob("*.pt"))


def load_config(config_name, gbticl_ckpt_name, coeff_ckpt_name):
    """Returns (gbticl_model_or_None, coeff_model, fixed_basis)."""
    if config_name == BASELINE_LABEL:
        return ContextGradientGBTICL().to(DEVICE), LaplaceCoeffModel().to(DEVICE), False
    if config_name == DCT_LABEL:
        return None, LaplaceCoeffModel().to(DEVICE), True

    ckpt_dir = SCRIPT_DIR / "checkpoints"
    gbticl_model = None
    if gbticl_ckpt_name and gbticl_ckpt_name != "(none)":
        gbticl_model = load_gbticl_for_ablation(ckpt_dir / gbticl_ckpt_name, DEVICE)
    coeff_model = LaplaceCoeffModel().to(DEVICE)
    if coeff_ckpt_name and coeff_ckpt_name != "(none)":
        coeff_model = load_coeff_for_ablation(ckpt_dir / coeff_ckpt_name, "distilgpt2", WIDE_RANGE, DEVICE)
    return gbticl_model, coeff_model, False


def crop_center(img_rgb, crop_size):
    h, w = img_rgb.shape[:2]
    size = min(int(crop_size), h, w)
    size = max((size // BLOCK_SIZE) * BLOCK_SIZE, BLOCK_SIZE)
    r0 = (h - size) // 2
    c0 = (w - size) // 2
    r0 -= r0 % BLOCK_SIZE
    c0 -= c0 % BLOCK_SIZE
    return img_rgb[r0:r0 + size, c0:c0 + size, :]


def run_image(image, config_name, gbticl_ckpt_name, coeff_ckpt_name, crop_size, quant_step,
              progress=gr.Progress()):
    if image is None:
        return None, None, "Upload an image first."

    img_rgb = np.array(Image.fromarray(image).convert("RGB"))
    img_rgb = crop_center(img_rgb, crop_size)
    n_blocks = (img_rgb.shape[0] // BLOCK_SIZE) * (img_rgb.shape[1] // BLOCK_SIZE)
    est_s = n_blocks * SEC_PER_BLOCK_ESTIMATE * 2  # x2: encode + decode

    try:
        gbticl_model, coeff_model, fixed_basis = load_config(config_name, gbticl_ckpt_name, coeff_ckpt_name)
    except Exception as e:
        return None, None, f"Failed to load model config: {e}"

    img_ycbcr = rgb_to_ycbcr(img_rgb)
    progress(0.0, desc=f"Encoding ({n_blocks} blocks, est ~{est_s:.0f}s total)...")
    t0 = time.time()
    payload, meta = encode_image(
        img_ycbcr, block_size=BLOCK_SIZE, quant_step=float(quant_step),
        gbticl_model=gbticl_model, coeff_model=coeff_model,
        symbol_range=WIDE_RANGE, device=DEVICE, fixed_basis=fixed_basis,
    )
    t1 = time.time()
    progress(0.5, desc="Decoding...")
    recon_ycbcr = decode_image(payload, meta, gbticl_model=gbticl_model, coeff_model=coeff_model, device=DEVICE)
    t2 = time.time()
    recon_rgb = ycbcr_to_rgb(recon_ycbcr)

    y_psnr = psnr(img_ycbcr[..., 0], recon_ycbcr[..., 0])
    try:
        y_ssim = ssim(img_ycbcr[..., 0], recon_ycbcr[..., 0])
    except ImportError:
        y_ssim = float("nan")
    bpp = bits_per_pixel(payload, meta["H"], meta["W"])
    ratio = RAW_BPP / bpp if bpp > 0 else float("inf")

    report = (
        f"**Y-PSNR:** {y_psnr:.2f} dB &nbsp;&nbsp; **Y-SSIM:** {y_ssim:.4f}\n\n"
        f"**bpp:** {bpp:.3f} (raw: {RAW_BPP:.0f}) &nbsp;&nbsp; **Compression ratio:** {ratio:.2f}x\n\n"
        f"**Payload:** {len(payload):,} bytes &nbsp;&nbsp; "
        f"**Encode:** {t1-t0:.1f}s &nbsp;&nbsp; **Decode:** {t2-t1:.1f}s"
    )
    return img_rgb, recon_rgb, report


def extract_video_frames(video_path, max_frames, out_fps=2):
    """Extracts up to max_frames frames from an uploaded clip via the
    system ffmpeg (the same tool run_dataset_pipeline.py's reassemble_video
    uses in the other direction, already confirmed present on PATH) --
    avoids adding a new video-decoding dependency."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="gbticl_app_in_"))
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path), "-r", str(out_fps),
        "-frames:v", str(int(max_frames)), str(tmp_dir / "frame%04d.png"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed:\n{result.stderr[-1500:]}")
    frame_paths = sorted(tmp_dir.glob("*.png"))
    frames = [np.array(Image.open(p).convert("RGB")) for p in frame_paths]
    return frames


def run_video(video_path, config_name, gbticl_ckpt_name, coeff_ckpt_name, crop_size, quant_step,
              max_frames, progress=gr.Progress()):
    if video_path is None:
        return None, "Upload a video clip first."

    try:
        frames = extract_video_frames(video_path, max_frames)
    except Exception as e:
        return None, f"Could not read that video: {e}"
    if not frames:
        return None, "No frames could be extracted from that clip."

    frames = [crop_center(f, crop_size) for f in frames]
    n_blocks = (frames[0].shape[0] // BLOCK_SIZE) * (frames[0].shape[1] // BLOCK_SIZE)
    est_s = n_blocks * len(frames) * SEC_PER_BLOCK_ESTIMATE * 2

    try:
        gbticl_model, coeff_model, fixed_basis = load_config(config_name, gbticl_ckpt_name, coeff_ckpt_name)
    except Exception as e:
        return None, f"Failed to load model config: {e}"

    frames_ycbcr = [rgb_to_ycbcr(f) for f in frames]

    progress(0.0, desc=f"Encoding {len(frames)} frames ({n_blocks} blocks/frame, est ~{est_s:.0f}s total)...")
    t0 = time.time()
    payloads, metas = encode_video(
        frames_ycbcr, block_size=BLOCK_SIZE, quant_step=float(quant_step),
        gbticl_model=gbticl_model, coeff_model=coeff_model,
        symbol_range=WIDE_RANGE, device=DEVICE, fixed_basis=fixed_basis,
    )
    t1 = time.time()
    progress(0.5, desc="Decoding...")
    recon_ycbcr_frames = decode_video(payloads, metas, gbticl_model=gbticl_model, coeff_model=coeff_model,
                                       device=DEVICE)
    t2 = time.time()

    recon_rgb_frames = [ycbcr_to_rgb(f) for f in recon_ycbcr_frames]

    y_psnrs, bpps = [], []
    for img_yc, recon_yc, payload, meta in zip(frames_ycbcr, recon_ycbcr_frames, payloads, metas):
        y_psnrs.append(psnr(img_yc[..., 0], recon_yc[..., 0]))
        bpps.append(bits_per_pixel(payload, meta["H"], meta["W"]))

    out_dir = Path(tempfile.mkdtemp(prefix="gbticl_app_out_"))
    for i, f in enumerate(recon_rgb_frames):
        Image.fromarray(f).save(out_dir / f"frame{i:04d}.png")
    out_mp4 = out_dir / "reconstructed.mp4"
    reassemble_video(out_dir, out_mp4, fps=2.0)

    mean_bpp = float(np.mean(bpps))
    mean_y_psnr = float(np.mean(y_psnrs))
    ratio = RAW_BPP / mean_bpp if mean_bpp > 0 else float("inf")
    report = (
        f"**{len(frames)} frames**, {n_blocks} blocks/frame\n\n"
        f"**mean Y-PSNR:** {mean_y_psnr:.2f} dB &nbsp;&nbsp; **mean bpp:** {mean_bpp:.3f} "
        f"&nbsp;&nbsp; **Compression ratio:** {ratio:.2f}x\n\n"
        f"**Encode:** {t1-t0:.1f}s &nbsp;&nbsp; **Decode:** {t2-t1:.1f}s"
    )
    return (str(out_mp4) if out_mp4.exists() else None), report


def build_app():
    checkpoints = discover_checkpoints()
    ckpt_choices = ["(none)"] + checkpoints

    with gr.Blocks(title="GBT-ICL + LLM Codec Demo") as demo:
        gr.Markdown(
            "# GBT-ICL + LLM Coefficient Predictor -- compression demo\n"
            "Two frozen in-context predictors: **GBT-ICL** predicts the graph Laplacian from "
            "decoded pixel context (no graph signalling); its eigenvalue spectrum conditions the "
            "**LLM Coefficient Predictor**'s entropy coding (no probability-model signalling). "
            "See `architecture/final/gbticl_llm_unified_architecture.svg` for the full diagram.\n\n"
            "**Runtime note:** the entropy coder is CPU-only and strictly sequential -- runtime "
            "scales with block count (crop_size/8)² x frames. The sliders below are capped so a "
            "run finishes in a few minutes; a full 1920x1080 frame/clip is not realistic here "
            "(see run_dataset_pipeline.py for batch/background runs instead)."
        )

        with gr.Tabs():
            with gr.Tab("Image"):
                with gr.Row():
                    with gr.Column():
                        img_in = gr.Image(label="Input image", type="numpy")
                        config_dd = gr.Dropdown(PRESET_CONFIGS, value=BASELINE_LABEL, label="Model config")
                        gbticl_ckpt_dd = gr.Dropdown(ckpt_choices, value="(none)",
                                                      label="GBT-ICL checkpoint (Custom config only)")
                        coeff_ckpt_dd = gr.Dropdown(ckpt_choices, value="(none)",
                                                     label="Coefficient-model checkpoint (Custom config only)")
                        crop_slider = gr.Slider(8, 128, value=64, step=8,
                                                 label="Crop size (px, square, centered)")
                        quant_slider = gr.Slider(1, 32, value=8, step=1,
                                                  label="Quantization step (higher = smaller file, lower quality)")
                        run_btn = gr.Button("Encode + Decode", variant="primary")
                    with gr.Column():
                        orig_out = gr.Image(label="Original (cropped)")
                        recon_out = gr.Image(label="Reconstructed")
                        report_out = gr.Markdown()
                run_btn.click(
                    run_image,
                    [img_in, config_dd, gbticl_ckpt_dd, coeff_ckpt_dd, crop_slider, quant_slider],
                    [orig_out, recon_out, report_out],
                )

            with gr.Tab("Video (short clip)"):
                with gr.Row():
                    with gr.Column():
                        vid_in = gr.Video(label="Input clip (short -- a few seconds)")
                        config_dd2 = gr.Dropdown(PRESET_CONFIGS, value=BASELINE_LABEL, label="Model config")
                        gbticl_ckpt_dd2 = gr.Dropdown(ckpt_choices, value="(none)",
                                                       label="GBT-ICL checkpoint (Custom config only)")
                        coeff_ckpt_dd2 = gr.Dropdown(ckpt_choices, value="(none)",
                                                      label="Coefficient-model checkpoint (Custom config only)")
                        crop_slider2 = gr.Slider(8, 64, value=32, step=8,
                                                  label="Crop size (px, square, centered)")
                        quant_slider2 = gr.Slider(1, 32, value=8, step=1, label="Quantization step")
                        frames_slider = gr.Slider(2, 8, value=4, step=1, label="Max frames (keep small!)")
                        run_btn2 = gr.Button("Encode + Decode video", variant="primary")
                    with gr.Column():
                        vid_out = gr.Video(label="Reconstructed")
                        report_out2 = gr.Markdown()
                run_btn2.click(
                    run_video,
                    [vid_in, config_dd2, gbticl_ckpt_dd2, coeff_ckpt_dd2, crop_slider2, quant_slider2, frames_slider],
                    [vid_out, report_out2],
                )

        gr.Markdown(f"Device: `{DEVICE}` &nbsp;&nbsp; Checkpoints found: {len(checkpoints)}")

    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue().launch()
