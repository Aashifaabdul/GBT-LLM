"""Per-frame MSE, PSNR and LPIPS of the 1080p reconstructions.

For Beauty and HoneyBee, the first 30 original frames are compared with the
reconstructions gpu_compiled_recon_<sequence>_frameNNNN_q8.png in
results/gpu_batch_compiled_1080p/ (written by
ablations/run_gpu_batch_compiled_codec_1280x720.py). Per frame it reports MSE
and RMSE in RGB, per-channel MSE (R, G, B, and BT.601 Y, Cb, Cr), RGB and Y
PSNR, and LPIPS (AlexNet backbone, inputs scaled to [-1, 1]). A CSV with an
AVERAGE row is written per sequence, plus a combined summary of the averages,
all in results/gpu_batch_compiled_1080p/. Takes no command-line arguments.

Usage:
    python evaluation/compute_1080p_mse_lpips.py
"""

import sys
import types
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import csv


# Vendored minimal AlexNet/VGG16 in place of torchvision.models: lpips only uses
# their `features` stacks, and the layer layout matches torchvision so the
# pretrained weights load. Registered before lpips is imported.
if "torchvision" not in sys.modules or "torchvision.models" not in sys.modules:
    tv = types.ModuleType("torchvision")
    tv_models = types.ModuleType("torchvision.models")

    class AlexNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )

    def alexnet(pretrained=True):
        model = AlexNet()
        if pretrained:
            # Same cache location and file name as torchvision's own downloader.
            cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
            cache_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = cache_dir / "alexnet-owt-7be5be79.pth"
            if not ckpt_path.exists():
                tmp_path = cache_dir / "alexnet-owt-7be5be79.pth.tmp"
                url = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"
                print(f"Downloading {url}...")
                import urllib.request
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req) as resp, open(tmp_path, "wb") as f:
                    total = int(resp.headers.get("Content-Length", 0))
                    downloaded = 0
                    chunk_size = 1024 * 1024
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            print(f"\rDownloaded {downloaded / (1024*1024):.1f}MB / {total / (1024*1024):.1f}MB ({downloaded/total*100:.1f}%)", end="", flush=True)
                        else:
                            print(f"\rDownloaded {downloaded / (1024*1024):.1f}MB", end="", flush=True)
                print("\nDownload complete! Moving temporary file...")
                tmp_path.replace(ckpt_path)
            state_dict = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
        return model

    class VGG16(nn.Module):
        def __init__(self):
            super().__init__()
            layers = [
                nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            ]
            self.features = nn.Sequential(*layers)

    def vgg16(pretrained=True):
        model = VGG16()
        if pretrained:
            state_dict = torch.hub.load_state_dict_from_url(
                "https://download.pytorch.org/models/vgg16-397923af.pth",
                progress=True,
            )
            model.load_state_dict(state_dict, strict=False)
        return model

    tv_models.alexnet = alexnet
    tv_models.vgg16 = vgg16
    tv.models = tv_models
    sys.modules["torchvision"] = tv
    sys.modules["torchvision.models"] = tv_models

import lpips


def rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB [0, 255] to full-range YCbCr (ITU-R BT.601), clipped to [0, 255]."""
    matrix = np.array(
        [[0.299, 0.587, 0.114],
         [-0.168736, -0.331264, 0.5],
         [0.5, -0.418688, -0.081312]],
        dtype=np.float64,
    )
    ycbcr = np.dot(rgb.astype(np.float64), matrix.T)
    ycbcr[..., 1:] += 128.0
    return np.clip(ycbcr, 0.0, 255.0)


def compute_metrics_for_pair(orig_np: np.ndarray, recon_np: np.ndarray, lpips_model, device):
    """Return a dict of MSE, RMSE, PSNR and LPIPS for one uint8 HxWx3 RGB image pair."""
    orig_f = orig_np.astype(np.float64)
    recon_f = recon_np.astype(np.float64)
    diff = orig_f - recon_f

    r_mse = float(np.mean(diff[..., 0] ** 2))
    g_mse = float(np.mean(diff[..., 1] ** 2))
    b_mse = float(np.mean(diff[..., 2] ** 2))
    rgb_mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(rgb_mse))

    orig_ycbcr = rgb_to_ycbcr(orig_np)
    recon_ycbcr = rgb_to_ycbcr(recon_np)
    y_diff = orig_ycbcr[..., 0] - recon_ycbcr[..., 0]
    cb_diff = orig_ycbcr[..., 1] - recon_ycbcr[..., 1]
    cr_diff = orig_ycbcr[..., 2] - recon_ycbcr[..., 2]

    y_mse = float(np.mean(y_diff ** 2))
    cb_mse = float(np.mean(cb_diff ** 2))
    cr_mse = float(np.mean(cr_diff ** 2))

    rgb_psnr = float("inf") if rgb_mse == 0 else float(20 * np.log10(255.0) - 10 * np.log10(rgb_mse))
    y_psnr = float("inf") if y_mse == 0 else float(20 * np.log10(255.0) - 10 * np.log10(y_mse))

    # LPIPS expects NCHW tensors scaled to [-1, 1].
    t_orig = torch.from_numpy(orig_np).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    t_recon = torch.from_numpy(recon_np).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    t_orig = t_orig.to(device)
    t_recon = t_recon.to(device)

    with torch.no_grad():
        lpips_val = float(lpips_model(t_orig, t_recon).squeeze().cpu().item())

    return {
        "rgb_mse": rgb_mse,
        "rmse": rmse,
        "r_mse": r_mse,
        "g_mse": g_mse,
        "b_mse": b_mse,
        "y_mse": y_mse,
        "cb_mse": cb_mse,
        "cr_mse": cr_mse,
        "rgb_psnr": rgb_psnr,
        "y_psnr": y_psnr,
        "lpips": lpips_val,
    }


def evaluate_sequence(seq_name: str, root_dir: Path, lpips_model, device, num_frames=30):
    """Score the first num_frames frames of a sequence against their reconstructions; returns a list of metric dicts."""
    orig_dir = root_dir / f"{seq_name}/frames"
    recon_dir = root_dir / "results/gpu_batch_compiled_1080p"

    all_orig = sorted(list(orig_dir.glob("frame*.png")))[:num_frames]
    results = []

    print(f"\nProcessing {seq_name} ({len(all_orig)} frames)...")
    for f_idx, orig_path in enumerate(all_orig):
        recon_path = recon_dir / f"gpu_compiled_recon_{seq_name}_frame{f_idx:04d}_q8.png"
        if not recon_path.exists():
            print(f"  Warning: {recon_path.name} does not exist. Skipping.")
            continue

        orig_img = np.array(Image.open(orig_path).convert("RGB"), dtype=np.uint8)
        recon_img = np.array(Image.open(recon_path).convert("RGB"), dtype=np.uint8)

        m = compute_metrics_for_pair(orig_img, recon_img, lpips_model, device)
        m["frame_idx"] = f_idx
        m["sequence"] = seq_name
        m["orig_frame"] = orig_path.name
        m["recon_frame"] = recon_path.name
        results.append(m)
        print(f"  Frame {f_idx:02d} ({orig_path.name}): RGB MSE={m['rgb_mse']:.4f}, Y MSE={m['y_mse']:.4f}, RGB PSNR={m['rgb_psnr']:.2f}dB, LPIPS={m['lpips']:.4f}")

    return results


def main():
    """Evaluate both sequences and write the per-sequence and combined CSV files."""
    root_dir = Path(__file__).resolve().parent.parent
    results_dir = root_dir / "results/gpu_batch_compiled_1080p"
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    print("Initializing LPIPS (AlexNet backbone)...")
    lpips_alex = lpips.LPIPS(net="alex").to(device)
    lpips_alex.eval()

    sequences = ["Beauty", "HoneyBee"]
    all_summary = {}

    for seq in sequences:
        metrics = evaluate_sequence(seq, root_dir, lpips_alex, device, num_frames=30)
        csv_file = results_dir / f"mse_lpips_metrics_{seq}_1080p.csv"

        fieldnames = [
            "frame_idx", "sequence", "orig_frame", "recon_frame",
            "rgb_mse", "rmse", "r_mse", "g_mse", "b_mse",
            "y_mse", "cb_mse", "cr_mse", "rgb_psnr", "y_psnr", "lpips"
        ]

        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in metrics:
                writer.writerow(r)

            avg_row = {
                "frame_idx": "AVERAGE",
                "sequence": seq,
                "orig_frame": "ALL",
                "recon_frame": "ALL",
                "rgb_mse": float(np.mean([r["rgb_mse"] for r in metrics])),
                "rmse": float(np.mean([r["rmse"] for r in metrics])),
                "r_mse": float(np.mean([r["r_mse"] for r in metrics])),
                "g_mse": float(np.mean([r["g_mse"] for r in metrics])),
                "b_mse": float(np.mean([r["b_mse"] for r in metrics])),
                "y_mse": float(np.mean([r["y_mse"] for r in metrics])),
                "cb_mse": float(np.mean([r["cb_mse"] for r in metrics])),
                "cr_mse": float(np.mean([r["cr_mse"] for r in metrics])),
                "rgb_psnr": float(np.mean([r["rgb_psnr"] for r in metrics])),
                "y_psnr": float(np.mean([r["y_psnr"] for r in metrics])),
                "lpips": float(np.mean([r["lpips"] for r in metrics])),
            }
            writer.writerow(avg_row)

        all_summary[seq] = {
            "metrics": metrics,
            "avg": avg_row,
            "csv_path": csv_file,
        }
        print(f"\nSaved CSV to: {csv_file}")
        print(f"=== {seq} 1080p Summary (Average across {len(metrics)} frames) ===")
        print(f"  RGB MSE:   {avg_row['rgb_mse']:.4f} (RMSE: {avg_row['rmse']:.4f})")
        print(f"  R MSE:     {avg_row['r_mse']:.4f}")
        print(f"  G MSE:     {avg_row['g_mse']:.4f}")
        print(f"  B MSE:     {avg_row['b_mse']:.4f}")
        print(f"  Y MSE:     {avg_row['y_mse']:.4f} (Luma MSE)")
        print(f"  Cb MSE:    {avg_row['cb_mse']:.4f}")
        print(f"  Cr MSE:    {avg_row['cr_mse']:.4f}")
        print(f"  RGB PSNR:  {avg_row['rgb_psnr']:.2f} dB")
        print(f"  Y PSNR:    {avg_row['y_psnr']:.2f} dB")
        print(f"  LPIPS:     {avg_row['lpips']:.4f}")

    combined_csv = results_dir / "mse_lpips_1080p_summary.csv"
    with open(combined_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for seq in sequences:
            writer.writerow(all_summary[seq]["avg"])
    print(f"\nCombined summary written to: {combined_csv}")


if __name__ == "__main__":
    main()
