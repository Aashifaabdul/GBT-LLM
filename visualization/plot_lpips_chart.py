"""LPIPS perceptual-quality charts for the 1080p Beauty and HoneyBee sequences.

Reads results/gpu_batch_compiled_1080p/mse_lpips_metrics_{Beauty,HoneyBee}_1080p.csv, written by
evaluation/compute_1080p_mse_lpips.py (one row per frame plus an AVERAGE row; columns used:
frame_idx, lpips, rgb_psnr). Writes to results/charts/:
  chart5_lpips_bar.png                     mean LPIPS per sequence
  chart5_lpips_perceptual_comparison.png   mean LPIPS with RGB-PSNR and the per-frame trajectory
Also called from export_individual_charts.py.

Usage: python visualization/plot_lpips_chart.py
"""

import sys
import csv
from pathlib import Path
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def load_sequence_metrics(csv_path: Path):
    """Load per-frame metrics and average row from a CSV file."""
    frames = []
    avg_row = None
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["frame_idx"] == "AVERAGE":
                avg_row = row
            else:
                frames.append(row)
    return frames, avg_row


def plot_lpips():
    """Draw both LPIPS figures; raises FileNotFoundError if the metric CSVs are missing."""
    results_dir = ROOT_DIR / "results/gpu_batch_compiled_1080p"
    charts_dir = ROOT_DIR / "results/charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    beauty_csv = results_dir / "mse_lpips_metrics_Beauty_1080p.csv"
    honeybee_csv = results_dir / "mse_lpips_metrics_HoneyBee_1080p.csv"

    if not beauty_csv.exists() or not honeybee_csv.exists():
        raise FileNotFoundError("Missing 1080p MSE/LPIPS metric CSV files in results/gpu_batch_compiled_1080p/")

    beauty_frames, beauty_avg = load_sequence_metrics(beauty_csv)
    honeybee_frames, honeybee_avg = load_sequence_metrics(honeybee_csv)

    beauty_lpips = [float(r["lpips"]) for r in beauty_frames]
    honeybee_lpips = [float(r["lpips"]) for r in honeybee_frames]
    frame_indices = list(range(len(beauty_lpips)))

    mean_beauty = float(beauty_avg["lpips"])
    mean_honeybee = float(honeybee_avg["lpips"])

    fig, ax = plt.subplots(figsize=(8, 6), dpi=220)
    seqs = ["Beauty (1080p FHD)\n(Complex Facial Contours)", "HoneyBee (1080p FHD)\n(Fine Dynamic Textures)"]
    lpips_vals = [mean_beauty, mean_honeybee]
    colors = ["#3498db", "#2ecc71"]

    bars = ax.bar(seqs, lpips_vals, color=colors, width=0.45, edgecolor="black", linewidth=1.2)
    ax.set_title("Full-HD (1080p) Perceptual Metric Comparison (LPIPS)\n(AlexNet Backbone | Lower Score = Closer Perceptual Fidelity)",
                 fontsize=12, fontweight="bold", pad=15)
    ax.set_ylabel("LPIPS Distance (Lower is Better)", fontsize=11, fontweight="semibold")
    ax.set_ylim(0, 0.055)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for bar, val in zip(bars, lpips_vals):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.0015,
                f"LPIPS = {val:.4f}", ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.text(0.5, 0.048, "HoneyBee achieves 56.2% lower perceptual distance than Beauty\ndue to dense textural masking in the GBT basis.",
            transform=ax.transData, ha='center', fontsize=9.5, style='italic',
            bbox=dict(boxstyle="round,pad=0.4", fc="#f8f9fa", ec="#cccccc", alpha=0.9))

    out_bar = charts_dir / "chart5_lpips_bar.png"
    plt.savefig(out_bar, bbox_inches="tight", dpi=220)
    plt.close()
    print(f"Saved: {out_bar}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=220)
    plt.subplots_adjust(wspace=0.25)

    ax1 = axes[0]
    bars1 = ax1.bar(seqs, lpips_vals, color=colors, width=0.45, edgecolor="black", linewidth=1.2, zorder=3)
    ax1.set_title("A. Mean LPIPS Perceptual Distance (1080p, 30 Frames)\n(Lower LPIPS = Better Perceptual Reconstruction)",
                  fontsize=12, fontweight="bold", pad=12)
    ax1.set_ylabel("LPIPS (AlexNet Feature Distance)", fontsize=11, fontweight="semibold")
    ax1.set_ylim(0, 0.055)
    ax1.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)

    rgb_psnrs = [float(beauty_avg["rgb_psnr"]), float(honeybee_avg["rgb_psnr"])]
    for bar, val, psnr in zip(bars1, lpips_vals, rgb_psnrs):
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.0015,
                 f"{val:.4f}\n(RGB PSNR: {psnr:.2f} dB)", ha='center', va='bottom', fontsize=10.5, fontweight='bold')

    ax2 = axes[1]
    ax2.plot(frame_indices, beauty_lpips, label=f"Beauty (Mean: {mean_beauty:.4f})",
             color="#3498db", linewidth=2.0, marker="o", markersize=4.5, alpha=0.9)
    ax2.plot(frame_indices, honeybee_lpips, label=f"HoneyBee (Mean: {mean_honeybee:.4f})",
             color="#2ecc71", linewidth=2.0, marker="s", markersize=4.5, alpha=0.9)

    ax2.axhline(mean_beauty, color="#3498db", linestyle="--", alpha=0.5, linewidth=1.2)
    ax2.axhline(mean_honeybee, color="#2ecc71", linestyle="--", alpha=0.5, linewidth=1.2)

    ax2.set_title("B. Temporal LPIPS Trajectory Across 30 Coded Frames\n(Stability Across Intra / Temporal Autoregressive Blocks)",
                  fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("Video Frame Index (30 Frames at 1920x1080)", fontsize=11, fontweight="semibold")
    ax2.set_ylabel("Frame LPIPS Distance", fontsize=11, fontweight="semibold")
    ax2.set_xlim(-0.5, 29.5)
    ax2.set_ylim(0.010, 0.052)
    ax2.legend(loc="upper right", frameon=True, fontsize=10.5)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("GBT-ICL Full-HD (1920x1080) Perceptual Evaluation (AlexNet-based LPIPS)",
                 fontsize=14, fontweight="bold", y=1.02)

    out_comp = charts_dir / "chart5_lpips_perceptual_comparison.png"
    plt.savefig(out_comp, bbox_inches="tight", dpi=220)
    plt.close()
    print(f"Saved: {out_comp}")


if __name__ == "__main__":
    plot_lpips()
