"""Export the headline performance charts of the dissertation as individual figures.

Writes to results/charts/:
  chart1_bitrate_comparison.png        bitrate on Beauty (DCT, GBT-ICL without LLM, full model)
  chart2_rate_distortion_curve.png     Y-PSNR against bpp for the same three codecs
  chart3_bit_savings_waterfall.png     transform gain, LLM entropy gain and total saving
  chart4_multisequence_comparison.png  DCT vs full model on Beauty and HoneyBee
The values are hard-coded copies of the 256x256, Q=8.0 evaluation results; no CSV is read.
Chart 5 (LPIPS) is delegated to plot_lpips_chart.plot_lpips(), which reads the 1080p metric CSVs
and is skipped with a printed note if they are missing.

Usage: python visualization/export_individual_charts.py
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def export_individual_charts():
    """Render charts 1-4 to results/charts/ and then attempt the LPIPS chart."""
    out_dir = (ROOT_DIR / "results/charts")
    out_dir.mkdir(parents=True, exist_ok=True)

    models = ["Traditional DCT\n(JPEG Baseline)", "GBT-ICL Meta-Learner\n(No LLM)", "GBT-ICL + DistilGPT-2\n(Full Model)"]
    bpp_vals = [7.021, 3.342, 2.607]
    psnr_vals = [42.79, 42.95, 42.95]
    colors = ["#d9534f", "#f0ad4e", "#5cb85c"]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=220)
    bars = ax.bar(models, bpp_vals, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    ax.set_title("Bitrate Consumption (bpp) on Beauty Sequence\n(Lower Bitrate = Better Compression)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Bits Per Pixel (bpp)", fontsize=11, fontweight="semibold")
    ax.set_ylim(0, 8.5)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for bar, bpp in zip(bars, bpp_vals):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f"{bpp:.2f} bpp", ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.annotate("-52.4% Transform Gain", xy=(1, 3.342), xytext=(0.55, 5.5),
                arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=7),
                fontsize=10.5, fontweight="bold", ha="center", bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd", ec="#ffeeba"))
    ax.annotate("-22.0% LLM Gain\n(-62.9% Total vs DCT)", xy=(2, 2.607), xytext=(1.85, 4.5),
                arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=7),
                fontsize=10.5, fontweight="bold", ha="center", bbox=dict(boxstyle="round,pad=0.3", fc="#d4edda", ec="#c3e6cb"))

    c1_path = out_dir / "chart1_bitrate_comparison.png"
    plt.savefig(c1_path, bbox_inches="tight", dpi=220)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6), dpi=220)
    ax.scatter(bpp_vals, psnr_vals, c=colors, s=180, edgecolor="black", linewidth=1.5, zorder=5)

    labels = [
        "DCT Baseline\n(7.02 bpp, 42.79 dB)",
        "Meta-Learner No LLM\n(3.34 bpp, 42.95 dB)",
        "Full GBT-ICL + DistilGPT-2\n(2.61 bpp, 42.95 dB)"
    ]
    offsets = [(0.15, 0.03), (0.15, -0.06), (-0.15, 0.04)]
    aligns = ["left", "left", "right"]

    for i in range(3):
        ax.annotate(labels[i], (bpp_vals[i], psnr_vals[i]),
                    xytext=(bpp_vals[i] + offsets[i][0], psnr_vals[i] + offsets[i][1]),
                    fontsize=10, fontweight="semibold", ha=aligns[i],
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", alpha=0.85))

    ax.plot(bpp_vals, psnr_vals, linestyle=":", color="gray", alpha=0.7)
    ax.set_title("Rate-Distortion Space: PSNR vs. Bitrate\n(Top-Left Quadrant = Optimal Operating Point)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Bitrate (Bits Per Pixel, bpp)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Luminance Quality (Y-PSNR in dB)", fontsize=11, fontweight="semibold")
    ax.set_xlim(1.5, 8.2)
    ax.set_ylim(42.6, 43.15)
    ax.grid(True, linestyle="--", alpha=0.6)

    c2_path = out_dir / "chart2_rate_distortion_curve.png"
    plt.savefig(c2_path, bbox_inches="tight", dpi=220)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6), dpi=220)
    savings_categories = [
        "1. Transform Gain\n(Fixed DCT -> Adaptive GBT)",
        "2. LLM Entropy Gain\n(Laplace -> DistilGPT-2)",
        "3. Total Codec Savings\n(DCT -> GBT-ICL + LLM)"
    ]
    savings_percentages = [52.4, 22.0, 62.9]
    c3_colors = ["#17a2b8", "#6f42c1", "#28a745"]
    bar_c = ax.bar(savings_categories, savings_percentages, color=c3_colors, width=0.48, edgecolor="black", linewidth=1.2)

    ax.set_title("Incremental Compression Efficiency Gain Breakdown\n(% Bit Savings Over Baselines)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Bitrate Reduction (%)", fontsize=11, fontweight="semibold")
    ax.set_ylim(0, 78)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for bar, val in zip(bar_c, savings_percentages):
        ax.text(bar.get_x() + bar.get_width()/2.0, val + 1.8, f"+{val:.1f}%", ha='center', va='bottom', fontsize=12, fontweight='bold')

    c3_path = out_dir / "chart3_bit_savings_waterfall.png"
    plt.savefig(c3_path, bbox_inches="tight", dpi=220)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6), dpi=220)
    seq_labels = ["Beauty Sequence\n(High Texture / Facial Contours)", "HoneyBee Sequence\n(Dynamic Complex Edges)"]
    x = np.arange(len(seq_labels))
    width = 0.32

    dct_seq = [7.02, 7.15]
    full_seq = [2.61, 3.13]

    rects1 = ax.bar(x - width/2, dct_seq, width, label='Traditional DCT Baseline', color='#d9534f', edgecolor='black', linewidth=1.2)
    rects2 = ax.bar(x + width/2, full_seq, width, label='GBT-ICL + DistilGPT-2 (Ours)', color='#5cb85c', edgecolor='black', linewidth=1.2)

    ax.set_title("Multi-Sequence Performance Generalization\n(256x256 Aligned Crops, Q=8.0)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Bits Per Pixel (bpp)", fontsize=11, fontweight="semibold")
    ax.set_xticks(x)
    ax.set_xticklabels(seq_labels, fontsize=11, fontweight="semibold")
    ax.set_ylim(0, 8.5)
    ax.legend(loc="upper right", frameon=True, fontsize=10.5)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for rect in rects1:
        yval = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.0, yval + 0.18, f"{yval:.2f} bpp", ha='center', va='bottom', fontsize=10.5, fontweight='bold')
    for rect in rects2:
        yval = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.0, yval + 0.18, f"{yval:.2f} bpp", ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='darkgreen')

    c4_path = out_dir / "chart4_multisequence_comparison.png"
    plt.savefig(c4_path, bbox_inches="tight", dpi=220)
    plt.close()

    # chart 5 needs the 1080p LPIPS CSVs, which may not exist
    try:
        from visualization.plot_lpips_chart import plot_lpips
        plot_lpips()
        print(f"All 5 individual charts saved in {out_dir.resolve()}!")
    except Exception as e:
        print(f"All 4 individual charts saved in {out_dir.resolve()}! (Note: LPIPS chart skipped: {e})")

if __name__ == "__main__":
    export_individual_charts()
