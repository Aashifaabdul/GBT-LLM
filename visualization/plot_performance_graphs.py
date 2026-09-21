"""Four-panel overview: traditional DCT vs GBT-ICL (no LLM) vs GBT-ICL + DistilGPT-2.

A: bitrate on Beauty, B: Y-PSNR against bpp, C: bit savings of the transform, of the LLM entropy
model and in total, D: bitrate on Beauty and HoneyBee. Values are hard-coded from the 256x256,
Q=8.0 evaluation; no CSV is read. Writes results/graphical_performance_comparison.png.

Usage: python visualization/plot_performance_graphs.py
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import shutil

def plot_performance_metrics():
    """Render the four-panel comparison figure to results/."""
    out_dir = (ROOT_DIR / "results")
    out_dir.mkdir(parents=True, exist_ok=True)

    src_visual = (ROOT_DIR / "results/visual_comparison.png")
    dst_visual = out_dir / "visual_comparison.png"
    if src_visual.exists() and src_visual.resolve() != dst_visual.resolve():
        shutil.copy(src_visual, dst_visual)
        print(f"Saved visual comparison to {dst_visual.resolve()}")

    models = ["Traditional DCT\n(JPEG Baseline)", "GBT-ICL Meta-Learner\n(No LLM)", "GBT-ICL + DistilGPT-2\n(Full Model)"]
    bpp_vals = [7.021, 3.342, 2.607]
    psnr_vals = [42.79, 42.95, 42.95]
    colors = ["#d9534f", "#f0ad4e", "#5cb85c"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=200)
    plt.subplots_adjust(wspace=0.25, hspace=0.32)

    bars = axes[0, 0].bar(models, bpp_vals, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    axes[0, 0].set_title("A. Bitrate Consumption (bpp) on Beauty\n(Lower is Better)", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("Bits Per Pixel (bpp)", fontsize=11, fontweight="semibold")
    axes[0, 0].set_ylim(0, 8.5)
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.6)

    for bar, bpp in zip(bars, bpp_vals):
        yval = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f"{bpp:.2f} bpp", ha='center', va='bottom', fontsize=11, fontweight='bold')

    axes[0, 0].annotate("-52.4% Transform Gain", xy=(1, 3.342), xytext=(0.5, 5.5),
                        arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=7),
                        fontsize=10, fontweight="bold", ha="center", bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd", ec="#ffeeba"))
    axes[0, 0].annotate("-22.0% LLM Gain\n(-62.9% Total vs DCT)", xy=(2, 2.607), xytext=(1.85, 4.5),
                        arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=7),
                        fontsize=10, fontweight="bold", ha="center", bbox=dict(boxstyle="round,pad=0.3", fc="#d4edda", ec="#c3e6cb"))

    axes[0, 1].scatter(bpp_vals, psnr_vals, c=colors, s=160, edgecolor="black", linewidth=1.5, zorder=5)
    for i, txt in enumerate(["DCT (7.02 bpp, 42.79 dB)", "Meta-Learner No LLM (3.34 bpp, 42.95 dB)", "Full GBT-ICL + LLM (2.61 bpp, 42.95 dB)"]):
        offset_x = 0.15 if i != 2 else -0.15
        ha_align = "left" if i != 2 else "right"
        axes[0, 1].annotate(txt, (bpp_vals[i], psnr_vals[i]), xytext=(bpp_vals[i] + offset_x, psnr_vals[i] + 0.05),
                            fontsize=9.5, fontweight="semibold", ha=ha_align)

    axes[0, 1].plot(bpp_vals, psnr_vals, linestyle=":", color="gray", alpha=0.7)
    axes[0, 1].set_title("B. Rate-Distortion Space (PSNR vs. bpp)\n(Top-Left is Ideal: Higher PSNR at Lower bpp)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Bitrate (bpp)", fontsize=11, fontweight="semibold")
    axes[0, 1].set_ylabel("Luminance Y-PSNR (dB)", fontsize=11, fontweight="semibold")
    axes[0, 1].set_xlim(1.5, 8.0)
    axes[0, 1].set_ylim(42.5, 43.3)
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)

    savings_categories = ["Transform Gain\n(DCT -> GBT Graph)", "LLM Entropy Gain\n(Laplace -> DistilGPT-2)", "Total Codec Savings\n(DCT -> GBT-ICL+LLM)"]
    savings_percentages = [52.4, 22.0, 62.9]
    bar_c = axes[1, 0].bar(savings_categories, savings_percentages, color=["#17a2b8", "#6f42c1", "#28a745"], width=0.5, edgecolor="black", linewidth=1.2)
    axes[1, 0].set_title("C. Incremental Compression Efficiency Gain\n(% Bit Savings over Baselines)", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Bitrate Reduction (%)", fontsize=11, fontweight="semibold")
    axes[1, 0].set_ylim(0, 80)
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.6)
    for bar, val in zip(bar_c, savings_percentages):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2.0, val + 1.8, f"+{val:.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')

    seq_labels = ["Beauty Sequence\n(Low Motion, High Texture)", "HoneyBee Sequence\n(Dynamic Complex Detail)"]
    x = np.arange(len(seq_labels))
    width = 0.3

    dct_seq = [7.02, 7.15]
    full_seq = [2.61, 3.13]

    rects1 = axes[1, 1].bar(x - width/2, dct_seq, width, label='Traditional DCT', color='#d9534f', edgecolor='black', linewidth=1.2)
    rects2 = axes[1, 1].bar(x + width/2, full_seq, width, label='GBT-ICL + LLM (Ours)', color='#5cb85c', edgecolor='black', linewidth=1.2)

    axes[1, 1].set_title("D. Multi-Sequence Bitrate Comparison\n(256x256 Aligned Crops, Q=8.0)", fontsize=12, fontweight="bold")
    axes[1, 1].set_ylabel("Bits Per Pixel (bpp)", fontsize=11, fontweight="semibold")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(seq_labels, fontsize=10.5, fontweight="semibold")
    axes[1, 1].set_ylim(0, 8.5)
    axes[1, 1].legend(loc="upper right", frameon=True, fontsize=10)
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.6)

    for rect in rects1:
        yval = rect.get_height()
        axes[1, 1].text(rect.get_x() + rect.get_width()/2.0, yval + 0.15, f"{yval:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    for rect in rects2:
        yval = rect.get_height()
        axes[1, 1].text(rect.get_x() + rect.get_width()/2.0, yval + 0.15, f"{yval:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold', color='darkgreen')

    fig.suptitle("Comprehensive Performance & Rate-Distortion Evaluation: DCT vs. GBT-ICL + LLM", fontsize=15, fontweight="bold", y=0.98)

    graph_out = out_dir / "graphical_performance_comparison.png"
    plt.savefig(graph_out, bbox_inches="tight", dpi=200)

    plt.close()

if __name__ == "__main__":
    plot_performance_metrics()
