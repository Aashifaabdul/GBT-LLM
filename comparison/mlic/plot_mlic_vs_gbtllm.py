"""Figure 4.3 / Table 4.3: MLIC baseline against GBT-LLM.

Panel (a): bitrate on Beauty and HoneyBee 256x256 crops (log scale).
Panel (b): PSNR against bitrate on Beauty.

The values are those of Table 4.3 of the report. Sources:

  MLIC Q3  frame 0 of results/mlic/mlic_256x256_results.json
           (bpp; RGB PSNR for Beauty).
  MLIC Q6  mean over the patches in results/mlic/neural_lic_results.json
           (bpp); the plotted PSNR of 39.70 dB is the mean over both sequences.
  GBT-LLM  results/ablation_summary.csv, ablation `full`, quant step 8
           (bpp; PSNR is luma (Y) PSNR).

MLIC PSNR is computed on RGB, GBT-LLM PSNR on luma only.

Output: results/mlic/mlic_vs_gbtllm.png

Usage:
    python comparison/mlic/plot_mlic_vs_gbtllm.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "mlic"

CODECS = ['MLIC (Q3)', 'MLIC (Q6)', 'GBT-LLM\n(Ours)']
BEAUTY_BPP = [0.0436, 0.0698, 2.6068]
HONEYBEE_BPP = [0.1209, 0.3058, 3.1279]

# Beauty (bpp, PSNR in dB)
MLIC_POINTS = [(0.0436, 35.85), (0.0698, 39.70)]
GBT_LLM_POINT = (2.6068, 42.95)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10.5,
    'ytick.labelsize': 10.5,
    'legend.fontsize': 10,
})


def label_bars(ax, rects, colour):
    """Write the value above each bar."""
    for r in rects:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width() / 2, h * 1.15 if h < 1 else h + 0.25, f"{h:.2f}",
                ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=colour)


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel (a): bitrate per codec
    x = np.arange(len(CODECS))
    width = 0.35
    rects1 = ax1.bar(x - width / 2, BEAUTY_BPP, width, label='Beauty (256x256)',
                     color='#3182bd', edgecolor='black', linewidth=1)
    rects2 = ax1.bar(x + width / 2, HONEYBEE_BPP, width, label='HoneyBee (256x256)',
                     color='#e6550d', edgecolor='black', linewidth=1)
    label_bars(ax1, rects1, '#1c4a75')
    label_bars(ax1, rects2, '#8c2d04')

    ax1.set_ylabel("Bitrate (bpp) [Log Scale]", fontweight='bold')
    ax1.set_yscale('log')
    ax1.set_title("(a) Bitrate: MLIC vs GBT-LLM", fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(CODECS, fontweight='bold')
    ax1.set_ylim(0.01, 25)
    ax1.legend(loc='upper left', frameon=True)
    ax1.grid(True, which="both", ls="--", alpha=0.4)

    # Panel (b): PSNR against bitrate, Beauty
    ax2.scatter(*zip(*MLIC_POINTS), color='#2ca02c', s=120, marker='o',
                label='MLIC (Cheng Attention)', zorder=4)
    ax2.plot(*zip(*MLIC_POINTS), color='#2ca02c', linestyle='--', alpha=0.7)
    ax2.scatter([GBT_LLM_POINT[0]], [GBT_LLM_POINT[1]], color='#d62728', s=160, marker='*',
                label='GBT-LLM (Ours)', zorder=5)

    ax2.text(0.12, 35.4, "MLIC Q3\n(0.04 bpp, 35.9 dB)", fontsize=8.5, color='#1b611b', ha='left')
    ax2.text(0.12, 39.9, "MLIC Q6\n(0.07 bpp, 39.7 dB)", fontsize=8.5, color='#1b611b', ha='left')
    ax2.text(2.6068, 43.3, "GBT-LLM (Ours)\n(2.61 bpp, 42.95 dB)", fontsize=9, fontweight='bold',
             color='#a50f15', ha='center')

    ax2.set_xlabel("Bitrate (bpp)", fontweight='bold')
    ax2.set_ylabel("PSNR (dB)", fontweight='bold')
    ax2.set_title("(b) Rate vs PSNR on Beauty", fontweight='bold', pad=10)
    ax2.set_xlim(-0.3, 3.5)
    ax2.set_ylim(33, 46)
    ax2.legend(loc='lower right', frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.text(0.01, 0.985, "MLIC: RGB PSNR; GBT-LLM: luma (Y) PSNR", transform=ax2.transAxes,
             ha='left', va='top', fontsize=8.5, color='#555555', style='italic')

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_fig = OUT_DIR / "mlic_vs_gbtllm.png"
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    print(f"Saved {out_fig}")


if __name__ == "__main__":
    main()
