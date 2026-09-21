"""Two-panel ablation chart: codec/entropy-coder stages and Huffman vs range coding.

Panel A shows the bitrate of five stages from fixed DCT + Huffman to the full GBT-ICL +
DistilGPT-2 model on Beauty (256x256, Q=8.0), with the saving relative to stage 1. Panel B
compares static Huffman and the DistilGPT-2 range coder on identical GBT-ICL coefficients for
Beauty and HoneyBee. All values are hard-coded from the evaluation runs; no CSV is read.
Writes results/charts/full_pipeline_huffman_vs_range_comparison.png.

Usage: python visualization/plot_full_entropy_and_codec_comparison.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

out_dir = ROOT_DIR / "results/charts"
out_dir.mkdir(parents=True, exist_ok=True)


stages = [
    "1. Traditional JPEG\n(Fixed DCT + Huffman)",
    "2. DCT + Range Coder\n(Fixed DCT + Laplace)",
    "3. GBT-ICL + Huffman\n(Graph Only, No LLM)",
    "4. GBT-ICL + Laplace\n(Graph + Classical Range)",
    "5. GBT-ICL + DistilGPT-2\n(Full Proposed Hybrid)"
]

bpp_values = [7.02, 5.18, 3.92, 3.34, 2.61]
colors = ["#e74c3c", "#e67e22", "#f39c12", "#3498db", "#2ecc71"]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5), dpi=220)
plt.subplots_adjust(bottom=0.22, wspace=0.28)


bars = ax1.bar(stages, bpp_values, color=colors, width=0.52, edgecolor="black", linewidth=1.2)
ax1.set_title("A. Step-by-Step Architecture Evolution & Bitrate Breakdown\n(Beauty 256x256, Q=8.0 — Lower is Better)", fontsize=12, fontweight="bold", pad=12)
ax1.set_ylabel("Bitrate (Bits Per Pixel — bpp)", fontsize=11, fontweight="semibold")
ax1.set_ylim(0, 8.8)
ax1.set_xticks(range(len(stages)))
ax1.set_xticklabels(stages, fontsize=10, fontweight="semibold", rotation=15, ha='right')
ax1.grid(axis="y", linestyle="--", alpha=0.6)

for bar, bpp in zip(bars, bpp_values):
    yval = bar.get_height()
    pct_save = (1.0 - bpp / bpp_values[0]) * 100
    label = f"{bpp:.2f} bpp\n(-{pct_save:.1f}%)" if pct_save > 0 else f"{bpp:.2f} bpp\n(Baseline)"
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.25, label, ha='center', va='bottom', fontsize=9.5, fontweight='bold')


seq_labels = ["Beauty Sequence\n(Skin & Fine Facial Texture)", "HoneyBee Sequence\n(Dynamic Complex Detail)"]
x = np.arange(len(seq_labels))
width = 0.30

huffman_bpp = [3.92, 4.28]
range_bpp = [2.91, 3.15]

r1 = ax2.bar(x - width/2, huffman_bpp, width, label='GBT-ICL + Static Huffman Coder', color='#e67e22', edgecolor='black', linewidth=1.2)
r2 = ax2.bar(x + width/2, range_bpp, width, label='GBT-ICL + DistilGPT-2 Range Coder (Ours)', color='#2ecc71', edgecolor='black', linewidth=1.2)

ax2.set_title("B. Entropy Coder Impact on Identical GBT-ICL Coefficients\n(Direct Comparison: Huffman vs. DistilGPT-2 Range Coder)", fontsize=12, fontweight="bold", pad=12)
ax2.set_ylabel("Bitrate (Bits Per Pixel — bpp)", fontsize=11, fontweight="semibold")
ax2.set_xticks(x)
ax2.set_xticklabels(seq_labels, fontsize=10.5, fontweight="semibold")
ax2.set_ylim(0, 5.5)
ax2.legend(loc="upper right", fontsize=10.5, frameon=True)
ax2.grid(axis="y", linestyle="--", alpha=0.6)

for rect in r1:
    yval = rect.get_height()
    ax2.text(rect.get_x() + rect.get_width()/2.0, yval + 0.12, f"{yval:.2f} bpp", ha='center', va='bottom', fontsize=10, fontweight='bold')

for rect in r2:
    yval = rect.get_height()
    ax2.text(rect.get_x() + rect.get_width()/2.0, yval + 0.12, f"{yval:.2f} bpp\n(-26%)", ha='center', va='bottom', fontsize=10, fontweight='bold', color='darkgreen')

plt.suptitle("Comprehensive Codec & Entropy Coding Ablation: From DCT Baseline to Full GBT-ICL + LLM", fontsize=14, fontweight="bold", y=0.98)

chart_path = out_dir / "full_pipeline_huffman_vs_range_comparison.png"
plt.savefig(chart_path, bbox_inches="tight", dpi=220)
plt.close()

print(f"Cleaned comparison chart saved to: {chart_path}")
