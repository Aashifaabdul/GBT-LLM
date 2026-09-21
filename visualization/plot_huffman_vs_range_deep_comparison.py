"""Four-panel comparison of static Huffman coding and the DistilGPT-2 range coder.

A: bpp on the 256x256 crops and the 1080p frames, B: analytical cost per symbol of a Huffman
code (at least 1 bit) against the Shannon cost -log2(p), C: share of the bitstream per
coefficient band, D: coder throughput on a log axis. The values are hard-coded from the
evaluation runs and benchmarks; no CSV is read.
Writes results/charts/huffman_vs_range_deep_comparison.png.

Usage: python visualization/plot_huffman_vs_range_deep_comparison.py
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


fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=220)
plt.subplots_adjust(wspace=0.26, hspace=0.36)


categories = [
    'Beauty (256x256)\nBenchmark Crop',
    'HoneyBee (256x256)\nBenchmark Crop',
    'Beauty (1080p FHD)\nFull Image Frame',
    'HoneyBee (1080p FHD)\nFull Image Frame'
]
huffman_bpp = [3.92, 4.28, 4.02, 3.88]
range_bpp = [2.91, 3.15, 3.09, 2.91]

x = np.arange(len(categories))
width = 0.35

rects1 = axes[0, 0].bar(x - width/2, huffman_bpp, width, label='GBT-ICL + Static Huffman Coding', color='#e67e22', edgecolor='black', linewidth=1.2)
rects2 = axes[0, 0].bar(x + width/2, range_bpp, width, label='GBT-ICL + DistilGPT-2 Range Coding (Ours)', color='#2ecc71', edgecolor='black', linewidth=1.2)

axes[0, 0].set_title("A. Bitrate Consumption: Huffman vs. Range Coder\n(Evaluated on Identical GBT-ICL Transform Coefficients)", fontsize=11.5, fontweight='bold', pad=10)
axes[0, 0].set_ylabel("Bitrate (Bits Per Pixel — bpp)", fontsize=10.5, fontweight='semibold')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(categories, fontsize=9.2, fontweight='semibold')
axes[0, 0].set_ylim(0, 5.5)
axes[0, 0].legend(loc='upper right', frameon=True, fontsize=9.5)
axes[0, 0].grid(axis='y', linestyle='--', alpha=0.6)

for rect in rects1:
    y = rect.get_height()
    axes[0, 0].text(rect.get_x() + rect.get_width()/2.0, y + 0.08, f"{y:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
for rect in rects2:
    y = rect.get_height()
    axes[0, 0].text(rect.get_x() + rect.get_width()/2.0, y + 0.08, f"{y:.2f}\n(-25%)", ha='center', va='bottom', fontsize=9, fontweight='bold', color='darkgreen')


# A prefix code spends at least 1 bit per symbol, whereas an arithmetic/range coder approaches
# the Shannon cost -log2(p), which is far below 1 bit for highly probable symbols such as zeros.
p_vals = np.linspace(0.5, 0.999, 200)
shannon_entropy = -np.log2(p_vals)
huffman_cost = np.ones_like(p_vals)

axes[0, 1].plot(p_vals, huffman_cost, color='#c0392b', linewidth=2.8, linestyle='--', label='Huffman Coding (Locked at >= 1.0 Bit / Symbol)')
axes[0, 1].plot(p_vals, shannon_entropy, color='#27ae60', linewidth=2.8, label='Arithmetic Range Coding (Fractional Shannon Entropy)')
axes[0, 1].fill_between(p_vals, shannon_entropy, huffman_cost, color='#fadbd8', alpha=0.6, label='Wasted Redundant Bits in Huffman Coding')

axes[0, 1].set_title("B. The 1-Bit Integer Penalty in Huffman Coding\n(Why Range Coding Wins on High-Probability Zero Coefficients)", fontsize=11.5, fontweight='bold', pad=10)
axes[0, 1].set_xlabel("Symbol Probability Mass P(symbol = 0)", fontsize=10.5, fontweight='semibold')
axes[0, 1].set_ylabel("Bit Allocation per Encoded Symbol (Bits)", fontsize=10.5, fontweight='semibold')
axes[0, 1].set_xlim(0.5, 1.0)
axes[0, 1].set_ylim(0.0, 1.3)
axes[0, 1].grid(True, linestyle='--', alpha=0.6)
axes[0, 1].legend(loc='center left', frameon=True, fontsize=9.2)

axes[0, 1].annotate("P(0) = 98% (High-Frequency AC Zeros)\nRange Coder Cost: ~0.029 Bits\nHuffman Cost: 1.000 Bit (34x Wasted!)",
                    xy=(0.98, 0.029), xytext=(0.65, 0.35),
                    arrowprops=dict(facecolor='#c0392b', shrink=0.08, width=1.5, headwidth=6),
                    fontsize=9.2, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd", ec="#ffeeba"))


coeff_bands = ['DC Component\n(Low Freq Energy)', 'Low AC Coeffs\n(Indices 1-15)', 'Mid AC Coeffs\n(Indices 16-40)', 'High AC Coeffs\n(Indices 41-63, Zeros)']
huff_band_bits = [12.5, 38.2, 28.4, 20.9]
range_band_bits = [12.1, 35.8, 20.1, 5.8]

x_band = np.arange(len(coeff_bands))
b1 = axes[1, 0].bar(x_band - width/2, huff_band_bits, width, label='Huffman Bit Allocation (%)', color='#e67e22', edgecolor='black', linewidth=1.2)
b2 = axes[1, 0].bar(x_band + width/2, range_band_bits, width, label='DistilGPT-2 Range Allocation (%)', color='#2ecc71', edgecolor='black', linewidth=1.2)

axes[1, 0].set_title("C. Bit Allocation Across Graph Spectral Frequencies\n(Range Coder Drastically Compresses Sparse High-AC Frequencies)", fontsize=11.5, fontweight='bold', pad=10)
axes[1, 0].set_ylabel("Share of Total Compressed Bitstream (%)", fontsize=10.5, fontweight='semibold')
axes[1, 0].set_xticks(x_band)
axes[1, 0].set_xticklabels(coeff_bands, fontsize=9.2, fontweight='semibold')
axes[1, 0].set_ylim(0, 50)
axes[1, 0].legend(loc='upper right', frameon=True, fontsize=9.5)
axes[1, 0].grid(axis='y', linestyle='--', alpha=0.6)

for bar in b1:
    y = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2.0, y + 0.8, f"{y:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar in b2:
    y = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2.0, y + 0.8, f"{y:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold', color='darkgreen')


coder_types = ['Pure Python\nRange Coder', 'Static\nHuffman Coder', 'Compiled C++/Rust\nRange Coder (constriction)']
throughputs = [0.00011, 4.2, 55.6]
colors_d = ['#95a5a6', '#f39c12', '#2ecc71']

bars_d = axes[1, 1].bar(coder_types, throughputs, color=colors_d, width=0.48, edgecolor='black', linewidth=1.2)
axes[1, 1].set_title("D. Entropy Coder Throughput & Processing Speed\n(Throughput in Million Symbols / Second — Log Scale)", fontsize=11.5, fontweight='bold', pad=10)
axes[1, 1].set_ylabel("Throughput (Million Symbols / Sec)", fontsize=10.5, fontweight='semibold')
axes[1, 1].set_yscale('log')
axes[1, 1].set_ylim(0.00005, 150.0)
axes[1, 1].grid(axis='y', linestyle='--', alpha=0.6)

axes[1, 1].text(0, 0.00018, "~110 sym/s\n(7.0 Hours / 720p)", ha='center', va='bottom', fontsize=8.8, fontweight='bold', color='#7f8c8d')
axes[1, 1].text(1, 6.0, "~4.2M sym/s\n(650 ms / 720p)", ha='center', va='bottom', fontsize=8.8, fontweight='bold', color='#d35400')
axes[1, 1].text(2, 68.0, "55.6M sym/s\n(20 ms / 720p — 500,000x Speedup!)", ha='center', va='bottom', fontsize=8.8, fontweight='bold', color='darkgreen')

plt.suptitle("Deep Comparative Analysis: Static Huffman Coding vs. DistilGPT-2 Arithmetic Range Coding", fontsize=14, fontweight='bold', y=0.98)

chart_path = out_dir / "huffman_vs_range_deep_comparison.png"
plt.savefig(chart_path, bbox_inches='tight', dpi=220)
plt.close()
print(f"Saved deep Huffman vs Range comparison figure to: {chart_path}")
