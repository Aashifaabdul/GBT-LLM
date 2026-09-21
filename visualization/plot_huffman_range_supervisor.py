"""Huffman vs range coding bar charts from saved result CSVs, with measured and estimated values kept distinct.

Reads (columns used: sequence, the Huffman and range bpp columns, and the saving in percent):
  results/entropy_test/entropy_comparison_256x256.csv                     (comparison/compare_huffman_range.py)
  results/llm_huffman_vs_llm_range/llm_huffman_vs_llm_range_results.csv   (ablations/run_llm_huffman_vs_llm_range.py)
Panel A is fixed-DCT coefficients (both payloads measured); in panel B the Huffman bar is a code-length
estimate and is hatched. Writes results/charts/huffman_vs_range_supervisor.{png,svg}.

Usage: python visualization/plot_huffman_range_supervisor.py
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'charts'

def read(relative):
    """Return the rows of a CSV given relative to the repo root."""
    with (ROOT / relative).open(newline='') as source:
        return list(csv.DictReader(source))

def main():
    """Draw and save the two-panel figure."""
    dct = read('results/entropy_test/entropy_comparison_256x256.csv')
    llm = read('results/llm_huffman_vs_llm_range/llm_huffman_vs_llm_range_results.csv')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    fig.patch.set_facecolor('#f8fafc')
    fig.subplots_adjust(left=.075, right=.97, top=.73, bottom=.27, wspace=.28)
    fig.text(.075, .93, 'Huffman vs. range coding', fontsize=27, weight='bold', color='#10233f')
    fig.text(.075, .875, 'Saved payload comparisons on Beauty and HoneyBee  |  Lower bitrate is better',
             fontsize=13, color='#526176')
    configs = [
        (dct, 'A  Fixed DCT coefficients', 'Grayscale · 256 × 256 · quantization step 16',
         'huffman_bpp', 'range_bpp', 'Huffman payload', 1.65),
        (llm, 'B  GBT + LLM coefficients', 'Three channels · 256 × 256 · quantization step 8',
         'gbticl_llm_huffman_bpp', 'gbticl_llm_range_bpp', 'Huffman estimate*', 5.2),
    ]
    for ax, (rows, title, subtitle, hk, rk, label, ymax) in zip(axes, configs):
        ax.set_facecolor('#f8fafc')
        x = np.arange(len(rows))
        h = np.array([float(r[hk]) for r in rows])
        r = np.array([float(row[rk]) for row in rows])
        bars_h = ax.bar(x-.18, h, .34, color='#e5a044', label=label, zorder=3)
        bars_r = ax.bar(x+.18, r, .34, color='#168c8c', label='Range payload', zorder=3)
        if ax is axes[1]:
            for bar in bars_h:
                bar.set_hatch('//')
                bar.set_edgecolor('#8d5b15')
        for bars in (bars_h, bars_r):
            ax.bar_label(bars, fmt='%.3f', padding=6, fontsize=12, weight='bold')
        ax.set_ylim(0, ymax)
        ax.set_xticks(x, [row['sequence'] for row in rows])
        ax.set_ylabel('Payload bitrate (bits per pixel)')
        ax.grid(axis='y', color='#dce3eb', zorder=0)
        ax.spines['left'].set_color('#b6c2cf')
        ax.spines['bottom'].set_color('#b6c2cf')
        ax.set_title(title, loc='left', fontsize=16, weight='bold', pad=40, color='#10233f')
        ax.text(0, 1.04, subtitle, transform=ax.transAxes, fontsize=10.5, color='#526176')
        ax.legend(loc='upper left', frameon=False, fontsize=10)
        for i in range(len(rows)):
            saving = rows[i].get('reduction_pct', rows[i].get('range_savings_pct'))
            ax.text(i, -.14, f'{float(saving):.2f}% less with range',
                    transform=ax.get_xaxis_transform(), ha='center', color='#117575',
                    fontsize=11, weight='bold')
    fig.text(.075, .155, 'Read each panel separately: the colour format and quantization settings differ.',
             fontsize=11, weight='bold', color='#10233f')
    fig.text(.075, .11, 'A: Payload sizes exclude the shared histogram / codebook and symbol-mapping overhead.\n'
             '*B: Huffman is a heuristic code-length estimate, not a decoded stream; range output was not independently decoded.\n'
             'These are bitrate comparisons, not encoding-speed measurements. No experiments were rerun.',
             fontsize=10, color='#526176', va='top', linespacing=1.6)
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'svg'):
        fig.savefig(OUT / f'huffman_vs_range_supervisor.{suffix}', dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUT / 'huffman_vs_range_supervisor.png')

if __name__ == '__main__':
    main()
