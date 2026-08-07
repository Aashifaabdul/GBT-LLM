"""
Read the metrics.csv files produced by run_dataset_pipeline.py (one per
sequence) and produce the summary figures for your dissertation results
section: PSNR per frame, bits-per-pixel per frame, and a rate-distortion
scatter across all sequences.

No PyTorch needed -- pure pandas/matplotlib/numpy, so (unlike training.py
and run_dataset_pipeline.py) this one WAS actually executed and verified in
this environment, against a synthetic metrics.csv, before being handed to
you. See the bottom of this file's accompanying README section for how that
was checked.

USAGE
  python visualize_metrics.py --results-dir results/trained
  python visualize_metrics.py --results-dir results/baseline --compare results/trained
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_metrics(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "frame": row["frame"],
                "bpp": float(row["bpp"]),
                "psnr_db": float(row["psnr_db"]) if row["psnr_db"] not in ("inf", "Infinity") else float("inf"),
                "encode_s": float(row["encode_s"]),
                "decode_s": float(row["decode_s"]),
            })
    return rows


def find_sequence_metrics(results_dir):
    """results_dir/<Sequence>/metrics.csv for every sequence present."""
    results_dir = Path(results_dir)
    out = {}
    for seq_dir in sorted(results_dir.iterdir()):
        mp = seq_dir / "metrics.csv"
        if seq_dir.is_dir() and mp.exists():
            out[seq_dir.name] = read_metrics(mp)
    return out


def summarize(rows):
    finite = [r["psnr_db"] for r in rows if r["psnr_db"] != float("inf")]
    bpps = [r["bpp"] for r in rows]
    return dict(
        n_frames=len(rows),
        mean_psnr=sum(finite) / len(finite) if finite else float("nan"),
        mean_bpp=sum(bpps) / len(bpps) if bpps else float("nan"),
        mean_encode_s=sum(r["encode_s"] for r in rows) / len(rows),
        mean_decode_s=sum(r["decode_s"] for r in rows) / len(rows),
    )


def plot_per_frame(all_metrics, out_path, ylabel_key, title, ylabel):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for seq, rows in all_metrics.items():
        finite_rows = [(i, r[ylabel_key]) for i, r in enumerate(rows) if r[ylabel_key] != float("inf")]
        if not finite_rows:
            continue
        xs, ys = zip(*finite_rows)
        ax.plot(xs, ys, marker="o", markersize=3, label=seq)
    ax.set_xlabel("frame index (sampled order)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_rate_distortion(all_metrics, out_path, compare_metrics=None, compare_label="compare"):
    fig, ax = plt.subplots(figsize=(6, 6))
    for seq, rows in all_metrics.items():
        pts = [(r["bpp"], r["psnr_db"]) for r in rows if r["psnr_db"] != float("inf")]
        if pts:
            bpps, psnrs = zip(*pts)
            ax.scatter(bpps, psnrs, label=seq, alpha=0.75, s=28)
    if compare_metrics:
        for seq, rows in compare_metrics.items():
            pts = [(r["bpp"], r["psnr_db"]) for r in rows if r["psnr_db"] != float("inf")]
            if pts:
                bpps, psnrs = zip(*pts)
                ax.scatter(bpps, psnrs, label=f"{seq} ({compare_label})", alpha=0.4, s=28, marker="x")
    ax.set_xlabel("bits per pixel (lower = smaller file)")
    ax.set_ylabel("PSNR (dB) (higher = better quality)")
    ax.set_title("Rate-distortion: every encoded frame")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"saved {out_path}")


def write_summary_table(all_metrics, out_path):
    with open(out_path, "w") as f:
        f.write("sequence,n_frames,mean_psnr_db,mean_bpp,mean_encode_s,mean_decode_s\n")
        for seq, rows in all_metrics.items():
            s = summarize(rows)
            f.write(f"{seq},{s['n_frames']},{s['mean_psnr']:.3f},{s['mean_bpp']:.4f},"
                    f"{s['mean_encode_s']:.2f},{s['mean_decode_s']:.2f}\n")
    print(f"saved {out_path}")
    for seq, rows in all_metrics.items():
        s = summarize(rows)
        print(f"  {seq}: {s['n_frames']} frames, mean PSNR={s['mean_psnr']:.2f}dB, "
              f"mean bpp={s['mean_bpp']:.3f}, mean encode={s['mean_encode_s']:.2f}s, "
              f"mean decode={s['mean_decode_s']:.2f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=str, required=True,
                     help="e.g. results/trained (must contain <Sequence>/metrics.csv)")
    ap.add_argument("--compare", type=str, default=None,
                     help="Optional second results dir (e.g. results/baseline) to overlay on the RD plot.")
    ap.add_argument("--out", type=str, default=None, help="Defaults to <results-dir>/figures")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out) if args.out else results_dir / "figures"

    all_metrics = find_sequence_metrics(results_dir)
    if not all_metrics:
        raise SystemExit(f"No <sequence>/metrics.csv found under {results_dir}")

    compare_metrics = find_sequence_metrics(args.compare) if args.compare else None
    compare_label = Path(args.compare).name if args.compare else "compare"

    plot_per_frame(all_metrics, out_dir / "psnr_per_frame.png", "psnr_db",
                    "PSNR per frame", "PSNR (dB)")
    plot_per_frame(all_metrics, out_dir / "bpp_per_frame.png", "bpp",
                    "Bits-per-pixel per frame", "bits/pixel")
    plot_rate_distortion(all_metrics, out_dir / "rate_distortion.png", compare_metrics, compare_label)
    write_summary_table(all_metrics, out_dir / "summary.csv")


if __name__ == "__main__":
    main()
