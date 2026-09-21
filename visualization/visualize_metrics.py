"""Summary figures and tables from the metrics written by run_dataset_pipeline.py.

Per-sequence mode (--results-dir): reads <results-dir>/<Sequence>/metrics.csv and writes to
<results-dir>/figures (or --out): psnr_per_frame.png, bpp_per_frame.png, rate_distortion.png
(optionally overlaying the runs of --compare) and summary.csv.

Ablation mode (--compare-all): reads results/ablation_summary.csv (one row per ablation,
sequence and quantisation step), plots one rate-distortion figure per sequence with one curve per
ablation config and writes bd_rate_table.csv (BD-Rate against --bd-rate-reference). BD-Rate needs
at least 4 points per curve, i.e. a --quant-steps sweep in run_dataset_pipeline.py (see
gbticl_pipeline/evaluate.py::bd_rate).

Usage:
  python visualization/visualize_metrics.py --results-dir results/full
  python visualization/visualize_metrics.py --results-dir results/full --compare results/dct
  python visualization/visualize_metrics.py --compare-all results/ablation_summary.csv
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _to_float_or_inf(s):
    return float(s) if s not in ("inf", "Infinity", "") else float("inf")


def read_metrics(csv_path):
    """Read a per-sequence metrics.csv written by process_sequence().

    Accepts the current columns (frame, bpp, y_psnr_db, y_ssim, rgb_psnr_db; no per-frame timings,
    because the video codec times the whole sequence) and the older per-image columns
    (psnr_db, encode_s, decode_s). Missing values become NaN or inf (lossless PSNR)."""
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            r = {"frame": row["frame"], "bpp": float(row["bpp"])}
            if "y_psnr_db" in row:  # current (video-mode) format
                r["psnr_db"] = _to_float_or_inf(row["y_psnr_db"])
                r["y_ssim"] = float(row["y_ssim"]) if row.get("y_ssim") not in (None, "", "nan") else float("nan")
                r["rgb_psnr_db"] = _to_float_or_inf(row.get("rgb_psnr_db", "inf"))
            else:  # older (per-image-mode) format
                r["psnr_db"] = _to_float_or_inf(row["psnr_db"])
            r["encode_s"] = float(row["encode_s"]) if "encode_s" in row else float("nan")
            r["decode_s"] = float(row["decode_s"]) if "decode_s" in row else float("nan")
            rows.append(r)
    return rows


def find_sequence_metrics(results_dir):
    """Return {sequence name: rows} for every <results_dir>/<Sequence>/metrics.csv."""
    results_dir = Path(results_dir)
    out = {}
    for seq_dir in sorted(results_dir.iterdir()):
        mp = seq_dir / "metrics.csv"
        if seq_dir.is_dir() and mp.exists():
            out[seq_dir.name] = read_metrics(mp)
    return out


def summarize(rows):
    """Mean PSNR (finite frames only), bpp and encode/decode times of one sequence."""
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
    """Line plot of one metric (row key ylabel_key) against frame index, one line per sequence."""
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
    """Scatter of per-frame (bpp, PSNR); compare_metrics are drawn as crosses."""
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
    """Write one summary row per sequence to out_path and print the same numbers."""
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


def read_ablation_summary(csv_path):
    """Read run_dataset_pipeline.py's ablation_summary.csv (one row per ablation, sequence and
    quant_step, appended by _append_ablation_summary).

    Returns {(ablation, sequence): rows} with each list sorted by quant_step (a larger step
    gives a lower bpp)."""
    rows_by_key = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["ablation"], row["sequence"])
            rows_by_key.setdefault(key, []).append({
                "quant_step": float(row["quant_step"]),
                "mean_bpp": float(row["mean_bpp"]),
                "mean_y_psnr_db": float(row["mean_y_psnr_db"]),
                "mean_y_ssim": float(row["mean_y_ssim"]) if row["mean_y_ssim"] not in ("nan", "") else float("nan"),
            })
    for key in rows_by_key:
        rows_by_key[key].sort(key=lambda r: r["quant_step"])
    return rows_by_key


def plot_ablation_rate_distortion(rows_by_key, out_dir):
    """One rate-distortion plot per sequence with one curve per ablation config
    (DCT baseline, non-adaptive GBT, GBT-ICL without LLM, full model)."""
    sequences = sorted(set(seq for _, seq in rows_by_key))
    for seq in sequences:
        fig, ax = plt.subplots(figsize=(7, 6))
        for ablation, seq_ in sorted(rows_by_key):
            if seq_ != seq:
                continue
            rows = rows_by_key[(ablation, seq_)]
            if len(rows) < 2:
                continue  # need at least 2 points to draw a line
            bpps = [r["mean_bpp"] for r in rows]
            psnrs = [r["mean_y_psnr_db"] for r in rows]
            ax.plot(bpps, psnrs, marker="o", markersize=5, label=ablation)
        ax.set_xlabel("bits per pixel (lower = smaller file)")
        ax.set_ylabel("Y-PSNR (dB) (higher = better quality)")
        ax.set_title(f"Rate-distortion by ablation config -- {seq}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out_path = out_dir / f"ablation_rate_distortion_{seq}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        print(f"saved {out_path}")


def write_bd_rate_table(rows_by_key, out_path, reference="dct"):
    """Write the BD-Rate (gbticl_pipeline.evaluate.bd_rate, PSNR-based) of every ablation config
    against `reference`, per sequence.

    Curves with fewer than 4 quant_step points are skipped with a printed note."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gbticl_pipeline.evaluate import bd_rate

    sequences = sorted(set(seq for _, seq in rows_by_key))
    lines = ["sequence,ablation,bd_rate_pct_vs_" + reference]
    for seq in sequences:
        ref_rows = rows_by_key.get((reference, seq))
        if not ref_rows or len(ref_rows) < 4:
            print(f"[{seq}] skipping BD-Rate table: reference '{reference}' has "
                  f"{len(ref_rows) if ref_rows else 0} points (need >=4 -- run with --quant-steps)")
            continue
        ref_rate = np.array([r["mean_bpp"] for r in ref_rows])
        ref_dist = np.array([r["mean_y_psnr_db"] for r in ref_rows])

        for ablation, seq_ in sorted(rows_by_key):
            if seq_ != seq or ablation == reference:
                continue
            rows = rows_by_key[(ablation, seq_)]
            if len(rows) < 4:
                print(f"[{seq}] skipping '{ablation}': only {len(rows)} points (need >=4)")
                continue
            test_rate = np.array([r["mean_bpp"] for r in rows])
            test_dist = np.array([r["mean_y_psnr_db"] for r in rows])
            try:
                bd = bd_rate(ref_rate, ref_dist, test_rate, test_dist)
                lines.append(f"{seq},{ablation},{bd:.2f}")
                print(f"[{seq}] BD-Rate {ablation} vs {reference}: {bd:+.2f}% "
                      f"({'better' if bd < 0 else 'worse'})")
            except ValueError as e:
                print(f"[{seq}] BD-Rate {ablation} vs {reference}: skipped ({e})")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved {out_path}")


def main():
    """Parse the command line and write the per-sequence or ablation outputs."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=str, default=None,
                     help="e.g. results/trained (must contain <Sequence>/metrics.csv)")
    ap.add_argument("--compare", type=str, default=None,
                     help="Optional second results dir (e.g. results/baseline) to overlay on the RD plot.")
    ap.add_argument("--compare-all", type=str, default=None,
                     help="Path to results/ablation_summary.csv -- overlays every ablation config's "
                          "rate-distortion curve (one plot per sequence) + a BD-Rate table vs the DCT "
                          "baseline. Mutually exclusive with --results-dir.")
    ap.add_argument("--bd-rate-reference", type=str, default="dct",
                     help="Ablation name to use as the BD-Rate anchor (default: dct).")
    ap.add_argument("--out", type=str, default=None,
                     help="Defaults to <results-dir>/figures, or the ablation_summary.csv's own "
                          "directory / 'figures' for --compare-all.")
    args = ap.parse_args()

    if args.compare_all:
        summary_path = Path(args.compare_all)
        out_dir = Path(args.out) if args.out else summary_path.parent / "figures"
        rows_by_key = read_ablation_summary(summary_path)
        if not rows_by_key:
            raise SystemExit(f"No rows found in {summary_path}")
        plot_ablation_rate_distortion(rows_by_key, out_dir)
        write_bd_rate_table(rows_by_key, out_dir / "bd_rate_table.csv", reference=args.bd_rate_reference)
        return

    if not args.results_dir:
        raise SystemExit("pass either --results-dir or --compare-all")

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
