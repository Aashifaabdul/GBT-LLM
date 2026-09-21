"""Predicted GBT-ICL graph over a whole 1920x1080 frame (HoneyBee frame0000).

Same edge rule as visualize_graph.py (ContextGradientGBTICL: weight exp(-0.15 * |grey jump|)),
vectorised over the frame. A frame has ~32,400 blocks x 112 edges, but only the 14 border edges
of each block can differ from 1, and each of them sits on an 8-pixel block-boundary grid line and
depends only on the pixel gradient along that line. The full-frame map therefore needs one
gradient computation per image, keeping the values on the grid lines.

Reads data/HoneyBee/frames/frame0000.png. Writes to results/figures/:
  full_image_graph_heatmap.png  edge-weight map on the block-boundary lines over the frame
                                (green = strong/smooth, red = weak/likely content edge)
  full_image_graph_zoom.png     explicit node/edge drawing of the highest-variance 16x16-block
                                (128x128 px) patch

Usage: python visualization/visualize_graph_full_image.py
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image

BLOCK_SIZE = 8
EDGE_SENSITIVITY = 0.15
ZOOM_BLOCKS = 16  # zoom panel covers ZOOM_BLOCKS x ZOOM_BLOCKS blocks

SCRIPT_DIR = Path(__file__).resolve().parent
IMG_PATH = ROOT_DIR / "data" / "HoneyBee" / "frames" / "frame0000.png"
OUT_HEATMAP = ROOT_DIR / "results" / "figures" / "full_image_graph_heatmap.png"
OUT_ZOOM = ROOT_DIR / "results" / "figures" / "full_image_graph_zoom.png"


def edge_list(block_size):
    """4-neighbour grid edges (right and down) of a block as node-index pairs (row-major)."""
    edges = []
    for r in range(block_size):
        for c in range(block_size):
            node = r * block_size + c
            if c + 1 < block_size:
                edges.append((node, node + 1))
            if r + 1 < block_size:
                edges.append((node, node + block_size))
    return edges


def context_gradient_weights(top_ctx, left_ctx, block_size, sensitivity=EDGE_SENSITIVITY):
    """Per-block ContextGradientGBTICL edge weights (used for the zoom figure only); returns (weights, edges)."""
    edges = edge_list(block_size)
    weights = np.ones(len(edges), dtype=np.float64)
    top_gray = top_ctx.astype(np.float64).mean(axis=-1)
    left_gray = left_ctx.astype(np.float64).mean(axis=-1)
    for idx, (i, j) in enumerate(edges):
        ri, ci = divmod(i, block_size)
        rj, cj = divmod(j, block_size)
        if ri == 0 and rj == 0 and abs(ci - cj) == 1:
            jump = abs(top_gray[ci] - top_gray[cj])
            weights[idx] = np.exp(-sensitivity * jump)
        elif ci == 0 and cj == 0 and abs(ri - rj) == 1:
            jump = abs(left_gray[ri] - left_gray[rj])
            weights[idx] = np.exp(-sensitivity * jump)
    return weights, edges


def whole_image_weight_maps(gray, block_size=BLOCK_SIZE, sensitivity=EDGE_SENSITIVITY):
    """
    Vectorised equivalent of context_gradient_weights for every block boundary at once.

    Returns two (H, W) float arrays, NaN except at pixel pairs that correspond
    to a movable graph edge:
      - h_weight[r, c]: weight of the horizontal edge between (r,c) and
        (r,c+1), populated only on rows r that sit directly above a block
        row (r = k*block_size - 1).
      - v_weight[r, c]: weight of the vertical edge between (r,c) and
        (r+1,c), populated only on cols c that sit directly left of a block
        col (c = k*block_size - 1).
    """
    h, w = gray.shape
    h_weight = np.full((h, w), np.nan)
    v_weight = np.full((h, w), np.nan)

    # last row of each block row (it is top_ctx of the block below); the last block row has no block below
    boundary_rows = np.arange(block_size - 1, h - block_size, block_size)
    boundary_cols = np.arange(block_size - 1, w - block_size, block_size)

    if len(boundary_rows) > 0:
        row_strip = gray[boundary_rows, :]  # (n_rows, W)
        hgrad = np.abs(np.diff(row_strip, axis=1))  # (n_rows, W-1)
        hw = np.exp(-sensitivity * hgrad)
        h_weight[np.ix_(boundary_rows, np.arange(w - 1))] = hw

    if len(boundary_cols) > 0:
        col_strip = gray[:, boundary_cols]  # (H, n_cols)
        vgrad = np.abs(np.diff(col_strip, axis=0))  # (H-1, n_cols)
        vw = np.exp(-sensitivity * vgrad)
        v_weight[np.ix_(np.arange(h - 1), boundary_cols)] = vw

    return h_weight, v_weight


def make_heatmap_figure(img, gray):
    """Overlay the boundary edge-weight map on the frame and save OUT_HEATMAP."""
    h_weight, v_weight = whole_image_weight_maps(gray)
    n_h = np.sum(~np.isnan(h_weight))
    n_v = np.sum(~np.isnan(v_weight))
    print(f"whole-image movable edges: {n_h} horizontal + {n_v} vertical = {n_h + n_v} total")
    combined = np.nanmin(np.stack([h_weight, v_weight]), axis=0)  # weakest of the two, per pixel
    n_weak = np.sum(~np.isnan(combined) & (combined < 0.5))
    n_total = np.sum(~np.isnan(combined))
    print(f"weak (<0.5) movable edges: {n_weak} of {n_total} "
          f"({100*n_weak/max(n_total,1):.1f}%)")

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(img, interpolation="nearest")
    masked = np.ma.masked_invalid(combined)
    im = ax.imshow(masked, cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest", alpha=0.9)
    ax.set_title("Predicted graph weight at every block boundary, whole frame\n"
                  "green=strong/smooth (uniform-like), red=weak (likely content edge)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("predicted edge weight")
    fig.tight_layout()
    OUT_HEATMAP.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_HEATMAP, dpi=140, bbox_inches="tight")
    print(f"saved {OUT_HEATMAP}")


def find_high_variance_patch(img, patch_blocks=ZOOM_BLOCKS, block_size=BLOCK_SIZE):
    """Find the highest-variance patch_blocks x patch_blocks patch on a coarse grid.

    Returns ((block_row, block_col), variance). The search starts at block 1 so that every
    block in the patch has real top/left context."""
    h, w = img.shape[:2]
    n_bh, n_bw = h // block_size, w // block_size
    best, best_var = (1, 1), -1.0
    step = max(1, patch_blocks // 2)  # coarse search
    for bi in range(1, n_bh - patch_blocks, step):
        for bj in range(1, n_bw - patch_blocks, step):
            r0, c0 = bi * block_size, bj * block_size
            r1, c1 = r0 + patch_blocks * block_size, c0 + patch_blocks * block_size
            v = img[r0:r1, c0:c1, :].astype(float).var()
            if v > best_var:
                best_var, best = v, (bi, bj)
    return best, best_var


def make_zoom_figure(img):
    """Draw explicit nodes and edges for the highest-variance patch and save OUT_ZOOM."""
    (bi0, bj0), _ = find_high_variance_patch(img)
    n = ZOOM_BLOCKS
    r0, c0 = bi0 * BLOCK_SIZE, bj0 * BLOCK_SIZE
    patch = img[r0:r0 + n * BLOCK_SIZE, c0:c0 + n * BLOCK_SIZE, :]
    print(f"zoom patch: blocks ({bi0},{bj0}) to ({bi0+n},{bj0+n}), pixel origin ({r0},{c0})")

    edges = edge_list(BLOCK_SIZE)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(patch, interpolation="nearest", extent=(0, n * BLOCK_SIZE, n * BLOCK_SIZE, 0))

    segs, colors, lws = [], [], []
    for bi in range(n):
        for bj in range(n):
            r0b, c0b = r0 + bi * BLOCK_SIZE, c0 + bj * BLOCK_SIZE
            top_ctx = img[r0b - 1, c0b:c0b + BLOCK_SIZE, :]
            left_ctx = img[r0b:r0b + BLOCK_SIZE, c0b - 1, :]
            weights, _ = context_gradient_weights(top_ctx, left_ctx, BLOCK_SIZE)

            ox, oy = bj * BLOCK_SIZE, bi * BLOCK_SIZE  # offset of this block within the patch
            node_xy = np.array([[ox + c, oy + r] for r in range(BLOCK_SIZE) for c in range(BLOCK_SIZE)])
            for (i, j), w in zip(edges, weights):
                p1, p2 = node_xy[i], node_xy[j]
                segs.append([p1, p2])
                colors.append(plt.cm.RdYlGn(w))
                lws.append(0.3 + 1.2 * w)

    lc = LineCollection(segs, colors=colors, linewidths=lws)
    ax.add_collection(lc)
    ax.set_xlim(0, n * BLOCK_SIZE)
    ax.set_ylim(n * BLOCK_SIZE, 0)
    ax.set_title(f"Predicted graph, explicit nodes+edges, {n}x{n}-block patch\n"
                  "green=strong (smooth), red=weak (likely edge)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    OUT_ZOOM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_ZOOM, dpi=140, bbox_inches="tight")
    print(f"saved {OUT_ZOOM}")


def main():
    img = np.array(Image.open(IMG_PATH).convert("RGB"))
    gray = img.astype(np.float64).mean(axis=-1)

    make_heatmap_figure(img, gray)
    make_zoom_figure(img)


if __name__ == "__main__":
    main()
