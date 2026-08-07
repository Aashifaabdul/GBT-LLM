"""
Visualize the predicted GBT-ICL graph across an ENTIRE frame, not just one
block. Same ContextGradientGBTICL math as visualize_graph.py, but vectorized
instead of looped per-block, because a full 1920x1080 frame has ~32,400
blocks x 112 edges (~3.6M segments) -- far too many to draw individually and
mostly uninformative, since only the 14 border-touching edges per block can
ever move at all.

Key insight that makes this tractable: every edge that CAN move sits exactly
on an 8-pixel block-boundary grid line, and its weight depends only on the
pixel gradient at that exact line (top_ctx edges = horizontal gradient along
the row just above each block row of blocks; left_ctx edges = vertical
gradient along the column just left of each block column of blocks). So the
whole-image "predicted graph" is just: compute horizontal/vertical pixel
gradients once for the whole image (fully vectorized, no per-block loop),
then keep only the gradient values that fall exactly on those grid lines.

Produces two figures:
  1. full_image_graph_heatmap.png -- the whole frame, with the predicted
     edge-weight map overlaid only on the block-boundary grid lines (green =
     strong/smooth, red = weak/likely content edge). Shows at a glance where
     the adaptive graph would diverge from the uniform grid across the whole
     image, and whether that lines up with real object edges.
  2. full_image_graph_zoom.png -- an explicit node+edge rendering (like
     visualize_graph.py) but over a 16x16-block patch (128x128 px) instead of
     a single block, as a middle ground between the single-block and
     whole-image views.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image

BLOCK_SIZE = 8
EDGE_SENSITIVITY = 0.15
ZOOM_BLOCKS = 16  # zoom panel covers ZOOM_BLOCKS x ZOOM_BLOCKS blocks

SCRIPT_DIR = Path(__file__).resolve().parent
IMG_PATH = SCRIPT_DIR / "HoneyBee" / "frames" / "frame0000.png"
OUT_HEATMAP = SCRIPT_DIR / "figures" / "full_image_graph_heatmap.png"
OUT_ZOOM = SCRIPT_DIR / "figures" / "full_image_graph_zoom.png"


def edge_list(block_size):
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
    """Same per-block heuristic as ContextGradientGBTICL -- used only for the
    zoom panel, where drawing an explicit node/edge graph is still readable."""
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
    Vectorized equivalent of context_gradient_weights, applied to every block
    boundary in the image at once.

    Returns two (H, W) float arrays, mostly NaN (no edge there), with real
    weight values only on the pixel-pairs that correspond to an actual
    movable graph edge:
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

    # rows that are the last row of some block (these feed top_ctx for the
    # block below); skip row -1 (no block above the first block row)
    boundary_rows = np.arange(block_size - 1, h - block_size, block_size)
    boundary_cols = np.arange(block_size - 1, w - block_size, block_size)

    # horizontal gradient along every boundary row, for all columns at once
    if len(boundary_rows) > 0:
        row_strip = gray[boundary_rows, :]              # (n_rows, W)
        hgrad = np.abs(np.diff(row_strip, axis=1))       # (n_rows, W-1)
        hw = np.exp(-sensitivity * hgrad)
        h_weight[np.ix_(boundary_rows, np.arange(w - 1))] = hw

    # vertical gradient along every boundary column, for all rows at once
    if len(boundary_cols) > 0:
        col_strip = gray[:, boundary_cols]                # (H, n_cols)
        vgrad = np.abs(np.diff(col_strip, axis=0))         # (H-1, n_cols)
        vw = np.exp(-sensitivity * vgrad)
        v_weight[np.ix_(np.arange(h - 1), boundary_cols)] = vw

    return h_weight, v_weight


def make_heatmap_figure(img, gray):
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
    """Find the highest-variance ZOOM_BLOCKS x ZOOM_BLOCKS patch (in block
    units, offset by 1 block so every block inside has real top/left context)."""
    h, w = img.shape[:2]
    n_bh, n_bw = h // block_size, w // block_size
    best, best_var = (1, 1), -1.0
    step = max(1, patch_blocks // 2)  # coarse scan, this is just picking an example region
    for bi in range(1, n_bh - patch_blocks, step):
        for bj in range(1, n_bw - patch_blocks, step):
            r0, c0 = bi * block_size, bj * block_size
            r1, c1 = r0 + patch_blocks * block_size, c0 + patch_blocks * block_size
            v = img[r0:r1, c0:c1, :].astype(float).var()
            if v > best_var:
                best_var, best = v, (bi, bj)
    return best, best_var


def make_zoom_figure(img):
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
            block = img[r0b:r0b + BLOCK_SIZE, c0b:c0b + BLOCK_SIZE, :]
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
