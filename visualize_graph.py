"""
Visualize the actual predicted graph (nodes + weighted edges) for one real
block from the dataset, overlaid on the block's pixels, next to the context
it was derived from. Pure numpy re-implementation of ContextGradientGBTICL's
logic (torch isn't installable in this sandbox) -- identical math.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image

BLOCK_SIZE = 8
EDGE_SENSITIVITY = 0.15


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


# ---- load the real frame and grab the high-variance block we found ----
img = np.array(Image.open(
    "/sessions/great-upbeat-mccarthy/mnt/Dataset/gbticl_dataset/Beauty/frames/frame0000.png"
).convert("RGB"))

bi, bj = 124, 233  # block indices found earlier
r0, c0 = bi * BLOCK_SIZE, bj * BLOCK_SIZE

block = img[r0:r0 + BLOCK_SIZE, c0:c0 + BLOCK_SIZE, :]
top_ctx = img[r0 - 1, c0:c0 + BLOCK_SIZE, :]           # bottom row of block above
left_ctx = img[r0:r0 + BLOCK_SIZE, c0 - 1, :]          # right col of block to the left

weights, edges = context_gradient_weights(top_ctx, left_ctx, BLOCK_SIZE)
uniform_weights = np.ones(len(edges))

print(f"block ({bi},{bj}), pixel origin ({r0},{c0}), variance={block.astype(float).var():.1f}")
print(f"weights: min={weights.min():.3f}, max={weights.max():.3f}, "
      f"n_below_0.5={np.sum(weights < 0.5)} of {len(weights)} edges "
      f"(only the 14 border-touching edges can ever move)")

# ---- figure ----
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

# panel 1: the block itself, with context strips shown attached
canvas = np.full((BLOCK_SIZE + 1, BLOCK_SIZE + 1, 3), 255, dtype=np.uint8)
canvas[1:, 1:] = block
canvas[0, 1:] = top_ctx
canvas[1:, 0] = left_ctx
axes[0].imshow(canvas, interpolation="nearest")
axes[0].axhline(0.5, color="red", linewidth=1.5)
axes[0].axvline(0.5, color="red", linewidth=1.5)
axes[0].set_title("Block (bottom-right 8x8)\n+ context strips (red line = boundary)", fontsize=10)
axes[0].set_xticks([]); axes[0].set_yticks([])

def draw_graph(ax, block_img, edges, weights, title):
    ax.imshow(block_img, interpolation="nearest", extent=(-0.5, BLOCK_SIZE - 0.5, BLOCK_SIZE - 0.5, -0.5))
    node_xy = np.array([[c, r] for r in range(BLOCK_SIZE) for c in range(BLOCK_SIZE)])

    segs, colors, lws = [], [], []
    for (i, j), w in zip(edges, weights):
        p1, p2 = node_xy[i], node_xy[j]
        segs.append([p1, p2])
        colors.append(plt.cm.RdYlGn(w))   # red = weak/near-0, green = strong/near-1
        lws.append(0.5 + 3.5 * w)

    lc = LineCollection(segs, colors=colors, linewidths=lws)
    ax.add_collection(lc)
    ax.scatter(node_xy[:, 0], node_xy[:, 1], s=14, color="black", zorder=3)
    ax.set_xlim(-0.7, BLOCK_SIZE - 0.3)
    ax.set_ylim(BLOCK_SIZE - 0.3, -0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

draw_graph(axes[1], block, edges, uniform_weights, "UniformGBTICL\nevery edge weight = 1.0")
draw_graph(axes[2], block, edges, weights, "ContextGradientGBTICL\ngreen=strong (smooth), red=weak (likely edge)")

fig.suptitle(f"Predicted graph on a real block -- Beauty frame0000, block ({bi},{bj})", fontsize=12)
fig.tight_layout()
fig.savefig("/sessions/great-upbeat-mccarthy/mnt/outputs/graph_visualization.png", dpi=140, bbox_inches="tight")
print("saved graph_visualization.png")
