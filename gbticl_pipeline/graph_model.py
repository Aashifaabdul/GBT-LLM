"""
GBT-ICL: the in-context graph-Laplacian predictor.

STATUS: three implementations, in increasing order of capability.
  - UniformGBTICL / ContextGradientGBTICL: non-learned stand-ins (see their
    own docstrings). Kept as baselines/ablations -- your results should show
    the trained model beating both of these, not just existing.
  - GBTICLNet: the actual trainable model. A small MLP over the decoded
    context (top row + left column, both channels and a coarse gradient
    summary), predicting all 112 edge weights in one forward pass. This is
    genuinely trained via training.py -- nothing about it is hand-tuned.

All three subclass nn.Module so `.to(device)` and (for GBTICLNet) normal
optimizer/checkpoint machinery just work identically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def edge_list(block_size):
    """4-connected grid graph over a block_size x block_size block.

    Node index for pixel (r, c) is r * block_size + c.
    Returns a list of (node_i, node_j) tuples, one per edge.
    For block_size=8: 56 horizontal + 56 vertical = 112 edges.
    This is static index bookkeeping, not numerical work -- deliberately kept
    as plain Python/no tensors, it's the same on every device.
    """
    edges = []
    for r in range(block_size):
        for c in range(block_size):
            node = r * block_size + c
            if c + 1 < block_size:  # horizontal neighbour
                edges.append((node, node + 1))
            if r + 1 < block_size:  # vertical neighbour
                edges.append((node, node + block_size))
    return edges


class GBTICLPredictor(nn.Module):
    """Interface every GBT-ICL implementation (placeholder or trained) must satisfy."""

    def predict_edge_weights(self, top_ctx, left_ctx, valid_top, valid_left, block_size):
        """
        Args:
            top_ctx:  (block_size, 3) uint8 tensor, on this module's device --
                      bottom row of the block above (or padding)
            left_ctx: (block_size, 3) uint8 tensor, on this module's device --
                      right column of the block to the left (or padding)
            valid_top, valid_left: bool -- whether that context is real or padded
            block_size: int

        Returns:
            weights: (n_edges,) float64 tensor, on this module's device, one
                     weight per edge from edge_list(block_size), same order.
        """
        raise NotImplementedError


class UniformGBTICL(GBTICLPredictor):
    """Placeholder: equal weight everywhere (untrained baseline). See module docstring."""

    def __init__(self):
        super().__init__()
        # a parameter-free module still needs something for .to(device) to act on,
        # and it doubles as "where am I currently living" for the placeholder logic
        self.register_buffer("_device_anchor", torch.zeros(1))

    def predict_edge_weights(self, top_ctx, left_ctx, valid_top, valid_left, block_size):
        device = self._device_anchor.device
        n_edges = len(edge_list(block_size))
        return torch.ones(n_edges, dtype=torch.float64, device=device)


class ContextGradientGBTICL(GBTICLPredictor):
    """
    A second, still-non-learned placeholder that is at least context-sensitive,
    for sanity-checking the pipeline responds to context before a real model exists.

    Heuristic only (not the research contribution): edges near a border with a
    sharp brightness jump in the available context get down-weighted (mimicking
    "don't smooth across an edge"); everywhere else defaults to uniform weight.
    This is deliberately simple and is expected to be replaced.
    """

    def __init__(self, edge_sensitivity=0.15):
        super().__init__()
        self.edge_sensitivity = edge_sensitivity
        self.register_buffer("_device_anchor", torch.zeros(1))

    def predict_edge_weights(self, top_ctx, left_ctx, valid_top, valid_left, block_size):
        device = self._device_anchor.device
        edges = edge_list(block_size)
        weights = torch.ones(len(edges), dtype=torch.float64, device=device)

        top_gray = top_ctx.to(torch.float64).mean(dim=-1) if valid_top else None
        left_gray = left_ctx.to(torch.float64).mean(dim=-1) if valid_left else None

        for idx, (i, j) in enumerate(edges):
            ri, ci = divmod(i, block_size)
            rj, cj = divmod(j, block_size)
            # only down-weight edges that touch the first row/col, where we have
            # a real signal about a boundary discontinuity from context
            if ri == 0 and rj == 0 and top_gray is not None and abs(ci - cj) == 1:
                jump = torch.abs(top_gray[ci] - top_gray[cj])
                weights[idx] = torch.exp(-self.edge_sensitivity * jump)
            elif ci == 0 and cj == 0 and left_gray is not None and abs(ri - rj) == 1:
                jump = torch.abs(left_gray[ri] - left_gray[rj])
                weights[idx] = torch.exp(-self.edge_sensitivity * jump)
        return weights


class GBTICLNet(GBTICLPredictor):
    """
    The real, trainable GBT-ICL model. Predicts all n_edges edge weights in
    one forward pass from the decoded context, via a small MLP -- no hand
    tuning, no formula, weights are learned end to end by training.py.

    Input features per call, built from `top_ctx`/`left_ctx` (each
    (block_size, 3) uint8, or the (block_size,3) zero-padding tensor used
    for border blocks):
      - top_ctx, left_ctx flattened and normalised to [0,1]      -> 2*block_size*3
      - top_ctx / left_ctx local gradients (finite differences)  -> 2*(block_size-1)
      - valid_top, valid_left flags (0/1)                        -> 2
    For block_size=8 that's 48 + 14 + 2 = 64 input features -- small on
    purpose, since the whole point of a fixed 4-connected topology is that
    the prediction target (112 numbers) doesn't need a heavy model.

    forward() is the batched entry point used by training.py (operates on a
    batch of contexts at once, shape (B, feat_dim) -> (B, n_edges)).
    predict_edge_weights() is the single-context entry point the codec loop
    (encode_image/decode_image) actually calls, batch size 1.
    """

    def __init__(self, block_size=8, hidden=96, n_layers=2):
        super().__init__()
        self.block_size = block_size
        n_edges = len(edge_list(block_size))
        in_dim = 2 * block_size * 3 + 2 * (block_size - 1) + 2

        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers += [nn.Linear(d, n_edges)]
        self.net = nn.Sequential(*layers)
        self.in_dim = in_dim
        self.n_edges = n_edges

    def features_batch(self, top_ctx, left_ctx, valid_top, valid_left, device, dtype=torch.float32):
        """The single feature-extraction implementation used both by training
        (real batches, from training.py's DataLoader) and by inference
        (predict_edge_weights, batch size 1) -- kept as one code path so the
        two can never architecturally drift apart.

        Args:
            top_ctx, left_ctx: (B, block_size, 3) tensors (any numeric dtype)
            valid_top, valid_left: (B,) bool tensors (or plain Python bools,
                broadcast to a batch of 1)
        Returns:
            (B, in_dim) float tensor
        """
        bs = self.block_size
        top = top_ctx.to(device=device, dtype=dtype) / 255.0    # (B, bs, 3)
        left = left_ctx.to(device=device, dtype=dtype) / 255.0  # (B, bs, 3)
        b = top.shape[0]
        top_gray = top.mean(dim=-1)    # (B, bs)
        left_gray = left.mean(dim=-1)  # (B, bs)
        top_grad = top_gray[:, 1:] - top_gray[:, :-1]     # (B, bs-1)
        left_grad = left_gray[:, 1:] - left_gray[:, :-1]  # (B, bs-1)

        def _flags(v):
            if torch.is_tensor(v):
                return v.to(device=device, dtype=dtype).reshape(b, 1)
            return torch.full((b, 1), 1.0 if v else 0.0, device=device, dtype=dtype)

        flags = torch.cat([_flags(valid_top), _flags(valid_left)], dim=1)  # (B, 2)
        return torch.cat(
            [top.reshape(b, -1), left.reshape(b, -1), top_grad, left_grad, flags], dim=1
        )

    def forward(self, features_batch):
        """features_batch: (B, in_dim) -> raw logits (B, n_edges). Squashed to
        (0, 1] with an exponential (never exactly 0, so the Laplacian never
        becomes singular in a way that breaks eigendecomposition on an
        isolated node) via `to_weights`."""
        return self.net(features_batch)

    @staticmethod
    def to_weights(logits):
        """Map unconstrained network output to valid, positive edge weights.
        exp() rather than sigmoid: unbounded above (an edge can be predicted
        *stronger* than a default 1.0, not just weakened), always > 0."""
        return torch.exp(torch.clamp(logits, min=-8.0, max=3.0))

    def predict_edge_weights(self, top_ctx, left_ctx, valid_top, valid_left, block_size):
        assert block_size == self.block_size, (
            f"GBTICLNet was built for block_size={self.block_size}, got {block_size}"
        )
        device = next(self.parameters()).device
        feats = self.features_batch(
            top_ctx.unsqueeze(0), left_ctx.unsqueeze(0), valid_top, valid_left, device
        )
        logits = self.forward(feats).squeeze(0)
        return self.to_weights(logits).to(torch.float64)
