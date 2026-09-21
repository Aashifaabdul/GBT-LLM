"""
GBT-ICL: prediction of graph-Laplacian edge weights for a block.

All models predict one weight per edge of the 4-connected grid graph over a
block (112 edges for 8x8), given only causally decoded data, so encoder and
decoder obtain the same graph without transmitting it.

  - UniformGBTICL: all weights 1 (this gives a fixed grid-graph transform).
  - ContextGradientGBTICL: heuristic, down-weights border edges across
    brightness jumps seen in the decoded context.
  - GBTICLNet: MLP mapping the block's own context to edge weights
    (non-meta-learning ablation baseline).
  - GBTICLMetaLearner: the main model; attends over a support set of
    already-decoded blocks paired with reference graphs computed from them.

All models subclass nn.Module.
"""

import torch
import torch.nn as nn


def edge_list(block_size):
    """Edges of the 4-connected grid graph over a block_size x block_size block.

    Node index for pixel (r, c) is r * block_size + c. Returns a list of
    (node_i, node_j) tuples; for block_size=8 there are 56 horizontal and
    56 vertical edges (112 in total).
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


def reference_edge_weights(block, block_size, edge_sensitivity=0.15):
    """
    Closed-form reference graph for a fully known block.

    Each edge weight is exp(-edge_sensitivity * |gray_i - gray_j|), the usual
    pixel-similarity rule: edges across a sharp intensity jump are weakened.
    Used to label the support blocks of GBTICLMetaLearner; those blocks are
    already decoded, so encoder and decoder compute identical labels.

    Args:
        block: (block_size, block_size, 3) tensor of pixel values
        block_size: int
        edge_sensitivity: decay rate of the weight with intensity difference

    Returns:
        (n_edges,) float64 tensor on the device of `block`
    """
    device = block.device
    edges = edge_list(block_size)
    gray = block.to(torch.float64).mean(dim=-1).reshape(-1)  # (n,)
    idx_i = torch.tensor([e[0] for e in edges], device=device, dtype=torch.long)
    idx_j = torch.tensor([e[1] for e in edges], device=device, dtype=torch.long)
    diff = torch.abs(gray[idx_i] - gray[idx_j])
    return torch.exp(-edge_sensitivity * diff)


def reference_edge_weights_batch(blocks, block_size, edge_sensitivity=0.15):
    """Batched reference_edge_weights: (B, bs, bs, 3) blocks -> (B, n_edges) weights."""
    device = blocks.device
    edges = edge_list(block_size)
    b = blocks.shape[0]
    gray = blocks.to(torch.float64).mean(dim=-1).reshape(b, -1)  # (B, n)
    idx_i = torch.tensor([e[0] for e in edges], device=device, dtype=torch.long)
    idx_j = torch.tensor([e[1] for e in edges], device=device, dtype=torch.long)
    diff = torch.abs(gray[:, idx_i] - gray[:, idx_j])
    return torch.exp(-edge_sensitivity * diff)


class GBTICLPredictor(nn.Module):
    """Interface shared by all edge-weight predictors."""

    def predict_edge_weights(self, top_ctx, left_ctx, valid_top, valid_left, block_size, support=None):
        """
        Args:
            top_ctx:  (block_size, 3) uint8 bottom row of the block above (or padding)
            left_ctx: (block_size, 3) uint8 right column of the block to the left (or padding)
            valid_top, valid_left: whether that context is real or padding
            block_size: int
            support: optional support set from context.get_support_set
                (used by GBTICLMetaLearner only)

        Returns:
            (n_edges,) float64 tensor, one weight per edge of edge_list(block_size).
        """
        raise NotImplementedError


class UniformGBTICL(GBTICLPredictor):
    """Weight 1 on every edge (untrained baseline)."""

    def __init__(self):
        super().__init__()

        # A parameter-free module needs a buffer so .to(device) has something to move
        self.register_buffer("_device_anchor", torch.zeros(1))

    def predict_edge_weights(self, top_ctx, left_ctx, valid_top, valid_left, block_size, support=None):
        device = self._device_anchor.device
        n_edges = len(edge_list(block_size))
        return torch.ones(n_edges, dtype=torch.float64, device=device)


class ContextGradientGBTICL(GBTICLPredictor):
    """
    Non-learned, context-sensitive baseline.

    Edges along the first row/column of the block are down-weighted by
    exp(-edge_sensitivity * jump), where jump is the brightness step between the
    same two pixels in the decoded context; all other edges keep weight 1.
    """

    def __init__(self, edge_sensitivity=0.15):
        super().__init__()
        self.edge_sensitivity = edge_sensitivity
        self.register_buffer("_device_anchor", torch.zeros(1))

    def predict_edge_weights(self, top_ctx, left_ctx, valid_top, valid_left, block_size, support=None):
        device = self._device_anchor.device
        edges = edge_list(block_size)
        weights = torch.ones(len(edges), dtype=torch.float64, device=device)

        top_gray = top_ctx.to(torch.float64).mean(dim=-1) if valid_top else None
        left_gray = left_ctx.to(torch.float64).mean(dim=-1) if valid_left else None

        for idx, (i, j) in enumerate(edges):
            ri, ci = divmod(i, block_size)
            rj, cj = divmod(j, block_size)

            # only the first row/column has context to compare against
            if ri == 0 and rj == 0 and top_gray is not None and abs(ci - cj) == 1:
                jump = torch.abs(top_gray[ci] - top_gray[cj])
                weights[idx] = torch.exp(-self.edge_sensitivity * jump)
            elif ci == 0 and cj == 0 and left_gray is not None and abs(ri - rj) == 1:
                jump = torch.abs(left_gray[ri] - left_gray[rj])
                weights[idx] = torch.exp(-self.edge_sensitivity * jump)
        return weights


def context_features(top_ctx, left_ctx, valid_top, valid_left, block_size, device, dtype=torch.float32):
    """
    Context feature vector shared by GBTICLNet and GBTICLMetaLearner.

    Layout: top and left context values scaled to [0, 1] (2 * bs * 3), their
    finite-difference gradients along the grey-level profile (2 * (bs - 1)) and
    the two validity flags (2).

    Args:
        top_ctx, left_ctx: (B, block_size, 3) tensors (any numeric dtype)
        valid_top, valid_left: (B,) bool tensors, or plain bools (batch of 1)
    Returns:
        (B, in_dim) float tensor, in_dim = 2*block_size*3 + 2*(block_size-1) + 2
    """
    top = top_ctx.to(device=device, dtype=dtype) / 255.0  # (B, bs, 3)
    left = left_ctx.to(device=device, dtype=dtype) / 255.0  # (B, bs, 3)
    b = top.shape[0]
    top_gray = top.mean(dim=-1)  # (B, bs)
    left_gray = left.mean(dim=-1)  # (B, bs)
    top_grad = top_gray[:, 1:] - top_gray[:, :-1]  # (B, bs-1)
    left_grad = left_gray[:, 1:] - left_gray[:, :-1]  # (B, bs-1)

    def _flags(v):
        if torch.is_tensor(v):
            return v.to(device=device, dtype=dtype).reshape(b, 1)
        return torch.full((b, 1), 1.0 if v else 0.0, device=device, dtype=dtype)

    flags = torch.cat([_flags(valid_top), _flags(valid_left)], dim=1)  # (B, 2)
    return torch.cat(
        [top.reshape(b, -1), left.reshape(b, -1), top_grad, left_grad, flags], dim=1
    )


def context_feature_dim(block_size):
    """Length of the vector returned by context_features."""
    return 2 * block_size * 3 + 2 * (block_size - 1) + 2


class GBTICLNet(GBTICLPredictor):
    """
    MLP that maps the decoded context of one block directly to its edge weights.

    Input is the context_features vector (64 features for 8x8 blocks); output is
    one logit per edge, converted to a positive weight by to_weights. forward()
    is the batched entry point used for training; predict_edge_weights() is the
    single-block entry point used by the codec.
    """

    def __init__(self, block_size=8, hidden=96, n_layers=2):
        super().__init__()
        self.block_size = block_size
        n_edges = len(edge_list(block_size))
        in_dim = context_feature_dim(block_size)

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
        """Feature vectors for a batch of contexts (see context_features)."""
        return context_features(top_ctx, left_ctx, valid_top, valid_left, self.block_size, device, dtype)

    def forward(self, features_batch):
        """(B, in_dim) features -> (B, n_edges) logits; apply to_weights for edge weights."""
        return self.net(features_batch)

    @staticmethod
    def to_weights(logits):
        """Map logits to strictly positive edge weights, exp(clamp(logits, -8, 3)).

        The lower clamp keeps every weight above zero so no node becomes isolated
        in the Laplacian; exp (rather than sigmoid) lets an edge exceed weight 1.
        """
        return torch.exp(torch.clamp(logits, min=-8.0, max=3.0))

    def predict_edge_weights(self, top_ctx, left_ctx, valid_top, valid_left, block_size, support=None):
        assert block_size == self.block_size, (
            f"GBTICLNet was built for block_size={self.block_size}, got {block_size}"
        )
        device = next(self.parameters()).device
        feats = self.features_batch(
            top_ctx.unsqueeze(0), left_ctx.unsqueeze(0), valid_top, valid_left, device
        )
        logits = self.forward(feats).squeeze(0)
        return self.to_weights(logits).to(torch.float64)


class GBTICLMetaLearner(GBTICLPredictor):
    """
    In-context meta-learner for edge weights (the main GBT-ICL model).

    The edge weights of the query block are predicted by cross-attention over a
    support set of already-decoded blocks (context.get_support_set: up to 4
    spatial neighbours in the current frame and 4 temporal neighbours in the
    previous frame). Each support item is a (context features, reference graph)
    pair, where the reference graph is computed by reference_edge_weights from
    the support block's own decoded pixels. No weights are updated at inference;
    the support set is the only conditioning, and the model is meta-trained on
    many (query, support) episodes (see training.py).

    Missing support slots (for example the first block of a frame) are replaced
    by a learned "missing support" embedding. predict_edge_weights(support=None)
    uses an all-missing support set, so the model also works in the single-image
    path.
    """

    def __init__(self, block_size=8, d_model=64, n_heads=4, hidden=96):
        super().__init__()
        self.block_size = block_size
        self.n_edges = len(edge_list(block_size))
        ctx_dim = context_feature_dim(block_size)
        self.ctx_dim = ctx_dim
        self.d_model = d_model

        self.query_proj = nn.Sequential(
            nn.Linear(ctx_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.support_ctx_proj = nn.Sequential(
            nn.Linear(ctx_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.support_weight_proj = nn.Sequential(
            nn.Linear(self.n_edges, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

        # Learned embedding that stands in for an unavailable support item
        self.missing_support = nn.Parameter(torch.zeros(d_model))

        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.out_head = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, self.n_edges))

    def forward(self, query_feats, support_ctx_feats, support_weights, support_valid):
        """Batched forward pass used for meta-training.

        Args:
            query_feats:       (B, ctx_dim)
            support_ctx_feats: (B, K, ctx_dim)
            support_weights:   (B, K, n_edges) reference graphs of the support blocks
            support_valid:     (B, K) bool, False for missing support slots

        Returns:
            (B, n_edges) logits; apply to_weights for edge weights.
        """
        # Embed the query context and support examples
        q = self.query_proj(query_feats).unsqueeze(1)  # (B, 1, d)
        s_ctx = self.support_ctx_proj(support_ctx_feats)  # (B, K, d)
        s_w = self.support_weight_proj(support_weights.to(s_ctx.dtype))  # (B, K, d)
        kv = s_ctx + s_w  # (B, K, d)

        # Replace unavailable support items with a learned embedding
        missing = self.missing_support.to(kv.dtype)
        valid_mask = support_valid.unsqueeze(-1)  # (B, K, 1)

        # Missing slots get the learned embedding instead of being masked out,
        # so the model can use "no support here" as a signal
        kv = torch.where(valid_mask, kv, missing.expand_as(kv))

        # Attend to support examples and predict graph-edge weights
        attn_out, _ = self.attn(q, kv, kv)  # (B, 1, d)
        return self.out_head(attn_out.squeeze(1))  # (B, n_edges)

    @staticmethod
    def to_weights(logits):
        """Same mapping as GBTICLNet.to_weights."""
        return torch.exp(torch.clamp(logits, min=-8.0, max=3.0))

    def _empty_support(self, device):
        k = 8  # equals context.N_SUPPORT (literal to avoid importing context here)
        return dict(
            top=torch.full((k, self.block_size, 3), 128, dtype=torch.uint8, device=device),
            left=torch.full((k, self.block_size, 3), 128, dtype=torch.uint8, device=device),
            valid_top=torch.zeros(k, dtype=torch.bool, device=device),
            valid_left=torch.zeros(k, dtype=torch.bool, device=device),
            block=torch.full((k, self.block_size, self.block_size, 3), 128, dtype=torch.uint8, device=device),
            valid=torch.zeros(k, dtype=torch.bool, device=device),
        )

    def predict_edge_weights(self, top_ctx, left_ctx, valid_top, valid_left, block_size, support=None):
        assert block_size == self.block_size, (
            f"GBTICLMetaLearner was built for block_size={self.block_size}, got {block_size}"
        )
        device = next(self.parameters()).device
        if support is None:
            support = self._empty_support(device)

        # Convert query context into model features
        query_feats = context_features(
            top_ctx.unsqueeze(0), left_ctx.unsqueeze(0), valid_top, valid_left, block_size, device
        )  # (1, ctx_dim)

        support_ctx_feats = context_features(
            support["top"], support["left"], support["valid_top"], support["valid_left"],
            block_size, device,
        ).unsqueeze(0)  # (1, K, ctx_dim)

        # Derive reference graphs from decoded support blocks
        support_weights = reference_edge_weights_batch(
            support["block"].to(device), block_size
        ).unsqueeze(0)  # (1, K, n_edges)

        support_valid = support["valid"].to(device).unsqueeze(0)  # (1, K)

        # Predict positive weights for all graph edges
        logits = self.forward(query_feats, support_ctx_feats, support_weights, support_valid).squeeze(0)
        return self.to_weights(logits).to(torch.float64)
