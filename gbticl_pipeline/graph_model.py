"""
GBT-ICL: the in-context graph-Laplacian predictor.

STATUS: four implementations, in increasing order of capability.
  - UniformGBTICL / ContextGradientGBTICL: non-learned stand-ins (see their
    own docstrings). Kept as baselines/ablations -- your results should show
    the trained models beating both of these, not just existing.
  - GBTICLNet: a trainable MLP over the decoded context (top row + left
    column, both channels and a coarse gradient summary), predicting all 112
    edge weights in one forward pass. Genuinely trained via training.py.
    Kept as an explicit ablation baseline ("context-conditional regressor,
    non-meta-learning") against GBTICLMetaLearner below.
  - GBTICLMetaLearner: the primary GBT-ICL model. A genuine few-shot
    in-context meta-learner -- see its own docstring for the full design.

All four subclass nn.Module so `.to(device)` and normal optimizer/checkpoint
machinery just work identically.
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


def reference_edge_weights(block, block_size, edge_sensitivity=0.15):
    """
    Deterministic, closed-form 'target' graph for an ALREADY-DECODED block,
    computed directly from its true known pixels -- the classical GBT
    graph-construction rule (Gaussian/exponential pixel-similarity kernel
    over the 4-connected grid: edges spanning a sharp brightness jump get
    down-weighted, "don't smooth across an edge").

    This is used only to build support-set labels for GBTICLMetaLearner
    (below): every already-decoded block's true pixels are known to BOTH
    encoder and decoder, so this function's output is identically
    reproducible on both sides without transmitting anything. It is
    distinct from ContextGradientGBTICL, which only has access to a query
    block's *border* context (the block itself isn't decoded yet) and so
    can only estimate weights for edges touching that border -- here, the
    whole block is known, so every one of the n_edges edges gets a real,
    non-default estimate.

    Args:
        block: (block_size, block_size, 3) tensor (any numeric dtype), the
               support block's true/reconstructed pixels
        block_size: int
        edge_sensitivity: same role/units as ContextGradientGBTICL's

    Returns:
        weights: (n_edges,) float64 tensor, same device as `block`
    """
    device = block.device
    edges = edge_list(block_size)
    gray = block.to(torch.float64).mean(dim=-1).reshape(-1)  # (n,)
    idx_i = torch.tensor([e[0] for e in edges], device=device, dtype=torch.long)
    idx_j = torch.tensor([e[1] for e in edges], device=device, dtype=torch.long)
    diff = torch.abs(gray[idx_i] - gray[idx_j])
    return torch.exp(-edge_sensitivity * diff)


def reference_edge_weights_batch(blocks, block_size, edge_sensitivity=0.15):
    """Batched version of reference_edge_weights: blocks (B, bs, bs, 3) -> (B, n_edges)."""
    device = blocks.device
    edges = edge_list(block_size)
    b = blocks.shape[0]
    gray = blocks.to(torch.float64).mean(dim=-1).reshape(b, -1)  # (B, n)
    idx_i = torch.tensor([e[0] for e in edges], device=device, dtype=torch.long)
    idx_j = torch.tensor([e[1] for e in edges], device=device, dtype=torch.long)
    diff = torch.abs(gray[:, idx_i] - gray[:, idx_j])
    return torch.exp(-edge_sensitivity * diff)


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


def context_features(top_ctx, left_ctx, valid_top, valid_left, block_size, device, dtype=torch.float32):
    """The single context-feature-extraction implementation shared by
    GBTICLNet and GBTICLMetaLearner (both a query block's own context and,
    for the meta-learner, every support-set item's context go through this
    same function) -- so the two models can never architecturally drift
    apart on what "context" means. See GBTICLNet's docstring for the
    feature layout. Batched: (B, block_size, 3) -> (B, in_dim).

    Args:
        top_ctx, left_ctx: (B, block_size, 3) tensors (any numeric dtype)
        valid_top, valid_left: (B,) bool tensors (or plain Python bools,
            broadcast to a batch of 1)
    Returns:
        (B, in_dim) float tensor, in_dim = 2*block_size*3 + 2*(block_size-1) + 2
    """
    bs = block_size
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


def context_feature_dim(block_size):
    return 2 * block_size * 3 + 2 * (block_size - 1) + 2


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
        """The feature-extraction implementation used both by training (real
        batches, from training.py's DataLoader) and by inference
        (predict_edge_weights, batch size 1). Delegates to the module-level
        context_features() -- the same function GBTICLMetaLearner uses for
        every support-set item's context -- so no two models in this file
        can architecturally drift apart on what "context" means.
        """
        return context_features(top_ctx, left_ctx, valid_top, valid_left, self.block_size, device, dtype)

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


class GBTICLMetaLearner(GBTICLPredictor):
    """
    The primary GBT-ICL model: a genuine few-shot IN-CONTEXT meta-learner,
    not a context-conditional regressor (that's what GBTICLNet above is --
    kept as an explicit ablation baseline for exactly this comparison).

    THE KEY DIFFERENCE FROM GBTICLNet: GBTICLNet maps one block's own
    context straight to edge weights -- at inference it never looks at any
    other block, so there is no actual "learning from examples in context,"
    just a fixed function of local pixels. This model instead predicts the
    query block's edge weights by attending over a SUPPORT SET of other
    already-decoded blocks (context.py's get_support_set: up to 4 spatial
    neighbours in the current frame + up to 4 temporal neighbours in the
    previous frame), each paired with a REFERENCE edge-weight graph computed
    in closed form directly from that support block's own true pixels
    (reference_edge_weights(), above) -- exactly analogous to a few-shot
    prompt: "here are K examples of (local context -> good graph) pairs
    from content you've already seen; predict the graph for this new query
    context." No gradient updates happen at inference -- the support set
    itself IS the in-context conditioning, the same way an LLM conditions
    on a prompt without any weight update. The model is meta-trained (see
    training.py's episodic sampling) across many (query, support-set) pairs
    so it generalises to new image/video content purely through this
    in-context conditioning at test time, never fine-tuned per test
    sequence.

    Architecture: cross-attention (nn.MultiheadAttention, 1 layer) with the
    query block's own context as the attention query, and the support set's
    (context embedding + reference-weight embedding) as keys/values.
    Missing support slots (context.py marks these with valid=False -- e.g.
    the very first block of the very first frame, which has neither spatial
    nor temporal neighbours) are replaced with a learned "missing support"
    embedding, so the model degrades gracefully rather than needing special-
    cased code paths for partial support sets.

    BACKWARD COMPATIBLE call signature: predict_edge_weights() accepts an
    optional `support` argument (context.py's get_support_set() output). If
    omitted (support=None), an all-missing support set is synthesised
    internally, so this class is also a drop-in for the plain
    encode_image/decode_image path (codec.py) with graceful degradation to
    "no in-context evidence, use the learned fallback" -- exactly the
    intra-frame, first-block-of-a-sequence case anyway. encode_video/
    decode_video (below) are what actually supply a real support set.
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
        # learned "no support item here" fallback embedding (same trick as
        # TinyTransformerCoeffModel's `bos` and HFLoRACoeffModel's `bos`)
        self.missing_support = nn.Parameter(torch.zeros(d_model))

        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.out_head = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, self.n_edges))

    def forward(self, query_feats, support_ctx_feats, support_weights, support_valid):
        """Batched entry point used by training.py's episodic meta-training.

        Args:
            query_feats:        (B, ctx_dim) float
            support_ctx_feats:  (B, K, ctx_dim) float
            support_weights:    (B, K, n_edges) float -- reference weights
                                 from reference_edge_weights_batch(), or any
                                 placeholder value for invalid slots (masked
                                 out below regardless of its actual content)
            support_valid:      (B, K) bool

        Returns:
            logits: (B, n_edges) -- pass through to_weights() for valid
                    positive edge weights, same convention as GBTICLNet
        """
        q = self.query_proj(query_feats).unsqueeze(1)         # (B, 1, d)
        s_ctx = self.support_ctx_proj(support_ctx_feats)      # (B, K, d)
        s_w = self.support_weight_proj(support_weights.to(s_ctx.dtype))  # (B, K, d)
        kv = s_ctx + s_w                                       # (B, K, d)

        missing = self.missing_support.to(kv.dtype)
        valid_mask = support_valid.unsqueeze(-1)                # (B, K, 1)
        # soft substitution (not a hard attention mask): missing slots carry
        # the learned fallback embedding instead of being excluded outright,
        # so the model can itself learn what "no support was available"
        # means as a signal, and every query keeps a fixed-shape (B, K, d)
        # key/value tensor regardless of how many slots are actually valid
        kv = torch.where(valid_mask, kv, missing.expand_as(kv))

        attn_out, _ = self.attn(q, kv, kv)   # (B, 1, d)
        return self.out_head(attn_out.squeeze(1))  # (B, n_edges)

    @staticmethod
    def to_weights(logits):
        return torch.exp(torch.clamp(logits, min=-8.0, max=3.0))

    def _empty_support(self, device):
        k = 8  # matches context.N_SUPPORT; duplicated as a literal to avoid
               # a context.py -> graph_model.py import (context.py already
               # imports nothing from here, keep it that way)
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

        query_feats = context_features(
            top_ctx.unsqueeze(0), left_ctx.unsqueeze(0), valid_top, valid_left, block_size, device
        )  # (1, ctx_dim)

        k = support["top"].shape[0]
        support_ctx_feats = context_features(
            support["top"], support["left"], support["valid_top"], support["valid_left"],
            block_size, device,
        ).unsqueeze(0)  # (1, K, ctx_dim)

        support_weights = reference_edge_weights_batch(
            support["block"].to(device), block_size
        ).unsqueeze(0)  # (1, K, n_edges)

        support_valid = support["valid"].to(device).unsqueeze(0)  # (1, K)

        logits = self.forward(query_feats, support_ctx_feats, support_weights, support_valid).squeeze(0)
        return self.to_weights(logits).to(torch.float64)
