"""
LLM Coefficient Predictor: models p(coefficient) for entropy coding, conditioned
on the full eigenvalue spectrum Λ from GBT-ICL's eigendecomposition plus the
already-decoded coefficient history within this block/channel.

STATUS: three implementations, in increasing order of capability.
  - LaplaceCoeffModel: non-learned classical stand-in (see its own docstring).
    Kept as a baseline -- enough to drive the real range coder end to end and
    get honest bit costs, and something your trained model needs to beat.
  - TinyTransformerCoeffModel: the actual trainable "LLM-style" model -- a
    small causal transformer, built from scratch (no external weights/
    internet needed), trained via training.py.
  - HFLoRACoeffModel: an optional wrapper around a real pretrained causal LM
    (HuggingFace `transformers`) fine-tuned with LoRA (`peft`) -- closer to
    the original LLaMA-3 + LoRA design. Needs `pip install transformers peft`,
    a model download, and meaningfully more GPU memory/time than
    TinyTransformerCoeffModel. Provided as a documented extension path; not
    exercised in this environment (no internet/GPU here -- see README).

INTERFACE CHANGE from the earlier placeholder-only version: `symbol_probs`
now takes the *full* eigenvalue spectrum `eigvals` (shape (n,)) plus the
integer position `k` being predicted, instead of a single scalar eigenvalue.
Both encoder and decoder already compute the whole spectrum once per block
(`eigendecompose` runs before this loop) and it costs nothing extra to pass
all of it through -- and a real sequence model conditioned on "the shape of
the whole spectrum so far" is a meaningfully better match for "conditioned on
Λ" than one conditioned on a single number. codec.py's two call sites were
updated to match.

Subclasses nn.Module so `.to(device)`, `state_dict()`/`load_state_dict()`,
and optimizers all work identically across every implementation here.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoeffPredictor(nn.Module):
    def symbol_probs(self, eigvals, k, history, symbol_range):
        """
        Args:
            eigvals: (n,) float tensor -- the full graph eigenvalue spectrum Λ
                      for this block (n = block_size**2), identical on both
                      encoder and decoder since it's recomputed from context,
                      never transmitted
            k: int -- index (ascending-eigenvalue order) of the coefficient
                      being predicted right now
            history: list[int] -- previously decoded quantized coefficients in
                      this block/channel, positions 0..k-1, same order as eigvals
            symbol_range: (lo, hi) inclusive int bounds the coder supports

        Returns:
            probs: 1-d float64 tensor of length (hi - lo + 1), on this module's
                   device, sums to 1, probs[i] is P(symbol == lo + i). Every
                   entry must be > 0 (the coder needs nonzero probability mass
                   everywhere in range, so an unexpected symbol is still
                   encodable, just expensively).
        """
        raise NotImplementedError

    def forward_sequence(self, eigvals, true_values):
        """Optional: teacher-forced, parallel training pass over one full
        block/channel. eigvals, true_values: (n,) tensors. Returns logits
        (n, n_symbols), logits[t] predicts true_values[t]. Only models meant
        to be trained (TinyTransformerCoeffModel, HFLoRACoeffModel) implement
        this; LaplaceCoeffModel has no parameters to train."""
        raise NotImplementedError

    def precompute_encode_probs(self, eigvals, true_values, symbol_range):
        """
        ENCODER-ONLY speed optimization: unlike the decoder, the encoder
        already knows every true quantized coefficient in this block before
        it entropy-codes any of them. That means it never needs the
        incremental, one-symbol-at-a-time interface symbol_probs() provides
        (which exists for the decoder, which genuinely doesn't know future
        values yet) -- it can get the exact same distributions in one shot,
        teacher-forced, exactly like training's forward_sequence.

        Default implementation here just calls symbol_probs 3*n times (same
        as before, correct for any model including ones with no faster
        path). TinyTransformerCoeffModel overrides this with a real batched
        implementation -- see its docstring for the speedup this gives.

        Args:
            eigvals: (n,) eigenvalue spectrum for this block
            true_values: (n, 3) tensor -- the TRUE quantized coefficients
            symbol_range: (lo, hi)

        Returns:
            probs: (n, 3, n_symbols) float64 tensor. probs[k, ch] is exactly
                   what symbol_probs(eigvals, k, history_of_true_values_up_to_k[ch],
                   symbol_range) would have returned -- callers can swap this
                   in as a drop-in precomputed lookup.
        """
        n = true_values.shape[0]
        lo, hi = symbol_range
        n_symbols = hi - lo + 1
        out = torch.zeros(n, 3, n_symbols, dtype=torch.float64)
        for ch in range(3):
            history = []
            for k in range(n):
                out[k, ch] = self.symbol_probs(eigvals, k, history, symbol_range)
                history.append(int(true_values[k, ch].item()))
        return out


class LaplaceCoeffModel(CoeffPredictor):
    """Non-learned classical stand-in: a discrete Laplace distribution whose
    scale shrinks with the eigenvalue (high-frequency coefficients are
    expected to be small/sparse -- the one genuinely well-justified prior
    every transform-coding scheme relies on) and widens slightly if recent
    history had large magnitude. No training, no data -- a baseline."""

    def __init__(self, base_scale=3.0, min_scale=0.35, history_weight=0.15, floor=1e-6):
        super().__init__()
        self.base_scale = base_scale
        self.min_scale = min_scale
        self.history_weight = history_weight
        self.floor = floor
        self.register_buffer("_device_anchor", torch.zeros(1))

    def _scale(self, eigval, history):
        device = self._device_anchor.device
        eigval_t = eigval if torch.is_tensor(eigval) else torch.tensor(float(eigval), device=device)
        eigval_t = eigval_t.to(device=device, dtype=torch.float64)

        scale = self.base_scale / torch.sqrt(1.0 + eigval_t)
        if history:
            recent = sum(abs(h) for h in history[-4:]) / min(4, len(history))
            recent_t = torch.tensor(recent / 4.0, device=device, dtype=torch.float64)
            scale = scale * (1.0 + self.history_weight * torch.tanh(recent_t))
        return torch.clamp(scale, min=self.min_scale)

    def symbol_probs(self, eigvals, k, history, symbol_range):
        device = self._device_anchor.device
        lo, hi = symbol_range
        scale = self._scale(eigvals[k], history)
        symbols = torch.arange(lo, hi + 1, dtype=torch.float64, device=device)
        p = torch.exp(-torch.abs(symbols) / scale)
        p = p + self.floor
        p = p / p.sum()
        return p

    def batch_symbol_probs(self, eigvals, symbol_range):
        """Vectorised, history-free variant (kept for quick baseline eval)."""
        device = self._device_anchor.device
        lo, hi = symbol_range
        eigvals = eigvals.to(device=device, dtype=torch.float64)
        scale = torch.clamp(self.base_scale / torch.sqrt(1.0 + eigvals), min=self.min_scale)  # (k,)
        symbols = torch.arange(lo, hi + 1, dtype=torch.float64, device=device)  # (m,)
        p = torch.exp(-torch.abs(symbols)[None, :] / scale[:, None])  # (k, m), one batched op
        p = p + self.floor
        p = p / p.sum(dim=-1, keepdim=True)
        return p


class TinyTransformerCoeffModel(CoeffPredictor):
    """
    The real, trainable "LLM-style" coefficient predictor: a small causal
    (GPT-style) transformer, built from scratch, no external weights or
    internet access needed to train or run it.

    Architecture, standard autoregressive-LM shape: at position t, the input
    token is built from (a) an embedding of the coefficient value emitted at
    position t-1 (or a learned BOS embedding at t=0 -- exactly how a causal
    LM's input at step t is the token generated at step t-1), plus (b) an
    embedding of eigvals[t] (this position's graph eigenvalue -- the Λ
    conditioning), plus (c) a learned positional embedding. Causal
    self-attention lets position t see everything decoded before it, not
    just the immediately preceding value, via the usual attention
    aggregation through earlier layers.

    Two entry points:
      - forward_sequence: teacher-forced, whole block/channel in one forward
        pass -- what training.py uses (parallel, like training any causal LM).
      - symbol_probs: incremental, one coefficient at a time -- what the
        actual encoder/decoder loop uses, since a real decoder only has the
        history it's decoded so far, exactly like next-token generation.

    PERFORMANCE NOTE: symbol_probs recomputes attention over the whole
    prefix from scratch every call (no KV-cache) -- fine for correctness and
    for dissertation-scale evaluation, but the first thing to optimise if
    this needs to run over full-resolution video: cache the encoder's
    key/value projections across the 64 calls within a block instead of
    rebuilding them every time.
    """

    def __init__(self, block_size=8, symbol_range=(-2200, 2200), d_model=64,
                 n_heads=4, n_layers=2, ff_mult=4):
        super().__init__()
        self.block_size = block_size
        self.n = block_size * block_size
        self.symbol_range = tuple(symbol_range)
        n_symbols = symbol_range[1] - symbol_range[0] + 1

        self.value_embed = nn.Sequential(
            nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.eigval_embed = nn.Sequential(
            nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.pos_embed = nn.Embedding(self.n, d_model)
        self.bos = nn.Parameter(torch.zeros(d_model))  # learned "no history yet" embedding

        # Optional cross-frame conditioning (encode_video/decode_video, codec.py):
        # at position k, the previous frame's co-located block's DECODED
        # quantized coefficient at the same frequency index k, embedded the
        # same way as this frame's own history values and summed in. Off by
        # default (temporal_values=None everywhere below falls back to the
        # learned `no_temporal` embedding), so the plain per-image path
        # (encode_image/decode_image, no video/temporal context) is
        # completely unaffected -- this only activates when the caller
        # actually has cross-frame data to offer.
        #
        # KNOWN LIMITATION (documented, not hidden): GBTICLMetaLearner's
        # predicted eigenbasis can shift block-to-block, so "frequency index
        # k" is only an approximate cross-frame correspondence, not the same
        # physical basis vector every frame. See run_dataset_pipeline.py's
        # ablation matrix for an explicit on/off comparison of whether this
        # conditioning actually helps given that misalignment risk, rather
        # than assuming it does.
        self.temporal_value_embed = nn.Sequential(
            nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.no_temporal = nn.Parameter(torch.zeros(d_model))

        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=ff_mult * d_model,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.out_proj = nn.Linear(d_model, n_symbols)
        self.d_model = d_model

    def _value_feat(self, values):
        """Continuous value embedding input: raw scale + sign-log magnitude,
        so both small and large coefficients (this project uses a wide
        symbol_range like (-2200,2200) to cover near-lossless DC terms) are
        representable without a huge lookup table."""
        scaled = values / 256.0
        logmag = torch.sign(values) * torch.log1p(torch.abs(values)) / 8.0
        return torch.stack([scaled, logmag], dim=-1)

    def _tokens(self, eigvals, prev_values, device, temporal_values=None):
        """Batched token builder -- the single implementation both
        forward_sequence (training, real batch size) and symbol_probs
        (inference, batch size 1) go through, so the two can never drift
        into architecturally different code paths.

        eigvals, prev_values: (B, n_pos) -> tokens: (B, n_pos, d_model)
        temporal_values: optional (B, n_pos) -- previous frame's co-located
            block's decoded coefficient at each position (see class
            docstring); None means no cross-frame conditioning available,
            uses the learned no_temporal fallback for every position.
        """
        b, n_pos = eigvals.shape
        val_emb = self.value_embed(self._value_feat(prev_values))  # (B,n_pos,d)
        val_emb = val_emb.clone()
        val_emb[:, 0] = self.bos  # position 0 has no predecessor coefficient
        eig_emb = self.eigval_embed(eigvals.to(torch.float32).unsqueeze(-1))  # (B,n_pos,d)
        pos = self.pos_embed(torch.arange(n_pos, device=device)).unsqueeze(0).expand(b, -1, -1)

        if temporal_values is not None:
            temp_emb = self.temporal_value_embed(self._value_feat(temporal_values))  # (B,n_pos,d)
        else:
            temp_emb = self.no_temporal.to(val_emb.dtype).expand(b, n_pos, -1)

        return val_emb + eig_emb + pos + temp_emb

    def forward_sequence(self, eigvals, true_values, temporal_values=None):
        """Teacher-forced training pass.
        eigvals, true_values: (n,) for one sequence or (B, n) for a batch.
        temporal_values: optional, same shape as true_values -- see
            _tokens()'s docstring and the class docstring's cross-frame
            conditioning note.
        Returns logits: (n, n_symbols) or (B, n, n_symbols) matching input rank.
        """
        squeeze = eigvals.dim() == 1
        if squeeze:
            eigvals, true_values = eigvals.unsqueeze(0), true_values.unsqueeze(0)
            if temporal_values is not None:
                temporal_values = temporal_values.unsqueeze(0)
        device = eigvals.device
        b, n = eigvals.shape
        prev = torch.zeros(b, n, device=device, dtype=torch.float32)
        prev[:, 1:] = true_values[:, :-1].to(torch.float32)
        tokens = self._tokens(eigvals, prev, device, temporal_values=temporal_values)  # (B,n,d)
        mask = nn.Transformer.generate_square_subsequent_mask(n).to(device)
        out = self.encoder(tokens, mask=mask)
        logits = self.out_proj(out)  # (B,n,n_symbols)
        return logits.squeeze(0) if squeeze else logits

    def symbol_probs(self, eigvals, k, history, symbol_range, temporal_values=None):
        """
        Decoder-side, incremental (one symbol at a time -- the decoder
        genuinely doesn't know later values yet). NO KV-cache: this rebuilds
        and reattends over the whole 0..k prefix from scratch every call,
        which is correct but means total work across a block/channel is
        O(n^3) instead of O(n^2) -- the known, currently-unoptimised cost
        (see the class docstring's PERFORMANCE NOTE). For the encoder side,
        use precompute_encode_probs instead (below) -- it needs none of this
        because the encoder already has every true value up front.

        temporal_values: optional (n,) tensor -- the previous frame's
            co-located block's decoded coefficients at positions 0..n-1 (only
            0..k actually used here); None means no cross-frame conditioning.
        """
        assert tuple(symbol_range) == self.symbol_range, (
            f"TinyTransformerCoeffModel was built for symbol_range={self.symbol_range}, "
            f"got {tuple(symbol_range)}"
        )
        device = next(self.parameters()).device
        eigvals = eigvals.to(device=device, dtype=torch.float32)
        n_pos = k + 1
        prev = torch.zeros(n_pos, device=device, dtype=torch.float32)
        if history:
            prev[1:] = torch.tensor(history, device=device, dtype=torch.float32)
        temp = None
        if temporal_values is not None:
            temp = temporal_values[:n_pos].to(device=device, dtype=torch.float32).unsqueeze(0)
        tokens = self._tokens(eigvals[:n_pos].unsqueeze(0), prev.unsqueeze(0), device,
                               temporal_values=temp)  # (1,n_pos,d)
        mask = nn.Transformer.generate_square_subsequent_mask(n_pos).to(device)
        out = self.encoder(tokens, mask=mask)
        logits = self.out_proj(out[0, -1]).to(torch.float64)  # (n_symbols,)
        probs = F.softmax(logits, dim=-1)
        probs = probs + 1e-6
        return probs / probs.sum()

    def precompute_encode_probs(self, eigvals, true_values, symbol_range, temporal_values=None):
        """
        Encoder-side fast path: ONE batched forward pass (batch=3, one per
        colour channel) covering all 64 positions at once, teacher-forced
        with the true values -- instead of 192 separate incremental calls
        each redoing attention over a growing prefix. This is architecturally
        identical to symbol_probs (same _tokens()/encoder pipeline), just
        computed in parallel because the encoder is allowed to (it already
        has every true coefficient before it entropy-codes any of them).

        temporal_values: optional (n, 3) tensor -- previous frame's
            co-located block's decoded coefficients, same layout as
            true_values; None means no cross-frame conditioning.
        """
        assert tuple(symbol_range) == self.symbol_range, (
            f"TinyTransformerCoeffModel was built for symbol_range={self.symbol_range}, "
            f"got {tuple(symbol_range)}"
        )
        device = next(self.parameters()).device
        eigvals = eigvals.to(device=device, dtype=torch.float32)
        true_vals_rep = true_values.to(device=device, dtype=torch.float32).transpose(0, 1)  # (3, n)
        eigvals_rep = eigvals.unsqueeze(0).expand(3, -1)  # (3, n)
        temporal_rep = None
        if temporal_values is not None:
            temporal_rep = temporal_values.to(device=device, dtype=torch.float32).transpose(0, 1)  # (3, n)

        logits = self.forward_sequence(eigvals_rep, true_vals_rep, temporal_values=temporal_rep)  # (3, n, n_symbols)
        probs = F.softmax(logits.to(torch.float64), dim=-1) + 1e-6
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs.permute(1, 0, 2).contiguous()  # (n, 3, n_symbols)


class HFLoRACoeffModel(CoeffPredictor):
    """
    The real pretrained-LLM coefficient predictor: wraps a real pretrained
    causal LM (HuggingFace `transformers`), fine-tuned with LoRA adapters
    (`peft`) -- this is what satisfies "repurpose a real pretrained LLM" for
    coefficient prediction (as opposed to TinyTransformerCoeffModel, a small
    transformer trained from scratch only on this task; that class stays as
    an explicit ablation baseline for exactly this comparison).

    Default base model: distilgpt2 (82M params, 6 layers, hidden=768).
    Chosen because (a) it's unambiguously "a real pretrained LLM" while
    leaving large VRAM headroom on an 8GB card, (b) OpenAI's GPT-2 license
    permits this use, (c) `target_modules=["c_attn"]` below matches its
    architecture directly. `HuggingFaceTB/SmolLM2-135M` (Apache-2.0,
    `target_modules=["q_proj","v_proj"]`, already the fallback branch below)
    is a documented drop-in alternative for a more modern base model.

    Rather than tokenising coefficients as text, this feeds `inputs_embeds`
    directly (a standard technique): each position's embedding is built the
    same way as TinyTransformerCoeffModel's (value + eigval + position +
    optional cross-frame temporal value, see coeff_model.py's module-level
    note on that), projected up to the LM's hidden size, run through the
    frozen (LoRA-adapted) LM, and the final hidden state is projected back
    down to logits over symbol_range by a small trained head. The LM's own
    token-embedding matrix and vocabulary are not used at all -- only its
    pretrained transformer *body* and attention patterns are reused and
    lightly adapted.

    VRAM-fitting design (see class docstring continuation in __init__):
    frozen base loaded in bfloat16, gradient checkpointing enabled, only
    LoRA adapters + the small head modules (kept in fp32 for stable
    optimization) are trainable.

    KV-CACHING (symbol_probs): unlike TinyTransformerCoeffModel's
    from-scratch encoder (no native cache support, O(n^3) decode cost -- see
    its class docstring), this class uses the underlying HF model's native
    `past_key_values`/`use_cache=True` support to make incremental decoding
    O(n^2) like normal autoregressive generation.

    MULTI-SLOT, not single-slot (bug found and fixed during testing): both
    decode_image and decode_video call symbol_probs in INTERLEAVED order --
    for k in range(n): for ch in range(3): symbol_probs(...) -- i.e. three
    independent per-channel sequences advance one step at a time, round-
    robin, not one whole channel at a time. A single shared cache slot on
    the model instance is wrong here: it would attend with position-k's
    attention_mask length but only the *previous channel's* cached K/V,
    silently corrupting every channel-switch (confirmed: with a single-slot
    cache, this desynced ~98% of symbols within one block on real image
    data). Fixed with a small dict of caches keyed by `id(history)`:
    codec.py's decode loops build one persistent Python list per (block,
    channel) -- `history[ch]`, mutated via .append() across the k=0..n-1
    calls for that channel -- so the *same list object* recurs on every
    call within one channel's decode, letting each channel's cache be kept
    independently without changing codec.py's calling convention or the
    shared CoeffPredictor interface.

    Robustness against Python `id()` reuse (a freed list's memory address
    being reassigned to an unrelated new list): every cache entry also
    records the expected history length: a `len(history) != expected`
    mismatch is treated as a cache miss and triggers a safe "cold start"
    recompute of the full 0..k prefix in one call (correct, just slower --
    this is a defensive fallback for a scenario that shouldn't occur under
    normal codec usage, not the expected hot path). The cache dict is also
    capped in size (oldest entries evicted) so very long runs (thousands of
    blocks) don't grow it unbounded.
    """
    _MAX_CACHE_SLOTS = 8

    def __init__(self, base_model_name="distilgpt2", lora_r=8, lora_alpha=16,
                 lora_dropout=0.05, block_size=8, symbol_range=(-2200, 2200),
                 lm_dtype=None):
        super().__init__()
        try:
            from transformers import AutoModel
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            raise ImportError(
                "HFLoRACoeffModel needs `pip install transformers peft` "
                "(and, at import time, internet access to download "
                f"'{base_model_name}'). Not available in this environment -- "
                "use TinyTransformerCoeffModel instead, or install these "
                "packages and an internet-enabled/GPU machine to use this class."
            ) from e

        self.symbol_range = tuple(symbol_range)
        n_symbols = symbol_range[1] - symbol_range[0] + 1
        self.n = block_size * block_size
        # bfloat16 for the frozen base: well-supported and numerically stable
        # on Ada/Blackwell-class Tensor Cores (this project's target GPU),
        # no loss-scaler needed unlike fp16. CPU-only fallback uses fp32
        # (bf16 matmuls are slow/unsupported on many CPUs).
        self.lm_dtype = lm_dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)

        base = AutoModel.from_pretrained(base_model_name, dtype=self.lm_dtype)
        lora_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha,
            target_modules=["c_attn"] if base_model_name.startswith(("gpt2", "distilgpt2")) else ["q_proj", "v_proj"],
            lora_dropout=lora_dropout, bias="none",
        )
        # required for LoRA to receive gradients when the model's INPUT is
        # inputs_embeds (not token ids through the frozen embedding table,
        # which is what enable_input_require_grads() normally hooks) --
        # gradient checkpointing needs this too, or backprop silently stops
        # at the first checkpointed layer
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
        self.lm = get_peft_model(base, lora_cfg)  # base weights frozen; only LoRA adapters train
        d_model = base.config.hidden_size

        # heads kept in fp32 (not lm_dtype) for stable optimization -- cheap:
        # out_proj is the largest of these at d_model*n_symbols params, still
        # a few MB even at fp32 with Adam's two fp32 moment buffers
        self.value_embed = nn.Sequential(nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.eigval_embed = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.pos_embed = nn.Embedding(self.n, d_model)
        self.bos = nn.Parameter(torch.zeros(d_model))
        self.temporal_value_embed = nn.Sequential(nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.no_temporal = nn.Parameter(torch.zeros(d_model))
        self.out_proj = nn.Linear(d_model, n_symbols)

        # {id(history): (past_key_values, expected_len)} -- see class
        # docstring's KV-CACHING section for why this must be multi-slot
        self._kv_caches = {}

    def _value_feat(self, values):
        scaled = values / 256.0
        logmag = torch.sign(values) * torch.log1p(torch.abs(values)) / 8.0
        return torch.stack([scaled, logmag], dim=-1)

    def _embed_positions(self, eigvals, prev_values, pos_start, device, temporal_values=None):
        """Build embeddings for a contiguous range of positions
        [pos_start, pos_start+len(eigvals)) -- used both for a full-sequence
        forward pass (pos_start=0) and for a single new position during
        incremental KV-cached decoding (pos_start=k, len==1).

        eigvals, prev_values: (n_pos,). temporal_values: optional (n_pos,).
        Returns: (1, n_pos, d_model) in fp32 (caller casts to lm_dtype).
        """
        n_pos = eigvals.shape[0]
        val_emb = self.value_embed(self._value_feat(prev_values))
        if pos_start == 0:
            val_emb = val_emb.clone()
            val_emb[0] = self.bos  # position 0 has no predecessor coefficient
        eig_emb = self.eigval_embed(eigvals.to(torch.float32).unsqueeze(-1))
        positions = torch.arange(pos_start, pos_start + n_pos, device=device)
        pos = self.pos_embed(positions)
        if temporal_values is not None:
            temp_emb = self.temporal_value_embed(self._value_feat(temporal_values))
        else:
            temp_emb = self.no_temporal.to(val_emb.dtype).expand(n_pos, -1)
        return (val_emb + eig_emb + pos + temp_emb).unsqueeze(0)  # (1, n_pos, d)

    def _run_full(self, inputs_embeds):
        """Full-sequence forward pass, no cache -- used by forward_sequence
        (training / encoder-side precompute_encode_probs, teacher-forced,
        parallel over the whole block/channel in one call)."""
        attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
        out = self.lm(inputs_embeds=inputs_embeds.to(self.lm_dtype), attention_mask=attn)
        return out.last_hidden_state.to(torch.float32)  # (1, n_pos, d)

    def forward_sequence(self, eigvals, true_values, temporal_values=None):
        device = eigvals.device
        n = eigvals.shape[0]
        prev = torch.zeros(n, device=device, dtype=torch.float32)
        prev[1:] = true_values[:-1].to(torch.float32)
        embeds = self._embed_positions(eigvals, prev, 0, device, temporal_values=temporal_values)
        hidden = self._run_full(embeds)
        return self.out_proj(hidden[0])

    def _evict_cache_if_full(self):
        if len(self._kv_caches) > self._MAX_CACHE_SLOTS:
            # crude but safe: drop the oldest entry (dict preserves insertion
            # order in Python 3.7+); anything evicted just becomes a cache
            # miss later, handled correctly (if slower) by the cold-start
            # fallback below -- never a correctness issue, only a speed one
            oldest_key = next(iter(self._kv_caches))
            del self._kv_caches[oldest_key]

    def _cold_start(self, eigvals, history, k, device, temporal_values=None):
        """Rebuild the full 0..k prefix in one forward pass (no cache reuse)
        -- the safe fallback for a genuine cache miss (first call for this
        history, or a length-mismatch indicating a stale/evicted/colliding
        slot -- see class docstring). Returns (hidden_state_at_k, past_key_values)."""
        n_pos = k + 1
        prev = torch.zeros(n_pos, device=device, dtype=torch.float32)
        if history:
            prev[1:] = torch.tensor(history, device=device, dtype=torch.float32)
        temp = None
        if temporal_values is not None:
            temp = temporal_values[:n_pos].to(device=device, dtype=torch.float32)
        embed = self._embed_positions(eigvals[:n_pos], prev, 0, device, temporal_values=temp)
        attn = torch.ones((1, n_pos), dtype=torch.long, device=device)
        out = self.lm(inputs_embeds=embed.to(self.lm_dtype), attention_mask=attn, use_cache=True)
        return out.last_hidden_state[0, -1].to(torch.float32), out.past_key_values

    def symbol_probs(self, eigvals, k, history, symbol_range, temporal_values=None):
        """Decoder-side, incremental, KV-cached (see class docstring's
        MULTI-SLOT section: caches are keyed by id(history) since decode
        interleaves 3 channels' sequences round-robin, not one at a time).
        temporal_values, if given, is the full (n,) previous-frame-
        coefficient array for this block/channel -- only temporal_values[k]
        is used per call."""
        assert tuple(symbol_range) == self.symbol_range
        device = next(self.parameters()).device
        eigvals = eigvals.to(device=device, dtype=torch.float32)
        cache_key = id(history)

        if k == 0:
            self._kv_caches.pop(cache_key, None)  # fresh sequence, discard any stale/colliding entry
            temp_val = temporal_values[0:1].to(device=device, dtype=torch.float32) if temporal_values is not None else None
            embed = self._embed_positions(eigvals[0:1], torch.zeros(1, device=device, dtype=torch.float32),
                                           0, device, temporal_values=temp_val)
            attn = torch.ones((1, 1), dtype=torch.long, device=device)
            out = self.lm(inputs_embeds=embed.to(self.lm_dtype), attention_mask=attn, use_cache=True)
            hidden = out.last_hidden_state[0, -1].to(torch.float32)
            self._kv_caches[cache_key] = (out.past_key_values, 1)
            self._evict_cache_if_full()
        else:
            cached = self._kv_caches.get(cache_key)
            if cached is not None and cached[1] == k and len(history) == k:
                # hot path: reuse cache, feed only the single new token
                past, _ = cached
                prev_val = torch.tensor([float(history[-1])], device=device, dtype=torch.float32)
                temp_val = temporal_values[k:k + 1].to(device=device, dtype=torch.float32) if temporal_values is not None else None
                embed = self._embed_positions(eigvals[k:k + 1], prev_val, k, device, temporal_values=temp_val)
                attn = torch.ones((1, k + 1), dtype=torch.long, device=device)
                out = self.lm(inputs_embeds=embed.to(self.lm_dtype), attention_mask=attn,
                               past_key_values=past, use_cache=True)
                hidden = out.last_hidden_state[0, -1].to(torch.float32)
                self._kv_caches[cache_key] = (out.past_key_values, k + 1)
            else:
                # cold path: cache miss (first call at k>0 for this key, or a
                # length mismatch -- see class docstring's id()-reuse note)
                hidden, past = self._cold_start(eigvals, history, k, device, temporal_values=temporal_values)
                self._kv_caches[cache_key] = (past, k + 1)
                self._evict_cache_if_full()

        logits = self.out_proj(hidden).to(torch.float64)
        probs = F.softmax(logits, dim=-1) + 1e-6
        return probs / probs.sum()

    def precompute_encode_probs(self, eigvals, true_values, symbol_range, temporal_values=None):
        """
        ENCODER-ONLY, but deliberately NOT a parallel/batched shortcut
        (unlike TinyTransformerCoeffModel's, or an earlier version of this
        method): loops through symbol_probs exactly like decode_image/
        decode_video will, one symbol at a time per channel, reusing its
        KV-cache. This guarantees BIT-IDENTICAL probabilities to the
        decoder, which turned out to matter here specifically: a real
        pretrained LM run in bfloat16 does NOT give bit-identical results
        between "one parallel forward pass over the whole teacher-forced
        sequence" and "many incremental KV-cached forward calls, one new
        position at a time" -- confirmed by direct testing, differences of
        ~0.005-0.02 in the resulting probabilities, easily enough to shift
        the integer frequency table (range_coder.py's TOTAL_FREQ=16384
        precision) and silently desync the decoder starting from the very
        first symbol (observed: PSNR collapsing to ~5dB on real image data).
        TinyTransformerCoeffModel does NOT have this problem (verified
        directly, exact match at real coefficient scale) because it runs in
        plain fp32 and recomputes attention from scratch on every call
        either way -- there is no separate "batched" vs "cached" code path
        for its numerics to diverge between. This method is still
        meaningfully faster than "no cache at all" thanks to symbol_probs'
        own id(history)-keyed KV-caching, just not as fast as a true single
        parallel forward pass would be if it were numerically safe here.
        """
        assert tuple(symbol_range) == self.symbol_range
        device = next(self.parameters()).device
        eigvals = eigvals.to(device=device, dtype=torch.float32)
        n = eigvals.shape[0]
        lo, hi = symbol_range
        n_symbols = hi - lo + 1
        out = torch.zeros(n, 3, n_symbols, dtype=torch.float64, device=device)
        for ch in range(3):
            history = []
            temp_ch = temporal_values[:, ch] if temporal_values is not None else None
            kwargs = {"temporal_values": temp_ch} if temp_ch is not None else {}
            for k in range(n):
                out[k, ch] = self.symbol_probs(eigvals, k, history, symbol_range, **kwargs)
                history.append(int(true_values[k, ch].item()))
        return out
