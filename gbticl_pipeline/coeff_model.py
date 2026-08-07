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

    def _tokens(self, eigvals, prev_values, device):
        """Batched token builder -- the single implementation both
        forward_sequence (training, real batch size) and symbol_probs
        (inference, batch size 1) go through, so the two can never drift
        into architecturally different code paths.

        eigvals, prev_values: (B, n_pos) -> tokens: (B, n_pos, d_model)
        """
        b, n_pos = eigvals.shape
        val_emb = self.value_embed(self._value_feat(prev_values))  # (B,n_pos,d)
        val_emb = val_emb.clone()
        val_emb[:, 0] = self.bos  # position 0 has no predecessor coefficient
        eig_emb = self.eigval_embed(eigvals.to(torch.float32).unsqueeze(-1))  # (B,n_pos,d)
        pos = self.pos_embed(torch.arange(n_pos, device=device)).unsqueeze(0).expand(b, -1, -1)
        return val_emb + eig_emb + pos

    def forward_sequence(self, eigvals, true_values):
        """Teacher-forced training pass.
        eigvals, true_values: (n,) for one sequence or (B, n) for a batch.
        Returns logits: (n, n_symbols) or (B, n, n_symbols) matching input rank.
        """
        squeeze = eigvals.dim() == 1
        if squeeze:
            eigvals, true_values = eigvals.unsqueeze(0), true_values.unsqueeze(0)
        device = eigvals.device
        b, n = eigvals.shape
        prev = torch.zeros(b, n, device=device, dtype=torch.float32)
        prev[:, 1:] = true_values[:, :-1].to(torch.float32)
        tokens = self._tokens(eigvals, prev, device)  # (B,n,d)
        mask = nn.Transformer.generate_square_subsequent_mask(n).to(device)
        out = self.encoder(tokens, mask=mask)
        logits = self.out_proj(out)  # (B,n,n_symbols)
        return logits.squeeze(0) if squeeze else logits

    def symbol_probs(self, eigvals, k, history, symbol_range):
        """
        Decoder-side, incremental (one symbol at a time -- the decoder
        genuinely doesn't know later values yet). NO KV-cache: this rebuilds
        and reattends over the whole 0..k prefix from scratch every call,
        which is correct but means total work across a block/channel is
        O(n^3) instead of O(n^2) -- the known, currently-unoptimised cost
        (see the class docstring's PERFORMANCE NOTE). For the encoder side,
        use precompute_encode_probs instead (below) -- it needs none of this
        because the encoder already has every true value up front.
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
        tokens = self._tokens(eigvals[:n_pos].unsqueeze(0), prev.unsqueeze(0), device)  # (1,n_pos,d)
        mask = nn.Transformer.generate_square_subsequent_mask(n_pos).to(device)
        out = self.encoder(tokens, mask=mask)
        logits = self.out_proj(out[0, -1]).to(torch.float64)  # (n_symbols,)
        probs = F.softmax(logits, dim=-1)
        probs = probs + 1e-6
        return probs / probs.sum()

    def precompute_encode_probs(self, eigvals, true_values, symbol_range):
        """
        Encoder-side fast path: ONE batched forward pass (batch=3, one per
        colour channel) covering all 64 positions at once, teacher-forced
        with the true values -- instead of 192 separate incremental calls
        each redoing attention over a growing prefix. This is architecturally
        identical to symbol_probs (same _tokens()/encoder pipeline), just
        computed in parallel because the encoder is allowed to (it already
        has every true coefficient before it entropy-codes any of them).
        """
        assert tuple(symbol_range) == self.symbol_range, (
            f"TinyTransformerCoeffModel was built for symbol_range={self.symbol_range}, "
            f"got {tuple(symbol_range)}"
        )
        device = next(self.parameters()).device
        eigvals = eigvals.to(device=device, dtype=torch.float32)
        true_vals_rep = true_values.to(device=device, dtype=torch.float32).transpose(0, 1)  # (3, n)
        eigvals_rep = eigvals.unsqueeze(0).expand(3, -1)  # (3, n)

        logits = self.forward_sequence(eigvals_rep, true_vals_rep)  # (3, n, n_symbols)
        probs = F.softmax(logits.to(torch.float64), dim=-1) + 1e-6
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs.permute(1, 0, 2).contiguous()  # (n, 3, n_symbols)


class HFLoRACoeffModel(CoeffPredictor):
    """
    Optional: wraps a real pretrained causal LM (via HuggingFace
    `transformers`) fine-tuned with LoRA adapters (`peft`) as the coefficient
    predictor -- the closer match to the original LLaMA-3 + LoRA design.

    Rather than tokenising coefficients as text, this feeds `inputs_embeds`
    directly (a standard technique): each position's embedding is built the
    same way as TinyTransformerCoeffModel's (value + eigval + position),
    projected up to the LM's hidden size, run through the frozen
    (LoRA-adapted) LM, and the final hidden state is projected back down to
    logits over symbol_range by a small trained head. The LM's own
    token-embedding matrix and vocabulary are not used at all -- only its
    pretrained transformer *body* and attention patterns are being reused
    and lightly adapted.

    NOT exercised in this environment: instantiating this needs
    `pip install transformers peft`, an internet connection to download the
    base model, and materially more GPU memory than TinyTransformerCoeffModel.
    Import is deferred into __init__ specifically so the rest of this file
    (and the whole pipeline) still works with only plain PyTorch installed.
    """

    def __init__(self, base_model_name="gpt2", lora_r=8, lora_alpha=16,
                 block_size=8, symbol_range=(-2200, 2200)):
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

        base = AutoModel.from_pretrained(base_model_name)
        lora_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha,
            target_modules=["c_attn"] if base_model_name.startswith("gpt2") else ["q_proj", "v_proj"],
            lora_dropout=0.05, bias="none",
        )
        self.lm = get_peft_model(base, lora_cfg)  # base weights frozen; only LoRA adapters train
        d_model = base.config.hidden_size

        self.value_embed = nn.Sequential(nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.eigval_embed = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.pos_embed = nn.Embedding(self.n, d_model)
        self.bos = nn.Parameter(torch.zeros(d_model))
        self.out_proj = nn.Linear(d_model, n_symbols)

    def _value_feat(self, values):
        scaled = values / 256.0
        logmag = torch.sign(values) * torch.log1p(torch.abs(values)) / 8.0
        return torch.stack([scaled, logmag], dim=-1)

    def _embeds(self, eigvals, prev_values, device):
        n_pos = eigvals.shape[0]
        val_emb = self.value_embed(self._value_feat(prev_values)).clone()
        val_emb[0] = self.bos
        eig_emb = self.eigval_embed(eigvals.to(torch.float32).unsqueeze(-1))
        pos = self.pos_embed(torch.arange(n_pos, device=device))
        return (val_emb + eig_emb + pos).unsqueeze(0)  # (1, n_pos, d_model)

    def _run(self, inputs_embeds):
        attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
        out = self.lm(inputs_embeds=inputs_embeds, attention_mask=attn)
        return out.last_hidden_state  # (1, n_pos, d_model)

    def forward_sequence(self, eigvals, true_values):
        device = eigvals.device
        n = eigvals.shape[0]
        prev = torch.zeros(n, device=device, dtype=torch.float32)
        prev[1:] = true_values[:-1].to(torch.float32)
        hidden = self._run(self._embeds(eigvals, prev, device))
        return self.out_proj(hidden[0])

    def symbol_probs(self, eigvals, k, history, symbol_range):
        assert tuple(symbol_range) == self.symbol_range
        device = next(self.parameters()).device
        eigvals = eigvals.to(device=device, dtype=torch.float32)
        n_pos = k + 1
        prev = torch.zeros(n_pos, device=device, dtype=torch.float32)
        if history:
            prev[1:] = torch.tensor(history, device=device, dtype=torch.float32)
        hidden = self._run(self._embeds(eigvals[:n_pos], prev, device))
        logits = self.out_proj(hidden[0, -1]).to(torch.float64)
        probs = F.softmax(logits, dim=-1) + 1e-6
        return probs / probs.sum()
