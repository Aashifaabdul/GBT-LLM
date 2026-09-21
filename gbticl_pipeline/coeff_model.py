"""
Entropy models for the quantised GFT coefficients.

Every model predicts p(q_k) for the k-th coefficient (ascending graph
frequency) of one colour channel of one block, conditioned on the full
eigenvalue spectrum of the block's Laplacian and on the coefficients already
decoded in the same block/channel. Encoder and decoder recompute the same
distribution, so nothing besides the range-coded symbols is transmitted.

Implementations:
  - LaplaceCoeffModel: non-learned discrete Laplace prior (baseline).
  - TinyTransformerCoeffModel: small causal transformer trained from scratch
    (ablation baseline for the pretrained model).
  - HFLoRACoeffModel: pretrained causal LM (DistilGPT-2 by default) adapted
    with LoRA; the model used for the reported GBT-LLM results.

All models subclass nn.Module, so .to(device), state_dict() and optimisers
behave identically across them.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoeffPredictor(nn.Module):
    """Interface shared by all coefficient entropy models."""

    def symbol_probs(self, eigvals, k, history, symbol_range):
        """
        Distribution of the k-th quantised coefficient given the ones before it.

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
        """
        Teacher-forced training pass over one block/channel.

        eigvals, true_values: (n,) tensors. Returns logits (n, n_symbols) where
        logits[t] predicts true_values[t]. Only trainable models implement it.
        """
        raise NotImplementedError

    def precompute_encode_probs(self, eigvals, true_values, symbol_range):
        """
        Distributions for every coefficient of a block, for use by the encoder.

        The encoder knows all quantised coefficients up front, so it does not
        need the one-symbol-at-a-time interface the decoder uses. The default
        implementation simply calls symbol_probs 3 * n times; subclasses may
        override it with a faster batched version.

        Args:
            eigvals: (n,) eigenvalue spectrum for this block
            true_values: (n, 3) quantised coefficients of the block
            symbol_range: (lo, hi)

        Returns:
            (n, 3, n_symbols) float64 tensor; probs[k, ch] equals what
            symbol_probs would return for coefficient k of channel ch.
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
    """
    Non-learned discrete Laplace prior.

    The scale shrinks with the graph eigenvalue (high-frequency coefficients
    are expected to be small) and widens slightly when the recent decoded
    coefficients were large. Used as the no-LLM baseline.
    """

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
    Small causal transformer trained from scratch on the coefficient task.

    The input token at position t is the sum of an embedding of the previous
    coefficient (a learned BOS embedding at t = 0), an embedding of the graph
    eigenvalue at t, and a learned positional embedding. Causal self-attention
    lets position t see every earlier coefficient in the block.

    forward_sequence is the teacher-forced pass used for training;
    symbol_probs is the incremental pass used by the decoder. The latter
    caches per-layer keys/values across the calls for one block/channel.
    """

    def __init__(self, block_size=8, symbol_range=(-2200, 2200), d_model=64,
                 n_heads=4, n_layers=2, ff_mult=4):
        super().__init__()
        self.block_size = block_size
        self.n = block_size * block_size
        self.symbol_range = tuple(symbol_range)
        self.n_heads = n_heads
        n_symbols = symbol_range[1] - symbol_range[0] + 1

        self.value_embed = nn.Sequential(
            nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.eigval_embed = nn.Sequential(
            nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.pos_embed = nn.Embedding(self.n, d_model)
        self.bos = nn.Parameter(torch.zeros(d_model))  # learned "no history yet" embedding

        # Optional cross-frame conditioning (video mode): the previous frame's
        # co-located block contributes its decoded coefficient at the same
        # frequency index. Without it, a learned `no_temporal` embedding is used,
        # so the single-image path is unaffected. The index only approximately
        # matches across frames because the predicted eigenbasis can change from
        # block to block; the ablations measure whether the conditioning helps.
        self.temporal_value_embed = nn.Sequential(
            nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.no_temporal = nn.Parameter(torch.zeros(d_model))

        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=ff_mult * d_model,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        assert not self.encoder.layers[0].norm_first, (
            "cached inference path (_layer_step) assumes post-norm "
            "(norm_first=False, the default) -- update it if this ever changes"
        )
        self.out_proj = nn.Linear(d_model, n_symbols)
        self.d_model = d_model

        self._kv_caches = {}

    _MAX_CACHE_SLOTS = 8

    def _evict_cache_if_full(self):
        # Drop the oldest entry; a miss later is recomputed, so this only costs speed.
        if len(self._kv_caches) > self._MAX_CACHE_SLOTS:
            oldest_key = next(iter(self._kv_caches))
            del self._kv_caches[oldest_key]

    def _value_feat(self, values):
        """Embedding input for a coefficient: linear scale and signed log magnitude.

        Keeps both small and very large values (the symbol range is wide enough
        for near-lossless DC terms) representable without a huge lookup table.
        """
        scaled = values / 256.0
        logmag = torch.sign(values) * torch.log1p(torch.abs(values)) / 8.0
        return torch.stack([scaled, logmag], dim=-1)

    def _tokens(self, eigvals, prev_values, device, temporal_values=None, pos_start=0):
        """Build input tokens for positions pos_start .. pos_start + n_pos - 1.

        Shared by forward_sequence (training) and symbol_probs (decoding) so both
        use the same architecture.

        eigvals, prev_values: (B, n_pos) -> tokens: (B, n_pos, d_model)
        temporal_values: optional (B, n_pos), previous-frame coefficients; when
            None the learned no_temporal embedding is used at every position.
        """
        b, n_pos = eigvals.shape
        val_emb = self.value_embed(self._value_feat(prev_values))  # (B,n_pos,d)
        val_emb = val_emb.clone()
        if pos_start == 0:
            val_emb[:, 0] = self.bos  # position 0 has no predecessor coefficient
        eig_emb = self.eigval_embed(eigvals.to(torch.float32).unsqueeze(-1))  # (B,n_pos,d)
        positions = torch.arange(pos_start, pos_start + n_pos, device=device)
        pos = self.pos_embed(positions).unsqueeze(0).expand(b, -1, -1)

        if temporal_values is not None:
            temp_emb = self.temporal_value_embed(self._value_feat(temporal_values))  # (B,n_pos,d)
        else:
            temp_emb = self.no_temporal.to(val_emb.dtype).expand(b, n_pos, -1)

        return val_emb + eig_emb + pos + temp_emb

    def forward_sequence(self, eigvals, true_values, temporal_values=None):
        """Teacher-forced pass: (n,) or (B, n) inputs -> logits (n, n_symbols) or (B, n, n_symbols)."""
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

    def _self_attn_step(self, layer, x_new, k_cache, v_cache):
        """Self-attention for one new position, appending its key/value to the cache."""
        attn = layer.self_attn
        d_model = x_new.shape[-1]
        num_heads = attn.num_heads
        head_dim = d_model // num_heads

        qkv = F.linear(x_new, attn.in_proj_weight, attn.in_proj_bias)
        q, k_new, v_new = qkv.chunk(3, dim=-1)

        k_all = k_new if k_cache is None else torch.cat([k_cache, k_new], dim=1)
        v_all = v_new if v_cache is None else torch.cat([v_cache, v_new], dim=1)
        t = k_all.shape[1]

        q_h = q.view(1, 1, num_heads, head_dim).transpose(1, 2)
        k_h = k_all.view(1, t, num_heads, head_dim).transpose(1, 2)
        v_h = v_all.view(1, t, num_heads, head_dim).transpose(1, 2)

        scores = torch.matmul(q_h, k_h.transpose(-1, -2)) / math.sqrt(head_dim)
        weights = F.softmax(scores, dim=-1)
        out_h = torch.matmul(weights, v_h)
        out = out_h.transpose(1, 2).reshape(1, 1, d_model)
        out = attn.out_proj(out)
        return out, k_all, v_all

    def _layer_step(self, layer, x_new, cache_entry):
        """One post-norm nn.TransformerEncoderLayer step (attention + feed-forward) for one position."""
        k_cache, v_cache = cache_entry if cache_entry is not None else (None, None)
        attn_out, k_all, v_all = self._self_attn_step(layer, x_new, k_cache, v_cache)
        x = layer.norm1(x_new + attn_out)
        ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
        x = layer.norm2(x + ff)
        return x, (k_all, v_all)

    def _forward_incremental(self, eigvals, k, prev_val, device, temporal_val=None, layer_caches=None):
        """Run position k through every encoder layer, reusing and extending the per-layer caches."""
        eig_k = eigvals[k:k + 1].unsqueeze(0)
        prev = prev_val.to(device=device, dtype=torch.float32).view(1, 1)
        temp = None
        if temporal_val is not None:
            temp = temporal_val.to(device=device, dtype=torch.float32).view(1, 1)
        x = self._tokens(eig_k, prev, device, temporal_values=temp, pos_start=k)

        new_caches = []
        for layer, cache_entry in zip(self.encoder.layers, layer_caches):
            x, new_entry = self._layer_step(layer, x, cache_entry)
            new_caches.append(new_entry)

        logits = self.out_proj(x[0, -1]).to(torch.float64)
        return logits, new_caches

    def symbol_probs(self, eigvals, k, history, symbol_range, temporal_values=None):
        """
        Incremental distribution for coefficient k, used by the decoder.

        Per-layer key/value caches are keyed by id(history); a cache is reused
        only when it was built for exactly k earlier positions, otherwise the
        prefix is recomputed.

        temporal_values: optional (n,) previous-frame coefficients for this
        block/channel (only index k is used per call).
        """
        assert tuple(symbol_range) == self.symbol_range, (
            f"TinyTransformerCoeffModel was built for symbol_range={self.symbol_range}, "
            f"got {tuple(symbol_range)}"
        )
        device = next(self.parameters()).device
        eigvals = eigvals.to(device=device, dtype=torch.float32)
        cache_key = id(history)
        prev_val = torch.tensor(0.0 if k == 0 else float(history[-1]))
        temp_val = None if temporal_values is None else temporal_values[k]

        cached = self._kv_caches.get(cache_key)
        if k == 0:
            layer_caches = [None] * len(self.encoder.layers)
        elif cached is not None and cached[1] == k and len(history) == k:
            layer_caches = cached[0]
        else:
            layer_caches = [None] * len(self.encoder.layers)
            for j in range(k):
                prev_j = torch.tensor(0.0 if j == 0 else float(history[j - 1]))
                temp_j = None if temporal_values is None else temporal_values[j]
                _, layer_caches = self._forward_incremental(
                    eigvals, j, prev_j, device, temporal_val=temp_j, layer_caches=layer_caches)

        logits, new_layer_caches = self._forward_incremental(
            eigvals, k, prev_val, device, temporal_val=temp_val, layer_caches=layer_caches)
        self._kv_caches[cache_key] = (new_layer_caches, k + 1)
        self._evict_cache_if_full()

        probs = F.softmax(logits, dim=-1)
        probs = probs + 1e-6
        return probs / probs.sum()

    def precompute_encode_probs(self, eigvals, true_values, symbol_range, temporal_values=None):
        """
        Encoder-side distributions for a whole block.

        Calls symbol_probs in decoder order so that the encoder and decoder see
        bit-identical probabilities.

        temporal_values: optional (n, 3) previous-frame coefficients.
        """
        assert tuple(symbol_range) == self.symbol_range, (
            f"TinyTransformerCoeffModel was built for symbol_range={self.symbol_range}, "
            f"got {tuple(symbol_range)}"
        )
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


class HFLoRACoeffModel(CoeffPredictor):
    """
    Pretrained causal language model adapted with LoRA as a coefficient predictor.

    Coefficients are not tokenised as text. Each position's embedding (previous
    coefficient + graph eigenvalue + position [+ previous-frame coefficient])
    is fed to the language model through `inputs_embeds`, and the final hidden
    state is mapped to logits over the symbol range by a small trained head.
    Only the LoRA adapters and these input/output modules are trained; the
    pretrained weights stay frozen.

    Default base model: distilgpt2 (82M parameters, 6 layers, hidden size 768).
    Other GPT-2 style models work directly; models using q_proj/v_proj
    attention (for example SmolLM2-135M) use those as LoRA target modules.

    Decoding uses the model's native key/value cache. The decoder interleaves
    the three colour channels (k outer loop, channel inner loop), so one cache is
    kept per channel, keyed by id(history). An entry is reused only if it was
    built for exactly k earlier positions; otherwise the prefix is recomputed
    ("cold start"). The number of cached entries is capped.
    """
    _MAX_CACHE_SLOTS = 8

    def __init__(self, base_model_name="distilgpt2", lora_r=8, lora_alpha=16,
                 lora_dropout=0.05, block_size=8, symbol_range=(-2200, 2200),
                 lm_dtype=None):
        super().__init__()
        # transformers/peft are imported lazily so the other models work without them
        try:
            from transformers import AutoModel
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            raise ImportError(
                "HFLoRACoeffModel requires the `transformers` and `peft` packages "
                f"(and access to the '{base_model_name}' weights); "
                "install them with `pip install transformers peft`."
            ) from e

        self.symbol_range = tuple(symbol_range)
        n_symbols = symbol_range[1] - symbol_range[0] + 1
        self.n = block_size * block_size

        # bfloat16 for the frozen base on GPU (no loss scaling needed, unlike fp16);
        # fp32 on CPU, where bf16 matmuls are slow or unsupported.
        self.lm_dtype = lm_dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)

        # Attach LoRA adapters to the attention projections; base weights stay frozen
        base = AutoModel.from_pretrained(base_model_name, dtype=self.lm_dtype)
        lora_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha,
            target_modules=["c_attn"] if base_model_name.startswith(("gpt2", "distilgpt2")) else ["q_proj", "v_proj"],
            lora_dropout=lora_dropout, bias="none",
        )

        # Gradient checkpointing with inputs_embeds needs enable_input_require_grads(),
        # otherwise no gradient reaches the LoRA layers.
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
        self.lm = get_peft_model(base, lora_cfg)  # base weights frozen; only LoRA adapters train
        d_model = base.config.hidden_size

        # Input embeddings and output head are kept in fp32 for stable optimisation
        self.value_embed = nn.Sequential(nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.eigval_embed = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.pos_embed = nn.Embedding(self.n, d_model)
        self.bos = nn.Parameter(torch.zeros(d_model))
        self.temporal_value_embed = nn.Sequential(nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.no_temporal = nn.Parameter(torch.zeros(d_model))
        self.out_proj = nn.Linear(d_model, n_symbols)

        # {id(history): (past_key_values, number of cached positions)}
        self._kv_caches = {}

    def _value_feat(self, values):
        scaled = values / 256.0
        logmag = torch.sign(values) * torch.log1p(torch.abs(values)) / 8.0
        return torch.stack([scaled, logmag], dim=-1)

    def _embed_positions(self, eigvals, prev_values, pos_start, device, temporal_values=None):
        """Embeddings for positions pos_start .. pos_start + len(eigvals) - 1.

        eigvals, prev_values, temporal_values: (n_pos,) -> (1, n_pos, d_model), fp32.
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

    def _embed_positions_batched(self, eigvals, prev_values, device, temporal_values=None):
        """Batched _embed_positions for full sequences: (B, n_pos) -> (B, n_pos, d_model)."""
        b, n_pos = eigvals.shape
        val_emb = self.value_embed(self._value_feat(prev_values))  # (B,n_pos,d)
        val_emb = val_emb.clone()
        val_emb[:, 0] = self.bos
        eig_emb = self.eigval_embed(eigvals.to(torch.float32).unsqueeze(-1))  # (B,n_pos,d)
        pos = self.pos_embed(torch.arange(n_pos, device=device)).unsqueeze(0).expand(b, -1, -1)
        if temporal_values is not None:
            temp_emb = self.temporal_value_embed(self._value_feat(temporal_values))
        else:
            temp_emb = self.no_temporal.to(val_emb.dtype).expand(b, n_pos, -1)
        return val_emb + eig_emb + pos + temp_emb  # (B, n_pos, d)

    def _run_full(self, inputs_embeds):
        """Full-sequence forward pass without cache; returns hidden states (B, n_pos, d) in fp32."""
        attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
        out = self.lm(inputs_embeds=inputs_embeds.to(self.lm_dtype), attention_mask=attn)
        return out.last_hidden_state.to(torch.float32)  # (B, n_pos, d)

    def forward_sequence(self, eigvals, true_values, temporal_values=None):
        """Teacher-forced pass: (n,) or (B, n) inputs -> logits (n, n_symbols) or (B, n, n_symbols)."""
        # Input at position t is the coefficient at t - 1
        squeeze = eigvals.dim() == 1
        if squeeze:
            eigvals, true_values = eigvals.unsqueeze(0), true_values.unsqueeze(0)
            if temporal_values is not None:
                temporal_values = temporal_values.unsqueeze(0)
        device = eigvals.device
        b, n = eigvals.shape
        prev = torch.zeros(b, n, device=device, dtype=torch.float32)
        prev[:, 1:] = true_values[:, :-1].to(torch.float32)
        embeds = self._embed_positions_batched(eigvals, prev, device, temporal_values=temporal_values)
        # Run the LLM and project hidden states to coefficient logits
        hidden = self._run_full(embeds)  # (B, n, d)
        logits = self.out_proj(hidden)  # (B, n, n_symbols)
        return logits.squeeze(0) if squeeze else logits

    def _evict_cache_if_full(self):
        if len(self._kv_caches) > self._MAX_CACHE_SLOTS:
            # Drop the oldest entry; a later miss falls back to _cold_start
            oldest_key = next(iter(self._kv_caches))
            del self._kv_caches[oldest_key]

    def _cold_start(self, eigvals, history, k, device, temporal_values=None):
        """Recompute positions 0..k in one pass (cache miss); returns (hidden state at k, past_key_values)."""
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
        """
        Incremental, KV-cached distribution for coefficient k (decoder side).

        temporal_values: optional (n,) previous-frame coefficients for this
        block/channel; only index k is used per call.
        """
        assert tuple(symbol_range) == self.symbol_range
        device = next(self.parameters()).device
        eigvals = eigvals.to(device=device, dtype=torch.float32)
        cache_key = id(history)

        # Update this channel's attention cache
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
                # cache hit: feed only the new position
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
                # cache miss: first call for this history, or a stale/evicted slot
                hidden, past = self._cold_start(eigvals, history, k, device, temporal_values=temporal_values)
                self._kv_caches[cache_key] = (past, k + 1)
                self._evict_cache_if_full()

        # Small floor keeps every symbol encodable
        logits = self.out_proj(hidden).to(torch.float64)
        probs = F.softmax(logits, dim=-1) + 1e-6
        return probs / probs.sum()

    def precompute_encode_probs(self, eigvals, true_values, symbol_range, temporal_values=None):
        """
        Encoder-side distributions for a whole block.

        Deliberately not a single teacher-forced pass: it calls symbol_probs in
        decoder order. A bfloat16 language model gives slightly different
        probabilities for one parallel pass and for incremental cached passes;
        after integer quantisation of the frequency table those differences
        desynchronise the range decoder. Using the same code path on both
        sides guarantees identical probabilities.

        temporal_values: optional (n, 3) previous-frame coefficients.
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
