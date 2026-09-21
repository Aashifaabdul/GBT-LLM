"""Byte-level LLM entropy estimate.

LLMCompressor: byte-stream baseline in the style of "Language Modeling Is
Compression" (Deletang et al., ICLR 2024). Every byte is fed to a causal
language model as a token id, and the ideal arithmetic-coding size is computed
from the model's next-token probabilities. No bitstream is written.
"""

import math

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMCompressor:
    """Ideal code length of a byte stream under a causal language model.

    bits = sum_t -log2 P(x_t | x_<t), with the first byte of each chunk costing
    8 bits. If the model cannot be loaded, evaluate_entropy_bits() falls back to
    an order-1 Markov entropy estimate instead.
    """

    def __init__(self, model_name="distilgpt2", device="cpu"):
        self.device = device
        self.model_name = model_name
        print(f"Loading LLM model '{model_name}' on {device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            self.model = None
            self.tokenizer = None

    def evaluate_entropy_bits(self, raw_bytes: bytes, max_seq_len: int = 512):
        """Return (total_bits, compressed_bytes) for `raw_bytes`.

        The stream is split into chunks of at most `max_seq_len` bytes that are
        scored independently. compressed_bytes adds 2 bytes of arithmetic-coder
        termination overhead.
        """
        if self.model is None:
            return self._markov_entropy_estimate(raw_bytes)

        # Each byte value is used directly as a token id of the model vocabulary.
        vocab_size = self.model.config.vocab_size
        token_ids = [b % vocab_size for b in raw_bytes]

        total_bits = 0.0
        num_tokens = len(token_ids)
        chunk_size = min(max_seq_len, 512)

        with torch.no_grad():
            for i in range(0, num_tokens, chunk_size):
                chunk = token_ids[i:i + chunk_size]
                if len(chunk) <= 1:
                    total_bits += len(chunk) * 8.0
                    continue

                input_tensor = torch.tensor([chunk], dtype=torch.long, device=self.device)
                logits = self.model(input_tensor).logits  # [1, seq_len, vocab_size]

                # Position t predicts token t+1.
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_tensor[:, 1:].contiguous()
                log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                target_log_probs = torch.gather(
                    log_probs, 2, shift_labels.unsqueeze(-1)
                ).squeeze(-1)

                bits = -target_log_probs / math.log(2.0)
                # The first byte of the chunk has no context and costs 8 bits.
                total_bits += 8.0 + bits.sum().item()

        total_compressed_bytes = math.ceil(total_bits / 8.0) + 2
        return total_bits, total_compressed_bytes

    def _markov_entropy_estimate(self, raw_bytes: bytes) -> tuple:
        """Order-1 conditional entropy of the byte stream (+16 bits of overhead)."""
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        n = len(arr)
        if n == 0:
            return 0.0, 0

        context_counts = {}
        pair_counts = {}
        for i in range(len(arr) - 1):
            c = arr[i]
            nxt = arr[i + 1]
            context_counts[c] = context_counts.get(c, 0) + 1
            pair_counts[(c, nxt)] = pair_counts.get((c, nxt), 0) + 1

        cond_entropy = 0.0
        for (c, nxt), count in pair_counts.items():
            p_c_nxt = count / (n - 1)
            p_nxt_given_c = count / context_counts[c]
            cond_entropy -= p_c_nxt * math.log2(p_nxt_given_c)

        total_bits = cond_entropy * n + 16.0
        return total_bits, math.ceil(total_bits / 8.0)
