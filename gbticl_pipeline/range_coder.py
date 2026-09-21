"""
Byte-oriented carry-less range coder (Subbotin style).

Produces real compressed bytes from the probability distributions supplied by
the coefficient models. It runs on the CPU only: range coding is a sequential
integer algorithm, so the codec moves each distribution from the GPU to the
host right before calling into this module.
"""

import numpy as np

TOP_VALUE = 1 << 24
BOTTOM_VALUE = 1 << 16
MASK32 = 0xFFFFFFFF
TOTAL_FREQ = 1 << 14  # precision of the probability model's integer frequencies


class RangeEncoder:
    """Encoder: encode(cum_freq, freq, tot_freq) per symbol, finish() returns the bytes."""

    def __init__(self):
        self.low = 0
        self.range = MASK32
        self.buf = bytearray()

    def encode(self, cum_freq, freq, tot_freq):
        r = self.range // tot_freq
        self.low = (self.low + r * cum_freq) & MASK32
        self.range = r * freq
        self._normalize()

    def _normalize(self):
        while True:
            if (self.low ^ (self.low + self.range)) & MASK32 < TOP_VALUE:
                pass
            elif self.range < BOTTOM_VALUE:
                self.range = (-self.low) & (BOTTOM_VALUE - 1)
            else:
                break
            self.buf.append((self.low >> 24) & 0xFF)
            self.low = (self.low << 8) & MASK32
            self.range = (self.range << 8) & MASK32

    def finish(self):
        for _ in range(4):
            self.buf.append((self.low >> 24) & 0xFF)
            self.low = (self.low << 8) & MASK32
        return bytes(self.buf)


class RangeDecoder:
    """Decoder for a RangeEncoder payload: get_freq() then decode() per symbol."""

    def __init__(self, data):
        self.data = data
        self.pos = 0
        self.low = 0
        self.range = MASK32
        self.code = 0
        self.r = 1
        for _ in range(4):
            self.code = ((self.code << 8) | self._read_byte()) & MASK32

    def _read_byte(self):
        if self.pos < len(self.data):
            b = self.data[self.pos]
            self.pos += 1
            return b
        return 0

    def get_freq(self, tot_freq):
        self.r = self.range // tot_freq
        val = (self.code - self.low) // self.r
        return int(min(val, tot_freq - 1))

    def decode(self, cum_freq, freq, tot_freq):
        self.low = (self.low + self.r * cum_freq) & MASK32
        self.range = self.r * freq
        self._normalize()

    def _normalize(self):
        while True:
            if (self.low ^ (self.low + self.range)) & MASK32 < TOP_VALUE:
                pass
            elif self.range < BOTTOM_VALUE:
                self.range = (-self.low) & (BOTTOM_VALUE - 1)
            else:
                break
            self.code = ((self.code << 8) | self._read_byte()) & MASK32
            self.low = (self.low << 8) & MASK32
            self.range = (self.range << 8) & MASK32


def probs_to_freqs(probs, total=TOTAL_FREQ):
    """
    Convert probabilities to integer frequencies that sum exactly to `total`.

    Every symbol receives a frequency of at least 1, so any symbol in range stays
    encodable. The encoder and decoder both assume tot_freq == total; a frequency
    table with a different sum makes the decoder diverge from the first symbol.

    Largest-remainder apportionment: one unit is reserved per symbol, the rest is
    distributed proportionally to the probabilities (rounded down), and the units
    left over go to the symbols with the largest fractional remainder. Rounding
    each bin independently and correcting only the largest bin fails for wide
    symbol ranges with peaked distributions (the correction can exceed the bin).
    """
    probs = np.asarray(probs, dtype=np.float64)
    n = len(probs)
    if n > total:
        raise ValueError(
            f"probs_to_freqs: {n} symbols cannot each get freq>=1 out of a "
            f"total budget of only {total} -- narrow the symbol range or "
            f"raise `total` (TOTAL_FREQ)."
        )
    remaining = total - n  # budget left after reserving 1 unit per symbol
    ideal = probs * remaining
    base = np.floor(ideal).astype(np.int64)
    leftover = remaining - int(base.sum())
    if leftover > 0:
        frac = ideal - base
        # Stable sort so ties resolve identically on encoder and decoder
        order = np.argsort(-frac, kind="stable")
        base[order[:leftover]] += 1
    freqs = 1 + base
    cum = np.concatenate([[0], np.cumsum(freqs)])
    return freqs, cum


def encode_symbol(encoder, probs, symbol_index, total=TOTAL_FREQ):
    """Range-code one symbol index under the distribution `probs`."""
    freqs, cum = probs_to_freqs(probs, total)
    encoder.encode(int(cum[symbol_index]), int(freqs[symbol_index]), total)


def decode_symbol(decoder, probs, total=TOTAL_FREQ):
    """Decode one symbol index under the distribution `probs`."""
    freqs, cum = probs_to_freqs(probs, total)
    f = decoder.get_freq(total)
    k = int(np.searchsorted(cum, f, side="right") - 1)
    k = max(0, min(k, len(freqs) - 1))
    decoder.decode(int(cum[k]), int(freqs[k]), total)
    return k


if __name__ == "__main__":
    # Self-test: round-trip random symbols and compare the size with the entropy
    rng = np.random.default_rng(0)
    n_symbols = 21  # e.g. representing integers -10..10
    true_probs = np.exp(-np.abs(np.arange(-10, 11)) / 3.0)
    true_probs /= true_probs.sum()

    n = 20000
    symbols = rng.choice(n_symbols, size=n, p=true_probs)

    enc = RangeEncoder()
    for s in symbols:
        encode_symbol(enc, true_probs, int(s))
    payload = enc.finish()

    dec = RangeDecoder(payload)
    decoded = [decode_symbol(dec, true_probs) for _ in range(n)]

    assert list(symbols) == decoded, "range coder round-trip FAILED"

    entropy_bits = -np.sum(true_probs * np.log2(true_probs)) * n
    actual_bits = len(payload) * 8
    print(f"round-trip OK over {n} symbols")
    print(f"theoretical entropy: {entropy_bits:.0f} bits ({entropy_bits/8:.0f} bytes)")
    print(f"actual coded size:   {actual_bits} bits ({len(payload)} bytes)")
    print(f"overhead vs entropy: {100*(actual_bits/entropy_bits - 1):.2f}%")
