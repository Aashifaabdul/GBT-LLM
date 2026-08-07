"""
A real, working byte-oriented range coder (Subbotin-style, carryless).
This is genuine arithmetic coding, not an entropy estimate -- it produces
actual bytes you can measure, store, and decode back losslessly, driven by
whatever probability model (coeff_model.py) hands it.

DELIBERATELY CPU-ONLY, no .to(device) here. Range coding is a stateful,
carry-propagating integer algorithm processed one symbol at a time -- there
is no GPU parallelism to exploit within it, and forcing tensor ops onto a GPU
for scalar integer bit-shuffling would be slower, not faster. codec.py pulls
plain Python ints/numpy arrays out of GPU tensors right before calling into
this module; see the "DEVICE BOUNDARY" note in codec.py's module docstring.
"""

import numpy as np

TOP_VALUE = 1 << 24
BOTTOM_VALUE = 1 << 16
MASK32 = 0xFFFFFFFF
TOTAL_FREQ = 1 << 14  # precision of the probability model's integer frequencies


class RangeEncoder:
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
    """Convert a probability array to integer frequencies summing exactly to `total`,
    with every symbol guaranteed freq >= 1 (so nothing is ever unencodable)."""
    freqs = np.maximum(1, np.round(np.asarray(probs) * total)).astype(np.int64)
    diff = total - int(freqs.sum())
    idx = int(np.argmax(freqs))
    freqs[idx] += diff
    if freqs[idx] < 1:
        # extremely unlikely with reasonable probs, but keep it safe
        freqs[idx] = 1
        freqs[np.argmax(freqs)] -= (1 - freqs[idx])
    cum = np.concatenate([[0], np.cumsum(freqs)])
    return freqs, cum


def encode_symbol(encoder, probs, symbol_index, total=TOTAL_FREQ):
    freqs, cum = probs_to_freqs(probs, total)
    encoder.encode(int(cum[symbol_index]), int(freqs[symbol_index]), total)


def decode_symbol(decoder, probs, total=TOTAL_FREQ):
    freqs, cum = probs_to_freqs(probs, total)
    f = decoder.get_freq(total)
    k = int(np.searchsorted(cum, f, side="right") - 1)
    k = max(0, min(k, len(freqs) - 1))
    decoder.decode(int(cum[k]), int(freqs[k]), total)
    return k


if __name__ == "__main__":
    # ---- self-test: round-trip a random sequence under a known distribution,
    # and check compressed size lands close to the theoretical entropy ----
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
