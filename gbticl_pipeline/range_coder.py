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
    """Convert a probability array to integer frequencies summing EXACTLY to
    `total`, with every symbol guaranteed freq >= 1 (so nothing is ever
    unencodable). RangeEncoder/RangeDecoder both hard-code `tot_freq=total`
    (they never read freqs.sum() themselves) -- if the returned freqs don't
    sum to exactly `total`, the encoder's cumulative-frequency intervals and
    the decoder's `range // tot_freq` division silently disagree, and the
    decoder starts returning wrong symbols from the very first call. This is
    a real, confirmed-reproduced bug fix, not defensive-only code: with a
    wide symbol_range (this project uses up to 4401 symbols, to cover
    near-lossless DC coefficients) and a highly-peaked distribution (e.g. a
    fine-quantization DC-term Laplace distribution), the OLD implementation
    (round each bin independently to >=1, then patch only the single largest
    bin to absorb the total rounding error) could need a correction bigger
    than the largest bin itself -- confirmed: a 4401-symbol Laplace(scale=3)
    distribution rounded to a pre-correction sum of 20718 against a budget
    of 16384 (a 4334 excess from ~4356 near-zero bins each floored up to 1),
    while the largest bin only held 2704 -- the old single-bin patch then
    went negative, silently clamped to 1, and left freqs summing to 18015,
    not 16384. Decoding under a mismatched total desynced on symbol one.

    Fixed via the standard "largest remainder" apportionment method: reserve
    exactly 1 unit for every symbol up front (guaranteeing freq>=1 for all
    n <= total symbols), then distribute the remaining budget across bins
    proportionally to probability, rounding down and handing the leftover
    few units to the bins with the largest fractional remainder. This always
    produces freqs.sum() == total exactly, for any probability distribution,
    as long as n_symbols <= total.
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
        # hand the few leftover units to the bins closest to their next
        # integer (largest fractional remainder) -- stable sort so ties
        # resolve deterministically (same on encoder and decoder)
        order = np.argsort(-frac, kind="stable")
        base[order[:leftover]] += 1
    freqs = 1 + base
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
