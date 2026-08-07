"""Promotes range_coder.py's __main__ self-test into an assertion-based test:
round-trips a real sequence through the actual byte-oriented arithmetic coder
and checks the compressed size lands close to the theoretical entropy."""

import numpy as np
import pytest

from gbticl_pipeline.range_coder import (
    RangeEncoder, RangeDecoder, encode_symbol, decode_symbol, probs_to_freqs, TOTAL_FREQ,
)


def test_range_coder_roundtrip_and_efficiency():
    rng = np.random.default_rng(0)
    n_symbols = 21  # representing integers -10..10
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

    assert list(symbols) == decoded, "range coder round-trip must be exact (lossless)"

    entropy_bits = -np.sum(true_probs * np.log2(true_probs)) * n
    actual_bits = len(payload) * 8
    overhead_pct = 100 * (actual_bits / entropy_bits - 1)
    # a real arithmetic coder should land within a couple percent of the
    # theoretical entropy bound -- large overhead would indicate a bug
    assert overhead_pct < 5.0, f"range coder overhead too high: {overhead_pct:.2f}%"


def test_range_coder_empty_sequence():
    enc = RangeEncoder()
    payload = enc.finish()
    assert isinstance(payload, bytes)
    # decoder must not crash on a trivial/empty-symbol-stream payload
    RangeDecoder(payload)


def _laplace_probs(n_symbols, scale):
    x = np.arange(n_symbols) - n_symbols // 2
    p = np.exp(-np.abs(x) / scale)
    return p / p.sum()


def test_probs_to_freqs_always_sums_to_total():
    """Regression test for a confirmed-reproduced bug: the old probs_to_freqs
    correction logic only patched a single (largest) bin, which could go
    negative -- and silently clamp -- when a wide symbol range combined with
    a highly peaked distribution pushed many near-zero bins up to the floor
    of 1 each, adding more excess mass than the single largest bin could
    absorb (confirmed case: 4401 symbols, scale=3 Laplace peaked at the
    centre -> pre-correction sum 20718 against a budget of 16384, old code
    left the final sum at 18015 instead of 16384, corrupting every decode
    that used that probability table)."""
    for n_symbols, scale in [(4401, 3.0), (21, 3.0), (4401, 500.0), (2, 1.0), (16384, 1.0)]:
        probs = _laplace_probs(n_symbols, scale)
        freqs, cum = probs_to_freqs(probs, total=TOTAL_FREQ)
        assert freqs.sum() == TOTAL_FREQ, f"n={n_symbols} scale={scale}: sum={freqs.sum()}"
        assert (freqs >= 1).all()
        assert cum[-1] == TOTAL_FREQ
        assert cum[0] == 0


def test_probs_to_freqs_rejects_more_symbols_than_budget():
    probs = np.full(TOTAL_FREQ + 1, 1.0 / (TOTAL_FREQ + 1))
    with pytest.raises(ValueError):
        probs_to_freqs(probs, total=TOTAL_FREQ)


def test_range_coder_roundtrip_peaked_distribution_wide_range():
    """End-to-end regression test for the exact failure mode found: encode
    and decode a value deep in the tail of a wide (4401-symbol), highly
    peaked (scale=3) distribution -- this desynced under the old
    probs_to_freqs before the fix."""
    n_symbols = 4401
    lo = -(n_symbols // 2)
    probs = _laplace_probs(n_symbols, scale=3.0)

    for true_val in [0, 1, -1, 500, -500, 1546, -1546, n_symbols // 2 - 1, -(n_symbols // 2)]:
        sym_idx = true_val - lo
        enc = RangeEncoder()
        encode_symbol(enc, probs, sym_idx)
        payload = enc.finish()
        dec = RangeDecoder(payload)
        decoded = decode_symbol(dec, probs)
        assert decoded == sym_idx, f"true_val={true_val}: decoded {decoded + lo} != {true_val}"
