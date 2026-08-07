"""Scalar quantization of GFT coefficients. Elementwise ops -- run on whatever
device `coeffs`/`q` already live on."""

import torch


def quantize(coeffs, step):
    """x̂_q = round(x̂ / step). Lossy for step > 0; step -> 0 approaches lossless."""
    return torch.round(coeffs / step).to(torch.int64)


def dequantize(q, step):
    """x̂_approx = x̂_q * step."""
    return q.to(torch.float64) * step
