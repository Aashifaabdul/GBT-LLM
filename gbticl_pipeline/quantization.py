"""Uniform scalar quantisation of GFT coefficients."""

import torch


def quantize(coeffs, step):
    """q = round(coeffs / step), as int64."""
    return torch.round(coeffs / step).to(torch.int64)


def dequantize(q, step):
    """Reconstruction coeffs ≈ q * step, as float64."""
    return q.to(torch.float64) * step
