"""Rate-distortion metrics."""

import numpy as np


def psnr(original, reconstructed, max_val=255.0):
    orig = original.astype(np.float64)
    recon = reconstructed.astype(np.float64)
    mse = np.mean((orig - recon) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(max_val) - 10 * np.log10(mse)


def bits_per_pixel(payload_bytes, height, width):
    return (len(payload_bytes) * 8) / (height * width)
