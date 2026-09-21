"""evaluate.py: PSNR, SSIM, BD-rate and LPIPS metrics."""

import numpy as np
import pytest

from gbticl_pipeline.evaluate import psnr, ssim, bd_rate, lpips


def test_psnr_identical_images_is_infinite():
    img = np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    assert psnr(img, img) == float("inf")


def test_ssim_identical_images_is_one():
    img = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    s = ssim(img, img)
    assert s == pytest.approx(1.0, abs=1e-6)


def test_ssim_noisy_image_is_lower():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    noisy = np.clip(img.astype(int) + rng.normal(0, 40, img.shape), 0, 255).astype(np.uint8)
    assert ssim(img, noisy) < ssim(img, img)
    assert 0.0 <= ssim(img, noisy) <= 1.0


def test_bd_rate_identical_curves_is_near_zero():
    rates = np.array([0.5, 1.0, 2.0, 4.0])
    dists = np.array([30.0, 35.0, 40.0, 45.0])
    result = bd_rate(rates, dists, rates, dists)
    assert result == pytest.approx(0.0, abs=1e-6)


def test_bd_rate_strictly_better_curve_is_negative():
    """Halving the rate at equal distortion gives a BD-rate of about -50%."""
    dists = np.array([30.0, 35.0, 40.0, 45.0])
    rate_ref = np.array([1.0, 2.0, 4.0, 8.0])
    rate_test = rate_ref / 2.0
    result = bd_rate(rate_ref, dists, rate_test, dists)
    assert result < -40  # roughly -50% since rate is uniformly halved


def test_bd_rate_requires_at_least_4_points():
    rates = np.array([1.0, 2.0, 4.0])
    dists = np.array([30.0, 35.0, 40.0])
    with pytest.raises(ValueError):
        bd_rate(rates, dists, rates, dists)


def test_bd_rate_requires_overlapping_distortion_ranges():
    rates = np.array([1.0, 2.0, 4.0, 8.0])
    dist_ref = np.array([10.0, 12.0, 14.0, 16.0])
    dist_test = np.array([50.0, 52.0, 54.0, 56.0])
    with pytest.raises(ValueError):
        bd_rate(rates, dist_ref, rates, dist_test)


def test_lpips_identical_images_is_zero():
    img = np.random.default_rng(42).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    val = lpips(img, img)
    assert val == pytest.approx(0.0, abs=1e-5)


def test_lpips_distorted_image_is_higher():
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    noisy = np.clip(img.astype(int) + rng.normal(0, 30, img.shape), 0, 255).astype(np.uint8)
    val = lpips(img, noisy)
    assert val > 0.0


def test_lpips_tensor_input_supported():
    """lpips also accepts a (C, H, W) uint8 torch tensor."""
    import torch
    t = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
    val = lpips(t, t)
    assert val == pytest.approx(0.0, abs=1e-5)
