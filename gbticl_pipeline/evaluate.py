"""Rate-distortion metrics: PSNR, bpp, SSIM, and BD-Rate for the ablation
matrix (run_dataset_pipeline.py's --ablation) and its rate-distortion plots
(visualize_metrics.py)."""

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


def ssim(original, reconstructed, max_val=255.0):
    """Structural similarity (Wang et al. 2004), the standard SSIM used
    throughout the image/video-compression literature. Needs scikit-image
    (`pip install scikit-image`, listed in requirements.txt)."""
    try:
        from skimage.metrics import structural_similarity
    except ImportError as e:
        raise ImportError(
            "ssim() needs scikit-image: pip install scikit-image"
        ) from e
    orig = original.astype(np.float64)
    recon = reconstructed.astype(np.float64)
    if orig.ndim == 3:
        return float(structural_similarity(orig, recon, channel_axis=-1, data_range=max_val))
    return float(structural_similarity(orig, recon, data_range=max_val))


def bd_rate(rate_ref, dist_ref, rate_test, dist_test):
    """
    Standard Bjøntegaard-Delta rate metric (Bjøntegaard, VCEG-M33, 2001):
    the average % bitrate difference between two rate-distortion curves at
    equal quality, computed via cubic polynomial fits of distortion vs.
    log(rate) and numerical integration over their overlapping distortion
    range. This is the standard single-number summary of an RD-curve
    comparison used throughout video/image coding literature (HM/VTM BD-
    rate reports use exactly this method).

    Args:
        rate_ref, dist_ref: rate (bpp) and distortion (PSNR or SSIM) arrays
            for the reference/anchor method. Need >=4 points for a stable
            cubic fit -- run the --quant-step sweep in run_dataset_pipeline.py
            to get enough rate points per config.
        rate_test, dist_test: same, for the method being compared against
            the reference.

    Returns:
        float: average % bitrate difference of `test` vs. `ref` at equal
        quality. Negative = test uses FEWER bits at the same quality (test
        is better) -- this follows the field's usual sign convention.

    Raises:
        ValueError: fewer than 4 points on either curve, or the two curves'
        distortion ranges don't overlap (BD-Rate is only defined where both
        curves span comparable quality levels).
    """
    rate_ref = np.asarray(rate_ref, dtype=np.float64)
    dist_ref = np.asarray(dist_ref, dtype=np.float64)
    rate_test = np.asarray(rate_test, dtype=np.float64)
    dist_test = np.asarray(dist_test, dtype=np.float64)

    if len(rate_ref) < 4 or len(rate_test) < 4:
        raise ValueError(
            f"BD-Rate needs >=4 rate/distortion points per curve for a stable cubic fit "
            f"(got {len(rate_ref)} ref, {len(rate_test)} test)"
        )
    if np.any(rate_ref <= 0) or np.any(rate_test <= 0):
        raise ValueError("BD-Rate requires strictly positive rates (log(rate) is undefined at 0)")

    log_rate_ref = np.log(rate_ref)
    log_rate_test = np.log(rate_test)

    # sort by distortion ascending -- required for polyfit/integration
    order_ref = np.argsort(dist_ref)
    order_test = np.argsort(dist_test)
    d_ref, lr_ref = dist_ref[order_ref], log_rate_ref[order_ref]
    d_test, lr_test = dist_test[order_test], log_rate_test[order_test]

    p_ref = np.polyfit(d_ref, lr_ref, 3)
    p_test = np.polyfit(d_test, lr_test, 3)

    d_min = max(d_ref.min(), d_test.min())
    d_max = min(d_ref.max(), d_test.max())
    if d_min >= d_max:
        raise ValueError(
            f"distortion ranges do not overlap (ref: [{d_ref.min():.2f},{d_ref.max():.2f}], "
            f"test: [{d_test.min():.2f},{d_test.max():.2f}]) -- cannot compute BD-Rate"
        )

    p_ref_int = np.polyint(p_ref)
    p_test_int = np.polyint(p_test)
    int_ref = np.polyval(p_ref_int, d_max) - np.polyval(p_ref_int, d_min)
    int_test = np.polyval(p_test_int, d_max) - np.polyval(p_test_int, d_min)

    avg_log_rate_diff = (int_test - int_ref) / (d_max - d_min)
    return float((np.exp(avg_log_rate_diff) - 1) * 100)  # percentage
