"""Rate-distortion metrics: PSNR, bits per pixel, SSIM, LPIPS and BD-Rate."""

import numpy as np


def psnr(original, reconstructed, max_val=255.0):
    """Peak signal-to-noise ratio in dB (inf for identical images)."""
    orig = original.astype(np.float64)
    recon = reconstructed.astype(np.float64)
    mse = np.mean((orig - recon) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(max_val) - 10 * np.log10(mse)


def bits_per_pixel(payload_bytes, height, width):
    """Compressed size in bits divided by the number of pixels."""
    return (len(payload_bytes) * 8) / (height * width)


def ssim(original, reconstructed, max_val=255.0):
    """Structural similarity index (Wang et al., 2004); requires scikit-image."""
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
    Bjøntegaard-Delta rate (Bjøntegaard, VCEG-M33, 2001).

    Average percentage bitrate difference between two rate-distortion curves at
    equal quality, from cubic fits of log(rate) against distortion integrated
    over the overlapping distortion range.

    Args:
        rate_ref, dist_ref: rate (bpp) and distortion (PSNR or SSIM) of the
            reference method; at least 4 points are needed for the cubic fit
            (use a quantisation-step sweep).
        rate_test, dist_test: same for the method being compared.

    Returns:
        Average % bitrate change of `test` relative to `ref`; negative means
        `test` needs fewer bits at the same quality.

    Raises:
        ValueError: fewer than 4 points on a curve, non-positive rates, or
        distortion ranges that do not overlap.
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

    # Sort by distortion (ascending) for fitting and integration
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


# Loaded LPIPS models, so the weights are read only once
_LPIPS_CACHE = {}


def _ensure_lpips_dependencies():
    """Make sure `torchvision.models` can be imported by lpips (falls back to a minimal stand-in)."""
    import sys
    if "torchvision" not in sys.modules or "torchvision.models" not in sys.modules:
        try:
            import torchvision.models  # type: ignore # noqa: F401
        except ImportError:
            import types
            import torch
            import torch.nn as nn
            from pathlib import Path

            tv = types.ModuleType("torchvision")
            tv_models = types.ModuleType("torchvision.models")

            class AlexNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=2),
                        nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=2),
                        nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                        nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                        nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=2),
                    )

            def alexnet(pretrained=True):
                model = AlexNet()
                if pretrained:
                    cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
                    ckpt_path = cache_dir / "alexnet-owt-7be5be79.pth"
                    if ckpt_path.exists():
                        state_dict = torch.load(ckpt_path, map_location="cpu")
                        model.load_state_dict(state_dict, strict=False)
                return model

            class VGG16(nn.Module):
                def __init__(self):
                    super().__init__()
                    layers = [
                        nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
                        nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
                        nn.MaxPool2d(2, 2),
                        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
                        nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
                        nn.MaxPool2d(2, 2),
                        nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
                        nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
                        nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
                        nn.MaxPool2d(2, 2),
                        nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
                        nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                        nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                        nn.MaxPool2d(2, 2),
                        nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                        nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                        nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                        nn.MaxPool2d(2, 2),
                    ]
                    self.features = nn.Sequential(*layers)

            def vgg16(pretrained=True):
                model = VGG16()
                if pretrained:
                    cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
                    ckpt_path = cache_dir / "vgg16-397923af.pth"
                    if ckpt_path.exists():
                        state_dict = torch.load(ckpt_path, map_location="cpu")
                        model.load_state_dict(state_dict, strict=False)
                return model

            tv_models.alexnet = alexnet
            tv_models.vgg16 = vgg16
            tv.models = tv_models
            sys.modules["torchvision"] = tv
            sys.modules["torchvision.models"] = tv_models


def _to_lpips_tensor(img, device=None, max_val=255.0):
    """Convert numpy array or torch tensor to [1, 3, H, W] normalized to [-1, 1]."""
    import torch
    if not isinstance(img, torch.Tensor):
        img = torch.from_numpy(np.asarray(img))

    t = img.float()

    if t.ndim == 2:
        t = t.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    elif t.ndim == 3:
        if t.shape[-1] in (1, 3):
            t = t.permute(2, 0, 1).unsqueeze(0)
            if t.shape[1] == 1:
                t = t.repeat(1, 3, 1, 1)
        elif t.shape[0] in (1, 3):
            t = t.unsqueeze(0)
            if t.shape[1] == 1:
                t = t.repeat(1, 3, 1, 1)
        else:
            raise ValueError(f"Unsupported 3D image shape for LPIPS: {img.shape}")
    elif t.ndim == 4:
        if t.shape[-1] in (1, 3):
            t = t.permute(0, 3, 1, 2)
            if t.shape[1] == 1:
                t = t.repeat(1, 3, 1, 1)
        elif t.shape[1] == 1:
            t = t.repeat(1, 3, 1, 1)

    # Normalize to [-1.0, 1.0].
    if t.max() > 1.0:
        t = t / (max_val / 2.0) - 1.0
    elif t.min() >= 0.0:
        t = t * 2.0 - 1.0
    t = torch.clamp(t, -1.0, 1.0)

    if device is not None:
        t = t.to(device)
    return t


def lpips(original, reconstructed, net="alex", device=None, model=None, max_val=255.0):
    """Learned Perceptual Image Patch Similarity (Zhang et al., 2018); lower means closer."""
    try:
        _ensure_lpips_dependencies()
        import lpips as lpips_lib
    except ImportError as e:
        raise ImportError(
            "lpips() requires lpips: pip install lpips"
        ) from e

    import torch

    if device is None:
        if isinstance(original, torch.Tensor) and original.is_cuda:
            device = original.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if model is None:
        cache_key = (net, str(device))
        if cache_key not in _LPIPS_CACHE:
            loss_fn = lpips_lib.LPIPS(net=net, verbose=False).to(device)
            loss_fn.eval()
            _LPIPS_CACHE[cache_key] = loss_fn
        model = _LPIPS_CACHE[cache_key]

    t_orig = _to_lpips_tensor(original, device=device, max_val=max_val)
    t_recon = _to_lpips_tensor(reconstructed, device=device, max_val=max_val)

    with torch.no_grad():
        dist = model(t_orig, t_recon)
        return float(dist.squeeze().cpu().item())
