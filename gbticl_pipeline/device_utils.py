"""Device selection and tolerant checkpoint loading."""

import torch


def get_device(prefer_gpu=True, verbose=True):
    """Return the best available device: CUDA, then Apple MPS, then CPU."""
    if prefer_gpu and torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device="cuda")
            if verbose:
                name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"[Device] Using CUDA GPU: {name} ({vram_gb:.2f} GB VRAM)")
            torch.backends.cudnn.benchmark = True
            return torch.device("cuda")
        except Exception:
            pass
    if prefer_gpu and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        if verbose:
            print("[Device] Using Apple Silicon MPS")
        return torch.device("mps")
    if verbose:
        print("[Device] WARNING: Running on CPU (no CUDA GPU detected). Model inference will be significantly slower.")
    return torch.device("cpu")


def load_state_dict_relaxed(model, state_dict, label=""):
    """
    load_state_dict(strict=False) that reports what did not match.

    Newer model versions add optional parameters (for example the temporal
    conditioning of the coefficient models) that older checkpoints lack. Missing
    keys keep their initial values; unexpected keys are dropped and reported, as
    they usually indicate a checkpoint of a different model class.
    """
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(f"  [{label or type(model).__name__}] checkpoint predates these params "
              f"(kept at random init): {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  [{label or type(model).__name__}] WARNING: checkpoint has keys this model "
              f"doesn't use (dropped -- possible architecture/checkpoint mismatch): "
              f"{result.unexpected_keys}")
    return result
