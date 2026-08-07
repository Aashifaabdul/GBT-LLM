"""Single source of truth for picking a device, used everywhere else via .to(device)."""

import torch


def get_device(prefer_gpu=True):
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_gpu and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    return torch.device("cpu")


def load_state_dict_relaxed(model, state_dict, label=""):
    """load_state_dict(strict=False) with a clear printed summary of what
    didn't match, instead of either a hard crash (strict=True) or silent
    data loss (strict=False with no visibility).

    Why this exists: architectures in this project have grown new OPTIONAL
    parameters over time (e.g. TinyTransformerCoeffModel/HFLoRACoeffModel's
    temporal cross-frame conditioning, added after the original
    checkpoints/gbticl_ckpt.pt was trained) -- new params with a documented
    zero/near-zero-impact default when unused (temporal_values=None simply
    never routes through them). A checkpoint saved before such a param
    existed should still load and run correctly for everything it WAS
    trained on; strict=True's hard failure ("Missing key(s)...") is overly
    conservative for this case and was confirmed to break loading
    pre-existing checkpoints (checkpoints/gbticl_ckpt.pt) with a confusing
    error in the Gradio app and run_dataset_pipeline.py alike.

    Missing keys keep their random initialization (fine for a genuinely new
    feature the old checkpoint never used). Unexpected keys are silently
    dropped -- would indicate the checkpoint is for a DIFFERENT model class
    entirely, which model-type auto-detection (gbticl_model_type/
    coeff_model_type in the checkpoint) is meant to prevent upstream; this
    is a second line of defence, not the primary safeguard.
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
