"""Regression test for a real bug hit interactively via the Gradio app:
loading checkpoints/gbticl_ckpt.pt (saved before TinyTransformerCoeffModel
gained temporal_value_embed/no_temporal params) crashed with "Missing
key(s) in state_dict" under strict=True. load_state_dict_relaxed fixes
this by loading what matches and leaving new params at their init."""

import torch

from gbticl_pipeline.device_utils import load_state_dict_relaxed
from gbticl_pipeline.coeff_model import TinyTransformerCoeffModel


def test_load_state_dict_relaxed_handles_missing_new_params(device):
    """Simulates loading an 'old' checkpoint (state_dict missing keys a
    newer model version added) into the current model class."""
    model = TinyTransformerCoeffModel(block_size=8, symbol_range=(-2200, 2200)).to(device)
    old_style_state_dict = {
        k: v for k, v in model.state_dict().items()
        if not k.startswith("temporal_value_embed") and k != "no_temporal"
    }
    assert len(old_style_state_dict) < len(model.state_dict())  # sanity: we actually dropped some keys

    fresh_model = TinyTransformerCoeffModel(block_size=8, symbol_range=(-2200, 2200)).to(device)
    result = load_state_dict_relaxed(fresh_model, old_style_state_dict, "TinyTransformerCoeffModel")

    assert "no_temporal" in result.missing_keys
    assert any(k.startswith("temporal_value_embed") for k in result.missing_keys)
    assert not result.unexpected_keys
    # every OTHER param should have loaded correctly (not left at fresh_model's own random init)
    assert torch.equal(fresh_model.value_embed[0].weight, model.value_embed[0].weight)


def test_load_state_dict_relaxed_flags_unexpected_keys():
    model = TinyTransformerCoeffModel(block_size=8, symbol_range=(-2200, 2200))
    bad_state_dict = dict(model.state_dict())
    bad_state_dict["totally_unrelated_param"] = torch.zeros(3)

    result = load_state_dict_relaxed(model, bad_state_dict, "TinyTransformerCoeffModel")
    assert "totally_unrelated_param" in result.unexpected_keys
