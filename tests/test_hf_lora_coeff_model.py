"""HFLoRACoeffModel: the real pretrained-LLM (distilgpt2 + LoRA) coefficient
predictor. Slower than the other tests (loads a real HF model) -- skipped
automatically if transformers/peft aren't installed or the model can't be
loaded (e.g. no internet on first run, before it's cached locally)."""

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("peft")

from gbticl_pipeline.coeff_model import HFLoRACoeffModel
from gbticl_pipeline.codec import encode_image, decode_image
from gbticl_pipeline.evaluate import psnr
from gbticl_pipeline.graph_model import ContextGradientGBTICL
from gbticl_pipeline.graph_utils import build_laplacian, eigendecompose
from gbticl_pipeline.gft import forward_gft
from gbticl_pipeline.quantization import quantize
from gbticl_pipeline.context import get_context, get_block

WIDE_RANGE = (-2200, 2200)


@pytest.fixture(scope="module")
def hf_model(device):
    try:
        model = HFLoRACoeffModel(base_model_name="distilgpt2", block_size=8, symbol_range=WIDE_RANGE).to(device)
    except Exception as e:  # pragma: no cover -- environment-dependent (no internet, etc.)
        pytest.skip(f"could not load distilgpt2: {e}")
    model.eval()
    return model


def test_hf_lora_model_trainable_params_are_small_fraction(hf_model):
    n_trainable = sum(p.numel() for p in hf_model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in hf_model.parameters())
    assert n_total > 50_000_000, "distilgpt2 should be a real ~82M-param pretrained model"
    assert n_trainable < n_total, "base LM weights must stay frozen -- only LoRA + heads trainable"


def test_hf_lora_forward_sequence_shape_and_finite(hf_model, device):
    n = 64
    eigvals = torch.linspace(0, 8, n, device=device)
    true_values = torch.randint(-50, 50, (n,), device=device).float()
    logits = hf_model.forward_sequence(eigvals, true_values)
    assert logits.shape == (n, WIDE_RANGE[1] - WIDE_RANGE[0] + 1)
    assert torch.isfinite(logits).all()


def test_hf_lora_gradient_flows_to_lora_only(hf_model, device):
    hf_model.zero_grad()
    n = 16
    eigvals = torch.linspace(0, 8, n, device=device)
    true_values = torch.randint(-30, 30, (n,), device=device).float()
    logits = hf_model.forward_sequence(eigvals, true_values)
    logits.sum().backward()
    lora_grads = [p.grad for n_, p in hf_model.named_parameters() if "lora" in n_.lower()]
    assert any(g is not None and g.abs().sum().item() > 0 for g in lora_grads)
    base_grads = [p.grad for n_, p in hf_model.named_parameters()
                  if "lora" not in n_.lower() and "base_model" in n_.lower()]
    assert all(g is None for g in base_grads), "frozen base weights must not receive gradients"


def test_hf_lora_batched_forward_sequence_not_bit_identical_to_incremental(hf_model, device):
    """Documents a confirmed, deliberate design constraint (not a bug):
    forward_sequence (one parallel batched pass -- used only for training)
    and symbol_probs (incremental, KV-cached -- used for decode) run in
    bfloat16 and are NOT guaranteed bit-identical, because a real pretrained
    LM's bf16 attention numerics differ slightly between "recompute
    everything in one batched shape" and "reuse cached K/V, only compute the
    newest position." This is exactly why precompute_encode_probs
    deliberately does NOT use forward_sequence as a fast path (see its
    docstring) -- it loops through symbol_probs instead, so encode and
    decode always take the IDENTICAL code path and therefore DO agree (see
    test_hf_lora_multichannel_interleaved_cache_matches_precompute below).
    This test exists so that re-introducing a batched shortcut for
    precompute_encode_probs without re-verifying this constraint doesn't
    silently reintroduce the encode/decode desync bug found during
    development (PSNR collapsing to ~5dB on real image data)."""
    n = 32
    torch.manual_seed(0)
    eigvals = torch.linspace(0, 8, n, device=device)
    true_values = torch.randint(-100, 100, (n,), device=device).float()

    with torch.no_grad():
        logits_full = hf_model.forward_sequence(eigvals, true_values)
        probs_full = torch.softmax(logits_full.double(), dim=-1)
        probs_full = (probs_full + 1e-6)
        probs_full = probs_full / probs_full.sum(dim=-1, keepdim=True)

        history = []
        any_mismatch = False
        for k in range(n):
            probs_k = hf_model.symbol_probs(eigvals, k, history, WIDE_RANGE)
            if not torch.allclose(probs_k, probs_full[k], atol=1e-4):
                any_mismatch = True
            history.append(int(true_values[k].item()))
        # NOT asserting equality -- the point of this test is that the gap
        # exists (confirmed, expected, bf16-precision-driven), which is
        # exactly the reason precompute_encode_probs avoids this code path
        assert any_mismatch, (
            "if these now match exactly, precompute_encode_probs' incremental-loop "
            "design may no longer be strictly necessary for correctness -- re-verify "
            "before reintroducing a batched fast path"
        )


def test_hf_lora_multichannel_interleaved_cache_matches_precompute(hf_model, device):
    """Regression test for the confirmed multi-slot-cache bug: decode_image/
    decode_video call symbol_probs in INTERLEAVED (k outer, channel inner)
    order across 3 independent per-channel sequences -- a single shared
    cache slot corrupted ~98% of symbols under this exact call pattern
    before the fix (multi-slot dict keyed by id(history))."""
    n = 24
    torch.manual_seed(1)
    eigvals = torch.linspace(0, 6, n, device=device)
    true_values = torch.randint(-40, 40, (n, 3), device=device)

    with torch.no_grad():
        probs_all = hf_model.precompute_encode_probs(eigvals, true_values, WIDE_RANGE)

        history = {0: [], 1: [], 2: []}
        for k in range(n):
            for ch in range(3):
                probs = hf_model.symbol_probs(eigvals, k, history[ch], WIDE_RANGE)
                assert torch.allclose(probs, probs_all[k, ch], atol=1e-4), f"mismatch at k={k} ch={ch}"
                history[ch].append(int(true_values[k, ch].item()))


def test_hf_lora_codec_roundtrip_real_block(hf_model, device):
    """End-to-end sanity check on a real block-sized image: PSNR must be
    governed by quant_step (i.e. reasonable, ~30-60dB), not collapse to
    single digits the way it did under both confirmed bugs above."""
    rng = np.random.default_rng(0)
    h = w = 16
    yy, xx = np.mgrid[0:h, 0:w]
    base = 128 + 50 * np.sin(xx / 4.0) + 30 * np.cos(yy / 5.0)
    img = np.clip(np.stack([base] * 3, axis=-1) + rng.normal(0, 5, size=(h, w, 3)), 0, 255).astype(np.uint8)

    gbticl_model = ContextGradientGBTICL().to(device)
    payload, meta = encode_image(
        img, block_size=8, quant_step=8.0, gbticl_model=gbticl_model, coeff_model=hf_model,
        symbol_range=WIDE_RANGE, device=device,
    )
    recon = decode_image(payload, meta, gbticl_model=gbticl_model, coeff_model=hf_model, device=device)

    p = psnr(img, recon)
    assert p > 20, f"PSNR collapsed to {p:.1f}dB -- likely an encode/decode desync regression"
