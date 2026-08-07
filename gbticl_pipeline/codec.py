"""
The full raster-order encode/decode loop: this is the actual GBT-ICL + LLM
Coefficient Predictor architecture from the diagram, wired end to end and
running on GPU (or CPU/MPS fallback) via `.to(device)`.

encode_image() walks every block in raster order, and for each block:
  1. reads context from the canvas (already-decoded neighbours only)          -- on `device`
  2. GBT-ICL predicts edge weights -> Laplacian -> eigendecomposition (U, Λ)  -- on `device`
  3. forward GFT using U -> coefficients                                     -- on `device`
  4. quantize                                                                -- on `device`
  5. entropy-code each coefficient with the LLM Coefficient Predictor's
     probability model (conditioned on Λ and this block's coefficient
     history), via the real range coder
  6. self-decode (dequantize + inverse GFT) and write the reconstruction into
     the canvas -- this is what makes it "already decoded" for later blocks

decode_image() mirrors this exactly, reading from the bitstream instead of
already knowing the coefficients, but computing GBT-ICL/eigendecomposition
identically from the same canvas-derived context -- no graph and no
probability model is ever transmitted, only the entropy-coded coefficients.

DEVICE BOUNDARY -- read this before assuming everything is GPU-resident:
steps 1-4 and 6 are ordinary tensor math and stay on `device` throughout
(GPU-accelerated automatically when device='cuda', via torch.linalg.eigh and
batched matmuls). Step 5's range coder is a different kind of algorithm --
a stateful, carry-propagating integer bitstream coder -- which is inherently
sequential and does not parallelise on GPU. The one deliberate device->host
transfer in this file is right before each call to the range coder
(`probs.detach().cpu().numpy()` / `int(q[k, ch].item())`). This is the same
pattern real learned-image-compression codecs use: batch the neural
probability model on GPU, do the actual entropy coding on CPU. See the
performance note in coeff_model.py for how to make that GPU batching real
once GBTICLPredictor/CoeffPredictor are trained models instead of placeholders.
"""

import torch

from .graph_model import UniformGBTICL
from .graph_utils import build_laplacian, eigendecompose
from .gft import forward_gft, inverse_gft
from .quantization import quantize, dequantize
from .coeff_model import LaplaceCoeffModel
from .context import get_context, get_block, set_block
from .range_coder import RangeEncoder, RangeDecoder, encode_symbol, decode_symbol
from .device_utils import get_device


def encode_image(rgb_np, block_size=8, quant_step=8.0,
                  gbticl_model=None, coeff_model=None, symbol_range=(-2200, 2200),
                  device=None):
    """
    Args:
        rgb_np: (H, W, 3) uint8 numpy array -- the only thing that starts on CPU;
                everything else is pushed onto `device` immediately.
        device: torch.device, or None to auto-pick GPU > MPS > CPU via device_utils.get_device()
    """
    device = device or get_device()

    gbticl_model = (gbticl_model or UniformGBTICL()).to(device)
    coeff_model = (coeff_model or LaplaceCoeffModel()).to(device)

    H, W, _ = rgb_np.shape
    assert H % block_size == 0 and W % block_size == 0, \
        "image dimensions must be multiples of block_size for this prototype"

    rgb = torch.as_tensor(rgb_np, device=device)  # host -> device, once, up front
    n_bh, n_bw = H // block_size, W // block_size
    canvas = torch.zeros((H, W, 3), dtype=torch.uint8, device=device)
    encoder = RangeEncoder()
    n_symbols = 0

    for i in range(n_bh):
        for j in range(n_bw):
            top, left, valid_top, valid_left = get_context(canvas, i, j, block_size)
            weights = gbticl_model.predict_edge_weights(top, left, valid_top, valid_left, block_size)
            L = build_laplacian(weights, block_size, device=device)
            eigvals, U = eigendecompose(L)  # torch.linalg.eigh -- GPU-accelerated when device='cuda'

            block = get_block(rgb, i, j, block_size)
            coeffs = forward_gft(block, U)
            q = quantize(coeffs, quant_step)

            # Encoder-only fast path: every true coefficient in this block is
            # already known (it's right there in `q`), so get every symbol's
            # probability distribution in one precompute_encode_probs call
            # instead of 192 incremental symbol_probs calls -- see that
            # method's docstring. Correct for any CoeffPredictor (the base
            # class falls back to the same incremental calls if a model
            # doesn't override this), just much faster for ones that do
            # (TinyTransformerCoeffModel).
            probs_all = coeff_model.precompute_encode_probs(eigvals, q, symbol_range)  # (n, 3, n_symbols)

            for k in range(block_size * block_size):
                for ch in range(3):
                    probs = probs_all[k, ch]
                    val = int(q[k, ch].item())  # <- device->host: range coder needs a plain int
                    sym_idx = val - symbol_range[0]
                    assert 0 <= sym_idx <= (symbol_range[1] - symbol_range[0]), (
                        f"coefficient {val} out of symbol_range {symbol_range} "
                        f"at block ({i},{j}) k={k} ch={ch} -- widen symbol_range or quant_step"
                    )
                    # <- device->host: range coder is CPU-only by design, see module docstring
                    encode_symbol(encoder, probs.detach().cpu().numpy(), sym_idx)
                    n_symbols += 1

            dq = dequantize(q, quant_step)
            recon = inverse_gft(dq, U, block_size)
            recon_u8 = torch.clamp(torch.round(recon), 0, 255).to(torch.uint8)
            set_block(canvas, i, j, block_size, recon_u8)

    payload = encoder.finish()
    meta = dict(H=H, W=W, block_size=block_size, quant_step=quant_step,
                symbol_range=symbol_range, n_symbols_coded=n_symbols, device=str(device))
    return payload, meta


def decode_image(payload, meta, gbticl_model=None, coeff_model=None, device=None):
    device = device or get_device()
    gbticl_model = (gbticl_model or UniformGBTICL()).to(device)
    coeff_model = (coeff_model or LaplaceCoeffModel()).to(device)

    H, W = meta["H"], meta["W"]
    block_size = meta["block_size"]
    quant_step = meta["quant_step"]
    symbol_range = meta["symbol_range"]
    n_bh, n_bw = H // block_size, W // block_size

    canvas = torch.zeros((H, W, 3), dtype=torch.uint8, device=device)
    decoder = RangeDecoder(payload)

    for i in range(n_bh):
        for j in range(n_bw):
            top, left, valid_top, valid_left = get_context(canvas, i, j, block_size)
            weights = gbticl_model.predict_edge_weights(top, left, valid_top, valid_left, block_size)
            L = build_laplacian(weights, block_size, device=device)
            eigvals, U = eigendecompose(L)

            q = torch.zeros((block_size * block_size, 3), dtype=torch.int64, device=device)
            history = {0: [], 1: [], 2: []}
            for k in range(block_size * block_size):
                for ch in range(3):
                    probs = coeff_model.symbol_probs(eigvals, k, history[ch], symbol_range)
                    sym_idx = decode_symbol(decoder, probs.detach().cpu().numpy())
                    val = sym_idx + symbol_range[0]
                    q[k, ch] = val
                    history[ch].append(val)

            dq = dequantize(q, quant_step)
            recon = inverse_gft(dq, U, block_size)
            recon_u8 = torch.clamp(torch.round(recon), 0, 255).to(torch.uint8)
            set_block(canvas, i, j, block_size, recon_u8)

    return canvas.cpu().numpy()  # host boundary at the very end, for PIL/PSNR/etc.
