"""
Raster-order encoder and decoder (image and video).

For every block, in raster order, the encoder
  1. reads the causal context from the canvas of already reconstructed blocks,
  2. predicts the edge weights (GBT-ICL), builds the Laplacian and
     eigendecomposes it to obtain the basis U and spectrum,
  3. applies the forward GFT and quantises the coefficients,
  4. range-codes every coefficient with the distribution of the coefficient
     model, conditioned on the spectrum and the coefficients coded before it,
  5. reconstructs the block (dequantise + inverse GFT) and writes it to the
     canvas, so later blocks see exactly what the decoder will see.

The decoder repeats steps 1-2 from its own canvas and reads the coefficients
from the bitstream. Neither the graph nor any probability model is transmitted.

Device boundary: steps 1-3 and 5 are tensor operations and run on `device`. The
range coder (step 4) is a sequential integer algorithm, so each probability
vector is moved to the CPU right before it is coded.

Video mode (encode_video / decode_video) applies the same per-block procedure
to each frame with the reconstructed previous frame available as extra context:
  - GBTICLMetaLearner receives a spatio-temporal support set (context.py);
  - TinyTransformerCoeffModel and HFLoRACoeffModel receive the co-located
    decoded coefficients of the previous frame.
The first frame has no previous frame, so the temporal slots are marked missing.
Models that do not use a support set or temporal values (for example
UniformGBTICL and LaplaceCoeffModel) simply do not receive those arguments.
Every frame is coded as an independent range-coder payload.
"""

import time

import torch

from .graph_model import UniformGBTICL, GBTICLMetaLearner
from .graph_utils import build_laplacian, eigendecompose, dct_basis_and_eigvals
from .gft import forward_gft, inverse_gft
from .quantization import quantize, dequantize
from .coeff_model import LaplaceCoeffModel, TinyTransformerCoeffModel, HFLoRACoeffModel
from .context import get_context, get_block, set_block, get_support_set
from .range_coder import RangeEncoder, RangeDecoder, encode_symbol, decode_symbol
from .device_utils import get_device


@torch.inference_mode()
def encode_image(rgb_np, block_size=8, quant_step=8.0,
                  gbticl_model=None, coeff_model=None, symbol_range=(-2200, 2200),
                  device=None, fixed_basis=False):
    """
    Encode one image.

    Args:
        rgb_np: (H, W, 3) uint8 array; height and width must be multiples of block_size
        block_size: transform block size
        quant_step: uniform quantisation step
        gbticl_model: edge-weight predictor (default UniformGBTICL)
        coeff_model: coefficient entropy model (default LaplaceCoeffModel)
        symbol_range: (lo, hi) inclusive range of quantised values the coder supports
        device: torch device; None selects CUDA, then MPS, then CPU
        fixed_basis: use the fixed 2D DCT-II basis in every block instead of a
            predicted graph basis (DCT baseline; gbticl_model is then ignored)

    Returns:
        payload: bytes, the range-coded bitstream
        meta: dict with the settings the decoder needs
    """
    # Load the models on the selected device
    device = device or get_device()

    gbticl_model = (gbticl_model or UniformGBTICL()).to(device)
    coeff_model = (coeff_model or LaplaceCoeffModel()).to(device)
    if fixed_basis:
        dct_eigvals, dct_U = dct_basis_and_eigvals(block_size, device=device)

    H, W, _ = rgb_np.shape
    assert H % block_size == 0 and W % block_size == 0, \
        "image dimensions must be multiples of block_size for this prototype"

    # Create the input tensor, decoded canvas, and range encoder
    rgb = torch.as_tensor(rgb_np, device=device)
    n_bh, n_bw = H // block_size, W // block_size
    canvas = torch.zeros((H, W, 3), dtype=torch.uint8, device=device)
    encoder = RangeEncoder()
    n_symbols = 0

    for i in range(n_bh):
        for j in range(n_bw):
            # Reproduce the DCT or context-predicted graph basis
            if fixed_basis:
                eigvals, U = dct_eigvals, dct_U
            else:
                top, left, valid_top, valid_left = get_context(canvas, i, j, block_size)
                if isinstance(gbticl_model, GBTICLMetaLearner):
                    support = get_support_set(canvas, None, i, j, block_size)
                    weights = gbticl_model.predict_edge_weights(top, left, valid_top, valid_left, block_size, support=support)
                else:
                    weights = gbticl_model.predict_edge_weights(top, left, valid_top, valid_left, block_size)
                L = build_laplacian(weights, block_size, device=device)
                eigvals, U = eigendecompose(L)

            # Transform and quantize the current block
            block = get_block(rgb, i, j, block_size)
            coeffs = forward_gft(block, U)
            q = quantize(coeffs, quant_step)

            # Predict symbol probabilities and range-code each coefficient
            # The encoder knows all coefficients of the block, so it obtains every
            # distribution with one precompute_encode_probs call
            probs_all = coeff_model.precompute_encode_probs(eigvals, q, symbol_range)  # (n, 3, n_symbols)

            for k in range(block_size * block_size):
                for ch in range(3):
                    probs = probs_all[k, ch]
                    val = int(q[k, ch].item())
                    sym_idx = val - symbol_range[0]
                    assert 0 <= sym_idx <= (symbol_range[1] - symbol_range[0]), (
                        f"coefficient {val} out of symbol_range {symbol_range} "
                        f"at block ({i},{j}) k={k} ch={ch} -- widen symbol_range or quant_step"
                    )

                    # device -> host transfer for the CPU range coder
                    encode_symbol(encoder, probs.detach().cpu().numpy(), sym_idx)
                    n_symbols += 1

            # Reconstruct the block for future causal context
            dq = dequantize(q, quant_step)
            recon = inverse_gft(dq, U, block_size)
            recon_u8 = torch.clamp(torch.round(recon), 0, 255).to(torch.uint8)
            set_block(canvas, i, j, block_size, recon_u8)

    # Bitstream plus what the decoder needs
    payload = encoder.finish()
    meta = dict(H=H, W=W, block_size=block_size, quant_step=quant_step,
                symbol_range=symbol_range, n_symbols_coded=n_symbols, device=str(device),
                fixed_basis=fixed_basis)
    return payload, meta


@torch.inference_mode()
def decode_image(payload, meta, gbticl_model=None, coeff_model=None, device=None):
    """Decode one image from `payload` and its `meta` dict; returns an (H, W, 3) uint8 array.

    The models must be the same as (and identically configured to) those used to encode.
    """
    # Load the models and decoding settings
    device = device or get_device()
    gbticl_model = (gbticl_model or UniformGBTICL()).to(device)
    coeff_model = (coeff_model or LaplaceCoeffModel()).to(device)

    H, W = meta["H"], meta["W"]
    block_size = meta["block_size"]
    quant_step = meta["quant_step"]
    symbol_range = meta["symbol_range"]
    fixed_basis = meta.get("fixed_basis", False)
    n_bh, n_bw = H // block_size, W // block_size
    if fixed_basis:
        dct_eigvals, dct_U = dct_basis_and_eigvals(block_size, device=device)

    # Create the decoded canvas and range decoder
    canvas = torch.zeros((H, W, 3), dtype=torch.uint8, device=device)
    decoder = RangeDecoder(payload)

    for i in range(n_bh):
        for j in range(n_bw):
            # Reproduce the same transform basis used by the encoder
            if fixed_basis:
                eigvals, U = dct_eigvals, dct_U
            else:
                top, left, valid_top, valid_left = get_context(canvas, i, j, block_size)
                if isinstance(gbticl_model, GBTICLMetaLearner):
                    support = get_support_set(canvas, None, i, j, block_size)
                    weights = gbticl_model.predict_edge_weights(top, left, valid_top, valid_left, block_size, support=support)
                else:
                    weights = gbticl_model.predict_edge_weights(top, left, valid_top, valid_left, block_size)
                L = build_laplacian(weights, block_size, device=device)
                eigvals, U = eigendecompose(L)

            # Predict probabilities and decode the coefficient sequence
            q = torch.zeros((block_size * block_size, 3), dtype=torch.int64, device=device)
            history = {0: [], 1: [], 2: []}
            for k in range(block_size * block_size):
                for ch in range(3):
                    probs = coeff_model.symbol_probs(eigvals, k, history[ch], symbol_range)
                    sym_idx = decode_symbol(decoder, probs.detach().cpu().numpy())
                    val = sym_idx + symbol_range[0]
                    q[k, ch] = val
                    history[ch].append(val)

            # Invert the transform and update causal context
            dq = dequantize(q, quant_step)
            recon = inverse_gft(dq, U, block_size)
            recon_u8 = torch.clamp(torch.round(recon), 0, 255).to(torch.uint8)
            set_block(canvas, i, j, block_size, recon_u8)

    return canvas.cpu().numpy()


# ------------------------------------------------------------------------------
# Video mode: see the module docstring.
# ------------------------------------------------------------------------------
def _supports_context_set(gbticl_model):
    """True if the model takes a support set (GBTICLMetaLearner)."""
    return isinstance(gbticl_model, GBTICLMetaLearner)


def _supports_temporal_coeffs(coeff_model):
    """True if the model accepts previous-frame coefficients as extra conditioning."""
    return isinstance(coeff_model, (TinyTransformerCoeffModel, HFLoRACoeffModel))


@torch.inference_mode()
def encode_video(frames_np, block_size=8, quant_step=8.0,
                  gbticl_model=None, coeff_model=None, symbol_range=(-2200, 2200),
                  device=None, fixed_basis=False, verbose=True, progress_interval_s=5.0):
    """
    Encode a sequence of frames, each as an independent payload.

    Args:
        frames_np: (T, H, W, 3) uint8 array, or a sequence of (H, W, 3) frames of equal shape
        verbose: print a progress line about every progress_interval_s seconds
        (remaining arguments as in encode_image)

    Returns:
        payloads: list of T bytes objects
        metas: list of T meta dicts (same layout as encode_image's)
    """
    # Load models and initialize temporal state
    device = device or get_device()
    gbticl_model = (gbticl_model or UniformGBTICL()).to(device)
    coeff_model = (coeff_model or LaplaceCoeffModel()).to(device)
    use_support = _supports_context_set(gbticl_model) and not fixed_basis
    use_temporal_coeffs = _supports_temporal_coeffs(coeff_model)
    if fixed_basis:
        dct_eigvals, dct_U = dct_basis_and_eigvals(block_size, device=device)

    payloads, metas = [], []
    prev_canvas = None
    prev_q = None  # (n_bh, n_bw, n, 3) coefficients of the previous frame

    n_frames_total = len(frames_np)
    blocks_done_total = 0
    run_start = time.time()
    last_print = run_start

    # Encode each frame into an independent payload
    for frame_idx, frame_np in enumerate(frames_np):
        H, W, _ = frame_np.shape
        assert H % block_size == 0 and W % block_size == 0, \
            "frame dimensions must be multiples of block_size for this prototype"

        rgb = torch.as_tensor(frame_np, device=device)
        n_bh, n_bw = H // block_size, W // block_size
        n_blocks_frame = n_bh * n_bw
        canvas = torch.zeros((H, W, 3), dtype=torch.uint8, device=device)
        encoder = RangeEncoder()
        n_symbols = 0
        n = block_size * block_size
        this_q = torch.zeros((n_bh, n_bw, n, 3), dtype=torch.int64, device=device)

        # Process blocks using spatial and previous-frame context
        for i in range(n_bh):
            for j in range(n_bw):
                if verbose:
                    now = time.time()
                    if now - last_print >= progress_interval_s:
                        block_idx_frame = i * n_bw + j
                        elapsed = now - run_start
                        blocks_seen = blocks_done_total + block_idx_frame
                        rate = blocks_seen / elapsed if elapsed > 0 else 0.0
                        print(f"  [encode] frame {frame_idx + 1}/{n_frames_total}, "
                              f"block {block_idx_frame}/{n_blocks_frame} "
                              f"({rate:.2f} blocks/s, {elapsed:.0f}s elapsed)")
                        last_print = now
                if fixed_basis:
                    eigvals, U = dct_eigvals, dct_U
                else:
                    top, left, valid_top, valid_left = get_context(canvas, i, j, block_size)

                    if use_support:
                        support = get_support_set(canvas, prev_canvas, i, j, block_size)
                        weights = gbticl_model.predict_edge_weights(
                            top, left, valid_top, valid_left, block_size, support=support
                        )
                    else:
                        weights = gbticl_model.predict_edge_weights(top, left, valid_top, valid_left, block_size)

                    L = build_laplacian(weights, block_size, device=device)
                    eigvals, U = eigendecompose(L)

                # Transform, quantize, predict, and encode the block
                block = get_block(rgb, i, j, block_size)
                coeffs = forward_gft(block, U)
                q = quantize(coeffs, quant_step)

                temporal_values = None
                if use_temporal_coeffs and prev_q is not None:
                    temporal_values = prev_q[i, j].to(torch.float32)  # (n, 3)

                probs_all = coeff_model.precompute_encode_probs(
                    eigvals, q, symbol_range,
                    **({"temporal_values": temporal_values} if use_temporal_coeffs else {}),
                )  # (n, 3, n_symbols)

                for k in range(n):
                    for ch in range(3):
                        probs = probs_all[k, ch]
                        val = int(q[k, ch].item())
                        sym_idx = val - symbol_range[0]
                        assert 0 <= sym_idx <= (symbol_range[1] - symbol_range[0]), (
                            f"coefficient {val} out of symbol_range {symbol_range} "
                            f"at block ({i},{j}) k={k} ch={ch} -- widen symbol_range or quant_step"
                        )
                        encode_symbol(encoder, probs.detach().cpu().numpy(), sym_idx)
                        n_symbols += 1

                this_q[i, j] = q

                # Reconstruct the block on the causal canvas
                dq = dequantize(q, quant_step)
                recon = inverse_gft(dq, U, block_size)
                recon_u8 = torch.clamp(torch.round(recon), 0, 255).to(torch.uint8)
                set_block(canvas, i, j, block_size, recon_u8)

        # Save the frame payload and temporal state
        payload = encoder.finish()
        meta = dict(H=H, W=W, block_size=block_size, quant_step=quant_step,
                    symbol_range=symbol_range, n_symbols_coded=n_symbols, device=str(device),
                    fixed_basis=fixed_basis)
        payloads.append(payload)
        metas.append(meta)

        blocks_done_total += n_blocks_frame
        prev_canvas = canvas.clone()
        prev_q = this_q

    return payloads, metas


@torch.inference_mode()
def decode_video(payloads, metas, gbticl_model=None, coeff_model=None, device=None,
                  verbose=True, progress_interval_s=5.0):
    """
    Decode the payloads produced by encode_video; the models must match those used for encoding.

    Returns:
        list of T (H, W, 3) uint8 arrays
    """
    # Load models and initialize temporal state
    device = device or get_device()
    gbticl_model = (gbticl_model or UniformGBTICL()).to(device)
    coeff_model = (coeff_model or LaplaceCoeffModel()).to(device)
    use_temporal_coeffs = _supports_temporal_coeffs(coeff_model)

    recon_frames = []
    prev_canvas = None
    prev_q = None

    n_frames_total = len(payloads)
    blocks_done_total = 0
    run_start = time.time()
    last_print = run_start

    # Decode each frame payload in sequence
    for frame_idx, (payload, meta) in enumerate(zip(payloads, metas)):
        H, W = meta["H"], meta["W"]
        block_size = meta["block_size"]
        quant_step = meta["quant_step"]
        symbol_range = meta["symbol_range"]
        fixed_basis = meta.get("fixed_basis", False)
        use_support = _supports_context_set(gbticl_model) and not fixed_basis
        n_bh, n_bw = H // block_size, W // block_size
        n_blocks_frame = n_bh * n_bw
        n = block_size * block_size
        if fixed_basis:
            dct_eigvals, dct_U = dct_basis_and_eigvals(block_size, device=device)

        canvas = torch.zeros((H, W, 3), dtype=torch.uint8, device=device)
        decoder = RangeDecoder(payload)
        this_q = torch.zeros((n_bh, n_bw, n, 3), dtype=torch.int64, device=device)

        # Reproduce context, probabilities, and coefficients per block
        for i in range(n_bh):
            for j in range(n_bw):
                if verbose:
                    now = time.time()
                    if now - last_print >= progress_interval_s:
                        block_idx_frame = i * n_bw + j
                        elapsed = now - run_start
                        blocks_seen = blocks_done_total + block_idx_frame
                        rate = blocks_seen / elapsed if elapsed > 0 else 0.0
                        print(f"  [decode] frame {frame_idx + 1}/{n_frames_total}, "
                              f"block {block_idx_frame}/{n_blocks_frame} "
                              f"({rate:.2f} blocks/s, {elapsed:.0f}s elapsed)")
                        last_print = now
                if fixed_basis:
                    eigvals, U = dct_eigvals, dct_U
                else:
                    top, left, valid_top, valid_left = get_context(canvas, i, j, block_size)

                    if use_support:
                        support = get_support_set(canvas, prev_canvas, i, j, block_size)
                        weights = gbticl_model.predict_edge_weights(
                            top, left, valid_top, valid_left, block_size, support=support
                        )
                    else:
                        weights = gbticl_model.predict_edge_weights(top, left, valid_top, valid_left, block_size)

                    L = build_laplacian(weights, block_size, device=device)
                    eigvals, U = eigendecompose(L)

                temporal_values = None
                if use_temporal_coeffs and prev_q is not None:
                    temporal_values = prev_q[i, j].to(torch.float32)  # (n, 3)

                q = torch.zeros((n, 3), dtype=torch.int64, device=device)
                history = {0: [], 1: [], 2: []}
                for k in range(n):
                    for ch in range(3):
                        temp_ch = temporal_values[:, ch] if temporal_values is not None else None
                        kwargs = {"temporal_values": temp_ch} if use_temporal_coeffs else {}
                        probs = coeff_model.symbol_probs(eigvals, k, history[ch], symbol_range, **kwargs)
                        sym_idx = decode_symbol(decoder, probs.detach().cpu().numpy())
                        val = sym_idx + symbol_range[0]
                        q[k, ch] = val
                        history[ch].append(val)

                this_q[i, j] = q

                # Reconstruct the block and update the canvas
                dq = dequantize(q, quant_step)
                recon = inverse_gft(dq, U, block_size)
                recon_u8 = torch.clamp(torch.round(recon), 0, 255).to(torch.uint8)
                set_block(canvas, i, j, block_size, recon_u8)

        # Store the frame and carry its state to the next frame
        recon_frames.append(canvas.cpu().numpy())
        blocks_done_total += n_blocks_frame
        prev_canvas = canvas.clone()
        prev_q = this_q

    return recon_frames
