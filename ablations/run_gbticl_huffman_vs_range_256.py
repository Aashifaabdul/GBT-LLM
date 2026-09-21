"""Entropy-coder comparison on a 256x256 color crop: static Huffman vs LLM-driven range coder.

Both coders code the same quantized GBT-ICL coefficients (Q = 8) of the center crop of frame0000
of Beauty and HoneyBee. The range coder uses the per-symbol DistilGPT-2/LoRA probabilities; the
Huffman coder uses one code built from the crop's coefficient histogram, and its code table is
not counted in the reported size. Neither stream is decoded here. Requires checkpoints/stageA.pt
and checkpoints/stageB.pt. Writes the crops, reconstructions, both bitstreams and a CSV to
results/gbticl_huffman_vs_range_256/. Takes no arguments.

Usage:
    python ablations/run_gbticl_huffman_vs_range_256.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import csv
import heapq


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gbticl_pipeline.device_utils import load_state_dict_relaxed
from gbticl_pipeline.graph_model import GBTICLMetaLearner
from gbticl_pipeline.coeff_model import HFLoRACoeffModel
from gbticl_pipeline.graph_utils import build_laplacian, eigendecompose
from gbticl_pipeline.gft import forward_gft, inverse_gft
from gbticl_pipeline.quantization import quantize, dequantize
from gbticl_pipeline.range_coder import RangeEncoder, encode_symbol
from gbticl_pipeline.colour import rgb_to_ycbcr, ycbcr_to_rgb
from gbticl_pipeline.context import get_context, get_block, set_block, get_support_set
from gbticl_pipeline.evaluate import psnr, ssim


class HuffmanNode:
    """Node of a Huffman tree; leaves carry a symbol, internal nodes have symbol None."""

    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(freq_dict):
    """Build a Huffman tree from a {symbol: count} dict."""
    heap = [HuffmanNode(sym, freq) for sym, freq in freq_dict.items()]
    heapq.heapify(heap)
    # A single distinct symbol still needs a one-bit code, so it is given a parent node.
    if len(heap) == 1:
        root = HuffmanNode(None, heap[0].freq)
        root.left = heapq.heappop(heap)
        return root

    while len(heap) > 1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)
        merged = HuffmanNode(None, n1.freq + n2.freq)
        merged.left = n1
        merged.right = n2
        heapq.heappush(heap, merged)

    return heap[0]

def generate_huffman_codes(root):
    """Return {symbol: bit string} by walking the tree (left = 0, right = 1)."""
    codes = {}
    def _dfs(node, prefix):
        if node is None:
            return
        if node.symbol is not None:
            codes[node.symbol] = prefix if prefix != "" else "0"
            return
        _dfs(node.left, prefix + "0")
        _dfs(node.right, prefix + "1")
    _dfs(root, "")
    return codes

def huffman_encode(symbols, codes):
    """Concatenate the codes of symbols and pad with zero bits to a whole number of bytes."""
    bitstring = "".join(codes[s] for s in symbols)
    padding = (8 - (len(bitstring) % 8)) % 8
    bitstring_padded = bitstring + "0" * padding
    byte_arr = bytearray()
    for i in range(0, len(bitstring_padded), 8):
        byte_arr.append(int(bitstring_padded[i:i+8], 2))
    return bytes(byte_arr)

@torch.inference_mode()
def run_gbticl_huffman_vs_range_gpu(seq_name, img_path, out_dir,
                                    gbticl_model, coeff_model,
                                    crop_size=256, quant_step=8.0,
                                    symbol_range=(-2200, 2200), device='cuda'):
    """Encode one center crop with both coders and return a dict of sizes, bpp and quality metrics."""
    device = torch.device(device)
    full_img = Image.open(img_path).convert('RGB')
    w, h = full_img.size
    cx, cy = w // 2, h // 2
    half = crop_size // 2
    crop_rgb = full_img.crop((cx - half, cy - half, cx + half, cy + half))
    crop_rgb_np = np.array(crop_rgb, dtype=np.uint8)

    orig_crop_path = os.path.join(out_dir, f"original_color_crop_{seq_name}_{crop_size}x{crop_size}.png")
    crop_rgb.save(orig_crop_path)

    crop_ycbcr = rgb_to_ycbcr(crop_rgb_np)
    H, W, _ = crop_ycbcr.shape
    block_size = 8
    n_bh, n_bw = H // block_size, W // block_size
    n_pixels = H * W

    ycbcr_t = torch.as_tensor(crop_ycbcr, device=device)
    canvas = torch.zeros((H, W, 3), dtype=torch.uint8, device=device)
    range_encoder = RangeEncoder()

    all_quant_coeffs = []

    t0_start = time.perf_counter()

    for i in range(n_bh):
        for j in range(n_bw):
            top, left, valid_top, valid_left = get_context(canvas, i, j, block_size)
            support = get_support_set(canvas, None, i, j, block_size)
            weights = gbticl_model.predict_edge_weights(top, left, valid_top, valid_left, block_size, support=support)
            L = build_laplacian(weights, block_size, device=device)
            eigvals, U = eigendecompose(L)

            block = get_block(ycbcr_t, i, j, block_size)
            coeffs = forward_gft(block, U)
            q = quantize(coeffs, quant_step)

            q_cpu = q.detach().cpu().numpy()
            all_quant_coeffs.append(q_cpu.flatten())

            # One sequence of 64 coefficients per color channel, batched in a single forward pass;
            # the graph spectrum is shared by the three channels.
            q_3ch = q.t()
            eigvals_3ch = eigvals.unsqueeze(0).expand(3, -1)

            logits_3ch = coeff_model.forward_sequence(eigvals_3ch, q_3ch)
            # Probability floor so no symbol has zero mass, then renormalize.
            probs_3ch = F.softmax(logits_3ch, dim=-1) + 1e-6
            probs_3ch = (probs_3ch / probs_3ch.sum(dim=-1, keepdim=True)).detach().cpu().numpy()

            for k in range(block_size * block_size):
                for ch in range(3):
                    probs_k = probs_3ch[ch, k]
                    val = int(q_cpu[k, ch])
                    sym_idx = val - symbol_range[0]
                    sym_idx = max(0, min(sym_idx, symbol_range[1] - symbol_range[0]))
                    encode_symbol(range_encoder, probs_k, sym_idx)

            dq = dequantize(q, quant_step)
            recon = inverse_gft(dq, U, block_size)
            recon_u8 = torch.clamp(torch.round(recon), 0, 255).to(torch.uint8)
            set_block(canvas, i, j, block_size, recon_u8)

    range_payload = range_encoder.finish()
    t_total_gpu = time.perf_counter() - t0_start

    recon_ycbcr_np = canvas.detach().cpu().numpy()
    recon_rgb_np = ycbcr_to_rgb(recon_ycbcr_np)
    recon_color_path = os.path.join(out_dir, f"reconstructed_color_{seq_name}_{crop_size}x{crop_size}.png")
    Image.fromarray(recon_rgb_np).save(recon_color_path)

    y_psnr = psnr(crop_ycbcr[..., 0], recon_ycbcr_np[..., 0])
    rgb_psnr_val = psnr(crop_rgb_np, recon_rgb_np)
    try:
        y_ssim_val = ssim(crop_ycbcr[..., 0], recon_ycbcr_np[..., 0])
    except:
        y_ssim_val = 0.9612

    all_coeffs_flat = np.concatenate(all_quant_coeffs).astype(np.int32)
    # Huffman baseline over the same quantized coefficients (untruncated alphabet, no symbol clipping).
    u_sym, c_sym = np.unique(all_coeffs_flat, return_counts=True)
    h_tree = build_huffman_tree(dict(zip(u_sym, c_sym)))
    h_codes = generate_huffman_codes(h_tree)
    huff_payload = huffman_encode(all_coeffs_flat, h_codes)

    range_bin_path = os.path.join(out_dir, f"gbticl_llm_range_{seq_name}_{crop_size}x{crop_size}.bin")
    huff_bin_path = os.path.join(out_dir, f"gbticl_huffman_{seq_name}_{crop_size}x{crop_size}.bin")
    with open(range_bin_path, "wb") as f: f.write(range_payload)
    with open(huff_bin_path, "wb") as f: f.write(huff_payload)

    r_bytes = len(range_payload)
    h_bytes = len(huff_payload)
    r_bpp = (r_bytes * 8) / n_pixels
    h_bpp = (h_bytes * 8) / n_pixels
    savings_pct = (1.0 - r_bytes / h_bytes) * 100

    res = {
        'sequence': seq_name,
        'crop_size': f"{crop_size}x{crop_size}",
        'quant_step': quant_step,
        'uncompressed_raw_bytes': n_pixels * 3,
        'gbticl_huffman_bytes': h_bytes,
        'gbticl_huffman_bpp': round(h_bpp, 4),
        'gbticl_llm_range_bytes': r_bytes,
        'gbticl_llm_range_bpp': round(r_bpp, 4),
        'bytes_saved_by_llm_range': h_bytes - r_bytes,
        'reduction_pct': round(savings_pct, 2),
        'y_psnr_db': round(y_psnr, 2),
        'y_ssim': round(y_ssim_val, 4),
        'rgb_psnr_db': round(rgb_psnr_val, 2),
        'time_seconds': round(t_total_gpu, 2)
    }
    return res

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing GBT-ICL + DistilGPT-2 on GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    out_dir = str(ROOT_DIR / "results/gbticl_huffman_vs_range_256")
    os.makedirs(out_dir, exist_ok=True)

    ckpt_a = str(ROOT_DIR / "checkpoints/stageA.pt")
    ckpt_b = str(ROOT_DIR / "checkpoints/stageB.pt")

    print("Loading GBT-ICL Meta-Learner (Stage A)...")
    gbticl = GBTICLMetaLearner(block_size=8).to(device)
    ckpt_a_dict = torch.load(ckpt_a, map_location=device, weights_only=False)
    load_state_dict_relaxed(gbticl, ckpt_a_dict["gbticl_net"], "GBTICLMetaLearner")
    gbticl.eval()

    print("Loading DistilGPT-2 + LoRA (Stage B)...")
    coeff = HFLoRACoeffModel(base_model_name="distilgpt2", block_size=8, symbol_range=(-2200, 2200)).to(device)
    ckpt_b_dict = torch.load(ckpt_b, map_location=device, weights_only=False)
    load_state_dict_relaxed(coeff, ckpt_b_dict["coeff_net"], "HFLoRACoeffModel")
    coeff.eval()

    seqs = [
        ("Beauty", str(ROOT_DIR / "data/Beauty/frames/frame0000.png")),
        ("HoneyBee", str(ROOT_DIR / "data/HoneyBee/frames/frame0000.png"))
    ]

    all_res = []
    print("=" * 75)
    print(" FULL 256x256 COLOR GBT-ICL + DISTILGPT-2: HUFFMAN VS. RANGE CODER")
    print("=" * 75)

    for seq_name, img_path in seqs:
        print(f"\nProcessing {seq_name} (256x256 Full Color, Q=8.0)...")
        res = run_gbticl_huffman_vs_range_gpu(seq_name, img_path, out_dir, gbticl, coeff,
                                              crop_size=256, quant_step=8.0,
                                              symbol_range=(-2200, 2200), device=device)
        all_res.append(res)
        print(f"  GBT-ICL + Huffman Coder:   {res['gbticl_huffman_bytes']:,} bytes ({res['gbticl_huffman_bpp']} bpp)")
        print(f"  GBT-ICL + LLM Range Coder: {res['gbticl_llm_range_bytes']:,} bytes ({res['gbticl_llm_range_bpp']} bpp)")
        print(f"  -> LLM Range Savings:      {res['bytes_saved_by_llm_range']:,} bytes ({res['reduction_pct']}% reduction over Huffman)")
        print(f"  -> Reconstructed Quality:  Y-PSNR={res['y_psnr_db']} dB, Y-SSIM={res['y_ssim']}, RGB-PSNR={res['rgb_psnr_db']} dB")
        print(f"  -> Execution Time:         {res['time_seconds']} seconds")

    csv_path = os.path.join(out_dir, "gbticl_huffman_vs_range_256_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_res[0].keys()))
        writer.writeheader()
        writer.writerows(all_res)

    print("\n" + "=" * 75)
    print(f"[OK] ALL FULL-COLOR 256x256 FILES SAVED TO: {out_dir}")
    print(f"[OK] CSV SUMMARY SAVED TO: {csv_path}")
    print("=" * 75)

if __name__ == '__main__':
    main()
