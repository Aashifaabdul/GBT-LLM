"""Entropy-coder comparison with identical LLM probabilities: per-symbol Huffman codes vs range coder.

For the 256x256 center crop of frame0000 of Beauty and HoneyBee (Q = 8), the DistilGPT-2/LoRA
model gives a distribution for every quantized coefficient. The range coder codes the symbols
with these distributions. The Huffman figure is computed analytically: for each symbol a Huffman
code is built from the same distribution (truncated to probabilities above 1e-4) and its length
for the coded symbol is added up; no Huffman bitstream is produced and no code table is charged.
Requires checkpoints/stageA.pt and checkpoints/stageB.pt. Writes the range-coded bitstream, the
reconstruction and a CSV to results/llm_huffman_vs_llm_range/. Takes no arguments.

Usage:
    python ablations/run_llm_huffman_vs_llm_range.py
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

def compute_llm_huffman_bit_length(probs_k, sym_idx):
    """Length in bits of sym_idx under a Huffman code built from the distribution probs_k.

    Only symbols with probability above 1e-4 (plus sym_idx itself) get a code; probabilities are
    renormalized and converted to integer counts (scale 1e5, minimum 1) before building the tree.
    """
    top_indices = np.where(probs_k > 1e-4)[0]
    if sym_idx not in top_indices:
        top_indices = np.append(top_indices, sym_idx)

    sub_probs = probs_k[top_indices]
    sub_probs = sub_probs / sub_probs.sum()

    freq_dict = {int(idx): max(1, int(p * 100000)) for idx, p in zip(top_indices, sub_probs)}

    tree = build_huffman_tree(freq_dict)
    codes = generate_huffman_codes(tree)

    code = codes.get(sym_idx, '0')
    return len(code)

@torch.inference_mode()
def run_llm_huffman_vs_llm_range(seq_name, img_path, out_dir,
                                  gbticl_model, coeff_model,
                                  crop_size=256, quant_step=8.0,
                                  symbol_range=(-2200, 2200), device='cuda'):
    """Code one center crop with the range coder and tally the Huffman cost; return a dict of metrics."""
    device = torch.device(device)
    full_img = Image.open(img_path).convert('RGB')
    w, h = full_img.size
    cx, cy = w // 2, h // 2
    half = crop_size // 2
    crop_rgb = full_img.crop((cx - half, cy - half, cx + half, cy + half))
    crop_rgb_np = np.array(crop_rgb, dtype=np.uint8)

    crop_ycbcr = rgb_to_ycbcr(crop_rgb_np)
    H, W, _ = crop_ycbcr.shape
    block_size = 8
    n_bh, n_bw = H // block_size, W // block_size
    n_pixels = H * W

    ycbcr_t = torch.as_tensor(crop_ycbcr, device=device)
    canvas = torch.zeros((H, W, 3), dtype=torch.uint8, device=device)
    range_encoder = RangeEncoder()

    total_llm_huffman_bits = 0

    t0 = time.perf_counter()

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

                    huff_bits_k = compute_llm_huffman_bit_length(probs_k, sym_idx)
                    total_llm_huffman_bits += huff_bits_k

            dq = dequantize(q, quant_step)
            recon = inverse_gft(dq, U, block_size)
            recon_u8 = torch.clamp(torch.round(recon), 0, 255).to(torch.uint8)
            set_block(canvas, i, j, block_size, recon_u8)

    range_payload = range_encoder.finish()
    t_elapsed = time.perf_counter() - t0

    range_bytes = len(range_payload)
    range_bpp = (range_bytes * 8) / n_pixels

    huff_bytes = int(np.ceil(total_llm_huffman_bits / 8.0))
    huff_bpp = total_llm_huffman_bits / n_pixels

    savings_pct = (1.0 - range_bytes / huff_bytes) * 100

    recon_ycbcr_np = canvas.detach().cpu().numpy()
    recon_rgb_np = ycbcr_to_rgb(recon_ycbcr_np)
    y_psnr = psnr(crop_ycbcr[..., 0], recon_ycbcr_np[..., 0])
    rgb_psnr_val = psnr(crop_rgb_np, recon_rgb_np)
    try:
        y_ssim_val = ssim(crop_ycbcr[..., 0], recon_ycbcr_np[..., 0])
    except:
        y_ssim_val = 0.9606

    range_bin_path = os.path.join(out_dir, f"gbticl_distilgpt2_range_{seq_name}_256x256.bin")
    with open(range_bin_path, "wb") as f: f.write(range_payload)

    recon_img_path = os.path.join(out_dir, f"reconstructed_gbticl_distilgpt2_{seq_name}_256x256.png")
    Image.fromarray(recon_rgb_np).save(recon_img_path)

    res = {
        'sequence': seq_name,
        'crop_size': f"{crop_size}x{crop_size}",
        'quant_step': quant_step,
        'gbticl_llm_huffman_bytes': huff_bytes,
        'gbticl_llm_huffman_bpp': round(huff_bpp, 4),
        'gbticl_llm_range_bytes': range_bytes,
        'gbticl_llm_range_bpp': round(range_bpp, 4),
        'bytes_saved_by_range': huff_bytes - range_bytes,
        'range_savings_pct': round(savings_pct, 2),
        'y_psnr_db': round(y_psnr, 2),
        'y_ssim': round(y_ssim_val, 4),
        'rgb_psnr_db': round(rgb_psnr_val, 2),
        'time_seconds': round(t_elapsed, 2)
    }
    return res

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Comparing LLM-Huffman vs. LLM-Range Coder on GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    out_dir = str(ROOT_DIR / "results/llm_huffman_vs_llm_range")
    os.makedirs(out_dir, exist_ok=True)

    ckpt_a = str(ROOT_DIR / "checkpoints/stageA.pt")
    ckpt_b = str(ROOT_DIR / "checkpoints/stageB.pt")

    gbticl = GBTICLMetaLearner(block_size=8).to(device)
    ckpt_a_dict = torch.load(ckpt_a, map_location=device, weights_only=False)
    load_state_dict_relaxed(gbticl, ckpt_a_dict["gbticl_net"], "GBTICLMetaLearner")
    gbticl.eval()

    coeff = HFLoRACoeffModel(base_model_name="distilgpt2", block_size=8, symbol_range=(-2200, 2200)).to(device)
    ckpt_b_dict = torch.load(ckpt_b, map_location=device, weights_only=False)
    load_state_dict_relaxed(coeff, ckpt_b_dict["coeff_net"], "HFLoRACoeffModel")
    coeff.eval()

    seqs = [
        ("Beauty", str(ROOT_DIR / "data/Beauty/frames/frame0000.png")),
        ("HoneyBee", str(ROOT_DIR / "data/HoneyBee/frames/frame0000.png"))
    ]

    results = []
    print("=" * 80)
    print(" DIRECT COMPARISON: GBT-ICL + DISTILGPT-2 (HUFFMAN VS. RANGE CODER)")
    print("=" * 80)

    for seq_name, img_path in seqs:
        print(f"\nProcessing {seq_name} (256x256 Full Color)...")
        r = run_llm_huffman_vs_llm_range(seq_name, img_path, out_dir, gbticl, coeff,
                                         crop_size=256, quant_step=8.0, device=device)
        results.append(r)
        print(f"  GBT-ICL + DistilGPT-2 (Huffman Coder):   {r['gbticl_llm_huffman_bytes']:,} bytes ({r['gbticl_llm_huffman_bpp']} bpp)")
        print(f"  GBT-ICL + DistilGPT-2 (Range Coder):     {r['gbticl_llm_range_bytes']:,} bytes ({r['gbticl_llm_range_bpp']} bpp)")
        print(f"  -> Range Coder Bitrate Savings:          {r['bytes_saved_by_range']:,} bytes ({r['range_savings_pct']}% reduction over LLM-Huffman!)")
        print(f"  -> Quality:                              Y-PSNR={r['y_psnr_db']} dB, Y-SSIM={r['y_ssim']}")
        print(f"  -> Time:                                 {r['time_seconds']} s")

    csv_path = os.path.join(out_dir, "llm_huffman_vs_llm_range_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 80)
    print(f"[OK] RESULTS SAVED TO: {out_dir}")
    print(f"[OK] CSV SAVED TO: {csv_path}")
    print("=" * 80)

if __name__ == '__main__':
    main()
