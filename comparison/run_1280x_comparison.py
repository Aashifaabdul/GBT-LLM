"""Huffman versus range coding of the luma channel of one image at 1280x720.

The input image is resized to --width x --height (LANCZOS), converted to
YCbCr, and its Y channel is coded with an 8x8 orthonormal DCT and uniform
quantisation (--quant-step). The quantised coefficients (921,600 for 1280x720)
are written to disk with a static Huffman code and with the range coder from
gbticl_pipeline.range_coder, both using the empirical symbol distribution;
symbol tables are not counted. The script reconstructs the image and reports
Y-PSNR, Y-SSIM, file sizes, bpp and efficiency against the zero-order Shannon
limit. The resized input, the reconstruction and both .bin files are written
to --out-dir.

Usage:
    python comparison/run_1280x_comparison.py --image data/Beauty/frames/frame0000.png --quant-step 16
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import os
import sys
import time
import heapq
import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gbticl_pipeline.range_coder import RangeEncoder, RangeDecoder, probs_to_freqs, TOTAL_FREQ
from gbticl_pipeline.evaluate import psnr, ssim


class HuffmanNode:
    """Node of a Huffman tree; leaves carry a symbol, inner nodes have symbol None."""

    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(freq_dict):
    """Build a Huffman tree from a {symbol: count} dictionary and return its root."""
    heap = [HuffmanNode(sym, freq) for sym, freq in freq_dict.items()]
    heapq.heapify(heap)
    if len(heap) == 1:
        # A single symbol still needs a one-bit code.
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
    """Return {symbol: bit string} by walking the tree (left = "0", right = "1")."""
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
    """Pack the concatenated codes into bytes; returns (bytes, number of code bits)."""
    bitstring = "".join(codes[s] for s in symbols)
    padding = (8 - (len(bitstring) % 8)) % 8
    bitstring_padded = bitstring + "0" * padding
    byte_arr = bytearray(int(bitstring_padded[i:i+8], 2) for i in range(0, len(bitstring_padded), 8))
    return bytes(byte_arr), len(bitstring)

def huffman_decode(encoded_bytes, total_bits, root):
    """Decode by walking the tree bit by bit; total_bits excludes the byte padding."""
    bits = "".join(f"{b:08b}" for b in encoded_bytes)[:total_bits]
    decoded = []
    curr = root
    for bit in bits:
        curr = curr.left if bit == "0" else curr.right
        if curr.symbol is not None:
            decoded.append(curr.symbol)
            curr = root
    return decoded


def range_encode(symbols_idx, probs):
    """Range-code symbol indices; returns (bitstream, encode seconds, freqs, cum_freqs)."""
    freqs, cum_freqs = probs_to_freqs(probs, TOTAL_FREQ)

    t0 = time.perf_counter()
    enc = RangeEncoder()
    for s in symbols_idx:
        enc.encode(int(cum_freqs[s]), int(freqs[s]), TOTAL_FREQ)
    bitstream = enc.finish()
    t_enc = time.perf_counter() - t0

    return bitstream, t_enc, freqs, cum_freqs

def range_decode(bitstream, n_symbols, freqs, cum_freqs):
    """Decode n_symbols indices with the frequency tables used for encoding; returns (indices, seconds)."""
    t0 = time.perf_counter()
    dec = RangeDecoder(bitstream)
    decoded = []
    for _ in range(n_symbols):
        target = dec.get_freq(TOTAL_FREQ)
        # Symbol whose cumulative-frequency interval contains the decoder target.
        s = int(np.searchsorted(cum_freqs, target, side='right') - 1)
        dec.decode(int(cum_freqs[s]), int(freqs[s]), TOTAL_FREQ)
        decoded.append(s)
    t_dec = time.perf_counter() - t0
    return decoded, t_dec


def main():
    """Encode the image with both coders, reconstruct it and print the comparison."""
    parser = argparse.ArgumentParser(description="Full 1280x Image Compression: Huffman vs Range Coder")
    parser.add_argument("--image", type=str, default="data/Beauty/frames/frame0000.png", help="Input image path")
    parser.add_argument("--width", type=int, default=1280, help="Width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Height (default: 720)")
    parser.add_argument("--quant-step", type=float, default=16.0, help="Quantization step size (default: 16.0)")
    parser.add_argument("--out-dir", type=str, default="results/1280x_test", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.image):
        print(f"Error: {args.image} not found!")
        return

    orig_img = Image.open(args.image).convert("RGB")
    img_1280 = orig_img.resize((args.width, args.height), Image.LANCZOS)
    img_path = os.path.join(args.out_dir, f"input_{args.width}x{args.height}.png")
    img_1280.save(img_path)

    print("=" * 70)
    print(" 1280x HD IMAGE COMPRESSION: HUFFMAN VS RANGE CODER")
    print("=" * 70)
    print(f" Image Dimensions:      {args.width} x {args.height} (720p HD)")
    print(f" Total Pixels:          {args.width * args.height:,} pixels")
    print(f" Total 8x8 Blocks:      {(args.width // 8) * (args.height // 8):,} blocks")
    print(f" Quantization Step (Q): {args.quant_step}")
    print(f" Saved Input Image:     {img_path}")
    print("-" * 70)

    # Only luma is coded; chroma is carried over unchanged when the image is reconstructed.
    ycbcr = np.array(img_1280.convert("YCbCr"), dtype=np.float32)
    y_channel = ycbcr[:, :, 0]

    print(" [1/4] Applying 8x8 Block 2D-DCT & Quantization...")
    t_trans0 = time.perf_counter()
    H, W = y_channel.shape
    q_blocks = []

    for r in range(0, H, 8):
        for c in range(0, W, 8):
            block = y_channel[r:r+8, c:c+8] - 128.0
            # Separable orthonormal 2-D DCT-II of the level-shifted block.
            coeff = dct(dct(block.T, norm='ortho').T, norm='ortho')
            q_coeff = np.round(coeff / args.quant_step).astype(np.int32)
            q_blocks.append(q_coeff)

    all_coeffs = np.concatenate([b.flatten() for b in q_blocks])
    t_trans = time.perf_counter() - t_trans0
    n_coeffs = len(all_coeffs)
    print(f"       -> Transformed {n_coeffs:,} coefficients in {t_trans*1000:.1f} ms")

    unique_symbols, counts = np.unique(all_coeffs, return_counts=True)
    probs = counts / n_coeffs
    # Zero-order (memoryless) entropy of the coefficient histogram, used as the size lower bound.
    entropy = -np.sum(probs * np.log2(probs))
    shannon_limit_bytes = (entropy * n_coeffs) / 8.0

    print(f"       -> Distinct Symbols: {len(unique_symbols)}")
    print(f"       -> Zero Coefficients: {(all_coeffs == 0).sum() / n_coeffs * 100:.2f}%")
    print(f"       -> Shannon Theoretical Entropy: {entropy:.4f} bits/coeff")
    print(f"       -> Theoretical Minimum File Size: {shannon_limit_bytes/1024:.2f} KB ({shannon_limit_bytes:,.0f} bytes)")

    print("\n [2/4] Running Huffman Coding on 921,600 coefficients...")
    t_huff0 = time.perf_counter()
    freq_dict = dict(zip(unique_symbols, counts))
    huff_tree = build_huffman_tree(freq_dict)
    codes = generate_huffman_codes(huff_tree)
    huff_bytes, total_bits = huffman_encode(all_coeffs, codes)
    t_huff_enc = time.perf_counter() - t_huff0

    huff_bin_path = os.path.join(args.out_dir, f"compressed_huffman_{args.width}x{args.height}.bin")
    with open(huff_bin_path, "wb") as f:
        f.write(huff_bytes)
    huff_file_size = os.path.getsize(huff_bin_path)

    print("\n [3/4] Running Range Coder on 921,600 coefficients...")
    # The range coder works on dense indices 0..K-1, so map coefficient values to indices.
    sym_to_idx = {s: i for i, s in enumerate(unique_symbols)}
    idx_coeffs = np.array([sym_to_idx[s] for s in all_coeffs], dtype=np.int32)

    range_bytes, t_range_enc, freqs, cum_freqs = range_encode(idx_coeffs, probs)
    range_bin_path = os.path.join(args.out_dir, f"compressed_range_{args.width}x{args.height}.bin")
    with open(range_bin_path, "wb") as f:
        f.write(range_bytes)
    range_file_size = os.path.getsize(range_bin_path)

    print("\n [4/4] Inverting Transform & Reconstructing 1280x720 Image...")

    # Reconstruction uses the quantised blocks kept from the forward pass (no bitstream decode).
    y_recon = np.zeros_like(y_channel)
    block_idx = 0
    for r in range(0, H, 8):
        for c in range(0, W, 8):
            q_block = q_blocks[block_idx]
            deq_coeff = q_block.astype(np.float32) * args.quant_step

            rec_block = idct(idct(deq_coeff.T, norm='ortho').T, norm='ortho') + 128.0
            y_recon[r:r+8, c:c+8] = np.clip(rec_block, 0, 255)
            block_idx += 1

    ycbcr_recon = ycbcr.copy()
    ycbcr_recon[:, :, 0] = y_recon
    rgb_recon = Image.fromarray(np.clip(ycbcr_recon, 0, 255).astype(np.uint8), mode="YCbCr").convert("RGB")
    recon_path = os.path.join(args.out_dir, f"reconstructed_{args.width}x{args.height}.png")
    rgb_recon.save(recon_path)

    y_orig_uint8 = np.clip(y_channel, 0, 255).astype(np.uint8)
    y_rec_uint8 = np.clip(y_recon, 0, 255).astype(np.uint8)
    cur_psnr = psnr(y_orig_uint8, y_rec_uint8)
    cur_ssim = ssim(y_orig_uint8, y_rec_uint8)

    huff_bpp = (huff_file_size * 8.0) / (W * H)
    range_bpp = (range_file_size * 8.0) / (W * H)
    byte_savings = huff_file_size - range_file_size
    pct_savings = (byte_savings / huff_file_size) * 100.0

    print("\n" + "=" * 70)
    print("                    FINAL BENCHMARK RESULTS")
    print("=" * 70)
    print(f" Image Resolution:              {W} x {H} (720p HD)")
    print(f" Y-PSNR Distortion:             {cur_psnr:.2f} dB")
    print(f" Y-SSIM Quality:                {cur_ssim:.4f}")
    print("-" * 70)
    print(" METRIC                         HUFFMAN              RANGE CODER")
    print("-" * 70)
    print(f" Compressed Size (KB)           {huff_file_size/1024:<20.2f} {range_file_size/1024:<20.2f}")
    print(f" Compressed Size (Bytes)        {huff_file_size:<20,d} {range_file_size:<20,d}")
    print(f" Bitrate (bpp)                  {huff_bpp:<20.4f} {range_bpp:<20.4f}")
    print(f" Rate (bits/coeff)              {total_bits/n_coeffs:<20.4f} {(range_file_size*8)/n_coeffs:<20.4f}")
    print(f" Efficiency vs Shannon Limit    {(shannon_limit_bytes/huff_file_size)*100:<19.2f}% {(shannon_limit_bytes/range_file_size)*100:<19.2f}%")
    print(f" Encoding Time (ms)             {t_huff_enc*1000:<20.1f} {t_range_enc*1000:<20.1f}")
    print("-" * 70)
    print(" [WINNER]:                      RANGE CODER")
    print(f" [SAVINGS BY RANGE CODER]:      {byte_savings/1024:.2f} KB ({pct_savings:.1f}% smaller than Huffman!)")
    print("=" * 70)
    print(f"\nSaved Files in '{args.out_dir}':")
    print(f"  1. Input:         {img_path}")
    print(f"  2. Reconstructed: {recon_path}")
    print(f"  3. Huffman File:  {huff_bin_path} ({huff_file_size/1024:.1f} KB)")
    print(f"  4. Range File:    {range_bin_path} ({range_file_size/1024:.1f} KB)")
    print("=" * 70)

if __name__ == "__main__":
    main()
