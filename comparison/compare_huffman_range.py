"""Static Huffman coding versus the range coder on a 256x256 luma crop.

For each sequence (Beauty, HoneyBee) the centre crop of
data/<sequence>/frames/frame0000.png is reduced to luma and entropy coded twice,
once as raw 8-bit pixels and once as 8x8 orthonormal DCT coefficients
quantised with a uniform step of 16. Both coders use the empirical symbol
distribution of the crop, and symbol tables are not counted in the sizes. The
range decoder output is asserted to equal its input.

Writes the .bin streams, reconstructions and entropy_comparison_256x256.csv to
results/entropy_test/. Takes no command-line arguments.

Usage:
    python comparison/compare_huffman_range.py
"""
import sys
import os
import time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
import csv
import heapq


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gbticl_pipeline.range_coder import RangeEncoder, RangeDecoder, probs_to_freqs, TOTAL_FREQ


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
    """Pack the concatenated codes into bytes; returns (bytes, number of code bits, padding bits)."""
    bitstring = "".join(codes[s] for s in symbols)
    padding = (8 - (len(bitstring) % 8)) % 8
    bitstring_padded = bitstring + "0" * padding
    byte_arr = bytearray()
    for i in range(0, len(bitstring_padded), 8):
        byte_arr.append(int(bitstring_padded[i:i+8], 2))
    return bytes(byte_arr), len(bitstring), padding

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


def range_encode_symbols(symbols, probs):
    """Range-code symbol indices with a static distribution, then decode them to verify.

    Args:
        symbols: integer indices into probs.
        probs: probability of each index (converted to integer frequencies over TOTAL_FREQ).

    Returns:
        (bitstream, decoded indices, encode seconds, decode seconds)
    """
    freqs, cum_freqs = probs_to_freqs(probs, TOTAL_FREQ)

    t0 = time.perf_counter()
    enc = RangeEncoder()
    for s in symbols:
        enc.encode(int(cum_freqs[s]), int(freqs[s]), TOTAL_FREQ)
    buf = enc.finish()
    t_enc = time.perf_counter() - t0

    t1 = time.perf_counter()
    dec = RangeDecoder(buf)
    decoded = []
    for _ in range(len(symbols)):
        target = dec.get_freq(TOTAL_FREQ)
        # Symbol whose cumulative-frequency interval contains the decoder target.
        s = int(np.searchsorted(cum_freqs, target, side='right') - 1)
        dec.decode(int(cum_freqs[s]), int(freqs[s]), TOTAL_FREQ)
        decoded.append(s)
    t_dec = time.perf_counter() - t1

    assert decoded == list(symbols), "Range decoder output mismatch!"
    return buf, decoded, t_enc, t_dec

def process_sequence(seq_name, img_path, out_dir, crop_size=256, q_step=16.0):
    """Code the centre crop of one frame with both coders and return a dict of sizes (or None if the image is missing)."""
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        return None

    full_img = Image.open(img_path).convert('RGB')
    w, h = full_img.size
    cx, cy = w // 2, h // 2
    half = crop_size // 2
    crop_img = full_img.crop((cx - half, cy - half, cx + half, cy + half))

    crop_save_path = os.path.join(out_dir, f"extracted_crop_{seq_name}_{crop_size}x{crop_size}.png")
    crop_img.save(crop_save_path)

    y_channel = np.array(crop_img.convert('L'), dtype=np.float32)
    n_pixels = y_channel.size

    # Raw luma pixels: 8-bit symbols, coded losslessly by both coders.
    raw_symbols = y_channel.astype(np.uint8).flatten()
    u_raw, c_raw = np.unique(raw_symbols, return_counts=True)
    p_raw = c_raw / len(raw_symbols)

    h_tree_raw = build_huffman_tree(dict(zip(u_raw, c_raw)))
    h_codes_raw = generate_huffman_codes(h_tree_raw)
    h_bytes_raw, h_bits_raw, _ = huffman_encode(raw_symbols, h_codes_raw)
    h_dec_raw = np.array(huffman_decode(h_bytes_raw, h_bits_raw, h_tree_raw), dtype=np.uint8).reshape(crop_size, crop_size)

    # The range coder works on dense indices 0..K-1, so map pixel values to indices and back.
    s_to_i_raw = {s: i for i, s in enumerate(u_raw)}
    i_to_s_raw = {i: s for i, s in enumerate(u_raw)}
    idx_raw = np.array([s_to_i_raw[s] for s in raw_symbols], dtype=np.int32)
    r_bytes_raw, r_dec_idx_raw, _, _ = range_encode_symbols(idx_raw, p_raw)
    r_dec_raw = np.array([i_to_s_raw[i] for i in r_dec_idx_raw], dtype=np.uint8).reshape(crop_size, crop_size)

    raw_huff_file = os.path.join(out_dir, f"raw_pixels_huffman_{seq_name}_{crop_size}x{crop_size}.bin")
    raw_range_file = os.path.join(out_dir, f"raw_pixels_range_{seq_name}_{crop_size}x{crop_size}.bin")
    with open(raw_huff_file, "wb") as f: f.write(h_bytes_raw)
    with open(raw_range_file, "wb") as f: f.write(r_bytes_raw)
    Image.fromarray(h_dec_raw).save(os.path.join(out_dir, f"recon_raw_huffman_{seq_name}_{crop_size}x{crop_size}.png"))
    Image.fromarray(r_dec_raw).save(os.path.join(out_dir, f"recon_raw_range_{seq_name}_{crop_size}x{crop_size}.png"))

    # Transform path: level shift, separable 8x8 orthonormal DCT-II, uniform quantisation.
    blocks_q = []
    blocks_recon = np.zeros_like(y_channel)
    for r in range(0, crop_size, 8):
        for c in range(0, crop_size, 8):
            b = y_channel[r:r+8, c:c+8] - 128.0
            coeff = dct(dct(b.T, norm='ortho').T, norm='ortho')
            q = np.round(coeff / q_step).astype(np.int32)
            blocks_q.append(q)

            dq = q.astype(np.float32) * q_step
            recon_b = idct(idct(dq.T, norm='ortho').T, norm='ortho') + 128.0
            blocks_recon[r:r+8, c:c+8] = np.clip(recon_b, 0, 255)

    # All quantised coefficients (block by block, raster order) form one symbol stream.
    all_coeffs = np.concatenate([b.flatten() for b in blocks_q])
    u_c, c_c = np.unique(all_coeffs, return_counts=True)
    p_c = c_c / len(all_coeffs)

    h_tree_c = build_huffman_tree(dict(zip(u_c, c_c)))
    h_codes_c = generate_huffman_codes(h_tree_c)
    h_bytes_c, _, _ = huffman_encode(all_coeffs, h_codes_c)

    c_to_i = {s: i for i, s in enumerate(u_c)}
    idx_coeffs = np.array([c_to_i[s] for s in all_coeffs], dtype=np.int32)
    r_bytes_c, _, _, _ = range_encode_symbols(idx_coeffs, p_c)

    tf_huff_file = os.path.join(out_dir, f"transform_coeffs_huffman_{seq_name}_{crop_size}x{crop_size}.bin")
    tf_range_file = os.path.join(out_dir, f"transform_coeffs_range_{seq_name}_{crop_size}x{crop_size}.bin")
    with open(tf_huff_file, "wb") as f: f.write(h_bytes_c)
    with open(tf_range_file, "wb") as f: f.write(r_bytes_c)
    Image.fromarray(blocks_recon.astype(np.uint8)).save(os.path.join(out_dir, f"reconstructed_transform_{seq_name}_{crop_size}x{crop_size}.png"))

    h_len = len(h_bytes_c)
    r_len = len(r_bytes_c)
    h_bpp = (h_len * 8) / n_pixels
    r_bpp = (r_len * 8) / n_pixels
    savings_pct = (1.0 - r_len / h_len) * 100

    res = {
        'sequence': seq_name,
        'crop_size': f"{crop_size}x{crop_size}",
        'pixels': n_pixels,
        'uncompressed_bytes': n_pixels,
        'huffman_raw_bytes': len(h_bytes_raw),
        'range_raw_bytes': len(r_bytes_raw),
        'huffman_transform_bytes': h_len,
        'huffman_bpp': round(h_bpp, 4),
        'range_transform_bytes': r_len,
        'range_bpp': round(r_bpp, 4),
        'bytes_saved': h_len - r_len,
        'reduction_pct': round(savings_pct, 2)
    }
    return res

def main():
    """Process both sequences, print the size comparison and write the CSV summary."""
    out_dir = str(ROOT_DIR / "results/entropy_test")
    os.makedirs(out_dir, exist_ok=True)

    seqs = [
        ("Beauty", str(ROOT_DIR / "data/Beauty/frames/frame0000.png")),
        ("HoneyBee", str(ROOT_DIR / "data/HoneyBee/frames/frame0000.png"))
    ]

    results = []
    print("=" * 70)
    print(" EXECUTING 256x256 HUFFMAN VS. RANGE CODING EXPERIMENT")
    print("=" * 70)

    for seq_name, img_path in seqs:
        print(f"\nProcessing {seq_name} (256x256)...")
        r = process_sequence(seq_name, img_path, out_dir, crop_size=256, q_step=16.0)
        if r:
            results.append(r)
            print("  [Transform 8x8 DCT Q=16]")
            print(f"  Huffman Output:  {r['huffman_transform_bytes']:,} bytes ({r['huffman_bpp']} bpp)")
            print(f"  Range Output:    {r['range_transform_bytes']:,} bytes ({r['range_bpp']} bpp)")
            print(f"  -> Savings:      {r['bytes_saved']:,} bytes ({r['reduction_pct']}% reduction over Huffman)")

    csv_path = os.path.join(out_dir, "entropy_comparison_256x256.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 70)
    print(f"[OK] ALL 256x256 FILES SAVED TO: {out_dir}")
    print(f"[OK] CSV SUMMARY: {csv_path}")
    print("=" * 70)

if __name__ == '__main__':
    main()
