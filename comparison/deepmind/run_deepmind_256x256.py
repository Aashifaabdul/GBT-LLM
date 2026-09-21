"""Byte-level LLM baseline on the 256x256 test crops.

Baseline in the style of "Language Modeling Is Compression" (Deletang et al.,
ICLR 2024): the raw RGB bytes are scored by DistilGPT-2 (byte_llm.LLMCompressor)
and the ideal code length is reported in bits per byte (bpb) and, for comparison
with the codecs in bits per pixel, as bpp = 3 x bpb. Only the first 4096 bytes of
each image are evaluated. The `model` field records whether DistilGPT-2 was
loaded or the order-1 Markov fallback was used.

Test images: see comparison/crops_256.py.
Output: results/deepmind/deepmind_256x256_results.json

Usage:
    python comparison/deepmind/run_deepmind_256x256.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from byte_llm import LLMCompressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crops_256 import collect_samples  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "deepmind"

LLM_EVAL_BYTES = 4096


def main():
    """Evaluate every test image and write deepmind_256x256_results.json."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    llm = LLMCompressor(model_name="distilgpt2", device="cpu")
    samples = collect_samples()
    print(f"Test images: {len(samples)}")

    results = []
    for sample in tqdm(samples, desc="Byte-level LLM"):
        img = Image.open(sample["path"]).convert("RGB")
        w, h = img.size
        chunk = np.array(img, dtype=np.uint8).tobytes()[:LLM_EVAL_BYTES]
        llm_bits, _ = llm.evaluate_entropy_bits(chunk)

        results.append({
            "sequence": sample["sequence"],
            "file": sample["path"].name,
            "width": w,
            "height": h,
            "model": "distilgpt2" if llm.model is not None else "markov-order1-fallback",
            "deepmind_bpb": llm_bits / len(chunk),
            "deepmind_bpp": 3.0 * llm_bits / len(chunk),
        })

    out_json = OUT_DIR / "deepmind_256x256_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_json}")
    return results


if __name__ == "__main__":
    main()
