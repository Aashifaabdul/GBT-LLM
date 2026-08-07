# GBT-ICL + LLM Coefficient Predictor -- full pipeline

This is the complete codec for the dissertation: a real GBT-ICL model
(`GBTICLNet`), a real LLM-style coefficient predictor
(`TinyTransformerCoeffModel`), a real range coder, and scripts that train the
models, run them over the extracted video frames end to end, and produce the
figures/metrics for the results section.

## What's honestly true about this code right now

This project's working environment does not have PyTorch installed (repeated
`pip install torch` attempts all returned 403 from every index tried), so
**nothing that imports `torch` in this file set has been executed here** --
only syntax-checked (`python -m py_compile`, every file, all pass) and
carefully hand-reviewed. `visualize_metrics.py` is the one exception: it has
no PyTorch dependency, and it *was* run here end to end against a synthetic
`metrics.csv` and its figures visually confirmed.

The earlier CPU/numpy version of this codec (baselines only, no trained
model) *was* run and verified end to end, achieving 58.91dB near-lossless
and 41.22dB lossy PSNR on a real 64x64 crop -- that gives good confidence
the surrounding plumbing (context extraction, Laplacian/eigendecomposition,
GFT, quantization, real range coding, encoder/decoder symmetry) is correct.
What's new and unverified-by-execution in this delivery is specifically:
`GBTICLNet`, `TinyTransformerCoeffModel`, `training.py`'s training loop, and
`run_dataset_pipeline.py`'s frame-by-frame driver.

**Before trusting any results from this pipeline for your dissertation:**
run `training.py` for a couple of quick epochs on a small
`--samples-per-frame` first and check the printed loss is actually
decreasing, then run `run_dataset_pipeline.py --max-frames 2` before a full
run. Also worth knowing: there's a previously-diagnosed GPU numerical issue
(degenerate/repeated Laplacian eigenvalues causing eigenvector gradients and
encoder/decoder symmetry to become unstable at fine quantization on some GPU
eigensolvers) that has NOT been fixed yet -- if you see NaNs in training or
a sudden PSNR collapse at small `--quant-step` on GPU, that's the likely
cause. Ask for the fix (symmetry-breaking perturbation + eigenvector sign
canonicalization) before relying on fine-quantization GPU runs.

## Run order

```bash
# 1. Train (needs your GPU machine -- see runtime notes in training.py)
python training.py --epochs 15 --samples-per-frame 3000 --lambda-rate 0.01 \
    --out checkpoints/gbticl_ckpt.pt

# 2. Run the trained model over every extracted frame: encode, decode,
#    save each reconstructed frame, reassemble into a video, log metrics.
python run_dataset_pipeline.py --sequence both --checkpoint checkpoints/gbticl_ckpt.pt

# (optional) also run the non-learned baseline for comparison in your results:
python run_dataset_pipeline.py --sequence both

# 3. Turn the metrics into figures for the dissertation
python visualize_metrics.py --results-dir results/trained --compare results/baseline
```

Outputs land under `results/<trained|baseline>/<Sequence>/`:
`reconstructed/frameXXXX.png` (each reconstructed frame, saved one by one),
`original_crop/frameXXXX.png`, `reconstructed_video.mp4`, `original_video.mp4`,
`metrics.csv`, plus `results/<label>/figures/<Sequence>_predicted_graph.png`
(the actual trained model's predicted graph on a real block) and, after step 3,
`psnr_per_frame.png`, `bpp_per_frame.png`, `rate_distortion.png`, `summary.csv`.

## Why frames are cropped by default

The range coder is a genuinely sequential, per-symbol arithmetic coder (see
`range_coder.py`'s docstring) -- there's no way to parallelise it on GPU, by
design, same as every real learned codec. A full 1920x1080 frame is 32,400
blocks x 192 coefficients = ~6.2 million symbol-level Python calls per frame.
`run_dataset_pipeline.py` defaults to a 256x256 center crop per frame so a
run finishes in a reasonable time; pass `--full-frame` once you have time to
let a full-resolution run go (consider running it as an overnight job, and
sanity-check on `--max-frames 2` first).

## What changed from the earlier placeholder-only version

- `graph_model.py`: added `GBTICLNet`, a real trainable MLP over context.
  `UniformGBTICL`/`ContextGradientGBTICL` are kept as baselines/ablations --
  your results should show the trained model beating both.
- `coeff_model.py`: added `TinyTransformerCoeffModel` (a real, small,
  from-scratch causal transformer -- genuinely "LLM-style": next-coefficient
  prediction conditioned on Λ and history, same autoregressive shape as
  language modelling) and `HFLoRACoeffModel` (optional, wraps a real
  pretrained HuggingFace LM + LoRA, closer to the original LLaMA-3+LoRA plan,
  needs `pip install transformers peft` + internet + more GPU memory; not
  exercised here). `LaplaceCoeffModel` is kept as the baseline.
- **Interface change:** `CoeffPredictor.symbol_probs` now takes the *full*
  eigenvalue spectrum `eigvals` + position `k`, not one scalar eigenvalue --
  both encoder and decoder already compute the whole spectrum before this
  loop runs, and a sequence model conditioned on the whole spectrum shape is
  a better match for "conditioned on Λ" than one scalar. `codec.py`'s two
  call sites were updated to match.
- `graph_utils.py` / `gft.py`: added batched variants
  (`build_laplacian_batch`, `forward_gft_batch`, `inverse_gft_batch`) used by
  `training.py` to process many sampled blocks per optimizer step instead of
  one at a time.
- New files: `training.py`, `run_dataset_pipeline.py`, `visualize_metrics.py`.

## Files

```
gbticl_pipeline/
  graph_model.py      GBTICLPredictor interface; Uniform/ContextGradient baselines; GBTICLNet (trainable)
  coeff_model.py       CoeffPredictor interface; LaplaceCoeffModel baseline; TinyTransformerCoeffModel (trainable); HFLoRACoeffModel (optional)
  graph_utils.py        Laplacian construction + eigendecomposition (single and batched)
  gft.py                 Graph Fourier Transform, forward/inverse (single and batched)
  quantization.py        scalar quantize/dequantize
  context.py              live canvas-based context extraction (the codec loop's version)
  range_coder.py           real arithmetic coder (unit-tested standalone, CPU by design)
  codec.py                  encode_image / decode_image -- the full raster-order loop
  evaluate.py                psnr, bits_per_pixel
  device_utils.py              get_device() -- GPU > MPS > CPU
training.py       joint differentiable training of GBTICLNet + TinyTransformerCoeffModel
run_dataset_pipeline.py    encode/decode every extracted frame, save reconstructions, reassemble video, log metrics
visualize_metrics.py         turn metrics.csv into PSNR/bpp/rate-distortion figures (verified working)
```
