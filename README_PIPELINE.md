# GBT-ICL + LLM Coefficient Predictor -- full pipeline

The complete codec for the dissertation: a genuine few-shot in-context
GBT-ICL model (`GBTICLMetaLearner`), a real pretrained-LLM + LoRA
coefficient predictor (`HFLoRACoeffModel`, distilgpt2 base), a real
arithmetic range coder, temporal (video) support, and scripts that train
the models, run them end to end over the extracted video frames, and
produce the figures/metrics/ablation matrix for the results section.

## Status (2026-08-07)

This has been built forward from an earlier version of this same codebase
and is **execution-verified on real hardware and real data**, not just
syntax-checked -- every stage below was actually run against real extracted
Beauty/HoneyBee frames on the dev machine (RTX 5050 laptop, CUDA) during
development, with `pytest tests/` (69 tests) covering the core mechanics.
Two real, previously-unknown correctness bugs were found and fixed via
this testing (not just the one degenerate-eigenvalue issue that was
already documented) -- see "Bugs found and fixed" below.

**What is NOT yet done, and needs a real run from you:** all training so
far has been short smoke runs (1-2 epochs, small `--samples-per-frame`) to
prove the mechanics work -- none of the checkpoints in `checkpoints/` (if
any) represent a properly converged model. Getting real dissertation
results needs real training runs (Stage A: 30-50 epochs; a `--lambda-rate`
sweep for the rate-distortion curve; Stage B/C similarly) and a real
ablation matrix run (`--quant-steps` sweep x all 5 configs x both
sequences) -- these take hours, not minutes, given the measured runtimes
below, and should be launched as background/overnight jobs.

## Run order

```bash
# 0. Environment (only needed once)
pip install -r requirements.txt

# 1. Stage A: meta-train GBT-ICL alone (paired with the frozen
#    LaplaceCoeffModel baseline). Sweep --lambda-rate for a proper
#    rate-distortion curve (needed for BD-Rate later).
python training.py --gbticl-model metalearner --coeff-model none \
    --epochs 30 --samples-per-frame 3000 --lambda-rate 0.01 \
    --out checkpoints/stageA.pt

# 2. Stage B: freeze Stage A, train the real pretrained-LLM coefficient
#    predictor against it.
python training.py --gbticl-model metalearner --freeze-gbticl \
    --gbticl-checkpoint checkpoints/stageA.pt \
    --coeff-model hf_lora --epochs 10 \
    --out checkpoints/stageB.pt

# 3. Stage C (optional): short joint fine-tune of both, low LR.
python training.py --gbticl-model metalearner --coeff-model hf_lora \
    --gbticl-checkpoint checkpoints/stageA.pt \
    --coeff-checkpoint checkpoints/stageB.pt \
    --lr 3e-5 --epochs 5 --out checkpoints/stageC.pt

# 4. Run the full ablation matrix (see ABLATION_CONFIGS in
#    run_dataset_pipeline.py) -- one invocation per config, sweeping
#    quant_step for a rate-distortion curve per config:
python run_dataset_pipeline.py --sequence both --ablation dct \
    --quant-steps 2,4,8,16,32
python run_dataset_pipeline.py --sequence both --ablation nonadaptive_gbt \
    --quant-steps 2,4,8,16,32
python run_dataset_pipeline.py --sequence both --ablation metalearner_no_llm \
    --gbticl-checkpoint checkpoints/stageA.pt --quant-steps 2,4,8,16,32
python run_dataset_pipeline.py --sequence both --ablation full \
    --gbticl-checkpoint checkpoints/stageA.pt \
    --coeff-checkpoint checkpoints/stageB.pt --quant-steps 2,4,8,16,32

# 5. Turn ablation_summary.csv into rate-distortion plots + a BD-Rate table
python visualize_metrics.py --compare-all results/ablation_summary.csv

# (optional) interactive demo
python app.py
```

Outputs land under `results/<ablation-name>/<Sequence>/`:
`reconstructed/frameXXXX.png`, `original_crop/frameXXXX.png`,
`reconstructed_video.mp4`, `original_video.mp4`, `metrics.csv`
(Y-PSNR/Y-SSIM/RGB-PSNR/bpp per frame), plus
`results/<name>/figures/<Sequence>_predicted_graph.png`. Step 4 also
appends to `results/ablation_summary.csv`; step 5 reads that file and
writes `results/figures/ablation_rate_distortion_<Sequence>.png` +
`results/figures/bd_rate_table.csv`.

## Runtime, read before running on full frames

The range coder is a genuinely sequential, per-symbol arithmetic coder
(see `range_coder.py`'s docstring) -- no GPU parallelism, by design, same
as every real learned codec. A full 1920x1080 frame is 32,400 8x8 blocks x
192 coefficients = ~6.2 million symbol-level Python calls per frame.
**Measured on this project's dev machine** (RTX 5050 laptop): an
UNTRAINED model at a 256x256 crop (1024 blocks) took ~140s to encode and
~800-1000s to decode ONE frame -- decode is much slower because
`TinyTransformerCoeffModel.symbol_probs` has no KV-cache (documented,
known O(n^3) cost); `HFLoRACoeffModel` does have one and is faster.
`run_dataset_pipeline.py` defaults to a 256x256 center crop so a run
finishes in a reasonable time; pass `--full-frame` only once you have time
to let a full-resolution run go (as a background/overnight job), and
sanity-check with `--max-frames 2` first.

## Bugs found and fixed during this build

1. **Degenerate-eigenvalue NaN gradients** (documented before this build,
   fixed now): `torch.linalg.eigh`'s backward pass produces NaN gradients
   when the Laplacian has repeated eigenvalues (e.g. a near-uniform
   predicted graph). Fixed in `graph_utils.py::eigendecompose` with a
   deterministic diagonal symmetry-breaking perturbation + eigenvector
   sign canonicalization, applied unconditionally so every call site
   (encoder, decoder, training) inherits the fix.
2. **`range_coder.py::probs_to_freqs` frequency-table bug** (found via
   testing, not previously known): with a wide `symbol_range` (this
   project uses up to 4401 symbols) and a peaked probability distribution,
   the old single-bin rounding-error correction could go negative and
   silently clamp, leaving the returned frequencies NOT summing to the
   coder's `TOTAL_FREQ` -- desyncing decode from the very first symbol.
   Fixed with the standard largest-remainder apportionment method. This
   was a REAL bug affecting real data: fixing it nearly doubled the
   compression ratio (1.69x -> 3.13x) on a real test crop at the same PSNR
   -- it had been silently wasting bits.
3. **`HFLoRACoeffModel` bf16 batched-vs-incremental precision mismatch**
   (found via testing): the encoder's batched teacher-forced probability
   computation and the decoder's incremental KV-cached computation gave
   probabilities differing by ~0.005-0.02 in bfloat16 -- enough to round
   to different integer frequencies and desync decode (PSNR collapsed to
   ~5.5dB). Fixed by making the encoder use the identical incremental code
   path the decoder uses (`precompute_encode_probs` now loops through
   `symbol_probs`, not a separate batched call).
4. **`HFLoRACoeffModel` single-slot KV-cache bug** (found via testing):
   `decode_image`/`decode_video` call `symbol_probs` in INTERLEAVED order
   across 3 colour channels (k outer, channel inner), not one channel at a
   time -- a single shared cache slot corrupted ~98% of symbols on every
   channel switch. Fixed with a small dict of caches keyed by
   `id(history)`, one slot per in-flight channel sequence.

See `tests/` for the regression test covering each of these directly.

## Design notes worth knowing

- **GBT-ICL is a few-shot in-context meta-learner** (`GBTICLMetaLearner`),
  not a context-conditional regressor -- it predicts a query block's edge
  weights by cross-attending over a support set of other already-decoded
  blocks (spatial neighbours in the current frame + temporal neighbours in
  the previous frame), each paired with a closed-form reference graph
  computed from that support block's own true pixels. No gradient updates
  at inference -- the support set itself is the in-context conditioning.
  The earlier `GBTICLNet` (context-only MLP) is kept as an explicit
  ablation baseline for exactly this comparison
  (`--ablation gbticl_no_llm` vs `--ablation metalearner_no_llm`).
- **Colour**: YCbCr (BT.601 full-range) before encoding, Y-PSNR/Y-SSIM as
  the headline metric (literature-comparable), full colour on
  reconstruction. No 4:2:0 chroma subsampling (v1 scope decision).
- **Video**: `encode_video`/`decode_video` thread the previous frame's
  reconstruction into the next frame's context/support set. Frame 0 of any
  sequence has no previous frame -- both GBT-ICL's temporal support slots
  and the coefficient model's temporal conditioning gracefully fall back
  to their learned "missing" embeddings, not a crash or special case.

## Files

```
gbticl_pipeline/
  graph_model.py     GBTICLPredictor interface; Uniform/ContextGradient baselines;
                      GBTICLNet (context regressor, ablation baseline);
                      GBTICLMetaLearner (primary model, few-shot in-context)
  coeff_model.py      CoeffPredictor interface; LaplaceCoeffModel baseline;
                      TinyTransformerCoeffModel (from-scratch, ablation baseline);
                      HFLoRACoeffModel (primary model, pretrained LLM + LoRA)
  graph_utils.py      Laplacian construction, eigendecomposition (fixed, see above),
                      dct_basis_and_eigvals (DCT baseline ablation)
  gft.py              Graph Fourier Transform, forward/inverse (single and batched)
  quantization.py     scalar quantize/dequantize
  context.py          canvas-based context extraction + get_support_set (spatiotemporal)
  colour.py           RGB <-> YCbCr (BT.601 full-range)
  range_coder.py      real arithmetic coder (fixed, see above), unit-tested standalone
  codec.py            encode_image/decode_image + encode_video/decode_video
  evaluate.py         psnr, ssim, bits_per_pixel, bd_rate
  device_utils.py     get_device() -- GPU > MPS > CPU
training.py           staged training (Stage A/B/C -- see --gbticl-model/--coeff-model/
                       --freeze-gbticl) + episodic MetaEpisodeDataset for the meta-learner
run_dataset_pipeline.py  video-mode encode/decode over real sequences, 5-config ablation
                          matrix (--ablation), YCbCr metrics, ablation_summary.csv
visualize_metrics.py  per-sequence figures + --compare-all ablation rate-distortion/BD-Rate
app.py                Gradio demo: upload image/short clip, see reconstruction + metrics
tests/                69 tests covering all of the above
```
