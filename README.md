# GBT-LLM

Graph-based signal-adaptive transforms and large language model entropy priors for adaptive image and video compression.

MSc dissertation, Aston University. Author: Aashifa Abdul Samath Parveen.

## Folder structure

```text
GBT-LLM/
├── training.py                 Training (Stages A, B, C)
├── run_dataset_pipeline.py     Encode, decode and evaluate sequences
├── run_final_evaluation.py     Final evaluation (proposed method)
├── requirements.txt
├── pytest.ini
├── gbticl_pipeline/            Codec library
│   ├── codec.py                Image and video encoder/decoder
│   ├── coeff_model.py          Laplace, TinyTransformer and DistilGPT-2 + LoRA coefficient models
│   ├── graph_model.py          GBT-ICL edge-weight predictors
│   ├── graph_utils.py          Laplacian, eigendecomposition, DCT basis
│   ├── gft.py                  Graph Fourier transform
│   ├── quantization.py
│   ├── range_coder.py
│   ├── context.py              Causal context and support sets
│   ├── colour.py               RGB <-> YCbCr
│   ├── evaluate.py             PSNR, SSIM, LPIPS, bpp, BD-Rate
│   └── device_utils.py
├── preprocessing/              Frame extraction, cropping, resizing, context extraction
├── evaluation/                 Codec smoke check, 1080p MSE/LPIPS metrics
├── ablations/                  Ablation studies and high-resolution codec experiments
├── comparison/                 Baseline comparisons
│   ├── run_dct.py              DCT baseline
│   ├── compare_huffman_range.py, run_1280x_comparison.py    Huffman vs range coding
│   ├── crops_256.py            256x256 test images shared by the comparisons below
│   ├── mlic/                   MLIC baseline vs GBT-LLM              -> results/mlic/
│   ├── classical_comparison.py WebP, PNG, LZMA, Gzip                 -> results/classical_comparison/
│   └── deepmind/               Byte-level LLM (DeepMind) baseline    -> results/deepmind/
├── visualization/              Figures and charts
├── tests/                      Pytest suite
├── data/                       Extracted frames (not tracked)
├── checkpoints/                Trained models (not tracked)
└── results/                    Outputs (not tracked)
```

## Requirements

- Python 3.11
- NVIDIA GPU with CUDA (recommended; CPU works for small crops)
- ffmpeg on `PATH` (optional, for MP4 output)
- Raw Beauty and HoneyBee sequences, 1920x1080, 8-bit 4:2:0 YUV

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

For a CUDA build of PyTorch, install `torch` and `torchvision` from https://pytorch.org before the last command.

## Data and checkpoints

```text
data/<Sequence>/frames/frameNNNN.png     Sequence: Beauty, HoneyBee
checkpoints/stageA.pt                    GBT-ICL meta-learner
checkpoints/stageB.pt                    DistilGPT-2 + LoRA coefficient model
checkpoints/stageC.pt                    Joint fine-tuning of both
checkpoints/gbticl_ckpt.pt               GBTICLNet + TinyTransformer baseline
```

The first run with DistilGPT-2 downloads `distilgpt2` from the Hugging Face Hub.

All commands are run from the repository root.

## Execution steps

### 1. Data preparation

```bash
python preprocessing/gbticl_frame_prep.py
python preprocessing/crop_frames.py --sequence Beauty --crop 256
python preprocessing/resize_frames.py --sequence Beauty --width 128 --height 72
python preprocessing/gbticl_context_extraction.py Beauty
```

`gbticl_frame_prep.py` reads the raw YUV files from the parent of the repository folder, or from `$GBTICL_DATASET_ROOT` (file layout in its CONFIG block), and writes `data/<Sequence>/frames/`. The last three commands are optional.

### 2. Verification

```bash
python evaluation/codec_smoke_check.py
python -m pytest tests
```

### 3. Training

```bash
python training.py --gbticl-model metalearner --coeff-model none --lambda-rate 0.01 --epochs 30 --out checkpoints/stageA.pt
python training.py --gbticl-model metalearner --freeze-gbticl --gbticl-checkpoint checkpoints/stageA.pt --coeff-model hf_lora --epochs 10 --out checkpoints/stageB.pt
python training.py --gbticl-model metalearner --coeff-model hf_lora --gbticl-checkpoint checkpoints/stageA.pt --coeff-checkpoint checkpoints/stageB.pt --lr 3e-5 --epochs 5 --out checkpoints/stageC.pt
```

Baseline (GBTICLNet + TinyTransformer), written to `checkpoints/gbticl_ckpt.pt`:

```bash
python training.py --gbticl-model net --coeff-model tiny
```

### 4. Evaluation of the proposed method

```bash
python run_dataset_pipeline.py --sequence Beauty --crop 256 --max-frames 3 --ablation full --gbticl-checkpoint checkpoints/stageA.pt --coeff-checkpoint checkpoints/stageB.pt --quant-step 8
python run_dataset_pipeline.py --sequence both --crop 256 --max-frames 3 --ablation full --gbticl-checkpoint checkpoints/stageA.pt --coeff-checkpoint checkpoints/stageB.pt --quant-steps 2,4,8,16,32
python ablations/run_single_image_eval.py --image data/Beauty/frames/frame0000.png --quant-step 2
python run_final_evaluation.py --dry-run
python run_final_evaluation.py --sequence both --crop 256 --max-frames 3 --quant-steps 8
```

`run_dataset_pipeline.py` writes `results/full/<Sequence>/` (`metrics.csv`, reconstructed frames, videos) and appends to `results/ablation_summary.csv`. `run_final_evaluation.py` writes `results/submission_results/dissertation_ablation_study/`.

### 5. Baselines and ablations

```bash
python comparison/run_dct.py --crop 256 --max-frames 3 --quant-steps 8
python run_dataset_pipeline.py --sequence Beauty --crop 256 --max-frames 3 --ablation nonadaptive_gbt --quant-step 8
python ablations/run_no_llm.py --crop 256 --max-frames 3 --quant-steps 8
python ablations/run_tiny_transformer.py --crop 256 --max-frames 3 --quant-steps 8
python ablations/run_ablation_study.py --crop 256 --max-frames 3 --quant-steps 4,8,16
```

Entropy coder comparisons (Huffman vs range coding):

```bash
python comparison/compare_huffman_range.py
python comparison/run_1280x_comparison.py --image data/Beauty/frames/frame0000.png --quant-step 16
python ablations/run_gbticl_huffman_vs_range_256.py
python ablations/run_llm_huffman_vs_llm_range.py
```

DCT baseline at 1280x720:

```bash
python ablations/encode_1280x720.py --image data/Beauty/frames/frame0000.png --quant-step 16
```

### 6. High-resolution codec experiments

```bash
python ablations/run_gbticl_llm_1280x720.py --sequence HoneyBee --quant-step 8
python ablations/run_compiled_coder_1280x720.py --sequence HoneyBee --quant-step 8
python ablations/run_gpu_batch_compiled_codec_1280x720.py --sequence HoneyBee --num-frames 30 --quant-step 8
python ablations/run_1080p_pure_python_codec.py --sequence HoneyBee --frames 30 --quant-step 8
python ablations/run_full_1080p_60frames_codec.py --sequences Beauty HoneyBee --frames 30 --quant-step 8
python ablations/run_decoder_256_gpu_batch.py --sequence Beauty --crop 256 --quant-step 8
python ablations/run_decoder_1080p_gpu_batch.py --sequence Beauty --frame 0 --quant-step 8
python evaluation/compute_1080p_mse_lpips.py
```

### 7. Figures

```bash
python visualization/visualize_metrics.py --results-dir results/full
python visualization/visualize_metrics.py --compare-all results/ablation_summary.csv
python visualization/visualize_graph.py
python visualization/visualize_graph_full_image.py
python visualization/plot_performance_graphs.py
python visualization/export_individual_charts.py
python visualization/plot_1080p_graphical_comparison.py
python visualization/plot_full_entropy_and_codec_comparison.py
python visualization/plot_huffman_range_supervisor.py
python visualization/plot_huffman_vs_range_deep_comparison.py
python visualization/plot_lpips_chart.py
python visualization/generate_visual_comparison.py
```

Charts are written to `results/charts/` and `results/figures/`.

### 8. MLIC, lossless-codec and byte-level LLM comparisons

```bash
python comparison/mlic/data_prep.py
python comparison/mlic/neural_compression_benchmark.py
python comparison/mlic/run_mlic_256x256.py
python comparison/mlic/plot_mlic_vs_gbtllm.py
python comparison/classical_comparison.py
python comparison/deepmind/run_deepmind_256x256.py
```

`comparison/mlic/data_prep.py` and `neural_compression_benchmark.py` read `data/<Sequence>/frames/`; the 256x256 runs read `data/Beauty/beauty_crop/` (step 1) and `data/HoneyBee/frames/`. The MLIC baseline is the pretrained CompressAI `cheng2020_attn` model (quality 3 and 6, downloaded on first use). Outputs are written to `results/mlic/`, `results/classical_comparison/` and `results/deepmind/`.
