# Project folder guide

Run the commands below from the `GBT-LLM` repository root.

| Folder | What belongs here |
| --- | --- |
| `preprocessing/` | Frame extraction, cropping, resizing, context extraction |
| `comparison/` | DCT baseline and DCT coder comparisons |
| `ablations/` | No-LLM, TinyTransformer, and other ablation studies |
| `evaluation/` | Standalone metric calculation and manual codec checks |
| `visualization/` | Plotting and chart-generation source code |
| `gbticl_pipeline/` | Reusable codec and model implementation |
| `data/Beauty/`, `data/HoneyBee/` | Input frames, crops, resized frames, blocks, contexts |
| `checkpoints/` | Model weights |
| `results/` | All saved run outputs, metrics, reconstructions, charts, logs |

| `docs/` | Documentation and the old-to-new move inventory |


The root launchers remain available for training, dataset evaluation, final
evaluation, and the interactive app. The manual `test_codec.py` script is now
`evaluation/codec_smoke_check.py`; it is separate from automatic test collection.

## Typical workflow

```powershell
# Inspect available options before choosing run settings.
python preprocessing/crop_frames.py --help
python preprocessing/resize_frames.py --help
python training.py --help
python run_dataset_pipeline.py --help
python ablations/run_ablation_study.py --help
python run_final_evaluation.py --help
python visualization/visualize_metrics.py --help

# Validate paths and plan the ablations without starting model runs.
python ablations/run_ablation_study.py --dry-run --experiment-name folder_check

# Plot existing metrics.
python visualization/visualize_metrics.py --compare-all results/ablation_summary.csv

# Run automated regression tests.
python -m pytest tests -q
```

Frame extraction reads raw YUV files from the parent dataset directory by
 default and writes extracted data under `data/`. `GBTICL_DATASET_ROOT` overrides
 the raw-input directory; `GBTICL_OUT_ROOT` overrides the extraction/context
 directory. Cropping, resizing, training, and evaluation use `data/` by default.
Pass explicit frame paths using `data/Beauty/frames` or `data/HoneyBee/frames`.

## Output locations

- Graph images: `results/figures/`.
- Comparison charts: `results/charts/` or the plotter's existing directory below `results/`.
- Dissertation comparison outputs: `results/dissertation_comparison/`.
- Managed ablation/final-evaluation runs: `results/submission_results/<experiment-name>/`.
- Manual codec-check images: `results/codec_smoke_check/`.
- Historical full-run log: `results/logs/full_128x72_run.log`.
- Separate final-model bundle outputs: `results/final_gbticl_llm/`.

The separate final-model bundle retains its own modules and command conventions.
Its output defaults now point to the central results folder; explicit `--out-dir`
arguments still override that location. When running from inside that bundle,
use `../results/final_gbticl_llm/` to access its saved results.

Existing measurements and model weights were retained. No experiments were
rerun to replace saved dissertation results during organization. Some historical
plot scripts contain fixed values; reorganizing them does not revalidate those
measurements. Existing manifests and logs retain their historical path text.

See `reorganization_moves.json` for exact old and new locations, and
[README_PIPELINE.md](README_PIPELINE.md) for the detailed pipeline notes.
