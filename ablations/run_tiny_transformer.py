"""TinyTransformer ablation: GBTICLNet with the small transformer entropy predictor instead of DistilGPT-2.

Wrapper around run_ablation_study.py that fixes the ablation to gbt_icl_tiny_transformer.
The combined GBTICLNet + TinyTransformer checkpoint defaults to checkpoints/gbticl_ckpt.pt
(--legacy-tiny-checkpoint). Accepts the other options of run_ablation_study.py.
Results go to results/submission_results/tiny_transformer_ablation/.

Usage:
    python ablations/run_tiny_transformer.py --sequence Beauty --crop 256 --max-frames 5 --quant-steps 4,8
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ablations.run_ablation_study import main as run_study


def main():
    run_study(fixed_ablations=['gbt_icl_tiny_transformer'], defaults={'experiment_name': 'tiny_transformer_ablation', 'legacy_tiny_checkpoint': 'checkpoints/gbticl_ckpt.pt'})


if __name__ == "__main__":
    main()
