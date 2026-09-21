"""No-LLM ablation: GBT-ICL transform with the closed-form Laplace entropy model.

Wrapper around run_ablation_study.py that fixes the ablation to gbt_icl_laplace and
sets the experiment name to no_llm_ablation. Requires the Stage A checkpoint
(--gbticl-checkpoint, default checkpoints/stageA.pt); accepts the other options of
run_ablation_study.py. Results go to results/submission_results/no_llm_ablation/.

Usage:
    python ablations/run_no_llm.py --sequence Beauty --crop 256 --max-frames 5 --quant-steps 4,8
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ablations.run_ablation_study import main as run_study


def main():
    run_study(fixed_ablations=['gbt_icl_laplace'], defaults={'experiment_name': 'no_llm_ablation'})


if __name__ == "__main__":
    main()
