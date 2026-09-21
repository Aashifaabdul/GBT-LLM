"""Final evaluation of the proposed method (GBT-ICL + DistilGPT-2/LoRA).

Runs ablations/run_ablation_study.py with the ablation fixed to gbt_icl_distilgpt2_lora.
Requires checkpoints/stageA.pt and checkpoints/stageC.pt unless overridden with
--gbticl-checkpoint / --coeff-checkpoint. Results go to
results/submission_results/<experiment-name>/ (default dissertation_ablation_study).

Usage:
    python run_final_evaluation.py --sequence both --crop 256 --max-frames 10 --quant-steps 4,8
"""

from __future__ import annotations

from ablations.run_ablation_study import main as run_ablation_study


def main() -> None:
    run_ablation_study(fixed_ablations=["gbt_icl_distilgpt2_lora"])


if __name__ == "__main__":
    main()
