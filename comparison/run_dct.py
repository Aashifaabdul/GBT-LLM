"""DCT + Laplace baseline, run through the ablation driver.

Runs the `dct_laplace` ablation of ablations/run_ablation_study.py (fixed DCT
basis with the closed-form Laplace entropy model) under the experiment name
`dct_comparison`. The options of the ablation driver (--crop, --max-frames,
--quant-steps, --output-root, ...) are accepted; by default the results go to
results/submission_results/dct_comparison/.

Usage:
    python comparison/run_dct.py --crop 256 --max-frames 3
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ablations.run_ablation_study import main as run_study


def main():
    run_study(fixed_ablations=['dct_laplace'], defaults={'experiment_name': 'dct_comparison'})


if __name__ == "__main__":
    main()
