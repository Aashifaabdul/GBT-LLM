"""Run the ablation matrix by calling run_dataset_pipeline.py once per configuration.

Ablations: dct_laplace, uniform_gbt_laplace, gbt_icl_laplace, gbt_icl_tiny_transformer
and gbt_icl_distilgpt2_lora (the proposed method). Each run writes a log, a run manifest,
codec outputs and rate-distortion figures to <output-root>/<experiment-name>/
(default results/submission_results/dissertation_ablation_study/).
The defaults are a smoke test (64x64 crop, one frame, one quantisation step).

Usage:
    python ablations/run_ablation_study.py --sequence Beauty --crop 256 --max-frames 5 --quant-steps 4,8
"""

from __future__ import annotations

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Any


PROJECT_ROOT = ROOT_DIR
PIPELINE_SCRIPT = PROJECT_ROOT / "run_dataset_pipeline.py"
PLOT_SCRIPT = PROJECT_ROOT / "visualization" / "visualize_metrics.py"


ABLATION_REGISTRY: dict[str, dict[str, Any]] = {
    "dct_laplace": {
        "pipeline_name": "dct",
        "description": "Fixed DCT basis with the closed-form Laplace entropy model.",
        "needs_graph_checkpoint": False,
        "needs_coefficient_checkpoint": False,
    },
    "uniform_gbt_laplace": {
        "pipeline_name": "nonadaptive_gbt",
        "description": "Uniform non-adaptive graph transform with the Laplace entropy model.",
        "needs_graph_checkpoint": False,
        "needs_coefficient_checkpoint": False,
    },
    "gbt_icl_laplace": {
        "pipeline_name": "metalearner_no_llm",
        "description": "GBT-ICL meta-learner with no learned coefficient predictor.",
        "needs_graph_checkpoint": True,
        "needs_coefficient_checkpoint": False,
    },
    "gbt_icl_tiny_transformer": {
        "pipeline_name": "trained",
        "description": "GBTICLNet with the project TinyTransformer entropy predictor.",
        "needs_graph_checkpoint": False,
        "needs_coefficient_checkpoint": False,
        "needs_legacy_tiny_checkpoint": True,
    },
    "gbt_icl_distilgpt2_lora": {
        "pipeline_name": "full",
        "description": "Proposed method: GBT-ICL + DistilGPT-2 adapted with LoRA.",
        "needs_graph_checkpoint": True,
        "needs_coefficient_checkpoint": True,
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_project_path(value: str) -> Path:
    """Resolve a project-relative path; absolute paths are returned unchanged."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_ablation_names(value: str) -> list[str]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = sorted(set(names) - set(ABLATION_REGISTRY))
    if unknown:
        raise SystemExit(f"Unknown ablation name(s): {', '.join(unknown)}. "
                         f"Choose from: {', '.join(ABLATION_REGISTRY)}")
    if not names:
        raise SystemExit("At least one ablation must be selected.")
    return names


def _validate_common_inputs(args: argparse.Namespace, ablations: list[str]) -> dict[str, Path]:
    """Check checkpoints, frame directories and option ranges before any model is run."""
    if not PIPELINE_SCRIPT.is_file() or not PLOT_SCRIPT.is_file():
        raise SystemExit("The lower-level pipeline scripts are missing from the project root.")
    if args.crop is not None and (args.crop <= 0 or args.crop % 8 != 0):
        raise SystemExit("--crop must be a positive multiple of 8.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive when supplied.")
    if args.frames_dir and args.sequence == "both":
        raise SystemExit("--frames-dir is per sequence; use Beauty or HoneyBee, not --sequence both.")

    graph_ckpt = _resolve_project_path(args.gbticl_checkpoint)
    coeff_ckpt = _resolve_project_path(args.coeff_checkpoint)
    legacy_tiny_ckpt = _resolve_project_path(args.legacy_tiny_checkpoint) if args.legacy_tiny_checkpoint else None

    required_graph = any(ABLATION_REGISTRY[name]["needs_graph_checkpoint"] is True for name in ablations)
    required_coeff = any(ABLATION_REGISTRY[name]["needs_coefficient_checkpoint"] for name in ablations)
    required_legacy_tiny = any(ABLATION_REGISTRY[name].get("needs_legacy_tiny_checkpoint") for name in ablations)
    if required_graph and not graph_ckpt.is_file():
        raise SystemExit(f"Missing GBT-ICL checkpoint: {graph_ckpt}")
    if required_coeff and not coeff_ckpt.is_file():
        raise SystemExit(f"Missing coefficient-model checkpoint: {coeff_ckpt}")
    if required_legacy_tiny and (legacy_tiny_ckpt is None or not legacy_tiny_ckpt.is_file()):
        raise SystemExit("gbt_icl_tiny_transformer requires --legacy-tiny-checkpoint containing both GBTICLNet and TinyTransformer weights.")

    if args.frames_dir:
        frames_dir = _resolve_project_path(args.frames_dir)
        if not any(frames_dir.glob("*.png")):
            raise SystemExit(f"No PNG frames found in --frames-dir: {frames_dir}")
    else:
        sequences = ("Beauty", "HoneyBee") if args.sequence == "both" else (args.sequence,)
        for sequence in sequences:
            frames_dir = PROJECT_ROOT / "data" / sequence / "frames"
            if not any(frames_dir.glob("*.png")):
                raise SystemExit(f"No PNG frames found in expected directory: {frames_dir}")

    return {"gbticl_checkpoint": graph_ckpt, "coeff_checkpoint": coeff_ckpt,
            "legacy_tiny_checkpoint": legacy_tiny_ckpt}


def _stream_command(command: list[str], log_path: Path, dry_run: bool) -> None:
    """Run a child process, echoing its output to the console and to log_path.

    With dry_run set, only the command line is written to the log.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_text = subprocess.list2cmdline(command)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {command_text}\n\n")
        if dry_run:
            print(f"[dry run] {command_text}")
            return
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        exit_code = process.wait()
    if exit_code != 0:
        raise RuntimeError(f"Command failed with exit code {exit_code}. See {log_path}")


def _build_pipeline_command(args: argparse.Namespace, paths: dict[str, Path], spec: dict[str, Any], output_root: Path) -> list[str]:
    """Build the run_dataset_pipeline.py command line for one ablation."""
    # The TinyTransformer variant is selected by its combined checkpoint, not by --ablation.
    command = [sys.executable, str(PIPELINE_SCRIPT)]
    if spec.get("needs_legacy_tiny_checkpoint"):
        command.extend(["--checkpoint", str(paths["legacy_tiny_checkpoint"])])
    else:
        command.extend(["--ablation", spec["pipeline_name"]])
    command.extend(["--sequence", args.sequence, "--out-dir", str(output_root),
               "--quant-steps", args.quant_steps, "--fps", str(args.fps)]
    )
    if args.full_frame:
        command.append("--full-frame")
    else:
        command.extend(["--crop", str(args.crop)])
    if args.max_frames is not None:
        command.extend(["--max-frames", str(args.max_frames)])
    if args.frames_dir:
        command.extend(["--frames-dir", str(_resolve_project_path(args.frames_dir))])
    if spec["needs_graph_checkpoint"] is True:
        command.extend(["--gbticl-checkpoint", str(paths["gbticl_checkpoint"])])
    if spec["needs_coefficient_checkpoint"]:
        command.extend(["--coeff-checkpoint", str(paths["coeff_checkpoint"]),
                        "--base-model-name", args.base_model_name])
    return command


def build_parser(fixed_ablations: list[str] | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-name", default="dissertation_ablation_study",
                        help="Name of the folder created below --output-root.")
    parser.add_argument("--output-root", default="results/submission_results",
                        help="Parent directory for experiment folders.")
    if fixed_ablations is None:
        parser.add_argument("--ablations", default="dct_laplace,uniform_gbt_laplace,gbt_icl_laplace,gbt_icl_distilgpt2_lora",
                            help="Comma-separated publication-facing ablation names.")
    parser.add_argument("--sequence", choices=["Beauty", "HoneyBee", "both"], default="both")
    parser.add_argument("--frames-dir", default=None, help="Optional replacement frame directory for one sequence.")
    parser.add_argument("--crop", type=int, default=64, help="Square centre crop in pixels; default is a safe smoke size.")
    parser.add_argument("--full-frame", action="store_true", help="Disable cropping; measure first because this can be very slow.")
    parser.add_argument("--max-frames", type=int, default=1, help="Frames per sequence; default is a safe smoke run.")
    parser.add_argument("--quant-steps", default="8", help="Comma-separated rate-distortion quantisation sweep.")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--gbticl-checkpoint", default="checkpoints/stageA.pt")
    parser.add_argument("--coeff-checkpoint", default="checkpoints/stageC.pt")
    parser.add_argument("--legacy-tiny-checkpoint", default=None,
                        help="Combined GBTICLNet + TinyTransformer checkpoint, required only for gbt_icl_tiny_transformer.")
    parser.add_argument("--base-model-name", default="distilgpt2")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and write planned commands without running models.")
    parser.add_argument("--resume", action="store_true", help="Reuse an existing experiment folder and skip completed ablations.")
    return parser


def main(fixed_ablations: list[str] | None = None, *, defaults: dict[str, Any] | None = None) -> None:
    """Run the selected ablations, or the fixed list passed by a wrapper script.

    Args:
        fixed_ablations: Ablation names to run; removes the --ablations option.
        defaults: Overrides for argparse defaults (e.g. the experiment name).
    """
    parser = build_parser(fixed_ablations)
    if defaults:
        parser.set_defaults(**defaults)
    args = parser.parse_args()
    ablations = fixed_ablations or _parse_ablation_names(args.ablations)
    paths = _validate_common_inputs(args, ablations)

    experiment_dir = _resolve_project_path(args.output_root) / args.experiment_name
    if experiment_dir.exists() and not args.resume:
        raise SystemExit(f"Experiment folder already exists: {experiment_dir}. Use a new --experiment-name or --resume.")
    previous_statuses: dict[str, str] = {}
    previous_manifest_path = experiment_dir / "run_manifest.json"
    if args.resume and previous_manifest_path.is_file():
        try:
            previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
            previous_statuses = {
                name: item.get("status", "pending")
                for name, item in previous_manifest.get("ablations", {}).items()
            }
        except json.JSONDecodeError:
            print("[resume] Existing manifest is unreadable; rerunning every requested ablation.")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    output_root = experiment_dir / "codec_outputs"

    manifest: dict[str, Any] = {
        "project": "GBT-ICL + DistilGPT-2/LoRA Graph Transform Codec",
        "created_utc": _utc_now(),
        "status": "planned" if args.dry_run else "running",
        "command": " ".join(sys.argv),
        "settings": vars(args),
        "resolved_paths": {name: str(path) if path else None for name, path in paths.items()},
        "ablations": {name: {**ABLATION_REGISTRY[name], "status": "pending"} for name in ablations},
    }
    _write_json(experiment_dir / "run_manifest.json", manifest)
    _write_json(experiment_dir / "configurations" / "experiment_config.json", manifest["settings"])

    try:
        for name in ablations:
            spec = ABLATION_REGISTRY[name]
            if args.resume and previous_statuses.get(name) == "completed":
                print(f"[resume] Skipping completed ablation: {name}")
                manifest["ablations"][name]["status"] = "skipped_completed"
                _write_json(experiment_dir / "run_manifest.json", manifest)
                continue

            command = _build_pipeline_command(args, paths, spec, output_root)
            print(f"\n=== {name}: {spec['description']} ===")
            _stream_command(command, experiment_dir / "logs" / f"{name}.log", args.dry_run)
            manifest["ablations"][name]["status"] = "planned" if args.dry_run else "completed"
            _write_json(experiment_dir / "run_manifest.json", manifest)

        summary_path = output_root / "ablation_summary.csv"
        if not args.dry_run and summary_path.is_file():
            plot_command = [sys.executable, str(PLOT_SCRIPT), "--compare-all", str(summary_path),
                            "--out", str(experiment_dir / "figures")]
            _stream_command(plot_command, experiment_dir / "logs" / "generate_figures.log", False)
            shutil.copy2(summary_path, experiment_dir / "ablation_summary.csv")
        manifest["status"] = "planned" if args.dry_run else "completed"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure"] = str(exc)
        _write_json(experiment_dir / "run_manifest.json", manifest)
        raise SystemExit(str(exc)) from exc

    _write_json(experiment_dir / "run_manifest.json", manifest)
    print(f"\nExperiment complete. Submission artefacts: {experiment_dir}")


if __name__ == "__main__":
    main()
