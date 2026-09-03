"""Run the reproducible DTSR training pipeline without bundled checkpoints.

All generated checkpoints, intermediate datasets, and evaluation files are
written below ``runs/`` so the source tree remains safe to commit.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(command: list[str]) -> None:
    print("+", " ".join(map(str, command)))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train DTSR from scratch and write every generated artifact under runs/."
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-episodes", type=int, default=20)
    parser.add_argument("--multiday-episodes", type=int, default=500)
    parser.add_argument("--train-scenes", type=int, default=500)
    parser.add_argument("--val-scenes", type=int, default=60)
    parser.add_argument("--eval-scenes", type=int, default=20)
    parser.add_argument(
        "--runs-dir", type=Path, default=ROOT / "runs",
        help="Untracked destination for models, intermediate datasets, and evaluation output.",
    )
    parser.add_argument(
        "--skip-evaluation", action="store_true",
        help="Stop after DTSR/DeT training instead of running the final smoke evaluation.",
    )
    args = parser.parse_args()

    if min(args.baseline_episodes, args.multiday_episodes, args.train_scenes, args.val_scenes) <= 0:
        raise ValueError("Episode and scenario counts must be positive.")

    runs_dir = args.runs_dir.resolve()
    baseline_dir = runs_dir / "baseline"
    multiday_dir = runs_dir / "multiday_ddpg"
    clean_dir = runs_dir / "clean"
    dtsr_dir = runs_dir / "dtsr"
    evaluation_path = runs_dir / "evaluation" / "dtsr_smoke.csv"
    baseline_actor = baseline_dir / f"actor_baseline_seed{args.seed}.pt"
    baseline_bundle = baseline_dir / f"baseline_bundle_seed{args.seed}.pt"
    multiday_actor = multiday_dir / "actor_multiday_best.pt"
    multiday_bundle = multiday_dir / "bundle_multiday_best.pt"

    python = sys.executable
    baseline_device_args = [] if args.device == "auto" else ["--cuda" if args.device == "cuda" else "--no-cuda"]
    run([
        python, "cli.py", "baseline-train",
        "--episodes", str(args.baseline_episodes), "--seed", str(args.seed),
        "--output-dir", str(baseline_dir), "--actor-model-name", baseline_actor.name,
        "--bundle-name", baseline_bundle.name, *baseline_device_args,
    ])
    run([
        python, "scripts/05_train_multiday_ddpg.py", "--device", args.device,
        "--seed", str(args.seed), "--episodes", str(args.multiday_episodes),
        "--init-mode", "baseline_bundle", "--actor-path", str(baseline_actor),
        "--bundle-path", str(baseline_bundle), "--output-dir", str(multiday_dir),
    ])
    run([
        python, "scripts/02_collect_multiday_clean.py", "--device", args.device,
        "--algorithm", "ddpg", "--seed", str(args.seed),
        "--bundle-path", str(multiday_bundle), "--actor-path", str(multiday_actor),
        "--train-scenes", str(args.train_scenes), "--val-scenes", str(args.val_scenes),
        "--output-dir", str(clean_dir),
    ])
    run([
        python, "scripts/03_train_multiday_dtsr.py", "--device", args.device,
        "--actor-path", str(multiday_actor), "--bundle-path", str(multiday_bundle),
        "--clean-train", str(clean_dir / "clean_train.npz"),
        "--clean-val", str(clean_dir / "clean_val.npz"), "--output-dir", str(dtsr_dir),
    ])
    if not args.skip_evaluation:
        run([
            python, "scripts/04_evaluate_multiday_dtsr.py", "--device", args.device,
            "--actor-path", str(multiday_actor), "--bundle-path", str(multiday_bundle),
            "--dtsr-dir", str(dtsr_dir), "--scenes", str(args.eval_scenes),
            "--output", str(evaluation_path),
        ])


if __name__ == "__main__":
    main()
