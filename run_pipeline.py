"""Run the complete four-module DTSR main experiment from scratch."""

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
        description="Train and evaluate DAE + DeT + Temporal Shield + UG-BCR DTSR from scratch."
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--train-scenes", type=int, default=500)
    parser.add_argument("--val-scenes", type=int, default=60)
    parser.add_argument("--eval-scenes", type=int, default=20)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    args = parser.parse_args()

    runs_dir = args.runs_dir.resolve()
    policy_dir = runs_dir / "ddpg"
    clean_dir = runs_dir / "clean"
    dtsr_dir = runs_dir / "dtsr"
    evaluation_dir = runs_dir / "evaluation"
    actor = policy_dir / "actor_multiday_best.pt"
    bundle = policy_dir / "bundle_multiday_best.pt"
    python = sys.executable

    run([
        python, "scripts/05_train_multiday_ddpg.py", "--device", args.device,
        "--seed", str(args.seed), "--episodes", str(args.episodes),
        "--init-mode", "scratch", "--output-dir", str(policy_dir),
    ])
    run([
        python, "scripts/02_collect_multiday_clean.py", "--device", args.device,
        "--actor-path", str(actor), "--train-scenes", str(args.train_scenes),
        "--val-scenes", str(args.val_scenes), "--seed", str(args.seed),
        "--output-dir", str(clean_dir),
    ])
    run([
        python, "scripts/03_train_multiday_dtsr.py", "--device", args.device,
        "--actor-path", str(actor), "--bundle-path", str(bundle),
        "--clean-train", str(clean_dir / "clean_train.npz"),
        "--clean-val", str(clean_dir / "clean_val.npz"), "--output-dir", str(dtsr_dir),
    ])
    for algorithm in ("opposite_pgd", "q_function"):
        run([
            python, "scripts/04_evaluate_multiday_dtsr.py", "--device", args.device,
            "--actor-path", str(actor), "--bundle-path", str(bundle),
            "--dtsr-dir", str(dtsr_dir), "--scenes", str(args.eval_scenes),
            "--algorithm", algorithm,
            "--output", str(evaluation_dir / f"dtsr_{algorithm}.csv"),
        ])
    run([
        python, "scripts/06_evaluate_long_horizon_dtsr.py", "--device", args.device,
        "--actor-path", str(actor), "--bundle-path", str(bundle),
        "--dtsr-dir", str(dtsr_dir), "--scenes", str(args.eval_scenes),
        "--seed", str(args.seed),
        "--output", str(evaluation_dir / "dtsr_long_horizon.csv"),
    ])


if __name__ == "__main__":
    main()
