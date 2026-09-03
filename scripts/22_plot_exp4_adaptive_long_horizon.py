from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import PACKAGE_ROOT

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from evc.exp4_visualization import create_exp4_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create publication figures for Experiment 4 from completed rollout pairs."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PACKAGE_ROOT / "results" / "exp4_module_aware_long_horizon_seed42_lite10",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_path = args.results_dir / "intermediate" / "exp4_rollouts.jsonl"
    summary = create_exp4_figures(raw_path, args.results_dir / "figures", seed=args.seed)
    print(
        "Experiment 4 figures written to "
        f"{args.results_dir / 'figures'} "
        f"(deadline paired scenarios={summary['deadline_common_scenarios']})."
    )


if __name__ == "__main__":
    main()
