"""DAE-only output integrity check — verify 2160 rollouts, no NaN, correct counts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _common import PACKAGE_ROOT, write_json


def main():
    parser = argparse.ArgumentParser(description="Check DAE-only evaluation outputs")
    parser.add_argument("--output-dir", default="runs/dae_only_fullstate_20scenes_seed42")
    parser.add_argument("--scenes", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    long_path = output_dir / "tables" / "dae_only_episode_metrics_long.csv"
    if not long_path.exists():
        raise FileNotFoundError(f"Long table not found at {long_path}")

    long_df = pd.read_csv(long_path)
    checks = {}

    expected_scenes = int(args.scenes)
    expected_rollouts = expected_scenes * 9 * 2
    checks[f"scenario_count_{expected_scenes}"] = int(long_df["scenario_id"].nunique()) == expected_scenes
    checks["attack_keys_9"] = int(long_df["attack_key"].nunique()) == 9
    checks["stage_keys_2"] = int(long_df["stage_key"].nunique()) == 2
    checks[f"total_rollouts_{expected_rollouts}"] = len(long_df) == expected_rollouts

    checks["no_nan_in_ep_reward"] = bool(long_df["ep_reward"].notna().all())
    checks["all_done_cnt_344"] = bool((long_df["done_cnt"] == 344).all())

    # Clean: attack_obs_count = 0
    clean_rows = long_df[long_df["attack_key"] == "clean"]
    checks["clean_attack_obs_zero"] = bool((clean_rows["attack_obs_count"] == 0).all()) if not clean_rows.empty else False

    # DAE stage: dae_input_count > 0 (for non-clean)
    dae_non_clean = long_df[(long_df["stage_key"] == "dae") & (long_df["attack_key"] != "clean")]
    checks["dae_stage_dae_input_positive"] = bool((dae_non_clean["dae_input_count"] > 0).all()) if not dae_non_clean.empty else False

    # No detector/shield/ug_bcr in results
    checks["no_fp_adaptive_deadline"] = "full_pipeline_adaptive_deadline" not in set(long_df["attack_key"])

    all_pass = all(checks.values())
    result = {"status": "PASS" if all_pass else "FAIL", "checks": checks}
    write_json(output_dir / "dae_only_output_integrity.json", result)
    print(f"[INTEGRITY] {'PASS' if all_pass else 'FAIL'}")

    if not all_pass:
        failed = [k for k, v in checks.items() if not v]
        print(f"  Failed: {failed}")


if __name__ == "__main__":
    main()
