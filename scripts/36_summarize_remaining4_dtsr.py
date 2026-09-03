from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from _common import PACKAGE_ROOT, write_json
from dtsr_multiday_common import safe_recovery


ATTACK_KEYS = (
    "opposite_fgsm",
    "electhacker_c",
    "electhacker_f",
    "electhacker_o",
)
DDPG_ATTACK_KEYS = {key: f"{key}_eps_0.100000" for key in ATTACK_KEYS}
T95_DF19 = 2.093024054408263
T95_DF79 = 1.9904502102301282

DEFAULT_DDPG = (
    PACKAGE_ROOT
    / "results"
    / "exp2_strength_no_adaptive_20scenes_seed42_newcal_v3_sealed"
    / "tables"
    / "exp2_strength_addendum_rollouts.csv"
)
DEFAULT_TD3 = (
    PACKAGE_ROOT
    / "results"
    / "native_td3_dtsr_seed42"
    / "remaining4_test_evaluation"
)
DEFAULT_SAC = (
    PACKAGE_ROOT
    / "results"
    / "native_sac_dtsr_seed42"
    / "remaining4_test_evaluation"
)
DEFAULT_OUTPUT = PACKAGE_ROOT / "results" / "remaining4_cross_backbone_dtsr_seed42"


def ddpg_paired(frame: pd.DataFrame) -> pd.DataFrame:
    clean_raw = frame[
        (frame["attack_key"] == "clean") & (frame["stage_key"] == "attack")
    ].set_index("scenario_id")
    clean_dtsr = frame[
        (frame["attack_key"] == "clean") & (frame["stage_key"] == "ug_bcr")
    ].set_index("scenario_id")
    rows = []
    for attack_key, source_key in DDPG_ATTACK_KEYS.items():
        raw = frame[
            (frame["attack_key"] == source_key) & (frame["stage_key"] == "attack")
        ].set_index("scenario_id")
        defended = frame[
            (frame["attack_key"] == source_key) & (frame["stage_key"] == "ug_bcr")
        ].set_index("scenario_id")
        common = clean_raw.index.intersection(clean_dtsr.index).intersection(raw.index).intersection(defended.index)
        for scenario_id in common:
            clean_raw_reward = float(clean_raw.at[scenario_id, "ep_reward"])
            attack_raw_reward = float(raw.at[scenario_id, "ep_reward"])
            clean_dtsr_reward = float(clean_dtsr.at[scenario_id, "ep_reward"])
            attack_dtsr_reward = float(defended.at[scenario_id, "ep_reward"])
            rows.append(
                {
                    "algorithm": "ddpg",
                    "scenario_id": scenario_id,
                    "attack_key": attack_key,
                    "clean_raw_reward": clean_raw_reward,
                    "attack_raw_reward": attack_raw_reward,
                    "clean_dtsr_reward": clean_dtsr_reward,
                    "attack_dtsr_reward": attack_dtsr_reward,
                    "attack_degradation": clean_raw_reward - attack_raw_reward,
                    "defense_reward_gain": attack_dtsr_reward - attack_raw_reward,
                    "recovery": safe_recovery(clean_raw_reward, attack_raw_reward, attack_dtsr_reward),
                    "clean_reward_delta": clean_dtsr_reward - clean_raw_reward,
                    "exit_vio_reduction": float(raw.at[scenario_id, "exit_vio"])
                    - float(defended.at[scenario_id, "exit_vio"]),
                    "run_vio_reduction": float(raw.at[scenario_id, "run_vio"])
                    - float(defended.at[scenario_id, "run_vio"]),
                    "raw_attack_linf_max": float(raw.at[scenario_id, "attack_delta_linf_max"]),
                    "dtsr_attack_linf_max": float(defended.at[scenario_id, "attack_delta_linf_max"]),
                }
            )
    return pd.DataFrame(rows)


def summarize_paired(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (algorithm, attack_key), group in paired.groupby(["algorithm", "attack_key"], sort=False):
        recovery = group["recovery"].to_numpy(dtype=float)
        recovery = recovery[np.isfinite(recovery)]
        mean = float(np.mean(recovery))
        std = float(np.std(recovery, ddof=1))
        half = T95_DF19 * std / math.sqrt(len(recovery))
        rows.append(
            {
                "algorithm": algorithm,
                "attack_key": attack_key,
                "scenario_count": int(group["scenario_id"].nunique()),
                "valid_recovery_count": int(len(recovery)),
                "attack_degradation_mean": float(group["attack_degradation"].mean()),
                "defense_reward_gain_mean": float(group["defense_reward_gain"].mean()),
                "recovery_mean": mean,
                "recovery_std": std,
                "recovery_ci95_low": mean - half,
                "recovery_ci95_high": mean + half,
                "clean_reward_delta_mean": float(group["clean_reward_delta"].mean()),
                "exit_vio_reduction_mean": float(group["exit_vio_reduction"].mean()),
                "run_vio_reduction_mean": float(group["run_vio_reduction"].mean()),
                "raw_attack_linf_max": float(group["raw_attack_linf_max"].max()),
                "dtsr_attack_linf_max": float(group["dtsr_attack_linf_max"].max()),
            }
        )
    return pd.DataFrame(rows)


def macro_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for algorithm, group in paired.groupby("algorithm", sort=False):
        recovery = group["recovery"].to_numpy(dtype=float)
        recovery = recovery[np.isfinite(recovery)]
        mean = float(np.mean(recovery))
        std = float(np.std(recovery, ddof=1))
        half = T95_DF79 * std / math.sqrt(len(recovery))
        rows.append(
            {
                "algorithm": algorithm,
                "attack_key": "macro_average",
                "scenario_attack_count": int(len(recovery)),
                "recovery_mean": mean,
                "recovery_std": std,
                "recovery_ci95_low": mean - half,
                "recovery_ci95_high": mean + half,
                "clean_reward_delta_mean": float(group["clean_reward_delta"].mean()),
                "attack_degradation_mean": float(group["attack_degradation"].mean()),
                "defense_reward_gain_mean": float(group["defense_reward_gain"].mean()),
                "exit_vio_reduction_mean": float(group["exit_vio_reduction"].mean()),
                "run_vio_reduction_mean": float(group["run_vio_reduction"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddpg-rollouts", type=Path, default=DEFAULT_DDPG)
    parser.add_argument("--td3-dir", type=Path, default=DEFAULT_TD3)
    parser.add_argument("--sac-dir", type=Path, default=DEFAULT_SAC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    ddpg_source = pd.read_csv(args.ddpg_rollouts)
    ddpg = ddpg_paired(ddpg_source)
    td3 = pd.read_csv(args.td3_dir / "paired_recovery.csv")
    sac = pd.read_csv(args.sac_dir / "paired_recovery.csv")
    paired = pd.concat([ddpg, td3, sac], ignore_index=True)
    expected_scenarios = {f"test_day_{index:04d}" for index in range(1, 21)}
    checks = []
    for (algorithm, attack_key), group in paired.groupby(["algorithm", "attack_key"]):
        scenarios = set(group["scenario_id"].astype(str))
        checks.append(
            {
                "algorithm": algorithm,
                "attack_key": attack_key,
                "scenario_count": int(len(scenarios)),
                "same_expected_test_set": scenarios == expected_scenarios,
                "all_attack_degradation_positive": bool((group["attack_degradation"] > 0).all()),
                "all_recovery_finite": bool(np.isfinite(group["recovery"].to_numpy(float)).all()),
                "raw_budget_respected": bool((group["raw_attack_linf_max"] <= 0.100001).all()),
                "dtsr_budget_respected": bool((group["dtsr_attack_linf_max"] <= 0.100001).all()),
            }
        )
    summary = summarize_paired(paired)
    macro = macro_summary(paired)
    for column in ("recovery_mean", "recovery_std", "recovery_ci95_low", "recovery_ci95_high"):
        summary[f"{column}_pct"] = summary[column] * 100.0
        macro[f"{column}_pct"] = macro[column] * 100.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.output_dir / "paired_recovery_all_algorithms.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "recovery_summary_all_algorithms.csv", index=False, encoding="utf-8-sig")
    macro.to_csv(args.output_dir / "macro_recovery_all_algorithms.csv", index=False, encoding="utf-8-sig")
    comparison_rows = []
    attack_labels = {
        "opposite_fgsm": "FGSM",
        "electhacker_c": "ElectHacker-C",
        "electhacker_f": "ElectHacker-F",
        "electhacker_o": "ElectHacker-O",
        "macro_average": "Macro average",
    }
    combined_summary = pd.concat([summary, macro], ignore_index=True, sort=False)
    for attack_key in (*ATTACK_KEYS, "macro_average"):
        row = {"attack": attack_labels[attack_key], "attack_key": attack_key}
        for algorithm in ("ddpg", "td3", "sac"):
            record = combined_summary[
                (combined_summary["algorithm"] == algorithm)
                & (combined_summary["attack_key"] == attack_key)
            ].iloc[0]
            row[f"{algorithm}_recovery_mean_pct"] = float(record["recovery_mean_pct"])
            row[f"{algorithm}_ci95_low_pct"] = float(record["recovery_ci95_low_pct"])
            row[f"{algorithm}_ci95_high_pct"] = float(record["recovery_ci95_high_pct"])
            row[f"{algorithm}_mean_ci95"] = (
                f"{float(record['recovery_mean_pct']):.1f}% "
                f"[{float(record['recovery_ci95_low_pct']):.1f}, "
                f"{float(record['recovery_ci95_high_pct']):.1f}]"
            )
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(
        args.output_dir / "recovery_comparison_table.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_json(
        args.output_dir / "checks.json",
        {
            "complete": all(
                row["same_expected_test_set"]
                and row["all_attack_degradation_positive"]
                and row["all_recovery_finite"]
                and row["raw_budget_respected"]
                and row["dtsr_budget_respected"]
                for row in checks
            ),
            "checks": checks,
            "sources": {
                "ddpg": str(args.ddpg_rollouts.resolve()),
                "td3": str(args.td3_dir.resolve()),
                "sac": str(args.sac_dir.resolve()),
            },
        },
    )
    print(summary[["algorithm", "attack_key", "recovery_mean_pct", "recovery_ci95_low_pct", "recovery_ci95_high_pct", "clean_reward_delta_mean"]].to_string(index=False))
    print("\nMacro:")
    print(macro[["algorithm", "recovery_mean_pct", "recovery_ci95_low_pct", "recovery_ci95_high_pct", "clean_reward_delta_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
