from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _common import (
    DEFAULT_ACTOR_PATH,
    PACKAGE_ROOT,
    REFERENCE_DATA_PATH,
    REFERENCE_SIGNAL_PATH,
    deterministic_subset,
    load_manifest,
    load_scenario,
    resolve_device,
    write_json,
)

sys.path.insert(0, str(PACKAGE_ROOT))
from evc.merged_core import ChargingEnv, TRAIN_PROFILE, load_actor_from_path


def fast_clean_rollout(arrivals, actor, signal_path, device):
    env = ChargingEnv(signal_path, TRAIN_PROFILE)
    env.reset()
    index = 0
    active = []

    while env.t < env.horizon:
        states = []
        stations = []
        while index < len(arrivals) and int(arrivals.loc[index, "Arrive_time"]) == env.t:
            states.append(env.build_initial_obs(int(arrivals.loc[index, "Duration_of_stay"])))
            stations.append(int(arrivals.loc[index, "Station"]))
            index += 1

        if active:
            states.extend([item.obs for item in active])
            stations.extend([item.station for item in active])

        if states:
            state_tensor = torch.as_tensor(
                np.asarray(states, dtype=np.float32),
                dtype=torch.float32,
                device=device,
            )
            with torch.no_grad():
                actions = actor(state_tensor).detach().cpu().numpy()
            for state, action, station in zip(states, actions, stations):
                env.enqueue(state, action, station)

        _, active, metrics = env.step()

    final_soc = np.asarray(metrics.final_soc_list, dtype=np.float64)
    return {
        "ep_reward": float(metrics.ep_reward),
        "ep_r1": float(metrics.ep_r1_cost_sum),
        "ep_r2": float(metrics.ep_r2_exit_penalty_sum),
        "ep_r3": float(metrics.ep_r3_running_penalty_sum),
        "exit_vio": int(metrics.exit_violation_count),
        "run_vio": int(metrics.running_violation_count),
        "mean_final_soc": float(final_soc.mean()),
        "std_final_soc": float(final_soc.std()),
        "min_final_soc": float(final_soc.min()),
        "max_final_soc": float(final_soc.max()),
        "done_count": int(metrics.done_count),
        "total_transitions": int(metrics.total_transitions),
    }


def summarize(frame: pd.DataFrame, reference_reward: float) -> dict:
    count = len(frame)
    total_sessions = max(count * 344, 1)
    total_transitions = max(int(frame["total_transitions"].sum()), 1)
    mean_reward = float(frame["ep_reward"].mean())
    reward_change = (mean_reward - reference_reward) / max(abs(reference_reward), 1e-9)
    return {
        "scenario_count": count,
        "mean_reward": mean_reward,
        "std_reward": float(frame["ep_reward"].std(ddof=0)),
        "reward_change_vs_reference": float(reward_change),
        "mean_exit_vio": float(frame["exit_vio"].mean()),
        "exit_violation_rate_per_vehicle": float(frame["exit_vio"].sum() / total_sessions),
        "mean_run_vio": float(frame["run_vio"].mean()),
        "running_violation_rate_per_transition": float(frame["run_vio"].sum() / total_transitions),
        "mean_final_soc": float(frame["mean_final_soc"].mean()),
        "minimum_daily_mean_final_soc": float(frame["mean_final_soc"].min()),
        "complete_rate": float((frame["done_count"] == 344).mean()),
    }


def decide(summary: dict) -> tuple[str, list[str]]:
    strong = (
        summary["complete_rate"] == 1.0
        and summary["mean_final_soc"] >= 0.96
        and summary["reward_change_vs_reference"] >= -0.08
        and summary["exit_violation_rate_per_vehicle"] <= 0.03
        and summary["running_violation_rate_per_transition"] <= 0.015
    )
    relaxed = (
        summary["complete_rate"] == 1.0
        and summary["mean_final_soc"] >= 0.95
        and summary["reward_change_vs_reference"] >= -0.15
        and summary["exit_violation_rate_per_vehicle"] <= 0.08
        and summary["running_violation_rate_per_transition"] <= 0.04
    )
    if strong:
        return "PASS", [
            "Existing DDPG can be frozen and used for multi-day DTSR retraining.",
            "Retain the validation report as the clean-policy compatibility record.",
        ]
    if relaxed:
        return "CONDITIONAL", [
            "Existing DDPG is structurally compatible and can be used for pilot DTSR retraining.",
            "Clean multi-day constraint violations are higher than on the reference day.",
            "For final paper results, report the clean multi-day baseline and reconsider DDPG fine-tuning if violations remain unacceptable.",
        ]
    return "FAIL", [
        "Do not train DTSR on this policy yet.",
        "Fine-tune or retrain DDPG on the multi-day training set before collecting Dnormal trajectories.",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--actor-path", type=Path, default=DEFAULT_ACTOR_PATH)
    parser.add_argument("--validation-scenes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--test-scenes", type=int, default=120)
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "results" / "ddpg_check")
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = resolve_device(args.device)
    actor = load_actor_from_path(args.actor_path, device)

    reference_arrivals = pd.read_csv(REFERENCE_DATA_PATH)
    reference = fast_clean_rollout(reference_arrivals, actor, REFERENCE_SIGNAL_PATH, device)
    reference.update({"scenario_id": "original_reference", "split": "reference"})

    rows = [reference]
    validation_manifest = deterministic_subset(load_manifest("val"), args.validation_scenes, args.seed)
    for _, row in validation_manifest.iterrows():
        arrivals, signal_path, scenario_id = load_scenario(row)
        result = fast_clean_rollout(arrivals, actor, signal_path, device)
        result.update({"scenario_id": scenario_id, "split": "val"})
        rows.append(result)

    if args.include_test:
        test_manifest = deterministic_subset(load_manifest("test"), args.test_scenes, args.seed)
        for _, row in test_manifest.iterrows():
            arrivals, signal_path, scenario_id = load_scenario(row)
            result = fast_clean_rollout(arrivals, actor, signal_path, device)
            result.update({"scenario_id": scenario_id, "split": "test"})
            rows.append(result)

    output = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / "scenario_results.csv", index=False)

    reference_reward = float(reference["ep_reward"])
    validation_summary = summarize(output[output["split"] == "val"], reference_reward)
    status, recommendations = decide(validation_summary)

    report = {
        "status": status,
        "actor_path": str(args.actor_path),
        "decision_split": "validation",
        "reference": reference,
        "validation": validation_summary,
        "recommendations": recommendations,
        "dtsr_training_allowed": status in {"PASS", "CONDITIONAL"},
    }
    if args.include_test:
        report["test_exploratory"] = summarize(output[output["split"] == "test"], reference_reward)
    write_json(args.output_dir / "recommendation.json", report)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
