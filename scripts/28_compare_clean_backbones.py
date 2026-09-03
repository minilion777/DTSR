from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _common import (
    PACKAGE_ROOT,
    deterministic_subset,
    load_manifest,
    load_scenario,
    resolve_device,
    write_json,
)

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from evc.merged_core import ChargingEnv, PROFILE_MAP, QueueItem, load_actor_from_path
from evc.offpolicy_backbones import load_backbone_bundle
from evc.ppo_backbone import load_ppo_bundle


DEFAULT_DDPG_ACTOR = (
    PACKAGE_ROOT / "models" / "multiday_ddpg_baseline_bundle" / "actor_multiday_best.pt"
)


def clean_rollout(arrivals, actor, signal_path, device, reward_profile) -> dict[str, float | int]:
    env = ChargingEnv(signal_path, reward_profile)
    env.reset()
    actor = actor.to(device).eval()
    index = 0
    active: list[QueueItem] = []
    actions_seen: list[np.ndarray] = []
    while env.t < env.horizon:
        states = []
        stations = []
        while index < len(arrivals) and int(arrivals.loc[index, "Arrive_time"]) == env.t:
            states.append(env.build_initial_obs(int(arrivals.loc[index, "Duration_of_stay"])))
            stations.append(int(arrivals.loc[index, "Station"]))
            index += 1
        if active:
            states.extend(item.obs for item in active)
            stations.extend(item.station for item in active)
        if states:
            state_tensor = torch.as_tensor(
                np.asarray(states, dtype=np.float32), dtype=torch.float32, device=device
            )
            with torch.no_grad():
                actions = actor(state_tensor).detach().cpu().numpy()
            actions_seen.append(actions.reshape(-1))
            for state, action, station in zip(states, actions, stations):
                env.enqueue(state, action, station)
        _, active, metrics = env.step()
    action_values = np.concatenate(actions_seen)
    return {
        "ep_reward": float(metrics.ep_reward),
        "ep_r1_cost": float(metrics.ep_r1_cost_sum),
        "ep_r2_exit_penalty": float(metrics.ep_r2_exit_penalty_sum),
        "ep_r3_running_penalty": float(metrics.ep_r3_running_penalty_sum),
        "ep_r4_dense_safety": float(metrics.ep_r4_dense_safety_penalty_sum),
        "exit_vio": int(metrics.exit_violation_count),
        "run_vio": int(metrics.running_violation_count),
        "mean_final_soc": float(np.mean(metrics.final_soc_list)),
        "min_final_soc": float(np.min(metrics.final_soc_list)),
        "done_count": int(metrics.done_count),
        "total_transitions": int(metrics.total_transitions),
        "action_mean": float(np.mean(action_values)),
        "action_std": float(np.std(action_values)),
        "action_min": float(np.min(action_values)),
        "action_max": float(np.max(action_values)),
    }


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for algorithm, group in details.groupby("algorithm", sort=False):
        rows.append(
            {
                "algorithm": algorithm,
                "scenario_count": int(len(group)),
                "reward_mean": float(group["ep_reward"].mean()),
                "reward_std": float(group["ep_reward"].std(ddof=0)),
                "r1_cost_mean": float(group["ep_r1_cost"].mean()),
                "r2_exit_penalty_mean": float(group["ep_r2_exit_penalty"].mean()),
                "r3_running_penalty_mean": float(group["ep_r3_running_penalty"].mean()),
                "r4_dense_safety_mean": float(group["ep_r4_dense_safety"].mean()),
                "exit_vio_mean": float(group["exit_vio"].mean()),
                "run_vio_mean": float(group["run_vio"].mean()),
                "mean_final_soc": float(group["mean_final_soc"].mean()),
                "minimum_final_soc": float(group["min_final_soc"].min()),
                "complete_rate": float((group["done_count"] == 344).mean()),
                "action_mean": float(
                    np.average(group["action_mean"], weights=group["total_transitions"])
                ),
                "action_std_mean": float(group["action_std"].mean()),
                "action_min": float(group["action_min"].min()),
                "action_max": float(group["action_max"].max()),
            }
        )
    return pd.DataFrame(rows)


def paired_comparison(
    details: pd.DataFrame,
    *,
    max_normalized_reward_drop: float,
    max_exit_vio_delta: float,
    max_run_vio_delta: float,
) -> pd.DataFrame:
    ddpg = details[details["algorithm"] == "ddpg"].set_index("scenario_id")
    rows = []
    for algorithm in tuple(x for x in details["algorithm"].unique() if x != "ddpg"):
        target = details[details["algorithm"] == algorithm].set_index("scenario_id")
        common = ddpg.index.intersection(target.index)
        if len(common) == 0:
            continue
        reward_drop = ddpg.loc[common, "ep_reward"] - target.loc[common, "ep_reward"]
        normalized = reward_drop / ddpg.loc[common, "ep_reward"].abs().clip(lower=1e-9)
        normalized_mean = float(normalized.mean())
        exit_delta = float(
            (target.loc[common, "exit_vio"] - ddpg.loc[common, "exit_vio"]).mean()
        )
        run_delta = float(
            (target.loc[common, "run_vio"] - ddpg.loc[common, "run_vio"]).mean()
        )
        reward_ready = normalized_mean <= max_normalized_reward_drop
        exit_ready = exit_delta <= max_exit_vio_delta
        run_ready = run_delta <= max_run_vio_delta
        rows.append(
            {
                "algorithm": algorithm,
                "scenario_count": int(len(common)),
                "reward_drop_vs_ddpg_mean": float(reward_drop.mean()),
                "reward_drop_vs_ddpg_std": float(reward_drop.std(ddof=0)),
                "normalized_reward_drop_vs_ddpg_mean": normalized_mean,
                "exit_vio_delta_vs_ddpg_mean": exit_delta,
                "run_vio_delta_vs_ddpg_mean": run_delta,
                "final_soc_delta_vs_ddpg_mean": float(
                    (
                        target.loc[common, "mean_final_soc"]
                        - ddpg.loc[common, "mean_final_soc"]
                    ).mean()
                ),
                "reward_readiness_pass": bool(reward_ready),
                "exit_readiness_pass": bool(exit_ready),
                "run_readiness_pass": bool(run_ready),
                "ready_for_attack_comparison": bool(reward_ready and exit_ready and run_ready),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare DDPG, TD3, SAC, and optionally PPO on identical clean scenarios."
    )
    parser.add_argument("--td3-bundle", type=Path, required=True)
    parser.add_argument("--sac-bundle", type=Path, required=True)
    parser.add_argument("--ppo-bundle", type=Path)
    parser.add_argument("--ddpg-actor", type=Path, default=DEFAULT_DDPG_ACTOR)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--scenes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection-mode", choices=["seeded", "first"], default="seeded")
    parser.add_argument("--reward-profile", choices=sorted(PROFILE_MAP), default="train")
    parser.add_argument("--max-normalized-reward-drop", type=float, default=0.10)
    parser.add_argument("--max-exit-vio-delta", type=float, default=5.0)
    parser.add_argument("--max-run-vio-delta", type=float, default=5.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "results" / "clean_backbone_comparison_seed42",
    )
    args = parser.parse_args()

    required_paths = [args.ddpg_actor, args.td3_bundle, args.sac_bundle]
    if args.ppo_bundle is not None:
        required_paths.append(args.ppo_bundle)
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = resolve_device(args.device)
    actors = {"ddpg": load_actor_from_path(args.ddpg_actor, device)}
    td3_actor, _, td3_payload = load_backbone_bundle(args.td3_bundle, device)
    sac_actor, _, sac_payload = load_backbone_bundle(args.sac_bundle, device)
    if str(td3_payload["algorithm"]).lower() != "td3":
        raise ValueError(f"Expected TD3 bundle: {args.td3_bundle}")
    if str(sac_payload["algorithm"]).lower() != "sac":
        raise ValueError(f"Expected SAC bundle: {args.sac_bundle}")
    actors.update({"td3": td3_actor, "sac": sac_actor})
    if args.ppo_bundle is not None:
        ppo_actor, _, ppo_payload = load_ppo_bundle(args.ppo_bundle, device)
        if str(ppo_payload["algorithm"]).lower() != "ppo":
            raise ValueError(f"Expected PPO bundle: {args.ppo_bundle}")
        actors["ppo"] = ppo_actor

    manifest = load_manifest(args.split).sort_values("Scenario_ID").reset_index(drop=True)
    manifest = (
        manifest.iloc[: args.scenes].reset_index(drop=True)
        if args.selection_mode == "first"
        else deterministic_subset(manifest, args.scenes, args.seed)
    )
    reward_profile = PROFILE_MAP[args.reward_profile]
    rows = []
    for _, scenario in manifest.iterrows():
        arrivals, signal_path, scenario_id = load_scenario(scenario)
        for algorithm, actor in actors.items():
            result = clean_rollout(arrivals, actor, signal_path, device, reward_profile)
            result.update({"algorithm": algorithm, "scenario_id": scenario_id})
            rows.append(result)
            print(
                f"[clean] {scenario_id} {algorithm} reward={result['ep_reward']:.3f} "
                f"exit={result['exit_vio']} run={result['run_vio']}",
                flush=True,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    details = pd.DataFrame(rows)
    summary = summarize(details)
    comparison = paired_comparison(
        details,
        max_normalized_reward_drop=args.max_normalized_reward_drop,
        max_exit_vio_delta=args.max_exit_vio_delta,
        max_run_vio_delta=args.max_run_vio_delta,
    )
    details.to_csv(args.output_dir / "clean_rollouts.csv", index=False)
    summary.to_csv(args.output_dir / "clean_summary.csv", index=False)
    comparison.to_csv(args.output_dir / "paired_comparison_vs_ddpg.csv", index=False)
    report = {
        "protocol": {
            "split": args.split,
            "scenes": int(len(manifest)),
            "selection_seed": int(args.seed),
            "selection_mode": args.selection_mode,
            "reward_profile": args.reward_profile,
            "stochastic_evaluation": False,
            "readiness_thresholds": {
                "max_normalized_reward_drop": float(args.max_normalized_reward_drop),
                "max_exit_vio_delta": float(args.max_exit_vio_delta),
                "max_run_vio_delta": float(args.max_run_vio_delta),
            },
        },
        "bundles": {
            "ddpg": str(args.ddpg_actor),
            "td3": str(args.td3_bundle),
            "sac": str(args.sac_bundle),
            **({"ppo": str(args.ppo_bundle)} if args.ppo_bundle is not None else {}),
        },
        "summary": summary.to_dict(orient="records"),
        "paired_comparison_vs_ddpg": comparison.to_dict(orient="records"),
    }
    write_json(args.output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
