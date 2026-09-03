from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from _common import (
    DEFAULT_ACTOR_PATH,
    DEFAULT_BUNDLE_PATH,
    PACKAGE_ROOT,
    load_manifest,
    load_scenario,
    resolve_device,
    write_json,
)

sys.path.insert(0, str(PACKAGE_ROOT))
from evc.merged_attacks import PGDStateAttacker
from evc.merged_core import (
    ATTACK_DEFAULTS,
    ChargingEnv,
    Critic,
    TRAIN_PROFILE,
    load_actor_critic_bundle,
    load_actor_from_path,
)
from evc.merged_pipeline import rollout_episode


CONDITION_SPECS = {
    "clean": {"label": "Clean", "algorithm": None, "scenario": "O"},
    "opposite_pgd": {
        "label": "Opposite-PGD",
        "algorithm": "opposite_pgd",
        "scenario": "O",
    },
    "opposite_fgsm": {
        "label": "Opposite-FGSM",
        "algorithm": "opposite_fgsm",
        "scenario": "O",
    },
    "q_function": {
        "label": "Q-function",
        "algorithm": "q_function",
        "scenario": "O",
    },
    "electhacker_C": {
        "label": "ElectHacker-C",
        "algorithm": "electhacker",
        "scenario": "C",
    },
    "electhacker_F": {
        "label": "ElectHacker-F",
        "algorithm": "electhacker",
        "scenario": "F",
    },
    "electhacker_O": {
        "label": "ElectHacker-O",
        "algorithm": "electhacker",
        "scenario": "O",
    },
}

DEFAULT_CONDITIONS = list(CONDITION_SPECS)


def fast_clean_rollout(arrivals, actor, signal_path, device):
    """Fast deterministic clean rollout without attack-routing diagnostics."""
    env = ChargingEnv(signal_path, TRAIN_PROFILE)
    env.reset()
    actor = actor.to(device).eval()
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
        "label": "clean_no_dae",
        "ep_reward": float(metrics.ep_reward),
        "ep_r1": float(metrics.ep_r1_cost_sum),
        "ep_r2": float(metrics.ep_r2_exit_penalty_sum),
        "ep_r3": float(metrics.ep_r3_running_penalty_sum),
        "ep_r4_dense": float(metrics.ep_r4_dense_safety_penalty_sum),
        "exit_vio": int(metrics.exit_violation_count),
        "run_vio": int(metrics.running_violation_count),
        "total_transitions": int(metrics.total_transitions),
        "done_cnt": int(metrics.done_count),
        "mean_fin_soc": float(final_soc.mean()),
        "std_finl_soc": float(final_soc.std()),
        "cost_cnt": int(len(metrics.costlist)),
        "fin_soc_count": int(len(metrics.final_soc_list)),
        "route_count": 0,
        "route_total": int(metrics.total_transitions),
        "route_rate": 0.0,
        "attack_obs_count": 0,
        "attack_obs_rate": 0.0,
        "attack_delta_count": 0,
        "attack_delta_linf_mean": 0.0,
        "attack_action_abs_diff_mean": 0.0,
    }


def parse_conditions(text: str):
    tokens = [token.strip() for token in str(text).split(",") if token.strip()]
    unknown = [token for token in tokens if token not in CONDITION_SPECS]
    if unknown:
        raise ValueError(
            f"Unknown conditions: {unknown}. Available: {list(CONDITION_SPECS)}"
        )
    if "clean" not in tokens:
        tokens.insert(0, "clean")
    return list(dict.fromkeys(tokens))


def scalar_summary(summary: dict) -> dict:
    return {
        key: value
        for key, value in summary.items()
        if not isinstance(value, (list, tuple, dict))
    }


def actor_states_match(actor, bundle_payload) -> bool:
    bundle_state = bundle_payload.get("actor_state_dict")
    if bundle_state is None:
        return False
    actor_state = actor.state_dict()
    if set(actor_state) != set(bundle_state):
        return False
    return all(
        torch.equal(actor_state[key].detach().cpu(), bundle_state[key].detach().cpu())
        for key in actor_state
    )


def build_attacker(
    *,
    actor,
    critic,
    env,
    device,
    algorithm,
    epsilon,
    pgd_alpha,
    pgd_iters,
    electhacker_alpha,
    electhacker_iters,
    state_scope,
    seed,
):
    low, high = env.observation_bounds(max_duration_of_stay=12)
    defaults = ATTACK_DEFAULTS[algorithm]

    if algorithm == "opposite_fgsm":
        alpha = float(epsilon)
        iters = 1
    elif algorithm == "electhacker":
        alpha = float(electhacker_alpha)
        iters = int(electhacker_iters)
    else:
        alpha = float(pgd_alpha)
        iters = int(pgd_iters)

    return PGDStateAttacker(
        actor,
        device=device,
        algorithm=algorithm,
        epsilon=float(epsilon),
        alpha=alpha if alpha is not None else defaults.alpha,
        iters=iters if iters is not None else defaults.iters,
        seed=int(seed),
        obs_low=low,
        obs_high=high,
        critic=critic if algorithm == "q_function" else None,
        attack_state_scope=state_scope,
    )


def save_atomic_csv(frame: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def create_wide_and_summary(long_frame: pd.DataFrame, condition_order, output_dir: Path):
    labels = {key: CONDITION_SPECS[key]["label"] for key in condition_order}

    pivot = long_frame.pivot_table(
        index=["episode", "scenario_id"],
        columns="condition_key",
        values="ep_reward",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    ordered_columns = ["episode", "scenario_id"] + [
        key for key in condition_order if key in pivot.columns
    ]
    pivot = pivot[ordered_columns].sort_values("episode").reset_index(drop=True)
    pivot = pivot.rename(columns=labels)
    save_atomic_csv(pivot, output_dir / "episode_returns_wide.csv")

    clean_rows = long_frame[long_frame["condition_key"] == "clean"][[
        "scenario_id", "ep_reward"
    ]].rename(columns={"ep_reward": "clean_reward"})
    merged = long_frame.merge(clean_rows, on="scenario_id", how="left")
    merged["paired_reward_degradation"] = merged["clean_reward"] - merged["ep_reward"]

    summary_rows = []
    for key in condition_order:
        subset = merged[merged["condition_key"] == key]
        if subset.empty:
            continue
        summary_rows.append(
            {
                "condition_key": key,
                "condition": labels[key],
                "scenario_count": int(len(subset)),
                "mean_reward": float(subset["ep_reward"].mean()),
                "std_reward": float(subset["ep_reward"].std(ddof=0)),
                "median_reward": float(subset["ep_reward"].median()),
                "min_reward": float(subset["ep_reward"].min()),
                "max_reward": float(subset["ep_reward"].max()),
                "mean_paired_reward_degradation_vs_clean": float(
                    subset["paired_reward_degradation"].mean()
                ),
                "mean_exit_vio": float(subset["exit_vio"].mean()),
                "mean_run_vio": float(subset["run_vio"].mean()),
                "mean_final_soc": float(subset["mean_fin_soc"].mean()),
                "mean_attack_delta_linf": float(
                    subset.get("attack_delta_linf_mean", pd.Series([0.0])).mean()
                ),
                "mean_attack_action_abs_diff": float(
                    subset.get("attack_action_abs_diff_mean", pd.Series([0.0])).mean()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    save_atomic_csv(summary, output_dir / "attack_summary.csv")
    return pivot, summary


def plot_returns(wide: pd.DataFrame, condition_order, output_dir: Path, title: str | None):
    label_order = [CONDITION_SPECS[key]["label"] for key in condition_order]
    available = [label for label in label_order if label in wide.columns]
    if not available:
        raise RuntimeError("No completed conditions are available for plotting.")

    fig, ax = plt.subplots(figsize=(12.0, 6.2))
    line_styles = ["-", "--", "-.", ":", "-", "--", "-."]
    for index, label in enumerate(available):
        ax.plot(
            wide["episode"],
            wide[label],
            label=label,
            linewidth=1.25,
            linestyle=line_styles[index % len(line_styles)],
            alpha=0.90,
        )

    ax.set_xlabel("Test episode / scenario index")
    ax.set_ylabel("Cumulative reward")
    max_episode = int(wide["episode"].max())
    ax.set_xlim(1, max(2, max_episode))
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "attack_cumulative_reward_120.png", dpi=300)
    fig.savefig(output_dir / "attack_cumulative_reward_120.pdf")
    fig.savefig(output_dir / "attack_cumulative_reward_120.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Clean and multiple observation attacks on the same fixed "
            "120 test scenarios, then draw the cumulative-reward line chart."
        )
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--actor-path", type=Path, default=DEFAULT_ACTOR_PATH)
    parser.add_argument("--bundle-path", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--scenes", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--pgd-alpha", type=float, default=0.01)
    parser.add_argument("--pgd-iters", type=int, default=10)
    parser.add_argument("--electhacker-alpha", type=float, default=0.005)
    parser.add_argument("--electhacker-iters", type=int, default=100)
    parser.add_argument("--state-scope", choices=["local", "global", "all"], default="all")
    parser.add_argument("--attack-ratio", type=float, default=1.0)
    parser.add_argument("--attack-scope", choices=["obs", "vehicle", "window"], default="obs")
    parser.add_argument(
        "--conditions",
        default=",".join(DEFAULT_CONDITIONS),
        help="Comma-separated condition keys. Clean is always added.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "results" / "attack_120",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-mismatched-critic", action="store_true")
    parser.add_argument(
        "--title",
        default="Cumulative reward under observation attacks across 120 test scenarios",
    )
    args = parser.parse_args()

    conditions = parse_conditions(args.conditions)
    if args.scenes <= 0 or args.scenes > 120:
        raise ValueError("--scenes must be in [1, 120]")
    if not 0.0 <= args.attack_ratio <= 1.0:
        raise ValueError("--attack-ratio must be in [0, 1]")

    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = resolve_device(args.device)
    actor = load_actor_from_path(args.actor_path, device)
    actor.eval()

    bundle_payload = load_actor_critic_bundle(args.bundle_path, device)
    critic = None
    if "q_function" in conditions:
        if bundle_payload.get("critic_state_dict") is None:
            raise RuntimeError("Q-function attack requires critic weights in --bundle-path")
        if not actor_states_match(actor, bundle_payload) and not args.allow_mismatched_critic:
            raise RuntimeError(
                "The actor in --actor-path does not match the actor stored in --bundle-path. "
                "Use the matching retrained actor/bundle pair, or pass "
                "--allow-mismatched-critic only for an explicitly intended ablation."
            )
        critic = Critic().to(device)
        critic.load_state_dict(bundle_payload["critic_state_dict"])
        critic.eval()

    config = {
        "actor_path": str(args.actor_path.resolve()),
        "bundle_path": str(args.bundle_path.resolve()),
        "scenes": args.scenes,
        "seed": args.seed,
        "epsilon": args.epsilon,
        "pgd_alpha": args.pgd_alpha,
        "pgd_iters": args.pgd_iters,
        "electhacker_alpha": args.electhacker_alpha,
        "electhacker_iters": args.electhacker_iters,
        "state_scope": args.state_scope,
        "attack_ratio": args.attack_ratio,
        "attack_scope": args.attack_scope,
        "conditions": conditions,
        "pairing_rule": (
            "For every episode, Clean and all attacks share the same vehicle CSV "
            "and signal JSON; only the observation attack changes."
        ),
    }
    config_path = args.output_dir / "experiment_config.json"
    if config_path.exists() and args.resume and not args.overwrite:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise RuntimeError(
                "Existing output directory contains a different experiment configuration. "
                "Use --overwrite or a new --output-dir."
            )
    write_json(config_path, config)

    test_manifest = (
        load_manifest("test")
        .sort_values("Scenario_ID")
        .head(args.scenes)
        .reset_index(drop=True)
    )
    test_manifest[["Scenario_ID", "Vehicle_File", "Signal_File"]].assign(
        episode=np.arange(1, len(test_manifest) + 1)
    ).to_csv(args.output_dir / "scenario_order.csv", index=False)

    long_path = args.output_dir / "attack_episode_metrics_long.csv"
    if args.resume and long_path.exists():
        long_frame = pd.read_csv(long_path)
        completed = set(
            zip(long_frame["scenario_id"].astype(str), long_frame["condition_key"].astype(str))
        )
        rows = long_frame.to_dict("records")
    else:
        completed = set()
        rows = []

    for episode_index, row in test_manifest.iterrows():
        episode = int(episode_index) + 1
        arrivals, signal_path, scenario_id = load_scenario(row)
        env_for_bounds = ChargingEnv(signal_path, TRAIN_PROFILE)

        for condition_index, condition_key in enumerate(conditions):
            if (scenario_id, condition_key) in completed:
                continue

            spec = CONDITION_SPECS[condition_key]
            algorithm = spec["algorithm"]
            attacker = None
            if algorithm is not None:
                attacker = build_attacker(
                    actor=actor,
                    critic=critic,
                    env=env_for_bounds,
                    device=device,
                    algorithm=algorithm,
                    epsilon=args.epsilon,
                    pgd_alpha=args.pgd_alpha,
                    pgd_iters=args.pgd_iters,
                    electhacker_alpha=args.electhacker_alpha,
                    electhacker_iters=args.electhacker_iters,
                    state_scope=args.state_scope,
                    seed=args.seed + episode * 1000 + condition_index,
                )

            if algorithm is None:
                summary = fast_clean_rollout(arrivals, actor, signal_path, device)
            else:
                summary = rollout_episode(
                    arrivals,
                    actor,
                    signal_path,
                    device,
                    TRAIN_PROFILE,
                    attack_enabled=True,
                    attack_scenario=spec["scenario"],
                    attacker=attacker,
                    route_mode="none",
                    attack_ratio=args.attack_ratio,
                    attack_scope=args.attack_scope,
                )
            item = scalar_summary(summary)
            item.update(
                {
                    "episode": episode,
                    "scenario_id": scenario_id,
                    "condition_key": condition_key,
                    "condition": spec["label"],
                    "algorithm": "clean" if algorithm is None else algorithm,
                    "attack_scenario": spec["scenario"],
                    "epsilon": 0.0 if algorithm is None else args.epsilon,
                    "state_scope": args.state_scope,
                    "attack_ratio": 0.0 if algorithm is None else args.attack_ratio,
                    "actor_path": str(args.actor_path),
                    "bundle_path": str(args.bundle_path),
                }
            )
            rows.append(item)
            completed.add((scenario_id, condition_key))
            long_frame = pd.DataFrame(rows).sort_values(
                ["episode", "condition_key"]
            ).reset_index(drop=True)
            save_atomic_csv(long_frame, long_path)
            print(
                f"[attack120] episode={episode:03d}/{args.scenes} "
                f"scenario={scenario_id} condition={spec['label']} "
                f"reward={float(item['ep_reward']):.2f}"
            )

    long_frame = pd.DataFrame(rows).sort_values(
        ["episode", "condition_key"]
    ).reset_index(drop=True)
    wide, summary = create_wide_and_summary(
        long_frame,
        conditions,
        args.output_dir,
    )
    required_labels = [CONDITION_SPECS[key]["label"] for key in conditions]
    complete_wide = wide.dropna(subset=required_labels)
    if len(complete_wide) != args.scenes:
        raise RuntimeError(
            f"Only {len(complete_wide)}/{args.scenes} scenarios have all requested conditions."
        )
    plot_returns(complete_wide, conditions, args.output_dir, args.title)

    completion = {
        "status": "completed",
        "scenario_count": args.scenes,
        "condition_count": len(conditions),
        "rollout_count": int(args.scenes * len(conditions)),
        "figure_png": str(args.output_dir / "attack_cumulative_reward_120.png"),
        "figure_pdf": str(args.output_dir / "attack_cumulative_reward_120.pdf"),
        "wide_results": str(args.output_dir / "episode_returns_wide.csv"),
        "long_results": str(long_path),
        "summary": summary.to_dict("records"),
    }
    write_json(args.output_dir / "completion_report.json", completion)
    print(json.dumps(completion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
