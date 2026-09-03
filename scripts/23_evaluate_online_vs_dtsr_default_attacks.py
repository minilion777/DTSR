from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from _common import PACKAGE_ROOT, load_manifest, load_scenario, resolve_device
from dtsr_multiday_common import EP100_BUNDLE_PATH, safe_recovery, set_all_seeds, to_scalar_summary

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from evc.formal_experimental_long_horizon import build_formal_experimental_long_horizon_attacker
from evc.merged_attacks import PGDStateAttacker
from evc.merged_core import ChargingEnv, Critic, TRAIN_PROFILE, load_actor_critic_bundle, load_actor_from_path
from evc.online_atla_ppo_lstm_sa import load_online_atla_ppo_lstm_sa_bundle, run_atla_policy_episode
from evc.ug_bcr import rollout_episode_with_ug_bcr


SHORT_EPSILON = 0.100
LONG_EPSILON = 0.055

ATTACKS: tuple[dict[str, Any], ...] = (
    {
        "key": "clean",
        "display": "Clean",
        "algorithm": None,
        "scope": "all",
        "seed_offset": 0,
        "epsilon": 0.0,
    },
    {
        "key": "opposite_pgd",
        "display": "PGD",
        "algorithm": "opposite_pgd",
        "scope": "all",
        "seed_offset": 100_000,
        "epsilon": SHORT_EPSILON,
    },
    {
        "key": "q_function",
        "display": "Q-function",
        "algorithm": "q_function",
        "scope": "all",
        "seed_offset": 200_000,
        "epsilon": SHORT_EPSILON,
    },
    {
        "key": "local_small_drift_q",
        "display": "Small-drift Q",
        "algorithm": "local_small_drift_q",
        "scope": "local",
        # Preserve the seed offsets used by the complete formal strength table.
        "seed_offset": 700_000,
        "epsilon": LONG_EPSILON,
    },
    {
        "key": "local_deadline_drift_pgd",
        "display": "Deadline-PGD",
        "algorithm": "local_deadline_drift_pgd",
        "scope": "local",
        "seed_offset": 800_000,
        "epsilon": LONG_EPSILON,
    },
)

MODEL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "method": "SA-DDPG",
        "kind": "standard",
        "path": PACKAGE_ROOT
        / "models"
        / "online_short_best_seed42"
        / "sa_ddpg"
        / "sa_ddpg_short_best_ep075_bundle.pt",
    },
    {
        "method": "WocaR",
        "kind": "standard",
        "path": PACKAGE_ROOT
        / "models"
        / "online_short_best_seed42"
        / "wocar"
        / "wocar_short_best_ep100_bundle.pt",
    },
    {
        "method": "ATLA",
        "kind": "atla",
        "path": PACKAGE_ROOT
        / "models"
        / "online_short_best_seed42"
        / "atla"
        / "atla_short_best_iter075_bundle.pt",
    },
)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sample_std(values: pd.Series) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def format_mean_std(mean: float, std: float, digits: int = 1) -> str:
    return f"{float(mean):.{digits}f}±{float(std):.{digits}f}"


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def build_standard_attacker(
    attack: dict[str, Any],
    *,
    actor,
    critic,
    arrivals: pd.DataFrame,
    signal_path: Path,
    device: torch.device,
    attack_seed: int,
):
    algorithm = attack["algorithm"]
    if algorithm is None:
        return None
    env = ChargingEnv(signals_path=signal_path, reward_profile=TRAIN_PROFILE)
    max_duration = max(12, int(arrivals["Duration_of_stay"].max()))
    low, high = env.observation_bounds(max_duration_of_stay=max_duration)
    if algorithm in {"opposite_pgd", "q_function"}:
        return PGDStateAttacker(
            actor,
            device=device,
            algorithm=str(algorithm),
            epsilon=SHORT_EPSILON,
            alpha=SHORT_EPSILON / 10.0,
            iters=10,
            seed=int(attack_seed),
            obs_low=low,
            obs_high=high,
            critic=critic if algorithm == "q_function" else None,
            attack_state_scope="all",
        )
    return build_formal_experimental_long_horizon_attacker(
        str(algorithm),
        actor=actor,
        device=device,
        obs_low=low,
        obs_high=high,
        critic=critic,
        seed=int(attack_seed),
        attack_state_scope="local",
    )


def run_standard_episode(
    *,
    actor,
    critic,
    attack: dict[str, Any],
    arrivals: pd.DataFrame,
    signal_path: Path,
    device: torch.device,
    attack_seed: int,
) -> dict[str, Any]:
    attacker = build_standard_attacker(
        attack,
        actor=actor,
        critic=critic,
        arrivals=arrivals,
        signal_path=signal_path,
        device=device,
        attack_seed=attack_seed,
    )
    summary = rollout_episode_with_ug_bcr(
        arrivals,
        actor,
        signal_path,
        device,
        TRAIN_PROFILE,
        attack_enabled=attack["algorithm"] is not None,
        attack_scenario="O",
        attacker=attacker,
        epsilon=float(attack["epsilon"]),
        state_scope=str(attack["scope"]),
        attack_ratio=1.0,
        attack_scope="obs",
        route_mode="none",
        enable_shield=False,
        enable_belief=False,
        enable_urgency_gate=False,
        label=f"online_{attack['key']}",
        repair_mode="full",
    )
    return to_scalar_summary(summary)


def run_atla_episode(
    *,
    agent,
    critic,
    attack: dict[str, Any],
    arrivals: pd.DataFrame,
    signal_path: Path,
    attack_seed: int,
) -> dict[str, Any]:
    summary = run_atla_policy_episode(
        arrivals,
        signal_path,
        agent,
        reward_profile=TRAIN_PROFILE,
        attack_mode="none" if attack["algorithm"] is None else str(attack["algorithm"]),
        critic=critic if attack["algorithm"] in {"q_function", "local_small_drift_q"} else None,
        epsilon=float(attack["epsilon"]),
        alpha=(SHORT_EPSILON / 10.0 if attack["algorithm"] in {"opposite_pgd", "q_function"} else None),
        iters=(10 if attack["algorithm"] in {"opposite_pgd", "q_function"} else None),
        attack_state_scope=str(attack["scope"]),
        seed=int(attack_seed),
        label=f"atla_{attack['key']}",
    )
    return to_scalar_summary(summary)


def cached_condition_mask(frame: pd.DataFrame, attack_key: str) -> pd.Series:
    if attack_key == "clean":
        return frame["attack_base_key"].eq("clean")
    epsilon = SHORT_EPSILON if attack_key in {"opposite_pgd", "q_function"} else LONG_EPSILON
    return frame["attack_base_key"].eq(attack_key) & np.isclose(
        pd.to_numeric(frame["epsilon"], errors="coerce"), epsilon, rtol=0.0, atol=1e-12
    )


def build_cached_rows(cached_path: Path, scenario_ids: list[str]) -> list[dict[str, Any]]:
    frame = pd.read_csv(cached_path)
    frame = frame[frame["scenario_id"].astype(str).isin(scenario_ids)].copy()
    output: list[dict[str, Any]] = []
    for attack in ATTACKS:
        condition = frame[cached_condition_mask(frame, str(attack["key"]))]
        for method, stage_key in (("DDPG", "attack"), ("DTSR", "ug_bcr")):
            selected = condition[condition["stage_key"].eq(stage_key)].copy()
            if len(selected) != len(scenario_ids):
                raise RuntimeError(
                    f"Cached {method}/{attack['key']} expected {len(scenario_ids)} rows, got {len(selected)}."
                )
            for _, source in selected.iterrows():
                output.append(
                    {
                        "scenario_id": str(source["scenario_id"]),
                        "episode_index": int(source["episode_index"]),
                        "method": method,
                        "model_path": "EP100 DDPG + offline DTSR" if method == "DTSR" else "EP100 DDPG",
                        "attack_key": str(attack["key"]),
                        "attack_display": str(attack["display"]),
                        "epsilon": float(attack["epsilon"]),
                        "attack_seed": int(source["attack_seed"]),
                        "ep_reward": float(source["ep_reward"]),
                        "run_vio": int(float(source["run_vio"])),
                        "exit_vio": int(float(source["exit_vio"])),
                        "done_cnt": int(float(source["done_cnt"])),
                        "source": "cached_latest_strength_table",
                    }
                )
    return output


def build_summary(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = long_df[long_df["attack_key"].eq("clean")].set_index(["scenario_id", "method"])
    baseline_clean = clean.xs("DDPG", level="method")["ep_reward"]
    baseline_attack = long_df[long_df["method"].eq("DDPG")].set_index(["scenario_id", "attack_key"])

    raw_rows: list[dict[str, Any]] = []
    for attack in ATTACKS[1:]:
        attack_key = str(attack["key"])
        base_attacked = baseline_attack.xs(attack_key, level="attack_key")
        for method in ("SA-DDPG", "WocaR", "ATLA", "DTSR"):
            defended = long_df[
                long_df["method"].eq(method) & long_df["attack_key"].eq(attack_key)
            ].set_index("scenario_id")
            method_clean = clean.xs(method, level="method")["ep_reward"]
            common = baseline_clean.index.intersection(base_attacked.index).intersection(defended.index).intersection(method_clean.index)
            if len(common) == 0:
                raise RuntimeError(f"No paired rows for {method}/{attack_key}.")
            recovery = pd.Series(
                [
                    100.0
                    * safe_recovery(
                        float(baseline_clean.loc[scenario_id]),
                        float(base_attacked.loc[scenario_id, "ep_reward"]),
                        float(defended.loc[scenario_id, "ep_reward"]),
                    )
                    for scenario_id in common
                ],
                dtype=float,
            )
            raw_rows.append(
                {
                    "attack_key": attack_key,
                    "attack": str(attack["display"]),
                    "epsilon": float(attack["epsilon"]),
                    "method": method,
                    "scenario_count": int(len(common)),
                    "baseline_clean_reward_mean": float(baseline_clean.loc[common].mean()),
                    "baseline_clean_reward_std": sample_std(baseline_clean.loc[common]),
                    "baseline_attack_reward_mean": float(base_attacked.loc[common, "ep_reward"].mean()),
                    "baseline_attack_reward_std": sample_std(base_attacked.loc[common, "ep_reward"]),
                    "method_clean_reward_mean": float(method_clean.loc[common].mean()),
                    "method_clean_reward_std": sample_std(method_clean.loc[common]),
                    "defended_reward_mean": float(defended.loc[common, "ep_reward"].mean()),
                    "defended_reward_std": sample_std(defended.loc[common, "ep_reward"]),
                    "recovery_mean_pct": float(recovery.mean()),
                    "recovery_std_pct": sample_std(recovery),
                    "baseline_attack_run_vio_mean": float(base_attacked.loc[common, "run_vio"].mean()),
                    "baseline_attack_run_vio_sum": int(base_attacked.loc[common, "run_vio"].sum()),
                    "defended_run_vio_mean": float(defended.loc[common, "run_vio"].mean()),
                    "defended_run_vio_sum": int(defended.loc[common, "run_vio"].sum()),
                    "baseline_attack_exit_vio_mean": float(base_attacked.loc[common, "exit_vio"].mean()),
                    "baseline_attack_exit_vio_sum": int(base_attacked.loc[common, "exit_vio"].sum()),
                    "defended_exit_vio_mean": float(defended.loc[common, "exit_vio"].mean()),
                    "defended_exit_vio_sum": int(defended.loc[common, "exit_vio"].sum()),
                }
            )

    raw = pd.DataFrame(raw_rows)
    paper = pd.DataFrame(
        {
            "攻击": raw["attack"],
            "方法": raw["method"],
            "无攻击基准奖励": [
                format_mean_std(mean, std)
                for mean, std in zip(raw["baseline_clean_reward_mean"], raw["baseline_clean_reward_std"])
            ],
            "攻击后奖励": [
                format_mean_std(mean, std)
                for mean, std in zip(raw["baseline_attack_reward_mean"], raw["baseline_attack_reward_std"])
            ],
            "防御后奖励": [
                format_mean_std(mean, std)
                for mean, std in zip(raw["defended_reward_mean"], raw["defended_reward_std"])
            ],
            "恢复率/%": [
                format_mean_std(mean, std)
                for mean, std in zip(raw["recovery_mean_pct"], raw["recovery_std_pct"])
            ],
            "运行违规/场（合计）": [
                f"{mean:.2f} ({total})"
                for mean, total in zip(raw["defended_run_vio_mean"], raw["defended_run_vio_sum"])
            ],
            "离站违规/场（合计）": [
                f"{mean:.2f} ({total})"
                for mean, total in zip(raw["defended_exit_vio_mean"], raw["defended_exit_vio_sum"])
            ],
        }
    )
    return raw, paper


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare retained online robust policies with strict no-leak DTSR.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["test"], default="test")
    parser.add_argument("--scenes", type=int, default=20)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument(
        "--cached-strength-path",
        type=Path,
        default=PACKAGE_ROOT
        / "results"
        / "exp2_strength_no_adaptive_20scenes_seed42"
        / "tables"
        / "exp2_strength_addendum_rollouts.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "results" / "online_vs_dtsr_default_attacks_20scenes_seed42",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-rollouts", type=int, default=0)
    args = parser.parse_args()

    if args.seed != 42:
        raise ValueError("This comparison is locked to seed 42.")
    if args.scenes <= 0 or args.scenes > 20:
        raise ValueError("This comparison supports 1..20 scenes because the cached latest DTSR table has 20 scenes.")
    for spec in MODEL_SPECS:
        if not Path(spec["path"]).exists():
            raise FileNotFoundError(f"Missing retained model: {spec['path']}")
    if not args.cached_strength_path.exists():
        raise FileNotFoundError(f"Missing latest strength metrics: {args.cached_strength_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_path = args.output_dir / "intermediate" / "online_rollouts.jsonl"
    if args.overwrite and intermediate_path.exists():
        intermediate_path.unlink()

    torch.set_num_threads(max(1, int(args.num_threads)))
    set_all_seeds(args.seed)
    device = resolve_device(args.device)

    baseline_payload = load_actor_critic_bundle(EP100_BUNDLE_PATH, device)
    critic_state = baseline_payload.get("critic_state_dict")
    if critic_state is None:
        raise RuntimeError("EP100 baseline bundle has no critic_state_dict.")
    shared_critic = Critic().to(device)
    shared_critic.load_state_dict(critic_state)
    shared_critic.eval()
    for parameter in shared_critic.parameters():
        parameter.requires_grad_(False)

    models: dict[str, tuple[str, Any, Path]] = {}
    for spec in MODEL_SPECS:
        path = Path(spec["path"])
        if spec["kind"] == "atla":
            model = load_online_atla_ppo_lstm_sa_bundle(path, device)
        else:
            model = load_actor_from_path(path, device).eval()
        models[str(spec["method"])] = (str(spec["kind"]), model, path)

    manifest = load_manifest(args.split).sort_values("Scenario_ID", kind="mergesort").head(args.scenes).reset_index(drop=True)
    scenario_ids = manifest["Scenario_ID"].astype(str).tolist()

    existing = load_jsonl(intermediate_path) if args.resume else []
    completed = {(str(row["scenario_id"]), str(row["method"]), str(row["attack_key"])) for row in existing}
    expected_online = args.scenes * len(MODEL_SPECS) * len(ATTACKS)
    new_rollouts = 0
    started = time.perf_counter()

    for episode_index, (_, scenario_row) in enumerate(manifest.iterrows(), start=1):
        arrivals, signal_path, scenario_id = load_scenario(scenario_row)
        for method, (kind, model, model_path) in models.items():
            for attack in ATTACKS:
                key = (scenario_id, method, str(attack["key"]))
                if key in completed:
                    continue
                attack_seed = int(args.seed + int(attack["seed_offset"]) + episode_index)
                set_all_seeds(attack_seed)
                rollout_started = time.perf_counter()
                if kind == "atla":
                    summary = run_atla_episode(
                        agent=model,
                        critic=shared_critic,
                        attack=attack,
                        arrivals=arrivals,
                        signal_path=signal_path,
                        attack_seed=attack_seed,
                    )
                else:
                    summary = run_standard_episode(
                        actor=model,
                        critic=shared_critic,
                        attack=attack,
                        arrivals=arrivals,
                        signal_path=signal_path,
                        device=device,
                        attack_seed=attack_seed,
                    )
                runtime_seconds = float(time.perf_counter() - rollout_started)
                row = {
                    "scenario_id": scenario_id,
                    "episode_index": int(episode_index),
                    "method": method,
                    "model_path": str(model_path),
                    "attack_key": str(attack["key"]),
                    "attack_display": str(attack["display"]),
                    "epsilon": float(attack["epsilon"]),
                    "attack_seed": attack_seed,
                    "runtime_seconds": runtime_seconds,
                    "source": "new_unified_online_evaluation",
                    **summary,
                }
                if int(row.get("done_cnt", -1)) != 344:
                    raise RuntimeError(f"Incomplete rollout {key}: done_cnt={row.get('done_cnt')}")
                append_jsonl(intermediate_path, row)
                existing.append(row)
                completed.add(key)
                new_rollouts += 1
                print(
                    f"[{len(completed):03d}/{expected_online}] ep={episode_index:02d} {method:7s} "
                    f"{attack['display']:13s} reward={float(row['ep_reward']):.2f} "
                    f"run/exit={int(row.get('run_vio', 0))}/{int(row.get('exit_vio', 0))} "
                    f"time={runtime_seconds:.2f}s",
                    flush=True,
                )
                if args.max_rollouts > 0 and new_rollouts >= args.max_rollouts:
                    return

    online_rows = [
        row
        for row in load_jsonl(intermediate_path)
        if str(row["scenario_id"]) in scenario_ids
        and str(row["method"]) in {spec["method"] for spec in MODEL_SPECS}
        and str(row["attack_key"]) in {attack["key"] for attack in ATTACKS}
    ]
    if len(online_rows) != expected_online:
        raise RuntimeError(f"Expected {expected_online} online rows, got {len(online_rows)}.")

    cached_rows = build_cached_rows(args.cached_strength_path, scenario_ids)
    long_df = pd.DataFrame(cached_rows + online_rows)
    long_df = long_df.sort_values(["episode_index", "attack_key", "method"], kind="mergesort").reset_index(drop=True)
    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(tables_dir / "online_vs_dtsr_episode_metrics_long.csv", index=False, float_format="%.6f")
    raw, paper = build_summary(long_df)
    raw.to_csv(tables_dir / "online_vs_dtsr_default_attacks_raw.csv", index=False, float_format="%.6f")
    paper.to_csv(tables_dir / "online_vs_dtsr_default_attacks_paper.csv", index=False, encoding="utf-8-sig")
    (tables_dir / "online_vs_dtsr_default_attacks_paper.md").write_text(
        "# 在线鲁棒防御与离线 DTSR 对比（默认长短时序攻击）\n\n"
        + markdown_table(paper)
        + "\n恢复率按每个测试场景计算后取均值："
        + "(防御后奖励-无防御攻击奖励)/(无攻击DDPG奖励-无防御攻击奖励)×100%。"
        + "短时序 ε=0.10，长时序 ε_long=0.055；数值为20个固定测试场景的均值±样本标准差。\n",
        encoding="utf-8",
    )
    config = {
        "seed": args.seed,
        "split": args.split,
        "scenes": args.scenes,
        "device": str(device),
        "num_threads": torch.get_num_threads(),
        "short_epsilon": SHORT_EPSILON,
        "long_epsilon": LONG_EPSILON,
        "attack_keys": [attack["key"] for attack in ATTACKS],
        "models": [{"method": spec["method"], "path": str(spec["path"])} for spec in MODEL_SPECS],
        "baseline_bundle": str(EP100_BUNDLE_PATH),
        "cached_latest_dtsr_metrics": str(args.cached_strength_path),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(paper.to_string(index=False), flush=True)
    print(f"saved: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
