from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from _common import PACKAGE_ROOT, load_manifest, load_scenario, resolve_device, write_json
from dtsr_multiday_common import REPAIR_MODE, set_all_seeds, to_scalar_summary
import _strength_eval_common as attack_common

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from evc.merged_core import ChargingEnv, TRAIN_PROFILE
from evc.native_backbone_attacks import build_native_attacker
from evc.offpolicy_backbones import load_evaluation_backbone
from evc.ug_bcr import rollout_episode_with_ug_bcr


DEFAULT_DDPG_BUNDLE = (
    PACKAGE_ROOT / "models" / "multiday_ddpg_baseline_bundle" / "bundle_multiday_best.pt"
)
DEFAULT_TD3_BUNDLE = (
    PACKAGE_ROOT / "models" / "independent_td3_seed42" / "bundle_selected_ep50.pt"
)
DEFAULT_SAC_BUNDLE = (
    PACKAGE_ROOT / "models" / "independent_sac_seed42" / "bundle_selected_ep40.pt"
)
DEFAULT_PPO_BUNDLE = (
    PACKAGE_ROOT / "models" / "independent_ppo_seed42" / "bundle_selected_ep15.pt"
)
DEFAULT_ATTACK_KEYS = (
    "opposite_pgd",
    "q_function",
    "local_small_drift_q",
    "local_deadline_drift_pgd",
)
DEFAULT_PRICE_THRESHOLD_FILE = (
    PACKAGE_ROOT
    / "results"
    / "attack120_short_horizon"
    / "ehc_threshold_fix"
    / "electhacker_c_price_threshold.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def epsilon_for_attack(attack_key: str, short_epsilon: float, long_epsilon: float) -> float:
    if attack_key == "clean":
        return 0.0
    if attack_key in {"local_small_drift_q", "local_deadline_drift_pgd"}:
        return float(long_epsilon)
    return float(short_epsilon)


def build_paired_degradation(rollouts: pd.DataFrame) -> pd.DataFrame:
    clean = rollouts[rollouts["attack_key"] == "clean"][
        ["algorithm", "scenario_id", "ep_reward", "exit_vio", "run_vio"]
    ].rename(
        columns={
            "ep_reward": "clean_reward",
            "exit_vio": "clean_exit_vio",
            "run_vio": "clean_run_vio",
        }
    )
    attacks = rollouts[rollouts["attack_key"] != "clean"].copy()
    paired = attacks.merge(clean, on=["algorithm", "scenario_id"], how="inner", validate="many_to_one")
    paired["reward_degradation"] = paired["clean_reward"] - paired["ep_reward"]
    paired["normalized_reward_degradation"] = (
        paired["reward_degradation"] / paired["clean_reward"].abs().clip(lower=1e-9)
    )
    paired["exit_vio_increment"] = paired["exit_vio"] - paired["clean_exit_vio"]
    paired["run_vio_increment"] = paired["run_vio"] - paired["clean_run_vio"]
    return paired


def summarize_attacks(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (algorithm, attack_key), group in paired.groupby(
        ["algorithm", "attack_key"], sort=False
    ):
        rows.append(
            {
                "algorithm": algorithm,
                "attack_key": attack_key,
                "scenario_count": int(len(group)),
                "clean_reward_mean": float(group["clean_reward"].mean()),
                "attack_reward_mean": float(group["ep_reward"].mean()),
                "reward_degradation_mean": float(group["reward_degradation"].mean()),
                "reward_degradation_std": float(group["reward_degradation"].std(ddof=0)),
                "normalized_reward_degradation_mean": float(
                    group["normalized_reward_degradation"].mean()
                ),
                "exit_vio_increment_mean": float(group["exit_vio_increment"].mean()),
                "run_vio_increment_mean": float(group["run_vio_increment"].mean()),
                "attack_obs_count_mean": float(group["attack_obs_count"].mean()),
                "attack_delta_linf_mean": float(group["attack_delta_linf_mean"].mean()),
                "attack_delta_linf_max": float(group["attack_delta_linf_max"].max()),
                "runtime_seconds_mean": float(group["runtime_seconds"].mean()),
            }
        )
    return pd.DataFrame(rows)


def compare_to_ddpg(summary: pd.DataFrame) -> pd.DataFrame:
    ddpg = summary[summary["algorithm"] == "ddpg"].set_index("attack_key")
    rows = []
    for algorithm in tuple(x for x in summary["algorithm"].unique() if x != "ddpg"):
        target = summary[summary["algorithm"] == algorithm].set_index("attack_key")
        for attack_key in target.index.intersection(ddpg.index):
            target_row = target.loc[attack_key]
            ddpg_row = ddpg.loc[attack_key]
            ddpg_degradation = float(ddpg_row["reward_degradation_mean"])
            ddpg_normalized = float(ddpg_row["normalized_reward_degradation_mean"])
            target_degradation = float(target_row["reward_degradation_mean"])
            target_normalized = float(target_row["normalized_reward_degradation_mean"])
            rows.append(
                {
                    "algorithm": algorithm,
                    "attack_key": attack_key,
                    "reward_degradation_difference_vs_ddpg": (
                        target_degradation - ddpg_degradation
                    ),
                    "reward_degradation_ratio_vs_ddpg": (
                        target_degradation / ddpg_degradation
                        if abs(ddpg_degradation) > 1e-9
                        else float("nan")
                    ),
                    "normalized_degradation_difference_vs_ddpg": (
                        target_normalized - ddpg_normalized
                    ),
                    "normalized_degradation_ratio_vs_ddpg": (
                        target_normalized / ddpg_normalized
                        if abs(ddpg_normalized) > 1e-9
                        else float("nan")
                    ),
                    "exit_increment_difference_vs_ddpg": float(
                        target_row["exit_vio_increment_mean"]
                        - ddpg_row["exit_vio_increment_mean"]
                    ),
                    "run_increment_difference_vs_ddpg": float(
                        target_row["run_vio_increment_mean"]
                        - ddpg_row["run_vio_increment_mean"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def expectation_checks(
    rollouts: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    expected_rollouts: int,
    short_epsilon: float,
    long_epsilon: float,
) -> dict[str, Any]:
    checks = []
    for row in summary.to_dict(orient="records"):
        epsilon = epsilon_for_attack(row["attack_key"], short_epsilon, long_epsilon)
        checks.append(
            {
                "algorithm": row["algorithm"],
                "attack_key": row["attack_key"],
                "attack_effective": bool(row["reward_degradation_mean"] > 0.0),
                "attacked_observations_positive": bool(row["attack_obs_count_mean"] > 0.0),
                "linf_budget_respected": bool(
                    row["attack_delta_linf_max"] <= epsilon + 1e-5
                ),
                "epsilon": float(epsilon),
                "observed_linf_max": float(row["attack_delta_linf_max"]),
            }
        )
    clean = rollouts[rollouts["attack_key"] == "clean"]
    finite_columns = [
        "ep_reward",
        "exit_vio",
        "run_vio",
        "attack_delta_linf_mean",
        "attack_delta_linf_max",
    ]
    finite = bool(
        np.isfinite(rollouts[finite_columns].to_numpy(dtype=np.float64)).all()
    )
    return {
        "completed_rollouts": int(len(rollouts)),
        "expected_rollouts": int(expected_rollouts),
        "complete": bool(len(rollouts) == expected_rollouts),
        "clean_attack_obs_zero": bool((clean["attack_obs_count"] == 0).all()),
        "all_scalar_metrics_finite": finite,
        "all_attacks_effective": bool(all(row["attack_effective"] for row in checks)),
        "all_linf_budgets_respected": bool(
            all(row["linf_budget_respected"] for row in checks)
        ),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare clean-to-attack degradation across DDPG, TD3, SAC, and optionally PPO."
    )
    parser.add_argument("--ddpg-bundle", type=Path, default=DEFAULT_DDPG_BUNDLE)
    parser.add_argument("--td3-bundle", type=Path, default=DEFAULT_TD3_BUNDLE)
    parser.add_argument("--sac-bundle", type=Path, default=DEFAULT_SAC_BUNDLE)
    parser.add_argument("--ppo-bundle", type=Path)
    parser.add_argument(
        "--algorithms",
        default=None,
        help=(
            "Comma-separated subset of ddpg,td3,sac,ppo. By default all supplied "
            "bundles are evaluated. This is useful for appending a newly adapted "
            "backbone without rerunning completed baselines."
        ),
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--scenes", type=int, default=20)
    parser.add_argument("--attack-keys", default=",".join(DEFAULT_ATTACK_KEYS))
    parser.add_argument("--short-epsilon", type=float, default=0.10)
    parser.add_argument("--long-epsilon", type=float, default=0.055)
    parser.add_argument("--short-iters", type=int, default=10)
    parser.add_argument("--short-alpha", type=float, default=0.01)
    parser.add_argument(
        "--price-threshold-file",
        type=Path,
        default=DEFAULT_PRICE_THRESHOLD_FILE,
        help="Validation-frozen ElectHacker-C raw-price threshold.",
    )
    parser.add_argument(
        "--native-config",
        type=Path,
        default=None,
        help=(
            "Validation-calibrated attack config. When provided, TD3/SAC use "
            "their selected native profiles while DDPG remains canonical."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "results" / "attack_strength_backbones_seed42",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    available_algorithms = ["ddpg", "td3", "sac"]
    if args.ppo_bundle is not None:
        available_algorithms.append("ppo")
    if args.algorithms is None:
        algorithms = tuple(available_algorithms)
    else:
        algorithms = tuple(
            token.strip().lower()
            for token in args.algorithms.split(",")
            if token.strip()
        )
        unknown_algorithms = sorted(set(algorithms) - set(available_algorithms))
        if unknown_algorithms:
            raise ValueError(
                f"Unavailable --algorithms entries: {unknown_algorithms}; "
                f"available={available_algorithms}"
            )
        if not algorithms:
            raise ValueError("At least one algorithm must be selected.")
    native_algorithms = tuple(
        algorithm for algorithm in algorithms if algorithm in {"td3", "sac", "ppo"}
    )

    if args.short_iters <= 0 or args.short_alpha <= 0.0:
        raise ValueError("Short-attack iterations and alpha must be positive.")
    price_threshold = attack_common.load_price_threshold_from_path(
        args.price_threshold_file
    )
    attack_keys = tuple(key.strip() for key in args.attack_keys.split(",") if key.strip())
    if not attack_keys:
        raise ValueError("At least one attack key is required.")
    attack_lookup = {spec["key"]: spec for spec in attack_common.ATTACK_SPECS}
    unknown = sorted(set(attack_keys) - set(attack_lookup))
    if unknown:
        raise ValueError(f"Unknown attack keys: {unknown}")
    specs = [attack_lookup["clean"], *(attack_lookup[key] for key in attack_keys)]

    native_config = None
    if args.native_config is not None:
        if not args.native_config.exists():
            raise FileNotFoundError(args.native_config)
        native_config = json.loads(args.native_config.read_text(encoding="utf-8"))
        if native_config.get("calibration_split") != "val":
            raise ValueError("Native attack config must be calibrated on the val split.")
        if int(native_config.get("seed", -1)) != int(args.seed):
            raise ValueError("Native attack config seed does not match --seed.")
        fixed_budgets = native_config.get("fixed_budgets", {})
        if not math.isclose(
            float(fixed_budgets.get("short_epsilon", -1.0)),
            float(args.short_epsilon),
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            float(fixed_budgets.get("long_epsilon", -1.0)),
            float(args.long_epsilon),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Native attack config budgets do not match the evaluation budgets."
            )
        selected = native_config.get("selected", {})
        # Only the four attacks with algorithm-specific profiles are expected in
        # the calibration file. FGSM and ElectHacker share their frozen canonical
        # definition across backbones and therefore need no profile entry.
        calibrated_attack_keys = {
            "opposite_pgd",
            "q_function",
            "local_small_drift_q",
            "local_deadline_drift_pgd",
        }
        missing = [
            f"{algorithm}/{attack_key}"
            for algorithm in native_algorithms
            for attack_key in attack_keys
            if attack_key in calibrated_attack_keys
            if attack_key not in selected.get(algorithm, {})
        ]
        if missing:
            raise ValueError(f"Native attack config is incomplete: {missing}")

    bundle_paths = {
        "ddpg": args.ddpg_bundle,
        "td3": args.td3_bundle,
        "sac": args.sac_bundle,
    }
    if args.ppo_bundle is not None:
        bundle_paths["ppo"] = args.ppo_bundle
    bundle_paths = {
        algorithm: bundle_paths[algorithm]
        for algorithm in algorithms
    }
    for path in bundle_paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = resolve_device(args.device)
    backbones = {}
    bundle_metadata = {}
    for algorithm, path in bundle_paths.items():
        actor, critic, payload = load_evaluation_backbone(algorithm, path, device)
        backbones[algorithm] = (actor, critic)
        bundle_metadata[algorithm] = payload.get("metadata")

    manifest = load_manifest(args.split).sort_values(
        "Scenario_ID", kind="mergesort"
    ).reset_index(drop=True)
    if args.scenes <= 0 or args.scenes > len(manifest):
        raise ValueError(f"Invalid --scenes={args.scenes}; available={len(manifest)}")
    manifest = manifest.iloc[: args.scenes].copy().reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_path = args.output_dir / "rollouts.jsonl"
    if not args.resume and intermediate_path.exists():
        intermediate_path.unlink()
    expected_keys = {
        (algorithm, str(row["Scenario_ID"]), spec["key"])
        for algorithm in backbones
        for _, row in manifest.iterrows()
        for spec in specs
    }
    rows = [
        row
        for row in (load_jsonl(intermediate_path) if args.resume else [])
        if (row["algorithm"], row["scenario_id"], row["attack_key"]) in expected_keys
    ]
    completed = {
        (row["algorithm"], row["scenario_id"], row["attack_key"]) for row in rows
    }
    if len(completed) != len(rows):
        raise RuntimeError("Duplicate rollout keys in resume file.")

    attack_offsets = {
        str(spec["key"]): (index + 1) * 100_000
        for index, spec in enumerate(attack_common.ATTACK_SPECS)
    }
    run_config = {
        "experiment": "attack_strength_cross_backbone",
        "seed": int(args.seed),
        "split": args.split,
        "scenario_count": int(len(manifest)),
        "scenario_ids": manifest["Scenario_ID"].astype(str).tolist(),
        "attack_keys": list(attack_keys),
        "short_epsilon": float(args.short_epsilon),
        "long_epsilon": float(args.long_epsilon),
        "short_iters": int(args.short_iters),
        "short_alpha": float(args.short_alpha),
        "attack_ratio": 1.0,
        "attack_scope": "obs",
        "electhacker_c_price_threshold": float(price_threshold),
        "electhacker_c_price_threshold_file": str(args.price_threshold_file),
        "attack_seed_rule": "seed + canonical_attack_offset + episode_index",
        "defense_enabled": False,
        "native_attack_config": (
            {
                "path": str(args.native_config),
                "sha256": sha256_file(args.native_config),
                "calibration_split": native_config["calibration_split"],
                "calibration_scenario_ids": native_config["scenario_ids"],
                "selected": native_config["selected"],
            }
            if native_config is not None
            else None
        ),
        "bundles": {
            algorithm: {
                "path": str(path),
                "sha256": sha256_file(path),
                "metadata": bundle_metadata[algorithm],
                "attack_critic": (
                    "single_q"
                    if algorithm == "ddpg"
                    else ("state_value_as_q" if algorithm == "ppo" else "min_twin_q")
                ),
            }
            for algorithm, path in bundle_paths.items()
        },
    }
    write_json(args.output_dir / "run_config.json", run_config)

    started = time.perf_counter()
    for episode_index, (_, scenario_row) in enumerate(manifest.iterrows(), start=1):
        arrivals, signal_path, scenario_id = load_scenario(scenario_row)
        max_duration = max(12, int(arrivals["Duration_of_stay"].max()))
        env = ChargingEnv(signal_path, TRAIN_PROFILE)
        obs_low, obs_high = env.observation_bounds(
            max_duration_of_stay=max_duration
        )
        for attack_spec in specs:
            attack_key = str(attack_spec["key"])
            epsilon = epsilon_for_attack(
                attack_key, args.short_epsilon, args.long_epsilon
            )
            attack_seed = int(
                args.seed + attack_offsets.get(attack_key, 0) + episode_index
            )
            for algorithm, (actor, critic) in backbones.items():
                key = (algorithm, scenario_id, attack_key)
                if key in completed:
                    continue
                set_all_seeds(attack_seed)
                native_entry = None
                if (
                    native_config is not None
                    and algorithm in {"td3", "sac", "ppo"}
                    and attack_key != "clean"
                ):
                    native_entry = (
                        native_config.get("selected", {})
                        .get(algorithm, {})
                        .get(attack_key)
                    )
                if native_entry is not None:
                    attacker = build_native_attacker(
                        attack_key,
                        native_entry["profile"],
                        actor=actor,
                        critic=critic,
                        device=device,
                        obs_low=obs_low,
                        obs_high=obs_high,
                        seed=attack_seed,
                    )
                    profile_id = str(native_entry["profile_id"])
                else:
                    attacker = attack_common.build_attacker_for_rollout(
                        attack_spec=attack_spec,
                        actor=actor,
                        critic=critic,
                        device=device,
                        arrivals=arrivals,
                        signal_path=signal_path,
                        attack_seed=attack_seed,
                        epsilon=epsilon,
                        dae=None,
                        detector_model=None,
                        detector_threshold=0.0,
                        shield_config=None,
                        ug_bcr_config=None,
                        formal_long_outer_epsilon=(
                            args.long_epsilon
                            if attack_key
                            in {"local_small_drift_q", "local_deadline_drift_pgd"}
                            else None
                        ),
                    )
                    profile_id = "clean" if attack_key == "clean" else "canonical_shared"
                if (
                    native_entry is None
                    and attack_key in {"opposite_pgd", "q_function"}
                ):
                    attacker.iters = int(args.short_iters)
                    attacker.alpha = float(args.short_alpha)
                rollout_started = time.perf_counter()
                summary = rollout_episode_with_ug_bcr(
                    arrivals,
                    actor,
                    signal_path,
                    device,
                    TRAIN_PROFILE,
                    attack_enabled=attack_key != "clean",
                    attack_scenario=str(attack_spec["scenario"]),
                    attacker=attacker,
                    epsilon=epsilon,
                    state_scope=str(attack_spec.get("scope", "all")),
                    route_mode="none",
                    enable_shield=False,
                    enable_belief=False,
                    enable_urgency_gate=False,
                    attack_ratio=1.0,
                    attack_scope="obs",
                    price_threshold=float(price_threshold),
                    label=f"{algorithm}__{attack_key}",
                    repair_mode=REPAIR_MODE,
                )
                result = {
                    "algorithm": algorithm,
                    "scenario_id": scenario_id,
                    "episode_index": int(episode_index),
                    "seed": int(args.seed),
                    "attack_seed": int(attack_seed),
                    "attack_key": attack_key,
                    "attack_display_name": str(attack_spec["display"]),
                    "attack_profile_id": profile_id,
                    "epsilon": float(epsilon),
                    "runtime_seconds": float(time.perf_counter() - rollout_started),
                    **to_scalar_summary(summary),
                }
                scalar_values = [
                    value
                    for value in result.values()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                ]
                if not all(math.isfinite(float(value)) for value in scalar_values):
                    raise RuntimeError(f"Non-finite rollout result: {key}")
                done_count = int(result.get("done_count", result.get("done_cnt", -1)))
                if done_count != 344:
                    raise RuntimeError(
                        f"Incomplete rollout for {key}: done_count={done_count}"
                    )
                append_jsonl(intermediate_path, result)
                rows.append(result)
                completed.add(key)
                print(
                    f"[attack-strength] {len(completed):03d}/{len(expected_keys)} "
                    f"{scenario_id} {attack_key} {algorithm} "
                    f"reward={result['ep_reward']:.3f} exit={result['exit_vio']} "
                    f"run={result['run_vio']} linf={result['attack_delta_linf_max']:.5f}",
                    flush=True,
                )

    rollouts = pd.DataFrame(rows)
    rollouts = rollouts[
        rollouts.apply(
            lambda row: (row["algorithm"], row["scenario_id"], row["attack_key"])
            in expected_keys,
            axis=1,
        )
    ].copy()
    rollouts = rollouts.sort_values(
        ["episode_index", "attack_key", "algorithm"], kind="mergesort"
    ).reset_index(drop=True)
    paired = build_paired_degradation(rollouts)
    summary = summarize_attacks(paired)
    comparison = compare_to_ddpg(summary)
    checks = expectation_checks(
        rollouts,
        summary,
        expected_rollouts=len(expected_keys),
        short_epsilon=args.short_epsilon,
        long_epsilon=args.long_epsilon,
    )
    checks["elapsed_seconds_this_invocation"] = float(time.perf_counter() - started)

    rollouts.to_csv(args.output_dir / "rollouts.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(
        args.output_dir / "paired_degradation.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        args.output_dir / "attack_strength_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparison.to_csv(
        args.output_dir / "comparison_to_ddpg.csv", index=False, encoding="utf-8-sig"
    )
    write_json(args.output_dir / "expectation_checks.json", checks)
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
