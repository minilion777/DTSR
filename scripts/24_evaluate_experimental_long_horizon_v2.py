from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from _common import load_manifest, load_scenario, resolve_device  # noqa: E402
from dtsr_multiday_common import (  # noqa: E402
    EP100_ACTOR_PATH,
    EP100_BUNDLE_PATH,
    set_all_seeds,
)
from evc.experimental_long_horizon_v2 import (  # noqa: E402
    EXPERIMENTAL_DEADLINE_PGD_V2,
    EXPERIMENTAL_SMALL_DRIFT_Q_V2,
    MomentumSmallDriftConfig,
    StealthDeadlineConfig,
    build_experimental_long_horizon_attacker,
)
from evc.long_horizon_attacks import build_long_horizon_attacker  # noqa: E402
from evc.merged_attacks import LOCAL_ATTACK_IDX, attack_batch_by_context  # noqa: E402
from evc.merged_core import (  # noqa: E402
    ChargingEnv,
    Critic,
    TRAIN_PROFILE,
    load_actor_critic_bundle,
    load_actor_from_path,
    to_numpy_1d,
)
from evc.merged_pipeline import _build_contexts, summarize_metrics  # noqa: E402
from evc.sequential_adversary import update_active_vehicle_ids  # noqa: E402


ATTACKS = {
    "clean": "clean",
    "original_small": "local_small_drift_q",
    "experimental_small": EXPERIMENTAL_SMALL_DRIFT_Q_V2,
    "original_deadline": "local_deadline_drift_pgd",
    "experimental_deadline": EXPERIMENTAL_DEADLINE_PGD_V2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate isolated experimental long-horizon attacks.")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--scenes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--attacks", default=",".join(ATTACKS))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "results" / "experimental_long_horizon_v2",
    )

    parser.add_argument("--small-epsilon", type=float, default=0.150)
    parser.add_argument("--small-step-size", type=float, default=0.039)
    parser.add_argument("--small-slew", type=float, default=0.030)
    parser.add_argument("--small-momentum", type=float, default=0.86)
    parser.add_argument("--small-current-direction-weight", type=float, default=0.24)
    parser.add_argument("--small-drift-decay", type=float, default=0.89)
    parser.add_argument("--small-base-delta-gain", type=float, default=2.55)
    parser.add_argument("--small-action-pressure-weight", type=float, default=2.50)

    parser.add_argument("--deadline-epsilon", type=float, default=0.085)
    parser.add_argument("--deadline-onset", type=float, default=0.52)
    parser.add_argument("--deadline-window-fraction", type=float, default=0.45)
    parser.add_argument("--deadline-min-steps", type=int, default=3)
    parser.add_argument("--deadline-max-steps", type=int, default=6)
    parser.add_argument("--deadline-full-fraction", type=float, default=0.72)
    parser.add_argument("--deadline-slew-start", type=float, default=0.004)
    parser.add_argument("--deadline-slew-end", type=float, default=0.020)
    parser.add_argument("--deadline-action-shift-start", type=float, default=0.04)
    parser.add_argument("--deadline-action-shift-end", type=float, default=0.70)
    parser.add_argument("--deadline-safety-min", type=float, default=0.17)
    parser.add_argument("--deadline-safety-max", type=float, default=1.0)
    return parser.parse_args()


def _actions(actor, device: torch.device, states: list[np.ndarray]) -> np.ndarray:
    if not states:
        return np.empty((0,), dtype=np.float32)
    with torch.no_grad():
        state_t = torch.as_tensor(np.asarray(states, dtype=np.float32), dtype=torch.float32, device=device)
        return actor(state_t).detach().cpu().numpy().reshape(-1).astype(np.float32)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _sample_std(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=np.float64)
    return float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0


def _build_attacker(
    attack_key: str,
    *,
    actor,
    critic,
    device: torch.device,
    arrivals: pd.DataFrame,
    signal_path: Path,
    seed: int,
    small_config: MomentumSmallDriftConfig,
    deadline_config: StealthDeadlineConfig,
):
    if attack_key == "clean":
        return None
    env = ChargingEnv(signals_path=signal_path, reward_profile=TRAIN_PROFILE)
    low, high = env.observation_bounds(max_duration_of_stay=max(12, int(arrivals["Duration_of_stay"].max())))
    if attack_key == "original_small":
        return build_long_horizon_attacker(
            "local_small_drift_q",
            actor=actor,
            critic=critic,
            device=device,
            obs_low=low,
            obs_high=high,
            seed=seed,
        )
    if attack_key == "original_deadline":
        return build_long_horizon_attacker(
            "local_deadline_drift_pgd",
            actor=actor,
            critic=critic,
            device=device,
            obs_low=low,
            obs_high=high,
            seed=seed,
        )
    if attack_key == "experimental_small":
        return build_experimental_long_horizon_attacker(
            EXPERIMENTAL_SMALL_DRIFT_Q_V2,
            actor=actor,
            critic=critic,
            device=device,
            obs_low=low,
            obs_high=high,
            seed=seed,
            config=small_config,
        )
    if attack_key == "experimental_deadline":
        return build_experimental_long_horizon_attacker(
            EXPERIMENTAL_DEADLINE_PGD_V2,
            actor=actor,
            device=device,
            obs_low=low,
            obs_high=high,
            seed=seed,
            config=deadline_config,
        )
    raise ValueError(f"unknown attack key: {attack_key}")


def rollout(
    *,
    arrivals: pd.DataFrame,
    signal_path: Path,
    actor,
    device: torch.device,
    attacker,
    attack_seed: int,
) -> dict[str, Any]:
    env = ChargingEnv(signals_path=signal_path, reward_profile=TRAIN_PROFILE)
    env.reset()
    if attacker is not None and hasattr(attacker, "reset"):
        attacker.reset()
    idx = 0
    active = []
    active_ids: list[int] = []
    delta_linf: list[float] = []
    adjacent_linf: list[float] = []
    cosine: list[float] = []
    action_shift: list[float] = []
    early_delta: list[float] = []
    early_action_shift: list[float] = []
    late_delta: list[float] = []
    attacked_observations = 0
    prev_delta: dict[int, np.ndarray] = {}
    nonterminal_run_vio = 0
    terminal_run_overlap = 0

    def process(
        states: list[np.ndarray],
        stations: list[int],
        vehicle_ids: list[int],
        *,
        is_new: bool,
    ) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
        nonlocal attacked_observations
        contexts = _build_contexts(env, states, stations, "O", is_new, 400.0, 0.5, 0.3, 1.0, -0.5)
        observed, flags = attack_batch_by_context(
            attacker,
            states,
            contexts,
            attack_ratio=1.0,
            attack_scope="obs",
            vehicle_ids=vehicle_ids,
            episode_index=0,
            seed=attack_seed,
        )
        clean_actions = _actions(actor, device, states)
        adv_actions = _actions(actor, device, observed)
        kinds: list[str] = []
        for clean, adv, clean_action, adv_action, vehicle_id, flag in zip(
            states, observed, clean_actions, adv_actions, vehicle_ids, flags
        ):
            clean_vec = to_numpy_1d(clean)
            adv_vec = to_numpy_1d(adv)
            delta = adv_vec - clean_vec
            linf = float(np.max(np.abs(delta)))
            delta_linf.append(linf)
            shift = float(abs(float(adv_action) - float(clean_action)))
            action_shift.append(shift)
            attacked_observations += int(bool(flag))
            duration = max(int(arrivals.loc[int(vehicle_id), "Duration_of_stay"]), 1)
            remaining = max(int(round(12.0 * float(clean_vec[1]))), 1)
            phase = float(np.clip((duration - remaining) / max(duration - 1, 1), 0.0, 1.0))
            if phase < 0.5:
                early_delta.append(linf)
                early_action_shift.append(shift)
            if phase >= 0.75:
                late_delta.append(linf)
            previous = prev_delta.get(int(vehicle_id))
            if previous is not None:
                adjacent_linf.append(float(np.max(np.abs(delta - previous))))
                local_cur = delta[list(LOCAL_ATTACK_IDX)]
                local_prev = previous[list(LOCAL_ATTACK_IDX)]
                denom = float(np.linalg.norm(local_cur) * np.linalg.norm(local_prev))
                if denom > 1e-10:
                    cosine.append(float(np.dot(local_cur, local_prev) / denom))
            prev_delta[int(vehicle_id)] = delta.astype(np.float32)
            kinds.append("new" if is_new else "active")
        return observed, adv_actions, kinds

    while env.t < env.horizon:
        new_states: list[np.ndarray] = []
        new_stations: list[int] = []
        new_ids: list[int] = []
        while idx < len(arrivals) and int(arrivals.loc[idx, "Arrive_time"]) == env.t:
            new_states.append(env.build_initial_obs(int(arrivals.loc[idx, "Duration_of_stay"])))
            new_stations.append(int(arrivals.loc[idx, "Station"]))
            new_ids.append(int(idx))
            idx += 1

        transition_kinds: list[str] = []
        if new_states:
            _, actions, kinds = process(new_states, new_stations, new_ids, is_new=True)
            for clean_obs, action, station in zip(new_states, actions, new_stations):
                env.enqueue(clean_obs, np.asarray([action], dtype=np.float32), station)
            transition_kinds.extend(kinds)
        if active:
            active_states = [item.obs for item in active]
            active_stations = [item.station for item in active]
            _, actions, kinds = process(active_states, active_stations, active_ids, is_new=False)
            for item, action in zip(active, actions):
                env.enqueue(item.obs, np.asarray([action], dtype=np.float32), item.station)
            transition_kinds.extend(kinds)

        step_ids = new_ids + active_ids
        transitions, active, _ = env.step()
        for transition in transitions:
            if bool(transition.done):
                terminal_run_overlap += int(transition.running_violation_count)
            else:
                nonterminal_run_vio += int(transition.running_violation_count)
        active_ids = update_active_vehicle_ids(step_ids, transitions)

    summary = summarize_metrics(env.metrics, "experimental_attack_rollout")
    total = int(summary["total_transitions"])
    done = int(summary["done_cnt"])
    summary.update(
        {
            "run_vio_nonterminal": int(nonterminal_run_vio),
            "terminal_run_overlap": int(terminal_run_overlap),
            "run_vio_rate": float(nonterminal_run_vio / max(total - done, 1)),
            "exit_vio_rate": float(summary["exit_vio"] / max(done, 1)),
            "attack_obs_rate": float(attacked_observations / max(total, 1)),
            "delta_linf_mean": _mean(delta_linf),
            "adjacent_linf_mean": _mean(adjacent_linf),
            "direction_cosine_mean": _mean(cosine),
            "action_shift_mean": _mean(action_shift),
            "early_delta_linf_mean": _mean(early_delta),
            "early_action_shift_mean": _mean(early_action_shift),
            "late_delta_linf_mean": _mean(late_delta),
        }
    )
    return summary


def main() -> None:
    args = parse_args()
    requested = [x.strip() for x in args.attacks.split(",") if x.strip()]
    unknown = sorted(set(requested) - set(ATTACKS))
    if unknown:
        raise ValueError(f"unknown attacks: {unknown}")
    if args.scenes <= 0:
        raise ValueError("scenes must be positive")

    small_config = MomentumSmallDriftConfig(
        epsilon=args.small_epsilon,
        step_size=args.small_step_size,
        slew_limit=args.small_slew,
        momentum=args.small_momentum,
        current_direction_weight=args.small_current_direction_weight,
        drift_decay=args.small_drift_decay,
        base_delta_gain=args.small_base_delta_gain,
        action_pressure_weight=args.small_action_pressure_weight,
    )
    deadline_config = StealthDeadlineConfig(
        epsilon=args.deadline_epsilon,
        minimum_onset_phase=args.deadline_onset,
        attack_window_fraction=args.deadline_window_fraction,
        min_attack_steps=args.deadline_min_steps,
        max_attack_steps=args.deadline_max_steps,
        full_strength_fraction=args.deadline_full_fraction,
        slew_limit_start=args.deadline_slew_start,
        slew_limit_end=args.deadline_slew_end,
        action_shift_start=args.deadline_action_shift_start,
        action_shift_end=args.deadline_action_shift_end,
        safety_soc_min=args.deadline_safety_min,
        safety_soc_max=args.deadline_safety_max,
    )
    small_config.validate()
    deadline_config.validate()

    set_all_seeds(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = resolve_device(args.device)
    actor = load_actor_from_path(EP100_ACTOR_PATH, device).eval()
    bundle = load_actor_critic_bundle(EP100_BUNDLE_PATH, device)
    critic = Critic().to(device)
    critic.load_state_dict(bundle["critic_state_dict"])
    critic.eval()
    for module in (actor, critic):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    manifest = load_manifest(args.split).sort_values("Scenario_ID", kind="mergesort").head(args.scenes)
    rows: list[dict[str, Any]] = []
    for scene_index, (_, scene_row) in enumerate(manifest.iterrows(), start=1):
        arrivals, signal_path, scenario_id = load_scenario(scene_row)
        for attack_index, attack_key in enumerate(requested):
            attack_seed = int(args.seed + scene_index + attack_index * 100_000)
            set_all_seeds(attack_seed)
            attacker = _build_attacker(
                attack_key,
                actor=actor,
                critic=critic,
                device=device,
                arrivals=arrivals,
                signal_path=Path(signal_path),
                seed=attack_seed,
                small_config=small_config,
                deadline_config=deadline_config,
            )
            metrics = rollout(
                arrivals=arrivals,
                signal_path=Path(signal_path),
                actor=actor,
                device=device,
                attacker=attacker,
                attack_seed=attack_seed,
            )
            row = {
                "scenario_id": scenario_id,
                "scene_index": scene_index,
                "attack_key": attack_key,
                "attack_name": ATTACKS[attack_key],
                "attack_seed": attack_seed,
                **{k: v for k, v in metrics.items() if not isinstance(v, list)},
            }
            rows.append(row)
            print(
                f"scene={scene_index:02d} attack={attack_key:<22} "
                f"reward={float(row['ep_reward']):8.2f} "
                f"run={int(row['run_vio_nonterminal']):4d} "
                f"exit={int(row['exit_vio']):3d} "
                f"mag={float(row['delta_linf_mean']):.4f} "
                f"adj={float(row['adjacent_linf_mean']):.4f}",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    metric_columns = [
        "ep_reward",
        "run_vio_nonterminal",
        "exit_vio",
        "run_vio_rate",
        "exit_vio_rate",
        "mean_fin_soc",
        "delta_linf_mean",
        "adjacent_linf_mean",
        "direction_cosine_mean",
        "action_shift_mean",
        "early_delta_linf_mean",
        "early_action_shift_mean",
        "late_delta_linf_mean",
    ]
    for attack_key, group in frame.groupby("attack_key", sort=False):
        out: dict[str, Any] = {"attack_key": attack_key, "scenes": int(len(group))}
        for column in metric_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            out[f"{column}_mean"] = float(values.mean(skipna=True))
            out[f"{column}_std"] = _sample_std(values)
        summary_rows.append(out)
    summary = pd.DataFrame(summary_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "episode_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    config_payload = {
        "split": args.split,
        "scenes": args.scenes,
        "seed": args.seed,
        "attacks": requested,
        "small_config": small_config.__dict__,
        "deadline_config": deadline_config.__dict__,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n" + summary.to_string(index=False), flush=True)
    print(f"Saved: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
