from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
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

from evc.merged_core import ChargingEnv, PROFILE_MAP, QueueItem, set_seed
from evc.ppo_backbone import (
    PPOAgent,
    load_ppo_bundle,
    physics_safety_action,
    save_ppo_bundle,
)


def clean_rollout(arrivals, actor, signal_path, device, reward_profile) -> dict[str, float | int]:
    env = ChargingEnv(signal_path, reward_profile)
    env.reset()
    actor = actor.to(device).eval()
    index = 0
    active: list[QueueItem] = []
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
                actions = actor(state_tensor).cpu().numpy()
            for state, action, station in zip(states, actions, stations):
                env.enqueue(state, action, station)
        _, active, metrics = env.step()
    return {
        "ep_reward": float(metrics.ep_reward),
        "exit_vio": int(metrics.exit_violation_count),
        "run_vio": int(metrics.running_violation_count),
        "mean_final_soc": float(np.mean(metrics.final_soc_list)),
        "min_final_soc": float(np.min(metrics.final_soc_list)),
        "done_count": int(metrics.done_count),
        "total_transitions": int(metrics.total_transitions),
    }


def evaluate_actor(actor, manifest, device, reward_profile, episode: int):
    rows = []
    for _, row in manifest.iterrows():
        arrivals, signal_path, scenario_id = load_scenario(row)
        result = clean_rollout(arrivals, actor, signal_path, device, reward_profile)
        result.update({"scenario_id": scenario_id, "checkpoint_episode": int(episode)})
        rows.append(result)
    frame = pd.DataFrame(rows)
    return {
        "checkpoint_episode": int(episode),
        "scenario_count": int(len(frame)),
        "mean_reward": float(frame["ep_reward"].mean()),
        "std_reward": float(frame["ep_reward"].std(ddof=0)),
        "mean_exit_vio": float(frame["exit_vio"].mean()),
        "mean_run_vio": float(frame["run_vio"].mean()),
        "mean_final_soc": float(frame["mean_final_soc"].mean()),
        "minimum_final_soc": float(frame["min_final_soc"].min()),
        "complete_rate": float((frame["done_count"] == 344).mean()),
    }, frame


def is_better(candidate: dict, best: dict | None, tolerance: float = 1e-6) -> bool:
    if best is None:
        return True
    for metric, lower_is_better in (
        ("mean_exit_vio", True),
        ("mean_run_vio", True),
        ("mean_reward", False),
    ):
        lhs = float(candidate[metric])
        rhs = float(best[metric])
        if abs(lhs - rhs) <= tolerance:
            continue
        return lhs < rhs if lower_is_better else lhs > rhs
    return False


def scenario_for_episode(manifest: pd.DataFrame, episode_index: int, seed: int):
    cycle = int(episode_index) // len(manifest)
    position = int(episode_index) % len(manifest)
    shuffled = manifest.sample(frac=1.0, random_state=int(seed) + cycle).reset_index(drop=True)
    return shuffled.iloc[position]


def collect_safety_prior_states(manifest, reward_profile) -> np.ndarray:
    collected: list[np.ndarray] = []
    for _, row in manifest.iterrows():
        arrivals, signal_path, _ = load_scenario(row)
        env = ChargingEnv(signal_path, reward_profile)
        env.reset()
        index = 0
        active: list[QueueItem] = []
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
                state_array = np.asarray(states, dtype=np.float32)
                collected.append(state_array)
                with torch.no_grad():
                    actions = physics_safety_action(torch.as_tensor(state_array)).cpu().numpy()
                for state, action, station in zip(states, actions, stations):
                    env.enqueue(state, action, station)
            _, active, _ = env.step()
    if not collected:
        raise RuntimeError("Safety-prior state collection produced no observations.")
    return np.concatenate(collected, axis=0).astype(np.float32)


def warm_start_actor(
    agent: PPOAgent,
    observations: np.ndarray,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, float]:
    optimizer = torch.optim.Adam(agent.actor.parameters(), lr=float(learning_rate))
    observation_tensor = torch.as_tensor(observations, dtype=torch.float32, device=agent.device)
    final_loss = float("nan")
    for _ in range(int(steps)):
        indices = torch.randint(0, observation_tensor.shape[0], (int(batch_size),), device=agent.device)
        batch = observation_tensor[indices]
        with torch.no_grad():
            target = physics_safety_action(batch)
        prediction = agent.actor(batch)
        loss = torch.nn.functional.mse_loss(prediction, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    with torch.no_grad():
        prediction = agent.actor(observation_tensor)
        target = physics_safety_action(observation_tensor)
        full_mse = float(torch.nn.functional.mse_loss(prediction, target).cpu())
        max_error = float(torch.max(torch.abs(prediction - target)).cpu())
    return {
        "state_count": int(observation_tensor.shape[0]),
        "steps": int(steps),
        "final_batch_mse": final_loss,
        "full_mse": full_mse,
        "max_abs_error": max_error,
    }


def save_checkpoint(agent: PPOAgent, path: Path, *, args, episode: int, summary: dict | None) -> None:
    save_ppo_bundle(
        agent,
        path,
        metadata={
            "seed": int(args.seed),
            "checkpoint_episode": int(episode),
            "initialization": "scratch_with_neutral_action_bias",
            "reward_profile": str(args.reward_profile),
            "validation_summary": summary,
            "checkpoint_selection": "exit_vio_then_run_vio_then_reward",
            "training_budget": {"episodes": int(args.episodes), "on_policy": True},
            "hyperparameters": {
                "gamma": float(args.gamma),
                "gae_lambda": float(args.gae_lambda),
                "clip_ratio": float(args.clip_ratio),
                "actor_lr": float(args.actor_lr),
                "value_lr": float(args.value_lr),
                "entropy_coef": float(args.entropy_coef),
                "value_coef": float(args.value_coef),
                "update_epochs": int(args.update_epochs),
                "minibatch_size": int(args.minibatch_size),
                "target_kl": float(args.target_kl),
                "prior_behavior_coef": float(args.prior_behavior_coef),
                "initial_action_mean": float(args.initial_action_mean),
                "initial_log_std": float(args.initial_log_std),
                "advantage_estimator": "trajectorywise_GAE_with_explicit_vehicle_ids",
                "actor_warm_start": "behavior_pretraining_to_physics_feasibility_prior",
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train continuous-action PPO for the DTSR backbone study.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--reward-profile", choices=sorted(PROFILE_MAP), default="train")
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--value-lr", type=float, default=1e-3)
    parser.add_argument("--entropy-coef", type=float, default=0.002)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--prior-behavior-coef", type=float, default=0.05)
    parser.add_argument("--initial-action-mean", type=float, default=0.45)
    parser.add_argument("--initial-log-std", type=float, default=-1.5)
    parser.add_argument("--warm-start-scenes", type=int, default=20)
    parser.add_argument("--warm-start-steps", type=int, default=1500)
    parser.add_argument("--warm-start-batch-size", type=int, default=512)
    parser.add_argument("--warm-start-lr", type=float, default=1e-3)
    parser.add_argument("--validation-every", type=int, default=5)
    parser.add_argument("--validation-scenes", type=int, default=10)
    parser.add_argument("--final-validation-scenes", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "models" / "independent_ppo_seed42",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.episodes <= 0 or args.update_epochs <= 0 or args.minibatch_size <= 0:
        raise ValueError("PPO training budget must be positive.")

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = resolve_device(args.device)
    set_seed(args.seed)
    reward_profile = PROFILE_MAP[args.reward_profile]
    train_manifest = load_manifest("train").sort_values("Scenario_ID").reset_index(drop=True)
    validation_manifest = deterministic_subset(
        load_manifest("val").sort_values("Scenario_ID").reset_index(drop=True),
        args.validation_scenes,
        args.seed + 101,
    )
    final_validation_manifest = deterministic_subset(
        load_manifest("val").sort_values("Scenario_ID").reset_index(drop=True),
        args.final_validation_scenes,
        args.seed + 202,
    )
    agent = PPOAgent(
        device,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        actor_lr=args.actor_lr,
        value_lr=args.value_lr,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        target_kl=args.target_kl,
        prior_behavior_coef=args.prior_behavior_coef,
        initial_action_mean=args.initial_action_mean,
        initial_log_std=args.initial_log_std,
    )

    warm_manifest = deterministic_subset(train_manifest, args.warm_start_scenes, args.seed + 303)
    warm_states = collect_safety_prior_states(warm_manifest, reward_profile)
    warm_start_summary = warm_start_actor(
        agent,
        warm_states,
        steps=args.warm_start_steps,
        batch_size=args.warm_start_batch_size,
        learning_rate=args.warm_start_lr,
    )
    write_json(args.output_dir / "warm_start_summary.json", warm_start_summary)
    print(f"[ppo] warm-start {json.dumps(warm_start_summary, ensure_ascii=False)}", flush=True)

    best_path = args.output_dir / "bundle_best.pt"
    latest_path = args.output_dir / "bundle_latest.pt"
    training_rows: list[dict] = []
    validation_rows: list[dict] = []
    best_summary: dict | None = None
    started = time.perf_counter()

    initial_summary, _ = evaluate_actor(agent.actor, validation_manifest, device, reward_profile, 0)
    validation_rows.append(initial_summary)
    print(
        f"[ppo] validation ep=000 reward={initial_summary['mean_reward']:.3f} "
        f"exit={initial_summary['mean_exit_vio']:.3f} run={initial_summary['mean_run_vio']:.3f}",
        flush=True,
    )

    for episode in range(1, args.episodes + 1):
        row = scenario_for_episode(train_manifest, episode - 1, args.seed)
        arrivals, signal_path, scenario_id = load_scenario(row)
        env = ChargingEnv(signal_path, reward_profile)
        env.reset()
        index = 0
        active: list[QueueItem] = []
        active_vehicle_ids: list[int] = []
        next_vehicle_id = 0
        transitions_all = []
        transition_vehicle_ids: list[int] = []
        while env.t < env.horizon:
            states = []
            stations = []
            vehicle_ids: list[int] = []
            while index < len(arrivals) and int(arrivals.loc[index, "Arrive_time"]) == env.t:
                states.append(env.build_initial_obs(int(arrivals.loc[index, "Duration_of_stay"])))
                stations.append(int(arrivals.loc[index, "Station"]))
                vehicle_ids.append(next_vehicle_id)
                next_vehicle_id += 1
                index += 1
            if active:
                states.extend(item.obs for item in active)
                stations.extend(item.station for item in active)
                vehicle_ids.extend(active_vehicle_ids)
            if states:
                state_tensor = torch.as_tensor(
                    np.asarray(states, dtype=np.float32), dtype=torch.float32, device=device
                )
                actions = agent.act_tensor(state_tensor, explore=True).cpu().numpy()
                for state, action, station in zip(states, actions, stations):
                    env.enqueue(state, action, station)
            transitions, active, metrics = env.step()
            if len(transitions) != len(vehicle_ids):
                raise RuntimeError("Vehicle identity tracking diverged from environment transitions.")
            transitions_all.extend(transitions)
            transition_vehicle_ids.extend(vehicle_ids)
            active_vehicle_ids = [
                vehicle_id
                for vehicle_id, transition in zip(vehicle_ids, transitions)
                if not transition.done
            ]

        rollout = {
            "observations": np.asarray([t.obs for t in transitions_all], dtype=np.float32),
            "next_observations": np.asarray([t.next_obs for t in transitions_all], dtype=np.float32),
            "actions": np.asarray([t.action for t in transitions_all], dtype=np.float32),
            "rewards": np.asarray([t.reward for t in transitions_all], dtype=np.float32),
            "dones": np.asarray([t.done for t in transitions_all], dtype=np.float32),
            "trajectory_ids": np.asarray(transition_vehicle_ids, dtype=np.int64),
        }
        update_metrics = agent.update(rollout)
        training_row = {
            "algorithm": "ppo",
            "episode": int(episode),
            "scenario_id": scenario_id,
            "ep_reward": float(metrics.ep_reward),
            "exit_vio": int(metrics.exit_violation_count),
            "run_vio": int(metrics.running_violation_count),
            "mean_final_soc": float(np.mean(metrics.final_soc_list)),
            **update_metrics,
        }
        training_rows.append(training_row)
        print(
            f"[ppo] ep={episode:03d}/{args.episodes} reward={training_row['ep_reward']:.3f} "
            f"exit={training_row['exit_vio']} run={training_row['run_vio']} "
            f"kl={training_row['approx_kl']:.5f}",
            flush=True,
        )

        if episode % args.validation_every == 0 or episode == args.episodes:
            summary, _ = evaluate_actor(agent.actor, validation_manifest, device, reward_profile, episode)
            validation_rows.append(summary)
            checkpoint_path = args.output_dir / f"bundle_ep{episode:03d}.pt"
            save_checkpoint(agent, checkpoint_path, args=args, episode=episode, summary=summary)
            print(
                f"[ppo] validation ep={episode:03d} reward={summary['mean_reward']:.3f} "
                f"exit={summary['mean_exit_vio']:.3f} run={summary['mean_run_vio']:.3f}",
                flush=True,
            )
            if is_better(summary, best_summary):
                best_summary = summary
                save_checkpoint(agent, best_path, args=args, episode=episode, summary=summary)
        save_checkpoint(agent, latest_path, args=args, episode=episode, summary=None)
        pd.DataFrame(training_rows).to_csv(args.output_dir / "training_history.csv", index=False)
        pd.DataFrame(validation_rows).to_csv(args.output_dir / "validation_history.csv", index=False)

    if best_summary is None:
        raise RuntimeError("PPO training produced no selectable checkpoint.")
    best_actor, _, best_payload = load_ppo_bundle(best_path, device)
    best_episode = int((best_payload.get("metadata") or {})["checkpoint_episode"])
    final_summary, final_detail = evaluate_actor(
        best_actor, final_validation_manifest, device, reward_profile, best_episode
    )
    final_detail.to_csv(args.output_dir / "final_validation_details.csv", index=False)
    write_json(args.output_dir / "final_validation_summary.json", final_summary)
    run_config = {
        "algorithm": "ppo",
        "seed": int(args.seed),
        "device": str(device),
        "episodes": int(args.episodes),
        "best_checkpoint_episode": best_episode,
        "best_validation_summary": best_summary,
        "final_validation_summary": final_summary,
        "elapsed_seconds": float(time.perf_counter() - started),
        "critic_adapter": "state_value_as_q; action_argument_ignored",
        "warm_start_summary": warm_start_summary,
    }
    write_json(args.output_dir / "run_config.json", run_config)
    print(json.dumps(run_config, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
