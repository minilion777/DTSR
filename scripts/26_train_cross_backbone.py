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

from evc.merged_core import ChargingEnv, PROFILE_MAP, QueueItem, ReplayBuffer, set_seed
from evc.offpolicy_backbones import (
    create_agent,
    initialize_from_ddpg_bundle,
    load_backbone_bundle,
    save_backbone_bundle,
)


DEFAULT_DDPG_BUNDLE = (
    PACKAGE_ROOT / "models" / "multiday_ddpg_baseline_bundle" / "bundle_multiday_best.pt"
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
        "complete_rate": float((frame["done_count"] == 344).mean()),
    }, frame


def is_better(candidate: dict, best: dict | None, tolerance: float = 1e-6) -> bool:
    if best is None:
        return True
    if candidate["mean_exit_vio"] < best["mean_exit_vio"] - tolerance:
        return True
    if candidate["mean_exit_vio"] > best["mean_exit_vio"] + tolerance:
        return False
    if candidate["mean_run_vio"] < best["mean_run_vio"] - tolerance:
        return True
    if candidate["mean_run_vio"] > best["mean_run_vio"] + tolerance:
        return False
    if candidate["mean_reward"] > best["mean_reward"] + tolerance:
        return True
    return False


def scenario_for_episode(manifest: pd.DataFrame, episode_index: int, seed: int):
    cycle = int(episode_index) // len(manifest)
    position = int(episode_index) % len(manifest)
    shuffled = manifest.sample(frac=1.0, random_state=int(seed) + cycle).reset_index(drop=True)
    return shuffled.iloc[position]


def save_checkpoint(agent, path: Path, *, args, episode: int, summary: dict | None) -> None:
    save_backbone_bundle(
        agent,
        path,
        metadata={
            "seed": int(args.seed),
            "checkpoint_episode": int(episode),
            "initialization": (
                "scratch"
                if args.init_mode == "scratch"
                else f"ddpg_transfer:{args.init_ddpg_bundle}"
            ),
            "reward_profile": str(args.reward_profile),
            "validation_summary": summary,
            "training_budget": {
                "episodes": int(args.episodes),
                "updates_per_transition": float(args.updates_per_transition),
                "learning_starts_transitions": int(args.learning_starts),
            },
            "checkpoint_selection": "exit_vio_then_run_vio_then_reward",
            "hyperparameters": {
                "gamma": float(args.gamma),
                "tau": float(args.tau),
                "actor_lr": float(args.actor_lr),
                "critic_lr": float(args.critic_lr),
                "td3_policy_noise": float(args.td3_policy_noise),
                "td3_noise_clip": float(args.td3_noise_clip),
                "td3_policy_delay": int(args.td3_policy_delay),
                "td3_exploration_noise_start": float(args.td3_exploration_noise_start),
                "td3_exploration_noise_end": float(args.td3_exploration_noise_end),
                "sac_alpha_lr": float(args.sac_alpha_lr),
                "sac_initial_alpha": float(args.sac_initial_alpha),
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a TD3 or SAC backbone for DTSR transfer.")
    parser.add_argument("--algorithm", required=True, choices=["td3", "sac"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--reward-profile", choices=sorted(PROFILE_MAP), default="train")
    parser.add_argument("--buffer-size", type=int, default=300000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=5000)
    parser.add_argument(
        "--updates-per-transition",
        type=float,
        default=1.0,
        help="Gradient update-to-data ratio after learning starts.",
    )
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--td3-policy-noise", type=float, default=0.2)
    parser.add_argument("--td3-noise-clip", type=float, default=0.5)
    parser.add_argument("--td3-policy-delay", type=int, default=2)
    parser.add_argument("--td3-exploration-noise-start", type=float, default=0.2)
    parser.add_argument("--td3-exploration-noise-end", type=float, default=0.05)
    parser.add_argument("--sac-alpha-lr", type=float, default=3e-4)
    parser.add_argument("--sac-initial-alpha", type=float, default=0.2)
    parser.add_argument("--validation-every", type=int, default=25)
    parser.add_argument("--validation-scenes", type=int, default=20)
    parser.add_argument("--final-validation-scenes", type=int, default=60)
    parser.add_argument(
        "--allow-initial-checkpoint",
        action="store_true",
        help="Allow the unmodified DDPG initialization to remain the selected best model.",
    )
    parser.add_argument(
        "--init-mode",
        choices=["ddpg_transfer", "scratch"],
        default="scratch",
    )
    parser.add_argument(
        "--init-ddpg-bundle",
        type=Path,
        default=DEFAULT_DDPG_BUNDLE,
        help="DDPG source bundle used when --init-mode=ddpg_transfer.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = (
            PACKAGE_ROOT / "models" / f"independent_{args.algorithm}_seed{args.seed}"
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.episodes <= 0 or args.updates_per_transition <= 0.0:
        raise ValueError("Training episodes and updates per transition must be positive.")
    if args.init_mode == "ddpg_transfer" and not args.init_ddpg_bundle.exists():
        raise FileNotFoundError(args.init_ddpg_bundle)

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
    agent_kwargs = {
        "gamma": args.gamma,
        "tau": args.tau,
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
    }
    if args.algorithm == "td3":
        agent_kwargs.update(
            {
                "policy_noise": args.td3_policy_noise,
                "noise_clip": args.td3_noise_clip,
                "policy_delay": args.td3_policy_delay,
                "exploration_noise": args.td3_exploration_noise_start,
            }
        )
    else:
        agent_kwargs.update(
            {
                "alpha_lr": args.sac_alpha_lr,
                "initial_alpha": args.sac_initial_alpha,
            }
        )
    agent = create_agent(args.algorithm, device, **agent_kwargs)
    if args.init_mode == "ddpg_transfer":
        initialize_from_ddpg_bundle(agent, args.init_ddpg_bundle)

    _, sample_signal, _ = load_scenario(train_manifest.iloc[0])
    sample_env = ChargingEnv(sample_signal, reward_profile)
    replay = ReplayBuffer(
        args.buffer_size, sample_env.obs_dim, sample_env.action_dim, device
    )
    best_path = args.output_dir / "bundle_best.pt"
    initial_path = args.output_dir / "bundle_initial.pt"
    latest_path = args.output_dir / "bundle_latest.pt"
    training_rows = []
    validation_rows = []
    validation_details = []
    total_transitions = 0
    total_env_steps = 0
    total_updates = 0
    update_budget = 0.0
    started = time.perf_counter()

    initial_summary, initial_detail = evaluate_actor(
        agent.actor, validation_manifest, device, reward_profile, 0
    )
    best_summary = initial_summary if args.allow_initial_checkpoint else None
    save_checkpoint(agent, initial_path, args=args, episode=0, summary=initial_summary)
    if args.allow_initial_checkpoint:
        save_checkpoint(agent, best_path, args=args, episode=0, summary=initial_summary)
    validation_rows.append(initial_summary)
    validation_details.append(initial_detail)
    print(
        f"[{args.algorithm}] validation ep=000 reward={initial_summary['mean_reward']:.3f} "
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
        last_update: dict[str, float] = {}
        if args.episodes == 1:
            progress = 1.0
        else:
            progress = (episode - 1) / (args.episodes - 1)
        td3_exploration_noise = (
            args.td3_exploration_noise_start
            + (args.td3_exploration_noise_end - args.td3_exploration_noise_start) * progress
        )
        if args.algorithm == "td3":
            agent.exploration_noise = float(td3_exploration_noise)
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
                if total_transitions < args.learning_starts:
                    actions = np.random.uniform(-1.0, 1.0, size=(len(states), 1)).astype(np.float32)
                else:
                    state_tensor = torch.as_tensor(
                        np.asarray(states, dtype=np.float32),
                        dtype=torch.float32,
                        device=device,
                    )
                    actions = agent.act_tensor(state_tensor, explore=True).cpu().numpy()
                for state, action, station in zip(states, actions, stations):
                    env.enqueue(state, action, station)
            transitions, active, metrics = env.step()
            for transition in transitions:
                replay.add(
                    transition.obs,
                    transition.next_obs,
                    transition.action,
                    transition.reward,
                    transition.done,
                )
            total_transitions += len(transitions)
            total_env_steps += 1
            if replay.size >= max(args.batch_size, args.learning_starts):
                update_budget += len(transitions) * args.updates_per_transition
                gradient_steps = int(update_budget)
                update_budget -= gradient_steps
                for _ in range(gradient_steps):
                    last_update = agent.update(replay.sample(args.batch_size))
                    total_updates += 1

        training_row = {
            "algorithm": args.algorithm,
            "episode": int(episode),
            "scenario_id": scenario_id,
            "ep_reward": float(metrics.ep_reward),
            "exit_vio": int(metrics.exit_violation_count),
            "run_vio": int(metrics.running_violation_count),
            "mean_final_soc": float(np.mean(metrics.final_soc_list)),
            "replay_size": int(replay.size),
            "total_transitions": int(total_transitions),
            "total_env_steps": int(total_env_steps),
            "total_updates": int(total_updates),
            "updates_per_transition": float(args.updates_per_transition),
            "exploration_noise": (
                float(agent.exploration_noise) if args.algorithm == "td3" else float("nan")
            ),
            **{key: float(value) for key, value in last_update.items()},
        }
        training_rows.append(training_row)
        print(
            f"[{args.algorithm}] ep={episode:03d}/{args.episodes} "
            f"reward={training_row['ep_reward']:.3f} exit={training_row['exit_vio']} "
            f"run={training_row['run_vio']} updates={total_updates}",
            flush=True,
        )

        if episode % args.validation_every == 0 or episode == args.episodes:
            summary, detail = evaluate_actor(
                agent.actor, validation_manifest, device, reward_profile, episode
            )
            validation_rows.append(summary)
            validation_details.append(detail)
            print(
                f"[{args.algorithm}] validation ep={episode:03d} "
                f"reward={summary['mean_reward']:.3f} exit={summary['mean_exit_vio']:.3f} "
                f"run={summary['mean_run_vio']:.3f}",
                flush=True,
            )
            if is_better(summary, best_summary):
                best_summary = summary
                save_checkpoint(agent, best_path, args=args, episode=episode, summary=summary)
        save_checkpoint(agent, latest_path, args=args, episode=episode, summary=None)
        pd.DataFrame(training_rows).to_csv(args.output_dir / "training_history.csv", index=False)
        pd.DataFrame(validation_rows).to_csv(args.output_dir / "validation_history.csv", index=False)

    best_payload = torch.load(best_path, map_location=device, weights_only=False)
    best_actor, _, _ = load_backbone_bundle(best_path, device)
    final_summary, final_detail = evaluate_actor(
        best_actor,
        final_validation_manifest,
        device,
        reward_profile,
        int(best_payload["metadata"]["checkpoint_episode"]),
    )
    final_detail.to_csv(args.output_dir / "final_validation_details.csv", index=False)
    write_json(args.output_dir / "final_validation_summary.json", final_summary)
    write_json(
        args.output_dir / "run_config.json",
        {
            "algorithm": args.algorithm,
            "seed": int(args.seed),
            "device": str(device),
            "episodes": int(args.episodes),
            "updates_per_transition": float(args.updates_per_transition),
            "initialization": (
                "scratch"
                if args.init_mode == "scratch"
                else f"ddpg_transfer:{args.init_ddpg_bundle}"
            ),
            "best_checkpoint_episode": int(best_payload["metadata"]["checkpoint_episode"]),
            "best_validation_summary": best_payload["metadata"]["validation_summary"],
            "final_validation_summary": final_summary,
            "elapsed_seconds": float(time.perf_counter() - started),
        },
    )
    print(json.dumps(final_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
