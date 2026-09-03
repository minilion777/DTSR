from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _common import (
    DEFAULT_ACTOR_PATH,
    DEFAULT_BUNDLE_PATH,
    PACKAGE_ROOT,
    deterministic_subset,
    load_manifest,
    load_scenario,
    resolve_device,
    write_json,
)

sys.path.insert(0, str(PACKAGE_ROOT))
from evc.merged_core import (
    Actor,
    ChargingEnv,
    DDPGAgent,
    PROFILE_MAP,
    QueueItem,
    ReplayBuffer,
    load_actor_critic_bundle,
    load_actor_from_path,
    save_actor,
    save_baseline_bundle,
    set_seed,
)


def fast_clean_rollout(arrivals, actor, signal_path, device, reward_profile):
    """Deterministic clean rollout used only for validation/checkpoint selection."""
    env = ChargingEnv(signal_path, reward_profile)
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
        "ep_reward": float(metrics.ep_reward),
        "ep_r1": float(metrics.ep_r1_cost_sum),
        "ep_r2": float(metrics.ep_r2_exit_penalty_sum),
        "ep_r3": float(metrics.ep_r3_running_penalty_sum),
        "ep_r4_dense": float(metrics.ep_r4_dense_safety_penalty_sum),
        "exit_vio": int(metrics.exit_violation_count),
        "run_vio": int(metrics.running_violation_count),
        "mean_final_soc": float(final_soc.mean()),
        "std_final_soc": float(final_soc.std()),
        "done_count": int(metrics.done_count),
        "total_transitions": int(metrics.total_transitions),
    }


def evaluate_actor(actor, manifest, device, reward_profile, checkpoint_episode):
    rows = []
    for _, row in manifest.iterrows():
        arrivals, signal_path, scenario_id = load_scenario(row)
        result = fast_clean_rollout(arrivals, actor, signal_path, device, reward_profile)
        result.update(
            {
                "scenario_id": scenario_id,
                "checkpoint_episode": int(checkpoint_episode),
            }
        )
        rows.append(result)

    frame = pd.DataFrame(rows)
    summary = {
        "checkpoint_episode": int(checkpoint_episode),
        "scenario_count": int(len(frame)),
        "mean_reward": float(frame["ep_reward"].mean()),
        "std_reward": float(frame["ep_reward"].std(ddof=0)),
        "mean_exit_vio": float(frame["exit_vio"].mean()),
        "mean_run_vio": float(frame["run_vio"].mean()),
        "mean_final_soc": float(frame["mean_final_soc"].mean()),
        "complete_rate": float((frame["done_count"] == 344).mean()),
    }
    return summary, frame


def is_better(candidate: dict, best: dict | None, tolerance: float = 1e-6) -> bool:
    """Reward is primary; violations are deterministic tie breakers."""
    if best is None:
        return True
    if candidate["mean_reward"] > best["mean_reward"] + tolerance:
        return True
    if abs(candidate["mean_reward"] - best["mean_reward"]) <= tolerance:
        if candidate["mean_exit_vio"] < best["mean_exit_vio"] - tolerance:
            return True
        if abs(candidate["mean_exit_vio"] - best["mean_exit_vio"]) <= tolerance:
            return candidate["mean_run_vio"] < best["mean_run_vio"] - tolerance
    return False


def build_agent(args, device):
    if args.resume_bundle is not None:
        bundle_path = Path(args.resume_bundle)
        payload = load_actor_critic_bundle(bundle_path, device)
        if payload.get("critic_state_dict") is None:
            raise RuntimeError(f"Bundle lacks critic weights: {bundle_path}")
        actor = Actor().to(device)
        actor.load_state_dict(payload["actor_state_dict"])
        agent = DDPGAgent(
            actor,
            device=device,
            gamma=args.gamma,
            tau=args.tau,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
        )
        agent.critic.load_state_dict(payload["critic_state_dict"])
        agent.actor_target.load_state_dict(agent.actor.state_dict())
        agent.critic_target.load_state_dict(agent.critic.state_dict())
        return agent, f"resume_bundle:{bundle_path}"

    if args.init_mode == "scratch":
        actor = Actor().to(device)
        agent = DDPGAgent(
            actor,
            device=device,
            gamma=args.gamma,
            tau=args.tau,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
        )
        return agent, "scratch"

    if args.init_mode == "baseline_actor":
        actor = load_actor_from_path(args.actor_path, device)
        agent = DDPGAgent(
            actor,
            device=device,
            gamma=args.gamma,
            tau=args.tau,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
        )
        return agent, f"baseline_actor:{args.actor_path}"

    payload = load_actor_critic_bundle(args.bundle_path, device)
    if payload.get("critic_state_dict") is None:
        raise RuntimeError(f"Baseline bundle lacks critic weights: {args.bundle_path}")
    actor = Actor().to(device)
    actor.load_state_dict(payload["actor_state_dict"])
    agent = DDPGAgent(
        actor,
        device=device,
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
    )
    agent.critic.load_state_dict(payload["critic_state_dict"])
    agent.actor_target.load_state_dict(agent.actor.state_dict())
    agent.critic_target.load_state_dict(agent.critic.state_dict())
    return agent, f"baseline_bundle:{args.bundle_path}"


def scenario_for_episode(train_manifest: pd.DataFrame, episode_index: int, seed: int):
    """Shuffle once per cycle, then use every training scenario before repeating."""
    cycle = int(episode_index) // len(train_manifest)
    position = int(episode_index) % len(train_manifest)
    shuffled = train_manifest.sample(frac=1.0, random_state=int(seed) + cycle).reset_index(drop=True)
    return shuffled.iloc[position]


def main():
    parser = argparse.ArgumentParser(
        description="Train or fine-tune DDPG on the multi-day training scenarios."
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument(
        "--init-mode",
        choices=["baseline_bundle", "baseline_actor", "scratch"],
        default="scratch",
        help="scratch is the reproducible default; other modes resume an existing local checkpoint.",
    )
    parser.add_argument("--actor-path", type=Path, default=DEFAULT_ACTOR_PATH)
    parser.add_argument("--bundle-path", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--resume-bundle", type=Path, default=None)
    parser.add_argument("--reward-profile", choices=sorted(PROFILE_MAP), default="train")
    parser.add_argument("--buffer-size", type=int, default=300000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=5000)
    parser.add_argument("--update-every", type=int, default=1)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--noise-start", type=float, default=0.35)
    parser.add_argument("--noise-end", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--validation-every", type=int, default=25)
    parser.add_argument("--validation-scenes", type=int, default=20)
    parser.add_argument("--final-validation-scenes", type=int, default=60)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "models" / "multiday_ddpg",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.validation_every <= 0:
        raise ValueError("--validation-every must be positive")
    if args.update_every <= 0 or args.updates_per_step <= 0:
        raise ValueError("update frequencies must be positive")

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {args.output_dir}. "
                "Use --overwrite or choose another directory."
            )
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    agent, initialization = build_agent(args, device)

    _, sample_signal, _ = load_scenario(train_manifest.iloc[0])
    sample_env = ChargingEnv(sample_signal, reward_profile)
    replay = ReplayBuffer(
        args.buffer_size,
        sample_env.obs_dim,
        sample_env.action_dim,
        device,
    )

    best_actor_path = args.output_dir / "actor_multiday_best.pt"
    best_bundle_path = args.output_dir / "bundle_multiday_best.pt"
    latest_actor_path = args.output_dir / "actor_multiday_latest.pt"
    latest_bundle_path = args.output_dir / "bundle_multiday_latest.pt"

    training_rows = []
    validation_rows = []
    validation_detail_frames = []
    total_transitions = 0
    total_updates = 0

    initial_summary, initial_details = evaluate_actor(
        agent.actor,
        validation_manifest,
        device,
        reward_profile,
        checkpoint_episode=0,
    )
    validation_rows.append(initial_summary)
    validation_detail_frames.append(initial_details)
    best_summary = initial_summary
    save_actor(agent.actor, best_actor_path)
    save_baseline_bundle(
        agent,
        best_bundle_path,
        metadata={
            "checkpoint_episode": 0,
            "initialization": initialization,
            "reward_profile": reward_profile.name,
            "validation_summary": initial_summary,
        },
    )
    print(
        "[validation] ep=0000 "
        f"reward={initial_summary['mean_reward']:.3f} "
        f"exit={initial_summary['mean_exit_vio']:.3f} "
        f"running={initial_summary['mean_run_vio']:.3f}"
    )

    for episode in range(1, args.episodes + 1):
        row = scenario_for_episode(train_manifest, episode - 1, args.seed)
        arrivals, signal_path, scenario_id = load_scenario(row)
        env = ChargingEnv(signal_path, reward_profile)
        env.reset()
        index = 0
        active: list[QueueItem] = []

        if args.episodes == 1:
            progress = 1.0
        else:
            progress = (episode - 1) / (args.episodes - 1)
        exploration_noise = (
            args.noise_start + (args.noise_end - args.noise_start) * progress
        )
        last_update = {"actor_loss": 0.0, "critic_loss": 0.0, "mean_q": 0.0}

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
                    actions = agent.actor(state_tensor).detach().cpu().numpy()
                if exploration_noise > 0.0:
                    actions += np.random.normal(
                        0.0,
                        exploration_noise,
                        size=actions.shape,
                    )
                actions = np.clip(actions, -1.0, 1.0)
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
                total_transitions += 1
                if (
                    replay.size >= max(args.batch_size, args.learning_starts)
                    and total_transitions % args.update_every == 0
                ):
                    for _ in range(args.updates_per_step):
                        last_update = agent.update(replay.sample(args.batch_size))
                        total_updates += 1

        training_row = {
            "episode": episode,
            "scenario_id": scenario_id,
            "exploration_noise": float(exploration_noise),
            "ep_reward": float(metrics.ep_reward),
            "ep_r1": float(metrics.ep_r1_cost_sum),
            "ep_r2": float(metrics.ep_r2_exit_penalty_sum),
            "ep_r3": float(metrics.ep_r3_running_penalty_sum),
            "ep_r4_dense": float(metrics.ep_r4_dense_safety_penalty_sum),
            "exit_vio": int(metrics.exit_violation_count),
            "run_vio": int(metrics.running_violation_count),
            "mean_final_soc": float(np.mean(metrics.final_soc_list)),
            "replay_size": int(replay.size),
            "total_transitions": int(total_transitions),
            "total_updates": int(total_updates),
            "actor_loss": float(last_update.get("actor_loss", 0.0)),
            "critic_loss": float(last_update.get("critic_loss", 0.0)),
            "mean_q": float(last_update.get("mean_q", 0.0)),
        }
        training_rows.append(training_row)

        if episode % args.print_every == 0 or episode in {1, args.episodes}:
            print(
                "[train-ddpg] "
                f"ep={episode:04d}/{args.episodes} "
                f"scenario={scenario_id} "
                f"reward={training_row['ep_reward']:.3f} "
                f"exit={training_row['exit_vio']} "
                f"running={training_row['run_vio']} "
                f"noise={exploration_noise:.4f}"
            )

        should_validate = episode % args.validation_every == 0 or episode == args.episodes
        if should_validate:
            summary, details = evaluate_actor(
                agent.actor,
                validation_manifest,
                device,
                reward_profile,
                checkpoint_episode=episode,
            )
            validation_rows.append(summary)
            validation_detail_frames.append(details)
            print(
                "[validation] "
                f"ep={episode:04d} "
                f"reward={summary['mean_reward']:.3f} "
                f"exit={summary['mean_exit_vio']:.3f} "
                f"running={summary['mean_run_vio']:.3f}"
            )
            if is_better(summary, best_summary):
                best_summary = summary
                save_actor(agent.actor, best_actor_path)
                save_baseline_bundle(
                    agent,
                    best_bundle_path,
                    metadata={
                        "checkpoint_episode": episode,
                        "initialization": initialization,
                        "reward_profile": reward_profile.name,
                        "validation_summary": summary,
                    },
                )
                print(f"[checkpoint] New best checkpoint at episode {episode}")

        save_actor(agent.actor, latest_actor_path)
        save_baseline_bundle(
            agent,
            latest_bundle_path,
            metadata={
                "checkpoint_episode": episode,
                "initialization": initialization,
                "reward_profile": reward_profile.name,
            },
        )
        pd.DataFrame(training_rows).to_csv(
            args.output_dir / "training_history.csv",
            index=False,
        )
        pd.DataFrame(validation_rows).to_csv(
            args.output_dir / "validation_history.csv",
            index=False,
        )

    validation_details = pd.concat(validation_detail_frames, ignore_index=True)
    validation_details.to_csv(
        args.output_dir / "validation_scenario_results.csv",
        index=False,
    )

    best_actor = load_actor_from_path(best_actor_path, device)
    final_summary, final_details = evaluate_actor(
        best_actor,
        final_validation_manifest,
        device,
        reward_profile,
        checkpoint_episode=int(best_summary["checkpoint_episode"]),
    )
    final_details.to_csv(
        args.output_dir / "best_model_full_validation.csv",
        index=False,
    )

    report = {
        "status": "completed",
        "device": str(device),
        "seed": args.seed,
        "episodes": args.episodes,
        "initialization": initialization,
        "reward_profile": reward_profile.name,
        "best_checkpoint_episode": int(best_summary["checkpoint_episode"]),
        "selection_validation": best_summary,
        "full_validation": final_summary,
        "best_actor": str(best_actor_path),
        "best_bundle": str(best_bundle_path),
        "latest_actor": str(latest_actor_path),
        "latest_bundle": str(latest_bundle_path),
        "total_transitions": int(total_transitions),
        "total_updates": int(total_updates),
        "notes": [
            "Validation scenes only are used for checkpoint selection.",
            "The 120 test scenarios are untouched until final attack evaluation.",
            "Episode 0 is retained as a candidate when fine-tuning from the baseline bundle.",
        ],
    }
    write_json(args.output_dir / "training_report.json", report)
    serialized_args = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    serialized_args["device_resolved"] = str(device)
    write_json(args.output_dir / "training_args.json", serialized_args)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
