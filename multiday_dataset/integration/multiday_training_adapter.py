from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from evc.merged_core import (
    TRAIN_PROFILE,
    Actor,
    ChargingEnv,
    DDPGAgent,
    QueueItem,
    ReplayBuffer,
    RewardProfile,
    load_actor_critic_bundle,
    load_actor_from_path,
    set_seed,
)
from integration.multiday_runtime import PairedScenarioDataset


def train_agent_multiday(
    dataset_root,
    device,
    split="train",
    seed=42,
    episodes=500,
    buffer_size=100000,
    batch_size=256,
    learning_starts=2500,
    exploration_noise=1.0,
    gamma=0.9,
    tau=0.005,
    actor_lr=3e-4,
    critic_lr=3e-4,
    print_every=1,
    init_actor_path=None,
    resume_bundle_path=None,
    freeze_actor=False,
    reward_profile: RewardProfile = TRAIN_PROFILE,
):
    """Train one persistent DDPG agent over changing daily scenarios."""
    set_seed(seed)

    dataset = PairedScenarioDataset(
        dataset_root,
        split=split,
        seed=seed,
        shuffle=True,
    )

    if init_actor_path is not None and resume_bundle_path is not None:
        raise ValueError(
            "Use either init_actor_path or resume_bundle_path, not both."
        )

    critic_state_dict = None
    if resume_bundle_path is not None:
        bundle = load_actor_critic_bundle(
            resume_bundle_path,
            device,
        )
        actor = Actor().to(device)
        actor.load_state_dict(bundle["actor_state_dict"])
        critic_state_dict = bundle["critic_state_dict"]
    else:
        actor = (
            load_actor_from_path(init_actor_path, device)
            if init_actor_path is not None
            else Actor().to(device)
        )

    agent = DDPGAgent(
        actor,
        device=device,
        gamma=gamma,
        tau=tau,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
    )

    if critic_state_dict is not None:
        agent.critic.load_state_dict(critic_state_dict)
        agent.critic_target.load_state_dict(critic_state_dict)

    # Observation/action dimensions are fixed for all scenarios.
    first = dataset.load_episode(0)
    first_env = ChargingEnv(
        signals_path=first["signal_path"],
        reward_profile=reward_profile,
    )
    replay = ReplayBuffer(
        buffer_size,
        first_env.obs_dim,
        first_env.action_dim,
        device,
    )

    rows = []
    current_noise = float(exploration_noise)

    for episode in range(1, int(episodes) + 1):
        scenario = dataset.load_episode(episode - 1)
        arrivals = scenario["arrivals"]

        env = ChargingEnv(
            signals_path=scenario["signal_path"],
            reward_profile=reward_profile,
        )
        env.reset()

        idx = 0
        active: list[QueueItem] = []
        last_update = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "mean_q": 0.0,
        }

        while env.t < env.horizon:
            while (
                idx < len(arrivals)
                and int(arrivals.loc[idx, "Arrive_time"]) == env.t
            ):
                obs = env.build_initial_obs(
                    int(arrivals.loc[idx, "Duration_of_stay"])
                )
                action = agent.act(
                    obs,
                    exploration_noise=current_noise,
                    deterministic=False,
                )
                env.enqueue(
                    obs,
                    action,
                    int(arrivals.loc[idx, "Station"]),
                )
                idx += 1

            for item in active:
                action = agent.act(
                    item.obs,
                    exploration_noise=current_noise,
                    deterministic=False,
                )
                env.enqueue(item.obs, action, item.station)

            transitions, active, metrics = env.step()

            for transition in transitions:
                replay.add(
                    transition.obs,
                    transition.next_obs,
                    transition.action,
                    transition.reward,
                    transition.done,
                )
                if replay.size >= max(batch_size, learning_starts):
                    last_update = agent.update(
                        replay.sample(batch_size),
                        freeze_actor=freeze_actor,
                    )

            current_noise *= (
                0.9999
                if current_noise > 0.1
                else 0.999977
            )

        row = {
            "episode": episode,
            "scenario_id": scenario["scenario_id"],
            "ep_reward": float(metrics.ep_reward),
            "ep_r1": float(metrics.ep_r1_cost_sum),
            "ep_r2": float(metrics.ep_r2_exit_penalty_sum),
            "ep_r3": float(metrics.ep_r3_running_penalty_sum),
            "exit_vio": int(metrics.exit_violation_count),
            "run_vio": int(metrics.running_violation_count),
            "replay_size": int(replay.size),
            "actor_loss": float(last_update["actor_loss"]),
            "critic_loss": float(last_update["critic_loss"]),
            "mean_q": float(last_update["mean_q"]),
        }
        rows.append(row)

        if (
            episode == 1
            or episode % int(print_every) == 0
            or episode == int(episodes)
        ):
            print(
                "[multiday-train] "
                f"ep={episode:04d}/{episodes} "
                f"scenario={scenario['scenario_id']} "
                f"reward={row['ep_reward']:.3f} "
                f"exit={row['exit_vio']} "
                f"running={row['run_vio']}"
            )

    return agent, rows


def iter_fixed_test_scenarios(dataset_root):
    """Yield the same 120 test pairs in a fixed order."""
    dataset = PairedScenarioDataset(
        dataset_root,
        split="test",
        seed=0,
        shuffle=False,
    )
    for index in range(len(dataset)):
        yield dataset.load_episode(index)
