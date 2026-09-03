from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch

from .merged_attacks import attack_indices_for_state_scope
from .merged_core import Actor, ChargingEnv, RewardProfile, TRAIN_PROFILE, resolve_max_duration_of_stay


def observation_bounds_for_arrivals(arrivals: pd.DataFrame, signals_path, reward_profile: RewardProfile = TRAIN_PROFILE) -> tuple[np.ndarray, np.ndarray]:
    env = ChargingEnv(signals_path=signals_path, reward_profile=reward_profile)
    max_duration = resolve_max_duration_of_stay(arrivals)
    return env.observation_bounds(max_duration_of_stay=max_duration)


def update_active_vehicle_ids(step_vehicle_ids: Sequence[int], transitions) -> list[int]:
    return [int(vehicle_id) for vehicle_id, tr in zip(step_vehicle_ids, transitions) if not bool(tr.done)]


def session_scope_mask(state_scope: str, device: torch.device, *, obs_dim: int = 11) -> torch.Tensor:
    mask = torch.zeros(int(obs_dim), dtype=torch.float32, device=device)
    mask[list(attack_indices_for_state_scope(state_scope))] = 1.0
    return mask


def collect_adversary_rollouts_for_actor(
    arrivals: pd.DataFrame,
    signals_path,
    actor: Actor,
    adversary,
    adversary_value,
    device: torch.device,
    *,
    reward_profile: RewardProfile = TRAIN_PROFILE,
    defender: torch.nn.Module | None = None,
    epsilon: float = 0.15,
    state_scope: str = 'all',
    phase_steps: int = 2048,
    start_episode_index: int = 0,
    obs_low: np.ndarray | None = None,
    obs_high: np.ndarray | None = None,
):
    raise NotImplementedError('Learned sequence adversary rollouts are not restored in the GRU-VAE temporal shield line yet.')


def train_learned_sequence_adversary_for_actor(
    arrivals: pd.DataFrame,
    signals_path,
    actor: Actor,
    device: torch.device,
    *,
    reward_profile: RewardProfile = TRAIN_PROFILE,
    defender: torch.nn.Module | None = None,
    epsilon: float = 0.15,
    state_scope: str = 'all',
    iters: int = 200,
    phase_steps: int = 2048,
    ppo_epochs: int = 10,
    num_minibatches: int = 32,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    adv_actor_lr: float = 1e-3,
    adv_critic_lr: float = 1e-5,
    adv_entropy_coeff: float = 1e-4,
    hidden_dim: int = 128,
    seed: int = 42,
    print_every: int = 10,
):
    raise NotImplementedError('Learned sequence adversary training is not restored in the GRU-VAE temporal shield line yet.')


__all__ = [
    'observation_bounds_for_arrivals',
    'update_active_vehicle_ids',
    'session_scope_mask',
    'collect_adversary_rollouts_for_actor',
    'train_learned_sequence_adversary_for_actor',
]
