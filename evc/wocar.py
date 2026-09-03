from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .merged_attacks import attack_indices_for_state_scope, build_state_attacker, canonical_attack_state_scope
from .merged_core import (
    Actor,
    ChargingEnv,
    Critic,
    QueueItem,
    RewardProfile,
    TRAIN_PROFILE,
    ensure_dir,
    load_actor_critic_bundle,
    load_actor_from_path,
    set_seed,
    to_numpy_1d,
)
from .sa_ddpg import (
    SAReplayBuffer,
    StateObservationAdversary,
    canonical_sa_train_attack,
    normalize_sa_train_attacks,
)
from .multiday_schedule import max_duration_across_scenarios, normalize_episode_scenarios, scenario_for_episode
from .robust_bounds import (
    actor_crown_action_bounds,
    actor_ibp_action_bounds,
    actor_split_crown_action_bounds,
    observation_bounds_across_scenarios,
    scalar_action_stability_loss,
)


DEFAULT_ONLINE_WOCAR_V1_TRAIN_ATTACKS = ('opposite_pgd', 'q_function')
ONLINE_WOCAR_REG_WEIGHT_MODES = ('uniform', 'q_gap')


def canonical_online_wocar_reg_weight_mode(value: str | None) -> str:
    token = str(value or 'uniform').strip().lower().replace('-', '_')
    aliases = {
        'none': 'uniform',
        'flat': 'uniform',
        'constant': 'uniform',
        'qgap': 'q_gap',
        'q_gap': 'q_gap',
        'gap': 'q_gap',
    }
    mode = aliases.get(token, token)
    if mode not in ONLINE_WOCAR_REG_WEIGHT_MODES:
        raise ValueError(f'Unsupported Online WocaR reg weight mode: {value}')
    return mode


def interval_q_extrema_1d(
    critic: Critic,
    observations: torch.Tensor,
    action_lower: torch.Tensor,
    action_upper: torch.Tensor,
    *,
    grid_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized Q extrema over a one-dimensional action interval."""
    grid_size = max(int(grid_size), 3)
    obs = observations.float()
    lower = action_lower.reshape(-1, 1)
    upper = action_upper.reshape(-1, 1)
    if lower.shape != upper.shape or lower.shape[0] != obs.shape[0]:
        raise ValueError('WocaR action intervals must have shape [batch, 1].')
    fractions = torch.linspace(0.0, 1.0, grid_size, dtype=obs.dtype, device=obs.device).reshape(1, -1, 1)
    actions = lower.unsqueeze(1) + fractions * (upper - lower).unsqueeze(1)
    repeated_obs = obs.unsqueeze(1).expand(-1, grid_size, -1)
    q_values = critic(
        repeated_obs.reshape(obs.shape[0] * grid_size, -1),
        actions.reshape(obs.shape[0] * grid_size, 1),
    ).reshape(obs.shape[0], grid_size)
    min_q, min_ids = torch.min(q_values, dim=1)
    max_q, max_ids = torch.max(q_values, dim=1)
    min_actions = actions[torch.arange(obs.shape[0], device=obs.device), min_ids]
    max_actions = actions[torch.arange(obs.shape[0], device=obs.device), max_ids]
    return min_q, max_q, min_actions, max_actions


def projected_q_extrema_1d(
    critic: Critic,
    observations: torch.Tensor,
    action_lower: torch.Tensor,
    action_upper: torch.Tensor,
    *,
    grid_size: int,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Grid-initialized projected action search used by the WocaR source."""
    grid_min_q, grid_max_q, min_action, max_action = interval_q_extrema_1d(
        critic,
        observations,
        action_lower,
        action_upper,
        grid_size=grid_size,
    )
    steps = max(int(steps), 0)
    if steps <= 0:
        return grid_min_q, grid_max_q, min_action, max_action

    obs = observations.detach()
    lower = action_lower.detach().reshape(-1, 1)
    upper = action_upper.detach().reshape(-1, 1)
    step_size = (upper - lower).clamp_min(1e-6) / float(steps)

    def refine(initial_action: torch.Tensor, *, maximize: bool) -> tuple[torch.Tensor, torch.Tensor]:
        action = initial_action.detach()
        with torch.enable_grad():
            for _ in range(steps):
                action = action.detach().requires_grad_(True)
                q_value = critic(obs, action).reshape(-1)
                gradient = torch.autograd.grad(q_value.sum(), action, retain_graph=False)[0]
                direction = gradient.sign() if maximize else -gradient.sign()
                action = torch.maximum(torch.minimum(action + step_size * direction, upper), lower)
            action = action.detach()
            value = critic(obs, action).reshape(-1).detach()
        return value, action

    refined_min_q, refined_min_action = refine(min_action, maximize=False)
    refined_max_q, refined_max_action = refine(max_action, maximize=True)
    use_refined_min = refined_min_q < grid_min_q.detach()
    use_refined_max = refined_max_q > grid_max_q.detach()
    min_q = torch.where(use_refined_min, refined_min_q, grid_min_q.detach())
    max_q = torch.where(use_refined_max, refined_max_q, grid_max_q.detach())
    min_actions = torch.where(use_refined_min.reshape(-1, 1), refined_min_action, min_action.detach())
    max_actions = torch.where(use_refined_max.reshape(-1, 1), refined_max_action, max_action.detach())
    return min_q, max_q, min_actions, max_actions


@dataclass
class OnlineWocaRV1TrainHistory:
    rows: list[dict]
    validation_rows: list[dict] | None = None


def _replay_buffer_state(buffer: SAReplayBuffer) -> dict:
    size = int(buffer.size)
    return {
        'capacity': int(buffer.capacity),
        'size': size,
        'pos': int(buffer.pos),
        'obs': buffer.obs[:size].copy(),
        'next_obs': buffer.next_obs[:size].copy(),
        'actions': buffer.actions[:size].copy(),
        'rewards': buffer.rewards[:size].copy(),
        'dones': buffer.dones[:size].copy(),
        'is_new_arrivals': buffer.is_new_arrivals[:size].copy(),
    }


def _restore_replay_buffer(buffer: SAReplayBuffer, state: dict) -> None:
    size = int(state.get('size', 0))
    capacity = int(state.get('capacity', buffer.capacity))
    if capacity != int(buffer.capacity):
        raise ValueError(
            f'WocaR resume replay capacity mismatch: checkpoint={capacity}, configured={buffer.capacity}.'
        )
    if not 0 <= size <= int(buffer.capacity):
        raise ValueError(f'Invalid WocaR resume replay size: {size}.')
    fields = ('obs', 'next_obs', 'actions', 'rewards', 'dones', 'is_new_arrivals')
    for name in fields:
        source = np.asarray(state[name], dtype=np.float32)
        target = getattr(buffer, name)
        if source.shape != target[:size].shape:
            raise ValueError(
                f'WocaR resume replay field {name} has shape {source.shape}, expected {target[:size].shape}.'
            )
        target[:size] = source
    buffer.size = size
    buffer.pos = int(state.get('pos', size % int(buffer.capacity)))


def _rng_state() -> dict:
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict) -> None:
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    if torch.cuda.is_available() and state.get('torch_cuda') is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state['torch_cuda']])


def _module_state_to_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _load_module_state(module: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    module.load_state_dict({key: value.detach().clone() for key, value in state.items()})


class OnlineWocaRV1Agent:
    """Minimal clean-replay online WocaR baseline.

    The only robust mechanism in V1 is the critic Bellman target:
    y = (1 - lambda) * y_clean + lambda * y_worst.
    Rollouts and replay storage remain clean.
    """

    def __init__(
        self,
        actor: Actor,
        device: torch.device,
        *,
        gamma: float = 0.9,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        adversary: StateObservationAdversary | None = None,
        target_lambda: float = 0.25,
        actor_beta: float = 0.0,
        actor_q_weight: float = 0.0,
        actor_reg_weight: float = 0.0,
        actor_reg_weight_mode: str = 'uniform',
        actor_reg_weight_clip: float = 0.0,
        separate_worst_critic: bool = False,
        clean_anchor_actor: Actor | None = None,
        actor_clean_anchor_weight: float = 0.0,
    ) -> None:
        self.device = device
        self.actor = actor.to(device)
        self.critic = Critic().to(device)
        self.actor_target = Actor().to(device)
        self.critic_target = Critic().to(device)
        self.separate_worst_critic = bool(separate_worst_critic)
        self.worst_critic = Critic().to(device) if self.separate_worst_critic else None
        self.worst_critic_target = Critic().to(device) if self.separate_worst_critic else None
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        if self.worst_critic is not None and self.worst_critic_target is not None:
            self.worst_critic.load_state_dict(self.critic.state_dict())
            self.worst_critic_target.load_state_dict(self.worst_critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(actor_lr))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(critic_lr))
        self.worst_critic_optimizer = (
            torch.optim.Adam(self.worst_critic.parameters(), lr=float(critic_lr))
            if self.worst_critic is not None
            else None
        )
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.adversary = adversary
        self.target_lambda = float(np.clip(float(target_lambda), 0.0, 1.0))
        self.actor_beta = float(np.clip(float(actor_beta), 0.0, 1.0))
        self.actor_q_weight = max(float(actor_q_weight), 0.0)
        self.actor_reg_weight = max(float(actor_reg_weight), 0.0)
        self.actor_reg_weight_mode = canonical_online_wocar_reg_weight_mode(actor_reg_weight_mode)
        self.actor_reg_weight_clip = max(float(actor_reg_weight_clip), 0.0)
        self.actor_clean_anchor_weight = max(float(actor_clean_anchor_weight), 0.0)
        self.clean_anchor_actor = None if clean_anchor_actor is None else clean_anchor_actor.to(device).eval()
        if self.clean_anchor_actor is not None:
            for param in self.clean_anchor_actor.parameters():
                param.requires_grad_(False)

    def _soft_update(self, src: torch.nn.Module, dst: torch.nn.Module) -> None:
        for s_param, d_param in zip(src.parameters(), dst.parameters()):
            d_param.data.copy_(self.tau * s_param.data + (1.0 - self.tau) * d_param.data)

    def attack_critic(self) -> Critic:
        if self.separate_worst_critic and self.worst_critic is not None:
            return self.worst_critic
        return self.critic

    def _target_attack_critic(self) -> Critic:
        if self.separate_worst_critic and self.worst_critic_target is not None:
            return self.worst_critic_target
        return self.critic_target

    def select_action(
        self,
        obs,
        *,
        exploration_noise: float = 0.0,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, dict[str, float]]:
        obs_t = torch.as_tensor(to_numpy_1d(obs), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_t).reshape(-1)
            if not deterministic and exploration_noise > 0.0:
                action = action + torch.normal(
                    mean=0.0,
                    std=float(exploration_noise),
                    size=action.shape,
                    device=self.device,
                )
            action = action.clamp(-1.0, 1.0).detach().cpu().numpy().astype(np.float32)
        return action, {'attacked_frac': 0.0, 'adv_linf_mean': 0.0, 'adv_l2_mean': 0.0}

    def _observation_candidates(
        self,
        obs_clean: torch.Tensor,
        *,
        actor: Actor,
        critic: Critic,
        attack_families: tuple[str, ...],
        enabled: bool,
        prefix: str,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        clean = obs_clean.detach()
        candidates: list[torch.Tensor] = [clean]
        stats = {
            f'{prefix}_candidate_count': 1.0,
            f'{prefix}_adv_frac': 0.0,
            f'{prefix}_adv_linf': 0.0,
            f'{prefix}_adv_l2': 0.0,
        }
        if (
            not enabled
            or self.adversary is None
            or float(getattr(self.adversary, 'epsilon', 0.0)) <= 0.0
        ):
            return torch.stack(candidates, dim=1), stats

        flags = torch.zeros(clean.shape[0], dtype=torch.float32, device=self.device)
        for family in attack_families:
            with torch.enable_grad():
                adv = self.adversary.perturb_tensor(
                    clean,
                    actor=actor,
                    critic=critic,
                    attack_family=family,
                    is_new_arrivals=flags,
                ).detach()
            candidates.append(adv)

        stacked = torch.stack(candidates, dim=1)
        deltas = stacked[:, 1:, :] - clean.unsqueeze(1)
        linf = torch.max(torch.abs(deltas), dim=2).values
        l2 = torch.linalg.vector_norm(deltas, ord=2, dim=2)
        stats = {
            f'{prefix}_candidate_count': float(stacked.shape[1]),
            f'{prefix}_adv_frac': 1.0 if stacked.shape[1] > 1 else 0.0,
            f'{prefix}_adv_linf': float(linf.mean().detach().cpu().item()) if linf.numel() > 0 else 0.0,
            f'{prefix}_adv_l2': float(l2.mean().detach().cpu().item()) if l2.numel() > 0 else 0.0,
        }
        for idx, family in enumerate(attack_families, start=1):
            stats[f'{prefix}_candidate_{canonical_sa_train_attack(family)}_index'] = float(idx)
        return stacked, stats

    def _next_action_candidates(
        self,
        next_obs_clean: torch.Tensor,
        *,
        attack_families: tuple[str, ...],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return self._observation_candidates(
            next_obs_clean,
            actor=self.actor_target,
            critic=self._target_attack_critic(),
            attack_families=attack_families,
            enabled=self.target_lambda > 0.0 or self.separate_worst_critic,
            prefix='target',
        )

    def update(
        self,
        batch: dict[str, torch.Tensor],
        *,
        attack_families: tuple[str, ...],
    ) -> dict[str, float]:
        obs_clean = batch['observations']
        next_obs_clean = batch['next_observations']
        rewards = batch['rewards']
        dones = batch['dones']

        next_obs_candidates, candidate_stats = self._next_action_candidates(
            next_obs_clean,
            attack_families=attack_families,
        )
        batch_size, candidate_count, obs_dim = next_obs_candidates.shape
        with torch.no_grad():
            flat_next_obs = next_obs_candidates.reshape(batch_size * candidate_count, obs_dim)
            flat_next_actions = self.actor_target(flat_next_obs)
            repeated_clean_next = next_obs_clean.unsqueeze(1).expand(-1, candidate_count, -1)
            repeated_clean_next = repeated_clean_next.reshape(batch_size * candidate_count, obs_dim)
            target_attack_critic = self._target_attack_critic()
            candidate_q = target_attack_critic(repeated_clean_next, flat_next_actions).reshape(batch_size, candidate_count)
            next_attack_clean_q = candidate_q[:, 0]
            if self.separate_worst_critic:
                clean_next_actions = self.actor_target(next_obs_clean)
                next_clean_q = self.critic_target(next_obs_clean, clean_next_actions).reshape(-1)
            else:
                next_clean_q = next_attack_clean_q
            next_worst_q = torch.min(candidate_q, dim=1).values
            y_clean = rewards + (1.0 - dones) * self.gamma * next_clean_q
            y_attack_clean = rewards + (1.0 - dones) * self.gamma * next_attack_clean_q
            y_worst = rewards + (1.0 - dones) * self.gamma * next_worst_q
            target_lambda = self.target_lambda
            q_target = y_clean if self.separate_worst_critic else (1.0 - target_lambda) * y_clean + target_lambda * y_worst
            worst_q_target = (1.0 - target_lambda) * y_attack_clean + target_lambda * y_worst

        q_pred = self.critic(obs_clean, batch['actions']).reshape(-1)
        critic_loss = F.mse_loss(q_pred, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        worst_critic_loss = q_pred.new_tensor(0.0)
        worst_q_pred = q_pred.detach()
        if self.separate_worst_critic:
            if self.worst_critic is None or self.worst_critic_optimizer is None:
                raise RuntimeError('Separate worst critic mode was enabled without a worst critic.')
            worst_q_pred = self.worst_critic(obs_clean, batch['actions']).reshape(-1)
            worst_critic_loss = F.mse_loss(worst_q_pred, worst_q_target)
            self.worst_critic_optimizer.zero_grad()
            worst_critic_loss.backward()
            self.worst_critic_optimizer.step()

        actor_beta = self.actor_beta
        actor_clean_actions = self.actor(obs_clean)
        actor_clean_q = self.critic(obs_clean, actor_clean_actions).reshape(-1)
        actor_value_critic = self.attack_critic()
        actor_value_clean_q = actor_value_critic(obs_clean, actor_clean_actions).reshape(-1)
        actor_clean_policy_loss = -actor_clean_q.mean()
        actor_worst_policy_loss = actor_clean_policy_loss
        actor_worst_q = actor_value_clean_q
        actor_q_policy_loss = actor_clean_policy_loss
        actor_q_q = actor_value_clean_q
        actor_q_candidate_index = 0
        actor_reg_loss = actor_clean_actions.new_tensor(0.0)
        actor_clean_anchor_loss = actor_clean_actions.new_tensor(0.0)
        actor_clean_anchor_action_mse_mean = 0.0
        actor_reg_per_sample = None
        actor_reg_action_mse_mean = 0.0
        actor_reg_candidate_count = 0.0
        actor_reg_sample_weight_mean = 0.0
        actor_reg_sample_weight_max = 0.0
        actor_reg_q_gap_weight_mean = 0.0
        actor_candidate_stats = {
            'actor_candidate_count': 1.0,
            'actor_adv_frac': 0.0,
            'actor_adv_linf': 0.0,
            'actor_adv_l2': 0.0,
        }
        actor_worst_ids = torch.zeros(obs_clean.shape[0], dtype=torch.long, device=self.device)
        if actor_beta > 0.0 or self.actor_q_weight > 0.0 or self.actor_reg_weight > 0.0:
            actor_obs_candidates, actor_candidate_stats = self._observation_candidates(
                obs_clean,
                actor=self.actor,
                critic=actor_value_critic,
                attack_families=attack_families,
                enabled=True,
                prefix='actor',
            )
            actor_batch_size, actor_candidate_count, actor_obs_dim = actor_obs_candidates.shape
            flat_actor_obs = actor_obs_candidates.reshape(actor_batch_size * actor_candidate_count, actor_obs_dim)
            flat_actor_actions = self.actor(flat_actor_obs)
            actor_candidate_actions = flat_actor_actions.reshape(actor_batch_size, actor_candidate_count, -1)
            if actor_candidate_count > 1:
                actor_action_deltas = actor_candidate_actions[:, 1:, :] - actor_clean_actions.unsqueeze(1)
                actor_reg_per_sample = actor_action_deltas.pow(2).mean(dim=(1, 2))
                actor_reg_action_mse_mean = float(actor_reg_per_sample.detach().mean().cpu().item())
                actor_reg_candidate_count = float(actor_candidate_count - 1)
            repeated_actor_clean_obs = obs_clean.unsqueeze(1).expand(-1, actor_candidate_count, -1)
            repeated_actor_clean_obs = repeated_actor_clean_obs.reshape(actor_batch_size * actor_candidate_count, actor_obs_dim)
            actor_candidate_q = actor_value_critic(repeated_actor_clean_obs, flat_actor_actions).reshape(
                actor_batch_size,
                actor_candidate_count,
            )
            actor_worst_q, actor_worst_ids = torch.min(actor_candidate_q, dim=1)
            actor_worst_policy_loss = -actor_worst_q.mean()
            q_index_value = actor_candidate_stats.get('actor_candidate_q_function_index')
            if q_index_value is not None:
                q_index = int(q_index_value)
                if 0 <= q_index < actor_candidate_count:
                    actor_q_candidate_index = q_index
                    actor_q_q = actor_candidate_q[:, q_index]
                    actor_q_policy_loss = -actor_q_q.mean()
            if actor_reg_per_sample is not None:
                reg_weights = torch.ones_like(actor_reg_per_sample)
                if self.actor_reg_weight_mode == 'q_gap':
                    q_gap_weight = (actor_value_clean_q.detach() - actor_worst_q.detach()).clamp_min(0.0)
                    q_gap_mean = q_gap_weight.mean().clamp_min(1e-6)
                    reg_weights = q_gap_weight / q_gap_mean
                    if self.actor_reg_weight_clip > 0.0:
                        reg_weights = reg_weights.clamp(max=self.actor_reg_weight_clip)
                    actor_reg_q_gap_weight_mean = float(reg_weights.detach().mean().cpu().item())
                actor_reg_loss = (reg_weights * actor_reg_per_sample).mean()
                actor_reg_sample_weight_mean = float(reg_weights.detach().mean().cpu().item())
                actor_reg_sample_weight_max = float(reg_weights.detach().max().cpu().item())
        if self.actor_clean_anchor_weight > 0.0 and self.clean_anchor_actor is not None:
            with torch.no_grad():
                anchor_actions = self.clean_anchor_actor(obs_clean).detach()
            actor_clean_anchor_per_sample = (actor_clean_actions - anchor_actions).pow(2).mean(dim=1)
            actor_clean_anchor_loss = actor_clean_anchor_per_sample.mean()
            actor_clean_anchor_action_mse_mean = float(actor_clean_anchor_per_sample.detach().mean().cpu().item())
        actor_base_loss = (1.0 - actor_beta) * actor_clean_policy_loss + actor_beta * actor_worst_policy_loss
        actor_q_weight = self.actor_q_weight
        actor_without_reg_loss = (
            (actor_base_loss + actor_q_weight * actor_q_policy_loss) / (1.0 + actor_q_weight)
            if actor_q_weight > 0.0 and actor_q_candidate_index > 0
            else actor_base_loss
        )
        actor_reg_weight = self.actor_reg_weight
        actor_loss = (
            actor_without_reg_loss
            + actor_reg_weight * actor_reg_loss
            + self.actor_clean_anchor_weight * actor_clean_anchor_loss
        )
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        if self.separate_worst_critic and self.worst_critic is not None and self.worst_critic_target is not None:
            self._soft_update(self.worst_critic, self.worst_critic_target)

        target_gap = y_attack_clean - y_worst
        return {
            'actor_loss': float(actor_loss.detach().cpu().item()),
            'critic_loss': float(critic_loss.detach().cpu().item()),
            'mean_q': float(q_pred.detach().mean().cpu().item()),
            'separate_worst_critic': float(self.separate_worst_critic),
            'worst_critic_loss': float(worst_critic_loss.detach().cpu().item()),
            'worst_mean_q': float(worst_q_pred.detach().mean().cpu().item()),
            'actor_policy_loss': float(actor_loss.detach().cpu().item()),
            'actor_beta': float(self.actor_beta),
            'actor_q_weight': float(self.actor_q_weight),
            'actor_reg_weight': float(self.actor_reg_weight),
            'actor_reg_weight_mode': str(self.actor_reg_weight_mode),
            'actor_reg_weight_clip': float(self.actor_reg_weight_clip),
            'actor_clean_anchor_weight': float(self.actor_clean_anchor_weight),
            'actor_clean_policy_loss': float(actor_clean_policy_loss.detach().cpu().item()),
            'actor_worst_policy_loss': float(actor_worst_policy_loss.detach().cpu().item()),
            'actor_q_policy_loss': float(actor_q_policy_loss.detach().cpu().item()),
            'actor_without_reg_policy_loss': float(actor_without_reg_loss.detach().cpu().item()),
            'actor_reg_loss': float(actor_reg_loss.detach().cpu().item()),
            'actor_clean_anchor_loss': float(actor_clean_anchor_loss.detach().cpu().item()),
            'actor_clean_anchor_term_active': float(self.actor_clean_anchor_weight > 0.0 and self.clean_anchor_actor is not None),
            'actor_clean_anchor_action_mse_mean': float(actor_clean_anchor_action_mse_mean),
            'actor_clean_q_mean': float(actor_clean_q.detach().mean().cpu().item()),
            'actor_worst_critic_clean_q_mean': float(actor_value_clean_q.detach().mean().cpu().item()),
            'actor_worst_q_mean': float(actor_worst_q.detach().mean().cpu().item()),
            'actor_q_q_mean': float(actor_q_q.detach().mean().cpu().item()),
            'actor_clean_to_worst_critic_gap_mean': float((actor_clean_q - actor_value_clean_q).detach().mean().cpu().item()),
            'actor_worst_gap_mean': float((actor_value_clean_q - actor_worst_q).detach().mean().cpu().item()),
            'actor_q_gap_mean': float((actor_value_clean_q - actor_q_q).detach().mean().cpu().item()),
            'actor_worst_nonclean_frac': float((actor_worst_ids != 0).float().mean().detach().cpu().item()),
            'actor_q_candidate_index': float(actor_q_candidate_index),
            'actor_q_term_active': float(actor_q_weight > 0.0 and actor_q_candidate_index > 0),
            'actor_reg_term_active': float(actor_reg_weight > 0.0 and actor_reg_candidate_count > 0.0),
            'actor_reg_action_mse_mean': float(actor_reg_action_mse_mean),
            'actor_reg_candidate_count': float(actor_reg_candidate_count),
            'actor_reg_sample_weight_mean': float(actor_reg_sample_weight_mean),
            'actor_reg_sample_weight_max': float(actor_reg_sample_weight_max),
            'actor_reg_q_gap_weight_mean': float(actor_reg_q_gap_weight_mean),
            'actor_candidate_count': float(actor_candidate_stats['actor_candidate_count']),
            'actor_adv_frac': float(actor_candidate_stats['actor_adv_frac']),
            'actor_adv_linf': float(actor_candidate_stats['actor_adv_linf']),
            'actor_adv_l2': float(actor_candidate_stats['actor_adv_l2']),
            'target_lambda': float(self.target_lambda),
            'target_candidate_count': float(candidate_stats['target_candidate_count']),
            'y_clean_mean': float(y_clean.detach().mean().cpu().item()),
            'y_attack_clean_mean': float(y_attack_clean.detach().mean().cpu().item()),
            'y_worst_mean': float(y_worst.detach().mean().cpu().item()),
            'q_target_mean': float(q_target.detach().mean().cpu().item()),
            'worst_q_target_mean': float(worst_q_target.detach().mean().cpu().item()),
            'next_clean_q_mean': float(next_clean_q.detach().mean().cpu().item()),
            'next_worst_q_mean': float(next_worst_q.detach().mean().cpu().item()),
            'target_gap_mean': float(target_gap.detach().mean().cpu().item()),
            'worst_target_le_clean_frac': float((y_worst <= y_attack_clean + 1e-6).float().mean().detach().cpu().item()),
            'target_adv_frac': float(candidate_stats['target_adv_frac']),
            'target_adv_linf': float(candidate_stats['target_adv_linf']),
            'target_adv_l2': float(candidate_stats['target_adv_l2']),
            'update_adv_frac': 0.0,
            'update_adv_linf': 0.0,
            'update_adv_l2': 0.0,
        }


class WocaR1DIntervalAgent(OnlineWocaRV1Agent):
    """WocaR adaptation using CROWN bounds plus attack-guided auxiliary losses."""

    def __init__(
        self,
        actor: Actor,
        device: torch.device,
        *,
        obs_low: np.ndarray | torch.Tensor,
        obs_high: np.ndarray | torch.Tensor,
        epsilon: float,
        state_scope: str,
        gamma: float = 0.9,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        worst_policy_weight: float = 0.25,
        state_reg_weight: float = 0.03,
        state_weight_clip: float = 3.0,
        worst_action_grid_size: int = 17,
        state_importance_grid_size: int = 17,
        candidate_adversary: StateObservationAdversary | None = None,
        candidate_worst_weight: float = 0.0,
        candidate_q_weight: float = 0.0,
        candidate_reg_weight: float = 0.0,
        candidate_reg_weight_clip: float = 1.0,
        candidate_interval_margin: float = 0.0,
        target_lambda: float = 1.0,
        clean_anchor_actor: Actor | None = None,
        actor_clean_anchor_weight: float = 0.0,
        crown_split_dimensions: int = 2,
        preserve_reachable_superset: bool = True,
        worst_action_pgd_steps: int = 10,
        state_importance_pgd_steps: int = 10,
        worst_critic_updates: int = 2,
    ) -> None:
        super().__init__(
            actor,
            device,
            gamma=gamma,
            tau=tau,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            adversary=candidate_adversary,
            target_lambda=target_lambda,
            actor_beta=worst_policy_weight,
            actor_q_weight=0.0,
            actor_reg_weight=state_reg_weight,
            actor_reg_weight_mode='q_gap',
            actor_reg_weight_clip=state_weight_clip,
            separate_worst_critic=True,
            clean_anchor_actor=clean_anchor_actor,
            actor_clean_anchor_weight=actor_clean_anchor_weight,
        )
        if int(actor.fc_mu.out_features) != 1:
            raise ValueError('WocaR1DIntervalAgent requires action_dim=1.')
        self.obs_low_t = torch.as_tensor(obs_low, dtype=torch.float32, device=device).reshape(1, -1)
        self.obs_high_t = torch.as_tensor(obs_high, dtype=torch.float32, device=device).reshape(1, -1)
        self.max_epsilon = max(float(epsilon), 0.0)
        self.bound_epsilon = 0.0
        self.state_scope = canonical_attack_state_scope(state_scope)
        self.attack_indices = tuple(int(i) for i in attack_indices_for_state_scope(self.state_scope))
        self.worst_action_grid_size = max(int(worst_action_grid_size), 3)
        self.state_importance_grid_size = max(int(state_importance_grid_size), 3)
        self.candidate_worst_weight = max(float(candidate_worst_weight), 0.0)
        self.candidate_q_weight = max(float(candidate_q_weight), 0.0)
        self.candidate_reg_weight = max(float(candidate_reg_weight), 0.0)
        self.candidate_reg_weight_clip = max(float(candidate_reg_weight_clip), 0.0)
        self.candidate_interval_margin = max(float(candidate_interval_margin), 0.0)
        self.crown_split_dimensions = max(int(crown_split_dimensions), 0)
        self.preserve_reachable_superset = bool(preserve_reachable_superset)
        self.worst_action_pgd_steps = max(int(worst_action_pgd_steps), 0)
        self.state_importance_pgd_steps = max(int(state_importance_pgd_steps), 0)
        self.worst_critic_updates = max(int(worst_critic_updates), 1)

    def set_bound_epsilon(self, epsilon: float) -> None:
        self.bound_epsilon = float(np.clip(float(epsilon), 0.0, self.max_epsilon))
        if self.adversary is not None:
            self.adversary.epsilon = self.bound_epsilon

    def set_policy_robustness(
        self,
        *,
        worst_policy_weight: float,
        state_reg_weight: float,
        target_lambda: float | None = None,
    ) -> None:
        self.actor_beta = max(float(worst_policy_weight), 0.0)
        self.actor_reg_weight = max(float(state_reg_weight), 0.0)
        if target_lambda is not None:
            self.target_lambda = float(np.clip(float(target_lambda), 0.0, 1.0))

    def tighten_action_bounds(
        self,
        lower: torch.Tensor,
        upper: torch.Tensor,
        candidate_actions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.preserve_reachable_superset or candidate_actions is None or candidate_actions.shape[1] <= 1:
            return lower, upper
        margin = min(float(self.candidate_interval_margin), float(self.bound_epsilon))
        candidate_lower = candidate_actions.detach().amin(dim=1) - margin
        candidate_upper = candidate_actions.detach().amax(dim=1) + margin
        tightened_lower = torch.maximum(lower, candidate_lower)
        tightened_upper = torch.minimum(upper, candidate_upper)
        invalid = tightened_lower > tightened_upper
        if bool(invalid.any()):
            tightened_lower = torch.where(invalid, lower, tightened_lower)
            tightened_upper = torch.where(invalid, upper, tightened_upper)
        return tightened_lower, tightened_upper

    def reachable_action_bounds(
        self,
        observations: torch.Tensor,
        *,
        actor: Actor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bound_fn = (
            actor_split_crown_action_bounds
            if self.crown_split_dimensions > 0
            else actor_crown_action_bounds
        )
        kwargs = {'split_dimensions': self.crown_split_dimensions} if self.crown_split_dimensions > 0 else {}
        return bound_fn(
            self.actor if actor is None else actor,
            observations,
            epsilon=self.bound_epsilon,
            obs_low=self.obs_low_t,
            obs_high=self.obs_high_t,
            attack_indices=self.attack_indices,
            **kwargs,
        )

    def update(
        self,
        batch: dict[str, torch.Tensor],
        *,
        attack_families: tuple[str, ...] = (),
    ) -> dict[str, float]:
        if self.worst_critic is None or self.worst_critic_target is None or self.worst_critic_optimizer is None:
            raise RuntimeError('WocaR1DIntervalAgent requires an independent worst-attack critic.')

        obs_clean = batch['observations']
        next_obs_clean = batch['next_observations']
        rewards = batch['rewards']
        dones = batch['dones']

        target_obs_candidates, target_candidate_stats = self._observation_candidates(
            next_obs_clean,
            actor=self.actor_target,
            critic=self.critic_target,
            attack_families=attack_families,
            enabled=bool(attack_families) and self.bound_epsilon > 0.0,
            prefix='target_empirical',
        )

        with torch.no_grad():
            next_clean_actions = self.actor_target(next_obs_clean)
            next_clean_q = self.critic_target(next_obs_clean, next_clean_actions).reshape(-1)
            next_crown_lower, next_crown_upper = self.reachable_action_bounds(
                next_obs_clean, actor=self.actor_target
            )
            target_candidate_actions = None
            if target_obs_candidates.shape[1] > 1:
                target_batch, target_count, target_obs_dim = target_obs_candidates.shape
                target_candidate_actions = self.actor_target(
                    target_obs_candidates.reshape(target_batch * target_count, target_obs_dim)
                ).reshape(target_batch, target_count, 1)
            next_lower, next_upper = self.tighten_action_bounds(
                next_crown_lower,
                next_crown_upper,
                target_candidate_actions,
            )
            next_worst_q, _, next_worst_actions, _ = projected_q_extrema_1d(
                self.worst_critic_target,
                next_obs_clean,
                next_lower,
                next_upper,
                grid_size=self.worst_action_grid_size,
                steps=self.worst_action_pgd_steps,
            )
            next_worst_clean_q = self.worst_critic_target(next_obs_clean, next_clean_actions).reshape(-1)
            clean_is_worse = next_worst_clean_q < next_worst_q
            next_worst_q = torch.minimum(next_worst_q, next_worst_clean_q)
            next_worst_actions = torch.where(clean_is_worse.reshape(-1, 1), next_clean_actions, next_worst_actions)
            interval_target_q = next_worst_q
            empirical_target_q = next_worst_q
            if target_candidate_actions is not None:
                empirical_obs = target_obs_candidates[:, 1:, :]
                batch_size, empirical_count, obs_dim = empirical_obs.shape
                empirical_actions = target_candidate_actions[:, 1:, :]
                repeated_clean = next_obs_clean.unsqueeze(1).expand(-1, empirical_count, -1)
                empirical_q_values = self.worst_critic_target(
                    repeated_clean.reshape(batch_size * empirical_count, obs_dim),
                    empirical_actions.reshape(batch_size * empirical_count, 1),
                ).reshape(batch_size, empirical_count)
                empirical_target_q, empirical_ids = torch.min(empirical_q_values, dim=1)
                empirical_worst_actions = empirical_actions[
                    torch.arange(batch_size, device=next_obs_clean.device), empirical_ids
                ]
                empirical_is_worse = empirical_target_q < next_worst_q
                next_worst_q = torch.minimum(next_worst_q, empirical_target_q)
                next_worst_actions = torch.where(
                    empirical_is_worse.reshape(-1, 1), empirical_worst_actions, next_worst_actions
                )
            y_clean = rewards + (1.0 - dones) * self.gamma * next_clean_q
            y_attack_clean = rewards + (1.0 - dones) * self.gamma * next_worst_clean_q
            scheduled_worst_q = (
                (1.0 - self.target_lambda) * next_worst_clean_q
                + self.target_lambda * next_worst_q
            )
            y_worst = rewards + (1.0 - dones) * self.gamma * scheduled_worst_q

        q_pred = self.critic(obs_clean, batch['actions']).reshape(-1)
        critic_loss = F.mse_loss(q_pred, y_clean)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        worst_critic_loss = obs_clean.new_tensor(0.0)
        worst_q_pred = self.worst_critic(obs_clean, batch['actions']).reshape(-1)
        for _ in range(self.worst_critic_updates):
            worst_q_pred = self.worst_critic(obs_clean, batch['actions']).reshape(-1)
            worst_critic_loss = F.mse_loss(worst_q_pred, y_worst)
            self.worst_critic_optimizer.zero_grad(set_to_none=True)
            worst_critic_loss.backward()
            self.worst_critic_optimizer.step()

        actor_actions = self.actor(obs_clean)
        actor_clean_q = self.critic(obs_clean, actor_actions).reshape(-1)
        actor_worst_q = self.worst_critic(obs_clean, actor_actions).reshape(-1)
        actor_clean_policy_loss = -actor_clean_q.mean()
        actor_worst_policy_loss = -actor_worst_q.mean()
        actor_without_reg_loss = actor_clean_policy_loss + self.actor_beta * actor_worst_policy_loss

        actor_obs_candidates, actor_candidate_stats = self._observation_candidates(
            obs_clean,
            actor=self.actor,
            critic=self.critic,
            attack_families=attack_families,
            enabled=bool(attack_families) and self.bound_epsilon > 0.0,
            prefix='actor_empirical',
        )
        candidate_worst_loss = actor_actions.new_tensor(0.0)
        candidate_q_loss = actor_actions.new_tensor(0.0)
        candidate_reg_loss = actor_actions.new_tensor(0.0)
        candidate_worst_q = actor_worst_q
        candidate_q_values = actor_worst_q
        candidate_worst_nonclean_frac = 0.0
        candidate_reg_action_mse_mean = 0.0
        candidate_reg_weight_mean = 0.0
        candidate_reg_weight_max = 0.0
        q_candidate_index = 0
        if actor_obs_candidates.shape[1] > 1:
            candidate_batch, candidate_count, candidate_obs_dim = actor_obs_candidates.shape
            flat_candidate_actions = self.actor(
                actor_obs_candidates.reshape(candidate_batch * candidate_count, candidate_obs_dim)
            )
            candidate_actions = flat_candidate_actions.reshape(candidate_batch, candidate_count, 1)
            repeated_actor_clean = obs_clean.unsqueeze(1).expand(-1, candidate_count, -1)
            candidate_values = self.worst_critic(
                repeated_actor_clean.reshape(candidate_batch * candidate_count, candidate_obs_dim),
                flat_candidate_actions,
            ).reshape(candidate_batch, candidate_count)
            candidate_worst_q, candidate_worst_ids = torch.min(candidate_values, dim=1)
            candidate_worst_loss = -candidate_worst_q.mean()
            candidate_worst_nonclean_frac = float((candidate_worst_ids > 0).float().mean().detach().cpu().item())

            q_index_value = actor_candidate_stats.get('actor_empirical_candidate_q_function_index')
            if q_index_value is not None:
                q_candidate_index = int(q_index_value)
                candidate_q_values = candidate_values[:, q_candidate_index]
                candidate_q_loss = -candidate_q_values.mean()

            action_deltas = candidate_actions[:, 1:, :] - actor_actions.unsqueeze(1)
            candidate_reg_per_sample = action_deltas.pow(2).mean(dim=2).amax(dim=1)
            candidate_reg_action_mse_mean = float(candidate_reg_per_sample.detach().mean().cpu().item())
            with torch.no_grad():
                q_gap_weights = (actor_worst_q.detach() - candidate_worst_q.detach()).clamp_min(0.0)
                q_gap_weights = q_gap_weights / q_gap_weights.mean().clamp_min(1e-6)
                if self.candidate_reg_weight_clip > 0.0:
                    q_gap_weights = q_gap_weights.clamp(max=self.candidate_reg_weight_clip)
            candidate_reg_loss = (q_gap_weights * candidate_reg_per_sample).mean()
            candidate_reg_weight_mean = float(q_gap_weights.mean().cpu().item())
            candidate_reg_weight_max = float(q_gap_weights.max().cpu().item())

        with torch.no_grad():
            full_lower = torch.full_like(actor_actions, -1.0)
            full_upper = torch.full_like(actor_actions, 1.0)
            state_min_q, state_max_q, _, _ = projected_q_extrema_1d(
                self.critic,
                obs_clean,
                full_lower,
                full_upper,
                grid_size=self.state_importance_grid_size,
                steps=self.state_importance_pgd_steps,
            )
            raw_state_weights = (state_max_q - state_min_q).clamp_min(0.0)
            state_weights = raw_state_weights / raw_state_weights.mean().clamp_min(1e-6)
            if self.actor_reg_weight_clip > 0.0:
                state_weights = state_weights.clamp(max=self.actor_reg_weight_clip)

        crown_action_lower, crown_action_upper = self.reachable_action_bounds(obs_clean, actor=self.actor)
        action_lower, action_upper = self.tighten_action_bounds(
            crown_action_lower,
            crown_action_upper,
            candidate_actions if actor_obs_candidates.shape[1] > 1 else None,
        )
        _, reg_per_sample = scalar_action_stability_loss(actor_actions, action_lower, action_upper)
        actor_reg_loss = (state_weights * reg_per_sample).mean()

        actor_clean_anchor_loss = actor_actions.new_tensor(0.0)
        actor_clean_anchor_action_mse_mean = 0.0
        if self.actor_clean_anchor_weight > 0.0 and self.clean_anchor_actor is not None:
            with torch.no_grad():
                anchor_actions = self.clean_anchor_actor(obs_clean).detach()
            actor_clean_anchor_loss = F.mse_loss(actor_actions, anchor_actions)
            actor_clean_anchor_action_mse_mean = float(actor_clean_anchor_loss.detach().cpu().item())

        robust_progress = 0.0 if self.max_epsilon <= 0.0 else self.bound_epsilon / self.max_epsilon
        effective_candidate_worst_weight = self.candidate_worst_weight * robust_progress
        effective_candidate_q_weight = self.candidate_q_weight * robust_progress
        effective_candidate_reg_weight = self.candidate_reg_weight * robust_progress
        actor_loss = (
            actor_without_reg_loss
            + self.actor_reg_weight * actor_reg_loss
            + effective_candidate_worst_weight * candidate_worst_loss
            + effective_candidate_q_weight * candidate_q_loss
            + effective_candidate_reg_weight * candidate_reg_loss
            + self.actor_clean_anchor_weight * actor_clean_anchor_loss
        )
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        self._soft_update(self.worst_critic, self.worst_critic_target)

        action_width = (action_upper - action_lower).detach().reshape(-1)
        target_width = (next_upper - next_lower).detach().reshape(-1)
        crown_action_width = (crown_action_upper - crown_action_lower).detach().reshape(-1)
        crown_target_width = (next_crown_upper - next_crown_lower).detach().reshape(-1)
        action_tightening = (crown_action_width - action_width).clamp_min(0.0)
        target_tightening = (crown_target_width - target_width).clamp_min(0.0)
        target_gap = y_attack_clean - y_worst
        actor_critic_gap = actor_clean_q.detach() - actor_worst_q.detach()
        nonclean_fraction = (torch.abs(next_worst_actions - next_clean_actions) > 1e-5).float().mean()
        return {
            'actor_loss': float(actor_loss.detach().cpu().item()),
            'critic_loss': float(critic_loss.detach().cpu().item()),
            'mean_q': float(q_pred.detach().mean().cpu().item()),
            'separate_worst_critic': 1.0,
            'worst_critic_loss': float(worst_critic_loss.detach().cpu().item()),
            'worst_mean_q': float(worst_q_pred.detach().mean().cpu().item()),
            'actor_policy_loss': float(actor_loss.detach().cpu().item()),
            'actor_beta': float(self.actor_beta),
            'actor_q_weight': float(effective_candidate_q_weight),
            'actor_reg_weight': float(self.actor_reg_weight),
            'actor_reg_weight_mode': 'q_gap',
            'actor_reg_weight_clip': float(self.actor_reg_weight_clip),
            'actor_clean_anchor_weight': float(self.actor_clean_anchor_weight),
            'actor_clean_policy_loss': float(actor_clean_policy_loss.detach().cpu().item()),
            'actor_worst_policy_loss': float(actor_worst_policy_loss.detach().cpu().item()),
            'actor_q_policy_loss': float(candidate_q_loss.detach().cpu().item()),
            'actor_without_reg_policy_loss': float(actor_without_reg_loss.detach().cpu().item()),
            'actor_reg_loss': float(actor_reg_loss.detach().cpu().item()),
            'actor_clean_anchor_loss': float(actor_clean_anchor_loss.detach().cpu().item()),
            'actor_clean_anchor_term_active': float(self.actor_clean_anchor_weight > 0.0),
            'actor_clean_anchor_action_mse_mean': actor_clean_anchor_action_mse_mean,
            'actor_clean_q_mean': float(actor_clean_q.detach().mean().cpu().item()),
            'actor_worst_critic_clean_q_mean': float(actor_worst_q.detach().mean().cpu().item()),
            'actor_worst_q_mean': float(actor_worst_q.detach().mean().cpu().item()),
            'actor_q_q_mean': float(candidate_q_values.detach().mean().cpu().item()),
            'actor_clean_to_worst_critic_gap_mean': float(actor_critic_gap.mean().cpu().item()),
            'actor_worst_gap_mean': float(actor_critic_gap.mean().cpu().item()),
            'actor_q_gap_mean': float(raw_state_weights.mean().cpu().item()),
            'actor_worst_nonclean_frac': candidate_worst_nonclean_frac,
            'actor_q_candidate_index': float(q_candidate_index),
            'actor_q_term_active': float(effective_candidate_q_weight > 0.0),
            'actor_reg_term_active': float(self.actor_reg_weight > 0.0),
            'actor_reg_action_mse_mean': float(reg_per_sample.detach().mean().cpu().item()),
            'actor_reg_candidate_count': 2.0,
            'actor_reg_sample_weight_mean': float(state_weights.mean().cpu().item()),
            'actor_reg_sample_weight_max': float(state_weights.max().cpu().item()),
            'actor_reg_q_gap_weight_mean': float(state_weights.mean().cpu().item()),
            'actor_candidate_count': float(actor_obs_candidates.shape[1]),
            'actor_adv_frac': float(self.bound_epsilon > 0.0),
            'actor_adv_linf': float(action_width.mean().cpu().item() * 0.5),
            'actor_adv_l2': float(action_width.mean().cpu().item() * 0.5),
            'target_lambda': float(self.target_lambda),
            'target_candidate_count': float(self.worst_action_grid_size + target_obs_candidates.shape[1] - 1),
            'y_clean_mean': float(y_clean.detach().mean().cpu().item()),
            'y_attack_clean_mean': float(y_attack_clean.detach().mean().cpu().item()),
            'y_worst_mean': float(y_worst.detach().mean().cpu().item()),
            'q_target_mean': float(y_clean.detach().mean().cpu().item()),
            'worst_q_target_mean': float(y_worst.detach().mean().cpu().item()),
            'next_clean_q_mean': float(next_clean_q.detach().mean().cpu().item()),
            'next_worst_q_mean': float(next_worst_q.detach().mean().cpu().item()),
            'target_gap_mean': float(target_gap.detach().mean().cpu().item()),
            'worst_target_le_clean_frac': float((y_worst <= y_attack_clean + 1e-6).float().mean().cpu().item()),
            'target_adv_frac': float(self.bound_epsilon > 0.0),
            'target_adv_linf': float(target_width.mean().cpu().item() * 0.5),
            'target_adv_l2': float(target_width.mean().cpu().item() * 0.5),
            'update_adv_frac': float(target_candidate_stats.get('target_empirical_adv_frac', 0.0)),
            'update_adv_linf': float(target_candidate_stats.get('target_empirical_adv_linf', 0.0)),
            'update_adv_l2': float(target_candidate_stats.get('target_empirical_adv_l2', 0.0)),
            'wocar_bound_epsilon': float(self.bound_epsilon),
            'reachable_action_width_mean': float(action_width.mean().cpu().item()),
            'target_action_width_mean': float(target_width.mean().cpu().item()),
            'crown_action_width_mean': float(crown_action_width.mean().cpu().item()),
            'crown_target_action_width_mean': float(crown_target_width.mean().cpu().item()),
            'candidate_interval_tightening_mean': float(action_tightening.mean().cpu().item()),
            'candidate_target_interval_tightening_mean': float(target_tightening.mean().cpu().item()),
            'state_importance_mean': float(raw_state_weights.mean().cpu().item()),
            'worst_action_grid_size': float(self.worst_action_grid_size),
            'worst_action_pgd_steps': float(self.worst_action_pgd_steps),
            'state_importance_pgd_steps': float(self.state_importance_pgd_steps),
            'worst_critic_updates': float(self.worst_critic_updates),
            'crown_split_dimensions': float(self.crown_split_dimensions),
            'preserve_reachable_superset': float(self.preserve_reachable_superset),
            'candidate_worst_weight': float(effective_candidate_worst_weight),
            'candidate_q_weight': float(effective_candidate_q_weight),
            'candidate_reg_weight': float(effective_candidate_reg_weight),
            'candidate_worst_loss': float(candidate_worst_loss.detach().cpu().item()),
            'candidate_reg_loss': float(candidate_reg_loss.detach().cpu().item()),
            'candidate_reg_action_mse_mean': candidate_reg_action_mse_mean,
            'candidate_reg_sample_weight_mean': candidate_reg_weight_mean,
            'candidate_reg_sample_weight_max': candidate_reg_weight_max,
            'candidate_target_tightening_mean': float(target_tightening.mean().cpu().item()),
            'candidate_attack_count': float(max(actor_obs_candidates.shape[1] - 1, 0)),
        }


def train_online_wocar_v1_agent(
    arrivals,
    signals_path,
    device: torch.device,
    *,
    seed: int = 42,
    episodes: int = 6,
    buffer_size: int = 100000,
    batch_size: int = 128,
    learning_starts: int = 512,
    update_every: int = 2,
    exploration_noise: float = 0.2,
    gamma: float = 0.9,
    tau: float = 0.005,
    actor_lr: float = 1e-4,
    critic_lr: float = 3e-4,
    print_every: int = 1,
    init_actor_path=None,
    resume_bundle_path=None,
    resume_training_path=None,
    resume_history_rows: list[dict] | None = None,
    clean_anchor_actor_path=None,
    reward_profile: RewardProfile = TRAIN_PROFILE,
    train_attacks=None,
    epsilon: float = 0.1,
    alpha: float | None = None,
    steps: int | None = None,
    noise_std: float = 0.0,
    state_scope: str = 'all',
    target_lambda: float = 0.25,
    actor_beta: float = 0.0,
    actor_q_weight: float = 0.0,
    actor_reg_weight: float = 0.0,
    actor_reg_weight_mode: str = 'uniform',
    actor_reg_weight_clip: float = 0.0,
    separate_worst_critic: bool = False,
    actor_clean_anchor_weight: float = 0.0,
    interval_wocar: bool = False,
    candidate_worst_weight: float = 0.0,
    candidate_q_weight: float = 0.0,
    candidate_reg_weight: float = 0.0,
    candidate_reg_weight_clip: float = 1.0,
    candidate_interval_margin: float = 0.0,
    candidate_update_interval: int = 1,
    epsilon_schedule_steps: int = 60000,
    worst_action_grid_size: int = 17,
    state_importance_grid_size: int = 17,
    crown_split_dimensions: int = 2,
    preserve_reachable_superset: bool = True,
    worst_action_pgd_steps: int = 10,
    state_importance_pgd_steps: int = 10,
    worst_critic_updates: int = 2,
    validation_every: int = 2,
    validation_attacks=None,
    validation_baseline_bundle_path=None,
    validation_clean_drop_hard_cap: float = 250.0,
    validation_clean_drop_weight: float = 0.0,
    log_name: str = 'train-online-wocar-v1',
    checkpoint_every: int = 0,
    checkpoint_dir: str | Path | None = None,
    checkpoint_prefix: str = 'wocar',
    checkpoint_metadata: dict | None = None,
    episode_scenarios=None,
) -> tuple[OnlineWocaRV1Agent, OnlineWocaRV1TrainHistory]:
    set_seed(seed)
    scenarios = normalize_episode_scenarios(arrivals, signals_path, episode_scenarios)
    env = ChargingEnv(signals_path=scenarios[0].signals_path, reward_profile=reward_profile)
    initialization_sources = sum(
        value is not None for value in (init_actor_path, resume_bundle_path, resume_training_path)
    )
    if initialization_sources > 1:
        raise ValueError(
            'train_online_wocar_v1_agent supports exactly one actor initialization or training-resume source.'
        )

    critic_state_dict = None
    worst_critic_state_dict = None
    resume_payload: dict | None = None
    resume_training_state: dict = {}
    resume_metadata: dict = {}
    if resume_training_path is not None:
        resume_payload = torch.load(Path(resume_training_path), map_location=device, weights_only=False)
        if not isinstance(resume_payload, dict) or resume_payload.get('actor_state_dict') is None:
            raise ValueError(f'Invalid WocaR training checkpoint: {resume_training_path}')
        actor = Actor().to(device)
        actor.load_state_dict(resume_payload['actor_state_dict'])
        critic_state_dict = resume_payload.get('critic_state_dict')
        worst_critic_state_dict = resume_payload.get('worst_critic_state_dict') or critic_state_dict
        resume_training_state = dict(resume_payload.get('training_state') or {})
        resume_metadata = dict(resume_payload.get('metadata') or {})
        if critic_state_dict is None:
            raise ValueError(f'WocaR training checkpoint has no critic weights: {resume_training_path}')
    elif resume_bundle_path is not None:
        bundle = load_actor_critic_bundle(resume_bundle_path, device)
        if bundle.get('critic_state_dict') is None:
            raise ValueError(f'resume_bundle_path does not contain critic weights: {resume_bundle_path}')
        actor = Actor().to(device)
        actor.load_state_dict(bundle['actor_state_dict'])
        critic_state_dict = bundle['critic_state_dict']
        worst_critic_state_dict = bundle.get('worst_critic_state_dict') or critic_state_dict
    else:
        actor = load_actor_from_path(init_actor_path, device) if init_actor_path is not None else Actor().to(device)

    clean_anchor_actor = None
    if float(actor_clean_anchor_weight) > 0.0:
        if clean_anchor_actor_path is not None:
            clean_anchor_actor = load_actor_from_path(clean_anchor_actor_path, device)
        else:
            clean_anchor_actor = Actor().to(device)
            clean_anchor_actor.load_state_dict({key: value.detach().clone() for key, value in actor.state_dict().items()})
        clean_anchor_actor.eval()

    max_duration = max_duration_across_scenarios(scenarios)
    obs_low, obs_high = observation_bounds_across_scenarios(
        scenarios,
        reward_profile=reward_profile,
        max_duration_of_stay=max_duration,
    )
    adversary = StateObservationAdversary(
        device=device,
        epsilon=epsilon,
        alpha=alpha,
        steps=steps,
        objective='q_function',
        noise_std=noise_std,
        obs_low=obs_low,
        obs_high=obs_high,
        attack_state_scope=state_scope,
    )
    if interval_wocar:
        agent = WocaR1DIntervalAgent(
            actor,
            device=device,
            obs_low=obs_low,
            obs_high=obs_high,
            epsilon=epsilon,
            state_scope=state_scope,
            gamma=gamma,
            tau=tau,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            worst_policy_weight=actor_beta,
            state_reg_weight=actor_reg_weight,
            state_weight_clip=actor_reg_weight_clip,
            worst_action_grid_size=worst_action_grid_size,
            state_importance_grid_size=state_importance_grid_size,
            candidate_adversary=adversary,
            candidate_worst_weight=candidate_worst_weight,
            candidate_q_weight=candidate_q_weight,
            candidate_reg_weight=candidate_reg_weight,
            candidate_reg_weight_clip=candidate_reg_weight_clip,
            candidate_interval_margin=candidate_interval_margin,
            target_lambda=target_lambda,
            clean_anchor_actor=clean_anchor_actor,
            actor_clean_anchor_weight=actor_clean_anchor_weight,
            crown_split_dimensions=crown_split_dimensions,
            preserve_reachable_superset=preserve_reachable_superset,
            worst_action_pgd_steps=worst_action_pgd_steps,
            state_importance_pgd_steps=state_importance_pgd_steps,
            worst_critic_updates=worst_critic_updates,
        )
        separate_worst_critic = True
        actor_q_weight = 0.0
        actor_reg_weight_mode = 'q_gap'
    else:
        agent = OnlineWocaRV1Agent(
            actor,
            device=device,
            gamma=gamma,
            tau=tau,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            adversary=adversary,
            target_lambda=target_lambda,
            actor_beta=actor_beta,
            actor_q_weight=actor_q_weight,
            actor_reg_weight=actor_reg_weight,
            actor_reg_weight_mode=actor_reg_weight_mode,
            actor_reg_weight_clip=actor_reg_weight_clip,
            separate_worst_critic=separate_worst_critic,
            clean_anchor_actor=clean_anchor_actor,
            actor_clean_anchor_weight=actor_clean_anchor_weight,
        )
    if critic_state_dict is not None:
        agent.critic.load_state_dict(critic_state_dict)
        agent.critic_target.load_state_dict(critic_state_dict)
    if separate_worst_critic and worst_critic_state_dict is not None:
        if agent.worst_critic is None or agent.worst_critic_target is None:
            raise RuntimeError('Separate worst critic mode failed to initialize worst critic modules.')
        agent.worst_critic.load_state_dict(worst_critic_state_dict)
        agent.worst_critic_target.load_state_dict(worst_critic_state_dict)

    candidate_guidance_enabled = bool(
        interval_wocar
        and (
            float(candidate_worst_weight) > 0.0
            or float(candidate_q_weight) > 0.0
            or float(candidate_reg_weight) > 0.0
        )
    )
    attack_families = (
        normalize_sa_train_attacks(train_attacks or DEFAULT_ONLINE_WOCAR_V1_TRAIN_ATTACKS)
        if not interval_wocar or candidate_guidance_enabled
        else tuple()
    )
    validation_families = normalize_sa_train_attacks(
        validation_attacks or DEFAULT_ONLINE_WOCAR_V1_TRAIN_ATTACKS
    )
    buffer = SAReplayBuffer(buffer_size, env.obs_dim, env.action_dim, device)
    resume_is_exact = int(resume_training_state.get('format_version', 0)) >= 2
    start_episode = 0
    if resume_training_path is not None:
        start_episode = int(
            resume_training_state.get(
                'completed_episode',
                resume_metadata.get('checkpoint_episode', resume_metadata.get('episode', 0)),
            )
        )
    if not 0 <= start_episode < int(episodes):
        raise ValueError(
            f'WocaR resume completed_episode={start_episode} must be below target episodes={episodes}.'
        )
    rows: list[dict] = list(
        resume_training_state.get('history_rows')
        or resume_history_rows
        or []
    )
    rows = [dict(row) for row in rows if int(float(row.get('episode', 0))) <= start_episode]
    validation_rows: list[dict] = [
        dict(row) for row in (resume_training_state.get('validation_rows') or [])
    ]
    current_noise = float(exploration_noise)
    total_steps = 0
    robust_update_count = 0
    candidate_update_count = 0
    last_candidate_update: dict[str, float] = {}
    candidate_update_interval = max(int(candidate_update_interval), 1)
    update_every = max(int(update_every), 1)
    validation_every = max(int(validation_every), 0)
    clean_drop_hard_cap = max(float(validation_clean_drop_hard_cap), 0.0)
    clean_drop_weight = max(float(validation_clean_drop_weight), 0.0)
    baseline_eval_actor: Actor | None = None
    baseline_eval_critic: Critic | None = None
    baseline_clean_reward: float | None = None
    baseline_clean_exit_vio: int | None = None
    baseline_attack_cache: dict[str, dict] = {}
    best_score = -math.inf
    best_validation: dict | None = None
    best_state: dict[str, dict[str, torch.Tensor]] | None = None
    train_started_at = time.perf_counter()
    prior_train_seconds = float(
        resume_training_state.get(
            'cumulative_train_seconds', resume_metadata.get('cumulative_train_seconds', 0.0)
        )
    )
    checkpoint_every = max(int(checkpoint_every), 0)
    checkpoint_dir_path = None if checkpoint_dir is None else ensure_dir(Path(checkpoint_dir))
    checkpoint_prefix = str(checkpoint_prefix or 'wocar')
    checkpoint_metadata = dict(checkpoint_metadata or {})

    if resume_training_path is not None and resume_is_exact:
        required = (
            'actor_target_state_dict',
            'critic_target_state_dict',
            'actor_optimizer_state_dict',
            'critic_optimizer_state_dict',
            'replay_buffer',
            'rng_state',
        )
        missing = [name for name in required if name not in resume_training_state]
        if agent.separate_worst_critic:
            missing.extend(
                name
                for name in ('worst_critic_target_state_dict', 'worst_critic_optimizer_state_dict')
                if name not in resume_training_state
            )
        if missing:
            raise ValueError(f'WocaR exact-resume checkpoint lacks fields: {missing}')
        agent.actor_target.load_state_dict(resume_training_state['actor_target_state_dict'])
        agent.critic_target.load_state_dict(resume_training_state['critic_target_state_dict'])
        agent.actor_optimizer.load_state_dict(resume_training_state['actor_optimizer_state_dict'])
        agent.critic_optimizer.load_state_dict(resume_training_state['critic_optimizer_state_dict'])
        if agent.separate_worst_critic:
            if agent.worst_critic_target is None or agent.worst_critic_optimizer is None:
                raise RuntimeError('WocaR exact resume requires the worst critic target and optimizer.')
            agent.worst_critic_target.load_state_dict(resume_training_state['worst_critic_target_state_dict'])
            agent.worst_critic_optimizer.load_state_dict(resume_training_state['worst_critic_optimizer_state_dict'])
        if agent.clean_anchor_actor is not None and resume_training_state.get('clean_anchor_state_dict') is not None:
            agent.clean_anchor_actor.load_state_dict(resume_training_state['clean_anchor_state_dict'])
        _restore_replay_buffer(buffer, resume_training_state['replay_buffer'])
        current_noise = float(resume_training_state['current_noise'])
        total_steps = int(resume_training_state['total_steps'])
        robust_update_count = int(resume_training_state.get('robust_update_count', 0))
        candidate_update_count = int(resume_training_state.get('candidate_update_count', 0))
        last_candidate_update = dict(resume_training_state.get('last_candidate_update') or {})
        best_score = float(resume_training_state.get('best_score', -math.inf))
        best_validation = resume_training_state.get('best_validation')
        best_state = resume_training_state.get('best_state')
        print(
            f'[{log_name}][resume] exact checkpoint={resume_training_path} '
            f'completed={start_episode} replay={buffer.size} total_steps={total_steps}',
            flush=True,
        )
    elif resume_training_path is not None:
        total_steps = int(resume_metadata.get('total_steps', 0))
        robust_update_count = max((total_steps - int(learning_starts)) // update_every, 0)
        candidate_update_count = robust_update_count // candidate_update_interval
        for _ in range(start_episode * int(env.horizon)):
            current_noise *= 0.9999 if current_noise > 0.1 else 0.999977
        print(
            f'[{log_name}][resume] legacy_weights_only checkpoint={resume_training_path} '
            f'completed={start_episode} total_steps={total_steps}; optimizer/replay were absent in the old bundle.',
            flush=True,
        )

    if validation_every > 0:
        if validation_baseline_bundle_path is None:
            raise ValueError('Online WocaR validation requires validation_baseline_bundle_path.')
        validation_bundle = load_actor_critic_bundle(validation_baseline_bundle_path, device)
        baseline_eval_actor = Actor().to(device)
        baseline_eval_actor.load_state_dict(validation_bundle['actor_state_dict'])
        baseline_eval_actor.eval()
        if validation_bundle.get('critic_state_dict') is not None:
            baseline_eval_critic = Critic().to(device)
            baseline_eval_critic.load_state_dict(validation_bundle['critic_state_dict'])
            baseline_eval_critic.eval()

    def _snapshot_agent_state() -> dict[str, dict[str, torch.Tensor]]:
        state = {
            'actor': _module_state_to_cpu(agent.actor),
            'critic': _module_state_to_cpu(agent.critic),
            'actor_target': _module_state_to_cpu(agent.actor_target),
            'critic_target': _module_state_to_cpu(agent.critic_target),
        }
        if agent.separate_worst_critic and agent.worst_critic is not None and agent.worst_critic_target is not None:
            state['worst_critic'] = _module_state_to_cpu(agent.worst_critic)
            state['worst_critic_target'] = _module_state_to_cpu(agent.worst_critic_target)
        return state

    def _restore_agent_state(state: dict[str, dict[str, torch.Tensor]]) -> None:
        _load_module_state(agent.actor, state['actor'])
        _load_module_state(agent.critic, state['critic'])
        _load_module_state(agent.actor_target, state['actor_target'])
        _load_module_state(agent.critic_target, state['critic_target'])
        if agent.separate_worst_critic and agent.worst_critic is not None and agent.worst_critic_target is not None:
            if 'worst_critic' in state and 'worst_critic_target' in state:
                _load_module_state(agent.worst_critic, state['worst_critic'])
                _load_module_state(agent.worst_critic_target, state['worst_critic_target'])

    def _build_training_state(completed_episode: int) -> dict:
        state = {
            'format_version': 2,
            'completed_episode': int(completed_episode),
            'total_steps': int(total_steps),
            'current_noise': float(current_noise),
            'robust_update_count': int(robust_update_count),
            'candidate_update_count': int(candidate_update_count),
            'last_candidate_update': dict(last_candidate_update),
            'actor_target_state_dict': agent.actor_target.state_dict(),
            'critic_target_state_dict': agent.critic_target.state_dict(),
            'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
            'replay_buffer': _replay_buffer_state(buffer),
            'rng_state': _rng_state(),
            'history_rows': list(rows),
            'validation_rows': list(validation_rows),
            'best_score': float(best_score),
            'best_validation': best_validation,
            'best_state': best_state,
            'resume_from_episode': int(start_episode),
            'stage_training_seconds': float(time.perf_counter() - train_started_at),
            'cumulative_train_seconds': float(
                prior_train_seconds + time.perf_counter() - train_started_at
            ),
        }
        if agent.worst_critic_target is not None and agent.worst_critic_optimizer is not None:
            state['worst_critic_target_state_dict'] = agent.worst_critic_target.state_dict()
            state['worst_critic_optimizer_state_dict'] = agent.worst_critic_optimizer.state_dict()
        if agent.clean_anchor_actor is not None:
            state['clean_anchor_state_dict'] = agent.clean_anchor_actor.state_dict()
        return state

    def _validation_attacker(policy_actor: Actor, policy_critic: Critic | None, family: str):
        canonical_family = canonical_sa_train_attack(family)
        critic_for_attack = policy_critic if canonical_family == 'q_function' else None
        if canonical_family == 'q_function' and critic_for_attack is None:
            raise ValueError('Online WocaR q_function validation requires a matching critic.')
        return build_state_attacker(
            policy_actor,
            device=device,
            algorithm=canonical_family,
            epsilon=epsilon,
            alpha=alpha,
            iters=steps,
            seed=seed,
            obs_low=obs_low,
            obs_high=obs_high,
            critic=critic_for_attack,
            attack_state_scope=state_scope,
        )

    def _run_validation(episode: int, *, initial: bool = False) -> dict:
        nonlocal baseline_clean_reward, baseline_clean_exit_vio
        if baseline_eval_actor is None:
            raise RuntimeError('Online WocaR V1 validation baseline actor is not initialized.')
        from .merged_pipeline import rollout_episode

        if baseline_clean_reward is None:
            baseline_clean = rollout_episode(
                arrivals,
                baseline_eval_actor,
                signals_path,
                device,
                reward_profile,
                attack_enabled=False,
                attack_scenario='O',
                attacker=None,
                route_mode='none',
                exploration_noise=0.0,
                attack_ratio=1.0,
                attack_scope='obs',
                detector_feature_mode='posterior',
            )
            baseline_clean_reward = float(baseline_clean['ep_reward'])
            baseline_clean_exit_vio = int(baseline_clean.get('exit_vio', 0))

        clean_rollout = rollout_episode(
            arrivals,
            agent.actor,
            signals_path,
            device,
            reward_profile,
            attack_enabled=False,
            attack_scenario='O',
            attacker=None,
            route_mode='none',
            exploration_noise=0.0,
            attack_ratio=1.0,
            attack_scope='obs',
            detector_feature_mode='posterior',
        )
        clean_reward = float(clean_rollout['ep_reward'])
        clean_exit_vio = int(clean_rollout.get('exit_vio', 0))
        clean_drop = float(baseline_clean_reward - clean_reward)
        clean_drop_ratio = 0.0 if abs(float(baseline_clean_reward)) < 1e-9 else clean_drop / abs(float(baseline_clean_reward))

        attack_rewards: dict[str, float] = {}
        baseline_attack_rewards: dict[str, float] = {}
        recovery_ratios: dict[str, float] = {}
        attack_exit_violations: dict[str, int] = {}
        baseline_attack_exit_violations: dict[str, int] = {}
        for family in validation_families:
            canonical_family = canonical_sa_train_attack(family)
            if canonical_family not in baseline_attack_cache:
                baseline_attacker = _validation_attacker(baseline_eval_actor, baseline_eval_critic, canonical_family)
                baseline_attack = rollout_episode(
                    arrivals,
                    baseline_eval_actor,
                    signals_path,
                    device,
                    reward_profile,
                    attack_enabled=True,
                    attack_scenario='O',
                    attacker=baseline_attacker,
                    route_mode='none',
                    exploration_noise=0.0,
                    attack_ratio=1.0,
                    attack_scope='obs',
                    detector_feature_mode='posterior',
                )
                baseline_attack_cache[canonical_family] = baseline_attack
            baseline_attack = baseline_attack_cache[canonical_family]
            policy_attacker = _validation_attacker(agent.actor, agent.critic, canonical_family)
            policy_attack = rollout_episode(
                arrivals,
                agent.actor,
                signals_path,
                device,
                reward_profile,
                attack_enabled=True,
                attack_scenario='O',
                attacker=policy_attacker,
                route_mode='none',
                exploration_noise=0.0,
                attack_ratio=1.0,
                attack_scope='obs',
                detector_feature_mode='posterior',
            )
            baseline_attack_reward = float(baseline_attack['ep_reward'])
            attack_reward = float(policy_attack['ep_reward'])
            attack_drop = float(baseline_clean_reward - baseline_attack_reward)
            recovery = 0.0 if abs(attack_drop) < 1e-9 else float((attack_reward - baseline_attack_reward) / attack_drop)
            baseline_attack_rewards[canonical_family] = baseline_attack_reward
            attack_rewards[canonical_family] = attack_reward
            recovery_ratios[canonical_family] = recovery
            baseline_attack_exit_violations[canonical_family] = int(baseline_attack.get('exit_vio', 0))
            attack_exit_violations[canonical_family] = int(policy_attack.get('exit_vio', 0))

        recovery_values = list(recovery_ratios.values())
        avg_recovery = float(np.mean(recovery_values)) if recovery_values else 0.0
        min_recovery = float(np.min(recovery_values)) if recovery_values else 0.0
        clean_drop_hard_excess = max(clean_drop - clean_drop_hard_cap, 0.0) if clean_drop_hard_cap > 0.0 else 0.0
        clean_drop_ok = int(clean_drop_hard_excess <= 0.0)
        clean_drop_penalty = (
            clean_drop_weight * max(clean_drop, 0.0) / max(clean_drop_hard_cap, 1.0)
            if clean_drop_hard_cap > 0.0
            else 0.0
        )
        score = float(min_recovery - clean_drop_penalty)
        if clean_drop_hard_excess > 0.0:
            score = float(-1000.0 - clean_drop_hard_excess / max(clean_drop_hard_cap, 1.0))

        row = {
            'episode': int(episode),
            'validation_initial': int(bool(initial)),
            'val_score': score,
            'val_min_recovery_ratio': min_recovery,
            'val_avg_recovery_ratio': avg_recovery,
            'val_clean_drop': clean_drop,
            'val_clean_drop_ratio': clean_drop_ratio,
            'val_clean_drop_hard_cap': float(clean_drop_hard_cap),
            'val_clean_drop_hard_excess': float(clean_drop_hard_excess),
            'val_clean_drop_ok': clean_drop_ok,
            'val_clean_exit_vio': int(clean_exit_vio),
            'val_baseline_clean_exit_vio': int(baseline_clean_exit_vio or 0),
            'val_baseline_clean_reward': float(baseline_clean_reward),
            'val_clean_reward': clean_reward,
        }
        for family in validation_families:
            canonical_family = canonical_sa_train_attack(family)
            row[f'val_{canonical_family}_baseline_attack_reward'] = float(baseline_attack_rewards[canonical_family])
            row[f'val_{canonical_family}_attack_reward'] = float(attack_rewards[canonical_family])
            row[f'val_{canonical_family}_recovery_ratio'] = float(recovery_ratios[canonical_family])
            row[f'val_{canonical_family}_baseline_attack_exit_vio'] = int(baseline_attack_exit_violations[canonical_family])
            row[f'val_{canonical_family}_attack_exit_vio'] = int(attack_exit_violations[canonical_family])
        return row

    if resume_training_path is not None and resume_is_exact:
        _restore_rng_state(resume_training_state['rng_state'])

    if validation_every > 0 and start_episode == 0:
        initial_validation = _run_validation(0, initial=True)
        validation_rows.append(initial_validation)
        best_score = float(initial_validation['val_score'])
        best_validation = dict(initial_validation)
        best_validation['best_source'] = 'initial'
        best_state = _snapshot_agent_state()
        print(
            f"[{log_name}][val] ep=000 score={initial_validation['val_score']:.4f} "
            f"min_recovery={initial_validation['val_min_recovery_ratio']:.4f} "
            f"clean_drop={initial_validation['val_clean_drop']:.2f} "
            f"clean_ok={initial_validation['val_clean_drop_ok']} best=*"
        )

    for episode in range(start_episode + 1, episodes + 1):
        episode_scenario = scenario_for_episode(scenarios, episode)
        episode_arrivals = episode_scenario.arrivals
        env = ChargingEnv(signals_path=episode_scenario.signals_path, reward_profile=reward_profile)
        env.reset()
        idx = 0
        active: list[QueueItem] = []
        last_update = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'mean_q': 0.0,
            'separate_worst_critic': float(separate_worst_critic),
            'worst_critic_loss': 0.0,
            'worst_mean_q': 0.0,
            'actor_policy_loss': 0.0,
            'actor_beta': float(actor_beta),
            'actor_q_weight': float(actor_q_weight),
            'actor_reg_weight': float(actor_reg_weight),
            'actor_reg_weight_mode': str(canonical_online_wocar_reg_weight_mode(actor_reg_weight_mode)),
            'actor_reg_weight_clip': float(max(float(actor_reg_weight_clip), 0.0)),
            'actor_clean_anchor_weight': float(max(float(actor_clean_anchor_weight), 0.0)),
            'actor_clean_policy_loss': 0.0,
            'actor_worst_policy_loss': 0.0,
            'actor_q_policy_loss': 0.0,
            'actor_without_reg_policy_loss': 0.0,
            'actor_reg_loss': 0.0,
            'actor_clean_anchor_loss': 0.0,
            'actor_clean_anchor_term_active': float(max(float(actor_clean_anchor_weight), 0.0) > 0.0),
            'actor_clean_anchor_action_mse_mean': 0.0,
            'actor_clean_q_mean': 0.0,
            'actor_worst_critic_clean_q_mean': 0.0,
            'actor_worst_q_mean': 0.0,
            'actor_q_q_mean': 0.0,
            'actor_clean_to_worst_critic_gap_mean': 0.0,
            'actor_worst_gap_mean': 0.0,
            'actor_q_gap_mean': 0.0,
            'actor_worst_nonclean_frac': 0.0,
            'actor_q_candidate_index': 0.0,
            'actor_q_term_active': 0.0,
            'actor_reg_term_active': 0.0,
            'actor_reg_action_mse_mean': 0.0,
            'actor_reg_candidate_count': 0.0,
            'actor_reg_sample_weight_mean': 0.0,
            'actor_reg_sample_weight_max': 0.0,
            'actor_reg_q_gap_weight_mean': 0.0,
            'actor_candidate_count': 1.0,
            'actor_adv_frac': 0.0,
            'actor_adv_linf': 0.0,
            'actor_adv_l2': 0.0,
            'target_lambda': float(target_lambda),
            'target_candidate_count': 1.0,
            'y_clean_mean': 0.0,
            'y_attack_clean_mean': 0.0,
            'y_worst_mean': 0.0,
            'q_target_mean': 0.0,
            'worst_q_target_mean': 0.0,
            'next_clean_q_mean': 0.0,
            'next_worst_q_mean': 0.0,
            'target_gap_mean': 0.0,
            'worst_target_le_clean_frac': 1.0,
            'target_adv_frac': 0.0,
            'target_adv_linf': 0.0,
            'target_adv_l2': 0.0,
            'update_adv_frac': 0.0,
            'update_adv_linf': 0.0,
            'update_adv_l2': 0.0,
            'wocar_bound_epsilon': 0.0,
            'reachable_action_width_mean': 0.0,
            'target_action_width_mean': 0.0,
            'state_importance_mean': 0.0,
            'worst_action_grid_size': float(worst_action_grid_size if interval_wocar else 0),
            'worst_action_pgd_steps': float(worst_action_pgd_steps if interval_wocar else 0),
            'state_importance_pgd_steps': float(state_importance_pgd_steps if interval_wocar else 0),
            'worst_critic_updates': float(worst_critic_updates if interval_wocar else 1),
            'crown_split_dimensions': float(crown_split_dimensions if interval_wocar else 0),
            'preserve_reachable_superset': float(bool(preserve_reachable_superset)),
            'candidate_worst_weight': 0.0,
            'candidate_q_weight': 0.0,
            'candidate_reg_weight': 0.0,
            'candidate_worst_loss': 0.0,
            'candidate_reg_loss': 0.0,
            'candidate_reg_action_mse_mean': 0.0,
            'candidate_reg_sample_weight_mean': 0.0,
            'candidate_reg_sample_weight_max': 0.0,
            'candidate_target_tightening_mean': 0.0,
            'candidate_attack_count': 0.0,
        }
        decision_count = 0

        while env.t < env.horizon:
            step_is_new_arrivals: list[bool] = []
            while idx < len(episode_arrivals) and int(episode_arrivals.loc[idx, 'Arrive_time']) == env.t:
                obs = env.build_initial_obs(int(episode_arrivals.loc[idx, 'Duration_of_stay']))
                action, _ = agent.select_action(obs, exploration_noise=current_noise, deterministic=False)
                decision_count += 1
                env.enqueue(obs, action, int(episode_arrivals.loc[idx, 'Station']))
                step_is_new_arrivals.append(True)
                idx += 1

            for item in active:
                action, _ = agent.select_action(item.obs, exploration_noise=current_noise, deterministic=False)
                decision_count += 1
                env.enqueue(item.obs, action, item.station)
                step_is_new_arrivals.append(False)

            transitions, active, metrics = env.step()
            if len(transitions) != len(step_is_new_arrivals):
                raise RuntimeError('Online WocaR V1 transition count does not match queued decision metadata.')
            for tr, is_new_arrival in zip(transitions, step_is_new_arrivals):
                buffer.add(tr.obs, tr.next_obs, tr.action, tr.reward, tr.done, is_new_arrival=is_new_arrival)
                total_steps += 1
                if buffer.size >= max(batch_size, learning_starts) and (total_steps % update_every == 0):
                    if interval_wocar:
                        schedule_steps = max(int(epsilon_schedule_steps), 1)
                        robust_steps = max(int(total_steps) - int(learning_starts), 0)
                        scheduled_epsilon = float(epsilon) * min(float(robust_steps) / float(schedule_steps), 1.0)
                        if not isinstance(agent, WocaR1DIntervalAgent):
                            raise RuntimeError('Interval WocaR training initialized the wrong agent type.')
                        agent.set_bound_epsilon(scheduled_epsilon)
                        robust_progress = min(float(robust_steps) / float(schedule_steps), 1.0)
                        scheduled_target_lambda = float(target_lambda) + (
                            1.0 - float(target_lambda)
                        ) * robust_progress
                        agent.set_policy_robustness(
                            worst_policy_weight=float(actor_beta) * robust_progress,
                            state_reg_weight=float(actor_reg_weight) * robust_progress,
                            target_lambda=scheduled_target_lambda,
                        )
                    robust_update_count += 1
                    run_candidate_guidance = bool(
                        attack_families
                        and (not interval_wocar or robust_update_count % candidate_update_interval == 0)
                    )
                    update_families = attack_families if run_candidate_guidance else tuple()
                    last_update = agent.update(buffer.sample(batch_size), attack_families=update_families)
                    if interval_wocar and run_candidate_guidance:
                        candidate_update_count += 1
                        last_candidate_update = {
                            key: float(value)
                            for key, value in last_update.items()
                            if key.startswith('candidate_') or key.startswith('update_adv_')
                        }
            current_noise *= 0.9999 if current_noise > 0.1 else 0.999977

        if interval_wocar and last_candidate_update:
            last_update.update(last_candidate_update)

        row = {
            'episode': episode,
            'scenario_id': str(episode_scenario.scenario_id),
            'ep_reward': float(metrics.ep_reward),
            'ep_r1': float(metrics.ep_r1_cost_sum),
            'ep_r2': float(metrics.ep_r2_exit_penalty_sum),
            'ep_r3': float(metrics.ep_r3_running_penalty_sum),
            'ep_r4_dense_safety': float(metrics.ep_r4_dense_safety_penalty_sum),
            'exit_vio': int(metrics.exit_violation_count),
            'run_vio': int(metrics.running_violation_count),
            'replay_size': int(buffer.size),
            'replay_is_clean': 1,
            'rollout_attack_rate': 0.0,
            'rollout_adv_linf': 0.0,
            'rollout_adv_l2': 0.0,
            'actor_loss': float(last_update.get('actor_loss', 0.0)),
            'critic_loss': float(last_update.get('critic_loss', 0.0)),
            'mean_q': float(last_update.get('mean_q', 0.0)),
            'separate_worst_critic': float(last_update.get('separate_worst_critic', separate_worst_critic)),
            'worst_critic_loss': float(last_update.get('worst_critic_loss', 0.0)),
            'worst_mean_q': float(last_update.get('worst_mean_q', 0.0)),
            'actor_policy_loss': float(last_update.get('actor_policy_loss', 0.0)),
            'actor_beta': float(last_update.get('actor_beta', actor_beta)),
            'actor_q_weight': float(last_update.get('actor_q_weight', actor_q_weight)),
            'actor_reg_weight': float(last_update.get('actor_reg_weight', actor_reg_weight)),
            'actor_reg_weight_mode': str(last_update.get('actor_reg_weight_mode', canonical_online_wocar_reg_weight_mode(actor_reg_weight_mode))),
            'actor_reg_weight_clip': float(last_update.get('actor_reg_weight_clip', actor_reg_weight_clip)),
            'actor_clean_anchor_weight': float(last_update.get('actor_clean_anchor_weight', actor_clean_anchor_weight)),
            'actor_clean_policy_loss': float(last_update.get('actor_clean_policy_loss', 0.0)),
            'actor_worst_policy_loss': float(last_update.get('actor_worst_policy_loss', 0.0)),
            'actor_q_policy_loss': float(last_update.get('actor_q_policy_loss', 0.0)),
            'actor_without_reg_policy_loss': float(last_update.get('actor_without_reg_policy_loss', 0.0)),
            'actor_reg_loss': float(last_update.get('actor_reg_loss', 0.0)),
            'actor_clean_anchor_loss': float(last_update.get('actor_clean_anchor_loss', 0.0)),
            'actor_clean_anchor_term_active': float(last_update.get('actor_clean_anchor_term_active', 0.0)),
            'actor_clean_anchor_action_mse_mean': float(last_update.get('actor_clean_anchor_action_mse_mean', 0.0)),
            'actor_clean_q_mean': float(last_update.get('actor_clean_q_mean', 0.0)),
            'actor_worst_critic_clean_q_mean': float(last_update.get('actor_worst_critic_clean_q_mean', 0.0)),
            'actor_worst_q_mean': float(last_update.get('actor_worst_q_mean', 0.0)),
            'actor_q_q_mean': float(last_update.get('actor_q_q_mean', 0.0)),
            'actor_clean_to_worst_critic_gap_mean': float(last_update.get('actor_clean_to_worst_critic_gap_mean', 0.0)),
            'actor_worst_gap_mean': float(last_update.get('actor_worst_gap_mean', 0.0)),
            'actor_q_gap_mean': float(last_update.get('actor_q_gap_mean', 0.0)),
            'actor_worst_nonclean_frac': float(last_update.get('actor_worst_nonclean_frac', 0.0)),
            'actor_q_candidate_index': float(last_update.get('actor_q_candidate_index', 0.0)),
            'actor_q_term_active': float(last_update.get('actor_q_term_active', 0.0)),
            'actor_reg_term_active': float(last_update.get('actor_reg_term_active', 0.0)),
            'actor_reg_action_mse_mean': float(last_update.get('actor_reg_action_mse_mean', 0.0)),
            'actor_reg_candidate_count': float(last_update.get('actor_reg_candidate_count', 0.0)),
            'actor_reg_sample_weight_mean': float(last_update.get('actor_reg_sample_weight_mean', 0.0)),
            'actor_reg_sample_weight_max': float(last_update.get('actor_reg_sample_weight_max', 0.0)),
            'actor_reg_q_gap_weight_mean': float(last_update.get('actor_reg_q_gap_weight_mean', 0.0)),
            'actor_candidate_count': float(last_update.get('actor_candidate_count', 1.0)),
            'actor_adv_frac': float(last_update.get('actor_adv_frac', 0.0)),
            'actor_adv_linf': float(last_update.get('actor_adv_linf', 0.0)),
            'actor_adv_l2': float(last_update.get('actor_adv_l2', 0.0)),
            'target_lambda': float(last_update.get('target_lambda', target_lambda)),
            'target_candidate_count': float(last_update.get('target_candidate_count', 1.0)),
            'y_clean_mean': float(last_update.get('y_clean_mean', 0.0)),
            'y_attack_clean_mean': float(last_update.get('y_attack_clean_mean', 0.0)),
            'y_worst_mean': float(last_update.get('y_worst_mean', 0.0)),
            'q_target_mean': float(last_update.get('q_target_mean', 0.0)),
            'worst_q_target_mean': float(last_update.get('worst_q_target_mean', 0.0)),
            'next_clean_q_mean': float(last_update.get('next_clean_q_mean', 0.0)),
            'next_worst_q_mean': float(last_update.get('next_worst_q_mean', 0.0)),
            'target_gap_mean': float(last_update.get('target_gap_mean', 0.0)),
            'worst_target_le_clean_frac': float(last_update.get('worst_target_le_clean_frac', 1.0)),
            'target_adv_frac': float(last_update.get('target_adv_frac', 0.0)),
            'target_adv_linf': float(last_update.get('target_adv_linf', 0.0)),
            'target_adv_l2': float(last_update.get('target_adv_l2', 0.0)),
            'update_adv_frac': float(last_update.get('update_adv_frac', 0.0)),
            'update_adv_linf': float(last_update.get('update_adv_linf', 0.0)),
            'update_adv_l2': float(last_update.get('update_adv_l2', 0.0)),
            'wocar_bound_mode': 'crown_action_interval_1d' if interval_wocar else 'attack_candidates',
            'wocar_bound_epsilon': float(last_update.get('wocar_bound_epsilon', 0.0)),
            'reachable_action_width_mean': float(last_update.get('reachable_action_width_mean', 0.0)),
            'target_action_width_mean': float(last_update.get('target_action_width_mean', 0.0)),
            'crown_action_width_mean': float(last_update.get('crown_action_width_mean', 0.0)),
            'crown_target_action_width_mean': float(last_update.get('crown_target_action_width_mean', 0.0)),
            'candidate_interval_tightening_mean': float(last_update.get('candidate_interval_tightening_mean', 0.0)),
            'candidate_target_interval_tightening_mean': float(last_update.get('candidate_target_interval_tightening_mean', 0.0)),
            'state_importance_mean': float(last_update.get('state_importance_mean', 0.0)),
            'worst_action_grid_size': float(last_update.get('worst_action_grid_size', 0.0)),
            'worst_action_pgd_steps': float(last_update.get('worst_action_pgd_steps', 0.0)),
            'state_importance_pgd_steps': float(last_update.get('state_importance_pgd_steps', 0.0)),
            'worst_critic_updates': float(last_update.get('worst_critic_updates', 1.0)),
            'crown_split_dimensions': float(last_update.get('crown_split_dimensions', 0.0)),
            'preserve_reachable_superset': float(last_update.get('preserve_reachable_superset', 0.0)),
            'candidate_worst_weight': float(last_update.get('candidate_worst_weight', 0.0)),
            'candidate_q_weight': float(last_update.get('candidate_q_weight', 0.0)),
            'candidate_reg_weight': float(last_update.get('candidate_reg_weight', 0.0)),
            'candidate_worst_loss': float(last_update.get('candidate_worst_loss', 0.0)),
            'candidate_reg_loss': float(last_update.get('candidate_reg_loss', 0.0)),
            'candidate_reg_action_mse_mean': float(last_update.get('candidate_reg_action_mse_mean', 0.0)),
            'candidate_reg_sample_weight_mean': float(last_update.get('candidate_reg_sample_weight_mean', 0.0)),
            'candidate_reg_sample_weight_max': float(last_update.get('candidate_reg_sample_weight_max', 0.0)),
            'candidate_target_tightening_mean': float(last_update.get('candidate_target_tightening_mean', 0.0)),
            'candidate_attack_count': float(last_update.get('candidate_attack_count', 0.0)),
            'candidate_update_interval': int(candidate_update_interval),
            'candidate_update_count': int(candidate_update_count),
            'train_attacks': (
                f"crown_interval+{','.join(attack_families)}" if interval_wocar else ','.join(attack_families)
            ),
            'state_scope': canonical_attack_state_scope(state_scope),
            'update_every': int(update_every),
            'total_steps': int(total_steps),
            'decision_count': int(decision_count),
        }
        rows.append(row)

        if validation_every > 0 and (episode % validation_every == 0 or episode == episodes):
            validation_row = _run_validation(episode)
            is_best = float(validation_row['val_score']) > float(best_score)
            if is_best:
                best_score = float(validation_row['val_score'])
                best_validation = dict(validation_row)
                best_validation['best_source'] = f'episode_{episode}'
                best_state = _snapshot_agent_state()
            validation_row['val_is_best'] = int(bool(is_best))
            validation_rows.append(validation_row)
            row.update(
                {
                    'val_score': float(validation_row['val_score']),
                    'val_min_recovery_ratio': float(validation_row['val_min_recovery_ratio']),
                    'val_avg_recovery_ratio': float(validation_row['val_avg_recovery_ratio']),
                    'val_clean_drop': float(validation_row['val_clean_drop']),
                    'val_clean_drop_ok': int(validation_row['val_clean_drop_ok']),
                    'val_clean_drop_hard_excess': float(validation_row['val_clean_drop_hard_excess']),
                    'val_clean_exit_vio': int(validation_row['val_clean_exit_vio']),
                    'val_is_best': int(bool(is_best)),
                }
            )
            print(
                f"[{log_name}][val] ep={episode:03d} score={validation_row['val_score']:.4f} "
                f"min_recovery={validation_row['val_min_recovery_ratio']:.4f} "
                f"avg_recovery={validation_row['val_avg_recovery_ratio']:.4f} "
                f"clean_drop={validation_row['val_clean_drop']:.2f} "
                f"clean_ok={validation_row['val_clean_drop_ok']} best={'*' if is_best else '-'}"
            )

        if episode == 1 or episode % print_every == 0 or episode == episodes:
            print(
                f"[{log_name}] ep={episode:03d}/{episodes} "
                f"reward={row['ep_reward']:.4f} exit={row['exit_vio']} running={row['run_vio']} "
                f"actor_loss={row['actor_loss']:.6f} critic_loss={row['critic_loss']:.6f} worst_critic_loss={row['worst_critic_loss']:.6f} "
                f"lambda={row['target_lambda']:.3f} sepcrit={row['separate_worst_critic']:.0f} beta={row['actor_beta']:.3f} q_w={row['actor_q_weight']:.3f} reg_w={row['actor_reg_weight']:.3f} anchor_w={row['actor_clean_anchor_weight']:.3f} "
                f"reg_mode={row['actor_reg_weight_mode']} reg_clip={row['actor_reg_weight_clip']:.2f} "
                f"target_gap={row['target_gap_mean']:.4f} actor_gap={row['actor_worst_gap_mean']:.4f} q_gap={row['actor_q_gap_mean']:.4f} reg={row['actor_reg_loss']:.4f} anchor={row['actor_clean_anchor_loss']:.4f} "
                f"cand_w={row['candidate_worst_weight']:.3f}/{row['candidate_q_weight']:.3f}/{row['candidate_reg_weight']:.3f} "
                f"cand_loss={row['candidate_worst_loss']:.4f}/{row['candidate_reg_loss']:.4f} tighten={row['candidate_target_tightening_mean']:.4f} "
                f"cand_updates={row['candidate_update_count']}@{row['candidate_update_interval']} "
                f"worst_ok={row['worst_target_le_clean_frac']:.3f} eps={row['wocar_bound_epsilon']:.4f} "
                f"action_width={row['reachable_action_width_mean']:.4f} replay_clean={row['replay_is_clean']}"
            )

        if checkpoint_every > 0 and checkpoint_dir_path is not None and (episode % checkpoint_every == 0 or episode == episodes):
            ckpt_path = checkpoint_dir_path / f"{checkpoint_prefix}_ep{int(episode):03d}_bundle.pt"
            training_state = _build_training_state(episode)
            save_online_wocar_v1_bundle(
                agent,
                ckpt_path,
                metadata={
                    **checkpoint_metadata,
                    'checkpoint_episode': int(episode),
                    'episode': int(episode),
                    'total_steps': int(total_steps),
                    'resume_from_episode': int(start_episode),
                    'stage_training_seconds': float(training_state['stage_training_seconds']),
                    'cumulative_train_seconds': float(training_state['cumulative_train_seconds']),
                    'checkpoint_kind': 'continuous_training_checkpoint',
                    'resume_mode': 'exact_training_state_v2',
                    'best_validation_so_far': best_validation,
                },
                training_state=training_state,
            )
            row['checkpoint_path'] = str(ckpt_path)
            row['cumulative_train_seconds'] = float(training_state['cumulative_train_seconds'])
            print(f"[{log_name}][ckpt] saved ep={episode:03d}: {ckpt_path}")

    agent.training_state = _build_training_state(episodes)
    if best_state is not None:
        _restore_agent_state(best_state)
        agent.best_validation = best_validation
        print(
            f"[{log_name}][val] restored best checkpoint from "
            f"{best_validation.get('best_source', 'unknown') if best_validation else 'unknown'} "
            f"score={best_score:.4f}"
        )
    else:
        agent.best_validation = None
    return agent, OnlineWocaRV1TrainHistory(rows, validation_rows=validation_rows)


def save_online_wocar_v1_bundle(
    agent: OnlineWocaRV1Agent,
    path: str | Path,
    *,
    metadata: dict | None = None,
    training_state: dict | None = None,
) -> Path:
    out_path = Path(path)
    ensure_dir(out_path.parent)
    payload = {
        'model_type': (
            'wocar_1d_crown_checkpoint'
            if isinstance(agent, WocaR1DIntervalAgent)
            else 'online_wocar_v1_bundle'
        ),
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'metadata': metadata or {},
        'training_state': dict(
            training_state if training_state is not None else getattr(agent, 'training_state', {}) or {}
        ),
    }
    if agent.worst_critic is not None:
        payload['worst_critic_state_dict'] = agent.worst_critic.state_dict()
    torch.save(payload, out_path)
    return out_path


def train_online_wocar_v2_agent(
    *args,
    actor_beta: float = 0.25,
    actor_q_weight: float = 0.1,
    actor_reg_weight: float = 0.0,
    actor_reg_weight_mode: str = 'uniform',
    actor_reg_weight_clip: float = 0.0,
    **kwargs,
) -> tuple[OnlineWocaRV1Agent, OnlineWocaRV1TrainHistory]:
    kwargs.setdefault('log_name', 'train-online-wocar-v2')
    return train_online_wocar_v1_agent(
        *args,
        actor_beta=actor_beta,
        actor_q_weight=actor_q_weight,
        actor_reg_weight=actor_reg_weight,
        actor_reg_weight_mode=actor_reg_weight_mode,
        actor_reg_weight_clip=actor_reg_weight_clip,
        **kwargs,
    )


def train_online_wocar_v3_agent(
    *args,
    actor_beta: float = 0.6,
    actor_q_weight: float = 0.1,
    actor_reg_weight: float = 0.03,
    actor_reg_weight_mode: str = 'uniform',
    actor_reg_weight_clip: float = 0.0,
    **kwargs,
) -> tuple[OnlineWocaRV1Agent, OnlineWocaRV1TrainHistory]:
    kwargs.setdefault('log_name', 'train-online-wocar-v3')
    return train_online_wocar_v1_agent(
        *args,
        actor_beta=actor_beta,
        actor_q_weight=actor_q_weight,
        actor_reg_weight=actor_reg_weight,
        actor_reg_weight_mode=actor_reg_weight_mode,
        actor_reg_weight_clip=actor_reg_weight_clip,
        **kwargs,
    )


def train_online_wocar_v4_agent(
    *args,
    actor_beta: float = 0.6,
    actor_q_weight: float = 0.1,
    actor_reg_weight: float = 0.03,
    actor_reg_weight_mode: str = 'q_gap',
    actor_reg_weight_clip: float = 1.0,
    **kwargs,
) -> tuple[OnlineWocaRV1Agent, OnlineWocaRV1TrainHistory]:
    kwargs.setdefault('log_name', 'train-wocar')
    return train_online_wocar_v1_agent(
        *args,
        actor_beta=actor_beta,
        actor_q_weight=actor_q_weight,
        actor_reg_weight=actor_reg_weight,
        actor_reg_weight_mode=actor_reg_weight_mode,
        actor_reg_weight_clip=actor_reg_weight_clip,
        **kwargs,
    )


def train_online_wocar_v5_agent(
    *args,
    actor_beta: float = 0.6,
    actor_q_weight: float = 0.1,
    actor_reg_weight: float = 0.03,
    actor_reg_weight_mode: str = 'q_gap',
    actor_reg_weight_clip: float = 1.0,
    actor_clean_anchor_weight: float = 0.015,
    **kwargs,
) -> tuple[OnlineWocaRV1Agent, OnlineWocaRV1TrainHistory]:
    kwargs.setdefault('log_name', 'train-online-wocar-v5')
    return train_online_wocar_v1_agent(
        *args,
        actor_beta=actor_beta,
        actor_q_weight=actor_q_weight,
        actor_reg_weight=actor_reg_weight,
        actor_reg_weight_mode=actor_reg_weight_mode,
        actor_reg_weight_clip=actor_reg_weight_clip,
        actor_clean_anchor_weight=actor_clean_anchor_weight,
        separate_worst_critic=True,
        **kwargs,
    )


def train_wocar_1d_interval_agent(
    *args,
    actor_beta: float = 0.5,
    actor_reg_weight: float = 0.1,
    actor_reg_weight_clip: float = 3.0,
    candidate_worst_weight: float = 0.15,
    candidate_q_weight: float = 0.1,
    candidate_reg_weight: float = 0.03,
    candidate_reg_weight_clip: float = 1.0,
    candidate_interval_margin: float = 0.0,
    candidate_update_interval: int = 8,
    worst_action_grid_size: int = 17,
    state_importance_grid_size: int = 17,
    crown_split_dimensions: int = 2,
    preserve_reachable_superset: bool = True,
    worst_action_pgd_steps: int = 10,
    state_importance_pgd_steps: int = 10,
    worst_critic_updates: int = 2,
    epsilon_schedule_steps: int = 60000,
    target_lambda_start: float = 0.25,
    **kwargs,
) -> tuple[OnlineWocaRV1Agent, OnlineWocaRV1TrainHistory]:
    """Project WocaR: CROWN reachable interval and Bellman minimization for scalar actions."""
    kwargs.setdefault('log_name', 'train-wocar-1d-crown')
    return train_online_wocar_v1_agent(
        *args,
        interval_wocar=True,
        target_lambda=target_lambda_start,
        actor_beta=actor_beta,
        actor_q_weight=0.0,
        actor_reg_weight=actor_reg_weight,
        actor_reg_weight_mode='q_gap',
        actor_reg_weight_clip=actor_reg_weight_clip,
        candidate_worst_weight=candidate_worst_weight,
        candidate_q_weight=candidate_q_weight,
        candidate_reg_weight=candidate_reg_weight,
        candidate_reg_weight_clip=candidate_reg_weight_clip,
        candidate_interval_margin=candidate_interval_margin,
        candidate_update_interval=candidate_update_interval,
        separate_worst_critic=True,
        worst_action_grid_size=worst_action_grid_size,
        state_importance_grid_size=state_importance_grid_size,
        crown_split_dimensions=crown_split_dimensions,
        preserve_reachable_superset=preserve_reachable_superset,
        worst_action_pgd_steps=worst_action_pgd_steps,
        state_importance_pgd_steps=state_importance_pgd_steps,
        worst_critic_updates=worst_critic_updates,
        epsilon_schedule_steps=epsilon_schedule_steps,
        **kwargs,
    )


def save_online_wocar_v2_bundle(
    agent: OnlineWocaRV1Agent,
    path: str | Path,
    *,
    metadata: dict | None = None,
) -> Path:
    out_path = Path(path)
    ensure_dir(out_path.parent)
    torch.save(
        {
            'model_type': 'online_wocar_v2_bundle',
            'actor_state_dict': agent.actor.state_dict(),
            'critic_state_dict': agent.critic.state_dict(),
            'metadata': metadata or {},
        },
        out_path,
    )
    return out_path


def save_online_wocar_v3_bundle(
    agent: OnlineWocaRV1Agent,
    path: str | Path,
    *,
    metadata: dict | None = None,
) -> Path:
    out_path = Path(path)
    ensure_dir(out_path.parent)
    torch.save(
        {
            'model_type': 'online_wocar_v3_bundle',
            'actor_state_dict': agent.actor.state_dict(),
            'critic_state_dict': agent.critic.state_dict(),
            'metadata': metadata or {},
        },
        out_path,
    )
    return out_path


def save_online_wocar_v4_bundle(
    agent: OnlineWocaRV1Agent,
    path: str | Path,
    *,
    metadata: dict | None = None,
) -> Path:
    out_path = Path(path)
    ensure_dir(out_path.parent)
    torch.save(
        {
            'model_type': 'online_wocar_v4_bundle',
            'actor_state_dict': agent.actor.state_dict(),
            'critic_state_dict': agent.critic.state_dict(),
            'metadata': metadata or {},
        },
        out_path,
    )
    return out_path


def save_online_wocar_v5_bundle(
    agent: OnlineWocaRV1Agent,
    path: str | Path,
    *,
    metadata: dict | None = None,
) -> Path:
    out_path = Path(path)
    ensure_dir(out_path.parent)
    payload = {
        'model_type': 'online_wocar_v5_separate_worst_critic_bundle',
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'metadata': metadata or {},
    }
    if agent.separate_worst_critic and agent.worst_critic is not None:
        payload['worst_critic_state_dict'] = agent.worst_critic.state_dict()
    torch.save(payload, out_path)
    return out_path


# Public Experiment 5 baseline: faithful one-dimensional WocaR adaptation.
train_wocar_agent = train_wocar_1d_interval_agent


def save_wocar_bundle(agent: OnlineWocaRV1Agent, path: str | Path, *, metadata: dict | None = None) -> Path:
    out_path = Path(path)
    ensure_dir(out_path.parent)
    meta = dict(metadata or {})
    meta.setdefault('algorithm', 'wocar')
    meta.setdefault('policy_tag', 'wocar')
    meta.setdefault('source_variant', 'wocar_1d_interval')
    meta.setdefault('source_provenance', 'wocar_1d_crown_independent_worst_critic')
    meta.setdefault('reachable_action_set', 'crown_interval')
    meta.setdefault('action_dim', 1)
    payload = {
        'model_type': 'wocar_1d_crown_bundle',
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'metadata': meta,
        'training_state': dict(getattr(agent, 'training_state', {}) or {}),
    }
    if agent.worst_critic is not None:
        payload['worst_critic_state_dict'] = agent.worst_critic.state_dict()
    torch.save(payload, out_path)
    return out_path
