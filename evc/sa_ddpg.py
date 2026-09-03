from __future__ import annotations

from dataclasses import dataclass
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .merged_core import (
    ATTACK_DEFAULTS,
    TRAIN_PROFILE,
    Actor,
    ChargingEnv,
    Critic,
    QueueItem,
    ReplayBuffer,
    RewardProfile,
    ensure_dir,
    load_actor_critic_bundle,
    load_actor_from_path,
    set_seed,
    to_numpy_1d,
)
from .merged_attacks import attack_indices_for_state_scope, build_state_attacker, canonical_attack_state_scope
from .multiday_schedule import max_duration_across_scenarios, normalize_episode_scenarios, scenario_for_episode
from .robust_bounds import (
    actor_crown_action_bounds,
    actor_split_crown_action_bounds,
)


@dataclass
class SADDPGTrainHistory:
    rows: list[dict]
    validation_rows: list[dict] | None = None


DEFAULT_SA_TRAIN_ATTACKS = ('opposite_pgd', 'q_function')


def canonical_sa_train_attack(name: str | None) -> str:
    token = str(name or 'q_function').strip().lower().replace('-', '_')
    if token in {'electhacker', 'electhacker_o', 'o', 'targeted_o'}:
        return 'electhacker_o'
    if token in {'pgd', 'opposite_pgd'}:
        return 'opposite_pgd'
    if token in {'fgsm', 'opposite_fgsm'}:
        return 'opposite_fgsm'
    if token in {'q', 'critic', 'min_q', 'q_function', 'q_function_attack'}:
        return 'q_function'
    if token in {'action', 'max_action_diff'}:
        return 'action'
    if token in {'sgld', 'sgld_maxdiff', 'sgld_mad', 'mad_sgld'}:
        return 'sgld_maxdiff'
    # Long-horizon inspired candidate families used by the WocaR-Long-NoGuard branch.
    # These are differentiable one-step surrogates for sequence-level attacks;
    # the full stateful versions are still used by the evaluation scripts.
    if token in {'local_small_drift_q', 'small_drift_q', 'drift_q'}:
        return 'local_small_drift_q'
    if token in {'local_deadline_drift_pgd', 'deadline_drift_pgd', 'deadline_pgd'}:
        return 'local_deadline_drift_pgd'
    if token in {'temporal_shift_attack', 'temporal_shift', 'local_temporal_shift_attack'}:
        return 'temporal_shift_attack'
    if token in {'full_pipeline_adaptive_deadline', 'fp_adaptive_deadline', 'adaptive_deadline'}:
        return 'full_pipeline_adaptive_deadline'
    raise ValueError(f'Unsupported SA-DDPG training attack: {name}')


def normalize_sa_train_attacks(names) -> tuple[str, ...]:
    if names is None:
        return tuple(DEFAULT_SA_TRAIN_ATTACKS)
    if isinstance(names, str):
        tokens = [names]
    else:
        tokens = list(names)
    out = tuple(canonical_sa_train_attack(token) for token in tokens if str(token).strip())
    if not out:
        raise ValueError('SA-DDPG training attack list cannot be empty.')
    return out


def scheduled_attack_probability(
    *,
    total_steps: int,
    warmup_steps: int,
    start_prob: float,
    final_prob: float,
    curriculum_steps: int,
) -> float:
    warmup = max(int(warmup_steps), 0)
    if int(total_steps) < warmup:
        return 0.0
    start = float(np.clip(float(start_prob), 0.0, 1.0))
    final = float(np.clip(float(final_prob), 0.0, 1.0))
    ramp = max(int(curriculum_steps), 0)
    if ramp <= 0:
        return final
    progress = min(max((float(total_steps) - float(warmup)) / float(ramp), 0.0), 1.0)
    return float(start + (final - start) * progress)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    if weights is None:
        return values.mean()
    w = weights
    while w.ndim < values.ndim:
        w = w.unsqueeze(-1)
    return (values * w).sum() / w.expand_as(values).sum().clamp_min(1e-6)


def _weighted_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    loss = (pred - target).pow(2)
    return _weighted_mean(loss, weights)


def _clean_preservation_weights(
    obs: torch.Tensor,
    *,
    target_soc: float,
    scale: float,
    max_weight: float,
) -> torch.Tensor | None:
    scale = max(float(scale), 0.0)
    if scale <= 0.0:
        return None
    soc = obs[:, 0].float()
    remaining_slots = torch.clamp(obs[:, 1].float(), min=0.0) * 12.0
    soc_gap = torch.clamp(float(target_soc) - soc, min=0.0)
    urgency = 1.0 / (remaining_slots + 1.0)
    risk = soc_gap * (1.0 + 3.0 * urgency)
    weights = 1.0 + scale * risk
    return torch.clamp(weights, min=1.0, max=max(float(max_weight), 1.0))


def _module_state_to_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _load_module_state(module: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    module.load_state_dict({key: value.detach().clone() for key, value in state.items()})


class SAReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: torch.device) -> None:
        super().__init__(capacity, obs_dim, action_dim, device)
        self.is_new_arrivals = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs, next_obs, action, reward: float, done: bool, *, is_new_arrival: bool = False) -> None:
        self.obs[self.pos] = to_numpy_1d(obs)
        self.next_obs[self.pos] = to_numpy_1d(next_obs)
        self.actions[self.pos] = to_numpy_1d(action)
        self.rewards[self.pos] = float(reward)
        self.dones[self.pos] = float(done)
        self.is_new_arrivals[self.pos] = float(bool(is_new_arrival))
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            'observations': torch.as_tensor(self.obs[idx], dtype=torch.float32, device=self.device),
            'next_observations': torch.as_tensor(self.next_obs[idx], dtype=torch.float32, device=self.device),
            'actions': torch.as_tensor(self.actions[idx], dtype=torch.float32, device=self.device),
            'rewards': torch.as_tensor(self.rewards[idx], dtype=torch.float32, device=self.device).reshape(-1),
            'dones': torch.as_tensor(self.dones[idx], dtype=torch.float32, device=self.device).reshape(-1),
            'is_new_arrivals': torch.as_tensor(self.is_new_arrivals[idx], dtype=torch.float32, device=self.device).reshape(-1),
        }


class StateObservationAdversary:
    def __init__(
        self,
        *,
        device: torch.device,
        epsilon: float = 0.1,
        alpha: float | None = None,
        steps: int | None = None,
        objective: str = 'q_function',
        noise_std: float = 0.0,
        soc_new_threshold: float = 0.5,
        soc_rollout_threshold: float = 0.3,
        obs_low: np.ndarray | torch.Tensor | None = None,
        obs_high: np.ndarray | torch.Tensor | None = None,
        attack_state_scope: str = 'all',
        attack_indices: tuple[int, ...] | list[int] | None = None,
    ) -> None:
        self.device = device
        self.epsilon = max(float(epsilon), 0.0)
        self.alpha = None if alpha is None else max(float(alpha), 0.0)
        self.steps = None if steps is None else max(int(steps), 0)
        self.objective = canonical_sa_train_attack(objective)
        self.noise_std = max(float(noise_std), 0.0)
        self.soc_new_threshold = float(soc_new_threshold)
        self.soc_rollout_threshold = float(soc_rollout_threshold)
        if self.objective not in {
            'q_function', 'action', 'opposite_pgd', 'opposite_fgsm', 'electhacker_o',
            'local_small_drift_q', 'local_deadline_drift_pgd', 'temporal_shift_attack',
            'full_pipeline_adaptive_deadline',
        }:
            raise ValueError(f'Unsupported SA-DDPG adversary objective: {objective}')
        self.attack_state_scope = canonical_attack_state_scope(attack_state_scope)
        self.attack_indices = tuple(
            int(v)
            for v in (attack_indices if attack_indices is not None else attack_indices_for_state_scope(self.attack_state_scope))
        )
        self.obs_low = None if obs_low is None else torch.as_tensor(obs_low, dtype=torch.float32, device=self.device).reshape(1, -1)
        self.obs_high = None if obs_high is None else torch.as_tensor(obs_high, dtype=torch.float32, device=self.device).reshape(1, -1)
        if (self.obs_low is None) != (self.obs_high is None):
            raise ValueError('StateObservationAdversary requires both obs_low and obs_high, or neither.')
        if self.obs_low is not None and self.obs_low.shape != self.obs_high.shape:
            raise ValueError('StateObservationAdversary obs bounds must share the same shape.')

    def _attack_mask(self, obs: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(obs)
        mask[..., list(self.attack_indices)] = 1.0
        return mask

    def _project(self, clean: torch.Tensor, proposal: torch.Tensor) -> torch.Tensor:
        if self.obs_low is not None and self.obs_high is not None:
            if self.obs_low.shape[1] != clean.shape[1]:
                raise ValueError(
                    f'SA-DDPG obs bounds dim mismatch: bounds={int(self.obs_low.shape[1])} obs={int(clean.shape[1])}'
                )
            projected = torch.maximum(torch.minimum(proposal, self.obs_high), self.obs_low)
        else:
            projected = torch.clamp(proposal, min=0.0, max=1.0)
        mask = self._attack_mask(clean)
        return clean * (1.0 - mask) + projected * mask

    def _long_surrogate_bias(self, clean: torch.Tensor, family: str) -> torch.Tensor:
        """One-step proxy for stateful long-horizon undercharge drift.

        The real long-horizon attacks keep per-vehicle memory and are used at
        evaluation time.  WocaR training samples replay batches without vehicle
        identity, so this branch exposes the actor/critic update to the same
        *direction* of long-horizon damage: SOC/time/cost are nudged upward so
        the policy sees a less urgent charging state.
        """
        bias = torch.zeros_like(clean)
        if clean.shape[1] <= 10:
            return bias
        remaining = clean[:, 1].float().clamp(0.0, 1.0)
        soc = clean[:, 0].float().clamp(0.0, 1.0)
        phase = (1.0 - remaining).clamp(0.0, 1.0)
        low_soc = ((0.62 - soc) / 0.62).clamp(0.0, 1.0)
        urgency = (0.35 + 0.65 * torch.maximum(phase, 0.55 * low_soc)).reshape(-1, 1)
        if family == 'local_small_drift_q':
            template = torch.tensor([0.18, 0.28, 0.16], dtype=clean.dtype, device=clean.device).reshape(1, 3)
            scale = 0.55
        elif family == 'temporal_shift_attack':
            template = torch.tensor([0.16, 0.26, 0.14], dtype=clean.dtype, device=clean.device).reshape(1, 3)
            scale = 0.50
            if clean.shape[1] > 9:
                price_curve = torch.tensor([0.90, 0.45, 0.00, -0.55, -1.00], dtype=clean.dtype, device=clean.device).reshape(1, 5)
                bias[:, 5:10] = self.epsilon * 0.22 * price_curve
        else:
            template = torch.tensor([0.28, 0.52, 0.34], dtype=clean.dtype, device=clean.device).reshape(1, 3)
            scale = 0.80
        local = self.epsilon * scale * urgency * template
        idx = torch.tensor([0, 1, 10], dtype=torch.long, device=clean.device)
        bias[:, idx] = local
        return bias * self._attack_mask(clean)

    def _family_hparams(self, family: str) -> tuple[int, float]:
        if family in {'opposite_pgd', 'opposite_fgsm', 'q_function', 'sgld_maxdiff'}:
            config_family = 'opposite_pgd' if family == 'sgld_maxdiff' else family
            cfg = ATTACK_DEFAULTS[config_family]
            default_steps = int(cfg.iters)
            default_alpha = float(cfg.alpha)
        elif family == 'electhacker_o':
            cfg = ATTACK_DEFAULTS['electhacker']
            default_steps = int(cfg.iters)
            default_alpha = float(cfg.alpha)
        elif family == 'local_small_drift_q':
            cfg = ATTACK_DEFAULTS['q_function']
            default_steps = int(cfg.iters)
            default_alpha = min(float(cfg.alpha), max(self.epsilon / 4.0, 1e-6))
        elif family == 'temporal_shift_attack':
            cfg = ATTACK_DEFAULTS['q_function']
            default_steps = int(cfg.iters)
            default_alpha = min(float(cfg.alpha), max(self.epsilon / 4.0, 1e-6))
        elif family in {'local_deadline_drift_pgd', 'full_pipeline_adaptive_deadline'}:
            cfg = ATTACK_DEFAULTS['opposite_pgd']
            default_steps = int(cfg.iters)
            default_alpha = min(float(cfg.alpha), max(self.epsilon / 4.0, 1e-6))
        else:
            default_steps = 5
            default_alpha = self.epsilon / float(max(default_steps, 1))
        step_count = default_steps if self.steps is None else int(self.steps)
        step_size = default_alpha if self.alpha is None else float(self.alpha)
        return max(step_count, 0), max(step_size, 0.0)

    def _objective_value(
        self,
        adv_obs: torch.Tensor,
        clean_obs: torch.Tensor,
        actor: Actor,
        critic: Critic,
    ) -> torch.Tensor:
        if self.objective == 'q_function':
            adv_action = actor(adv_obs)
            return critic(clean_obs, adv_action).mean()
        with torch.no_grad():
            clean_action = actor(clean_obs).detach()
        adv_action = actor(adv_obs)
        return -F.mse_loss(adv_action, clean_action)

    def perturb_tensor(
        self,
        clean_obs: torch.Tensor,
        *,
        actor: Actor,
        critic: Critic,
        attack_family: str | None = None,
        is_new_arrivals: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        clean = clean_obs.detach().clone().to(self.device)
        if clean.ndim == 1:
            clean = clean.unsqueeze(0)
        family = canonical_sa_train_attack(attack_family or self.objective)
        step_count, step_size = self._family_hparams(family)
        if self.epsilon <= 0.0 or step_count <= 0:
            return clean.detach()

        mask = self._attack_mask(clean)
        adv = clean + torch.empty_like(clean).uniform_(-self.epsilon, self.epsilon) * mask
        adv = self._project(clean, adv).detach()
        target_actions = None
        if family == 'electhacker_o':
            if is_new_arrivals is None:
                raise ValueError('electhacker_o SA training requires is_new_arrivals flags.')
            flags = torch.as_tensor(is_new_arrivals, dtype=torch.float32, device=self.device).reshape(-1)
            if flags.numel() != clean.shape[0]:
                raise ValueError('is_new_arrivals length must match clean_obs batch size.')
            new_thr = torch.full_like(clean[:, 0], self.soc_new_threshold)
            rollout_thr = torch.full_like(clean[:, 0], self.soc_rollout_threshold)
            thresholds = torch.where(flags > 0.5, new_thr, rollout_thr)
            target_actions = torch.where(
                clean[:, 0] < thresholds,
                -torch.ones_like(clean[:, 0]),
                torch.ones_like(clean[:, 0]),
            ).unsqueeze(1)
        elif family in {'local_deadline_drift_pgd', 'full_pipeline_adaptive_deadline'}:
            remaining = clean[:, 1].float().clamp(0.0, 1.0)
            soc = clean[:, 0].float().clamp(0.0, 1.0)
            phase = (1.0 - remaining).clamp(0.0, 1.0)
            low_soc = ((0.62 - soc) / 0.62).clamp(0.0, 1.0)
            target = -(0.20 + 0.45 * phase + 0.25 * low_soc).clamp(0.20, 1.0)
            target_actions = target.reshape(-1, 1)
        elif family in {'opposite_pgd', 'opposite_fgsm', 'action'}:
            with torch.no_grad():
                target_actions = actor(clean).detach()

        for _ in range(step_count):
            adv.requires_grad_(True)
            if family in {'electhacker_o', 'local_deadline_drift_pgd', 'full_pipeline_adaptive_deadline'}:
                loss = F.mse_loss(actor(adv), target_actions)
                grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]
                adv = adv - step_size * grad.sign() * mask
            elif family in {'opposite_pgd', 'opposite_fgsm', 'action'}:
                loss = F.mse_loss(actor(adv), target_actions)
                grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]
                adv = adv + step_size * grad.sign() * mask
            elif family == 'sgld_maxdiff':
                with torch.no_grad():
                    clean_action = actor(clean).detach()
                loss = F.mse_loss(actor(adv), clean_action)
                grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]
                noise_scale = math.sqrt(max(2.0 * step_size, 1e-12)) * 1e-5 / float(_ + 1)
                langevin_grad = grad + noise_scale * torch.randn_like(grad)
                adv = adv + step_size * langevin_grad.sign() * mask
            elif family in {'q_function', 'local_small_drift_q', 'temporal_shift_attack'}:
                adv_action = actor(adv)
                loss = critic(clean, adv_action).mean()
                grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]
                adv = adv - step_size * grad.sign() * mask
            else:
                loss = self._objective_value(adv, clean, actor, critic)
                grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]
                adv = adv - step_size * grad.sign() * mask
            if family in {'local_small_drift_q', 'local_deadline_drift_pgd', 'temporal_shift_attack', 'full_pipeline_adaptive_deadline'}:
                adv = adv + self._long_surrogate_bias(clean, family)
            if self.noise_std > 0.0:
                adv = adv + torch.randn_like(adv) * self.noise_std * mask
            delta = torch.clamp(adv - clean, min=-self.epsilon, max=self.epsilon) * mask
            adv = self._project(clean, clean + delta).detach()
        return adv

    def perturb_numpy(
        self,
        clean_obs: np.ndarray,
        *,
        actor: Actor,
        critic: Critic,
        attack_family: str | None = None,
        is_new_arrivals: torch.Tensor | np.ndarray | None = None,
    ) -> np.ndarray:
        obs_t = torch.as_tensor(np.asarray(clean_obs, dtype=np.float32), dtype=torch.float32, device=self.device)
        adv_t = self.perturb_tensor(
            obs_t,
            actor=actor,
            critic=critic,
            attack_family=attack_family,
            is_new_arrivals=is_new_arrivals,
        )
        return adv_t.detach().cpu().numpy().astype(np.float32)


class SADDPGAgent:
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
        rollout_attack_prob: float = 1.0,
        update_attack_prob: float = 1.0,
        actor_reg_weight: float = 0.0,
        anchor_actor: Actor | None = None,
        anchor_reg_weight: float = 0.0,
        anchor_clean_weight: float = 1.0,
        clean_policy_weight: float = 0.0,
        risk_weight_scale: float = 0.0,
        risk_weight_max: float = 3.0,
        risk_target_soc: float = 0.9,
    ) -> None:
        self.device = device
        self.actor = actor.to(device)
        self.critic = Critic().to(device)
        self.actor_target = Actor().to(device)
        self.critic_target = Critic().to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(actor_lr))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(critic_lr))
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.adversary = adversary
        self.rollout_attack_prob = float(np.clip(float(rollout_attack_prob), 0.0, 1.0))
        self.update_attack_prob = float(np.clip(float(update_attack_prob), 0.0, 1.0))
        self.actor_reg_weight = max(float(actor_reg_weight), 0.0)
        self.anchor_actor = None if anchor_actor is None else anchor_actor.to(device).eval()
        if self.anchor_actor is not None:
            for param in self.anchor_actor.parameters():
                param.requires_grad_(False)
        self.anchor_reg_weight = max(float(anchor_reg_weight), 0.0)
        self.anchor_clean_weight = max(float(anchor_clean_weight), 0.0)
        self.clean_policy_weight = max(float(clean_policy_weight), 0.0)
        self.risk_weight_scale = max(float(risk_weight_scale), 0.0)
        self.risk_weight_max = max(float(risk_weight_max), 1.0)
        self.risk_target_soc = float(risk_target_soc)

    def _soft_update(self, src: torch.nn.Module, dst: torch.nn.Module) -> None:
        for s, d in zip(src.parameters(), dst.parameters()):
            d.data.copy_(self.tau * s.data + (1.0 - self.tau) * d.data)

    def _attack_batch(
        self,
        obs_t: torch.Tensor,
        *,
        actor: Actor,
        critic: Critic,
        attack_prob: float | None,
        adversarial: bool,
        attack_family: str | None = None,
        is_new_arrivals: torch.Tensor | np.ndarray | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        clean = obs_t.detach()
        stats = {
            'attacked_frac': 0.0,
            'adv_linf_mean': 0.0,
            'adv_l2_mean': 0.0,
        }
        effective_prob = float(self.rollout_attack_prob if attack_prob is None else attack_prob)
        if (not adversarial) or self.adversary is None or effective_prob <= 0.0:
            return clean, stats

        if effective_prob >= 1.0:
            mask = torch.ones(clean.shape[0], dtype=torch.bool, device=clean.device)
        else:
            mask = torch.rand(clean.shape[0], device=clean.device) < effective_prob
        if not bool(mask.any()):
            return clean, stats

        attacked = clean.clone()
        subset_is_new_arrivals = None
        if is_new_arrivals is not None:
            flags = torch.as_tensor(is_new_arrivals, dtype=torch.float32, device=clean.device).reshape(-1)
            subset_is_new_arrivals = flags[mask]
        selected_clean = clean[mask]
        selected_flags = subset_is_new_arrivals
        if isinstance(attack_family, (tuple, list)) and len(attack_family) > 1:
            families = tuple(canonical_sa_train_attack(family) for family in attack_family)
            family_ids = torch.randint(len(families), (selected_clean.shape[0],), device=clean.device)
            attacked_subset = selected_clean.clone()
            family_fracs: dict[str, float] = {}
            for family_index, family in enumerate(families):
                family_mask = family_ids == int(family_index)
                family_count = int(family_mask.sum().detach().cpu().item())
                family_fracs[f'attack_family_{family}_frac'] = 0.0 if selected_clean.shape[0] == 0 else float(family_count) / float(selected_clean.shape[0])
                if family_count <= 0:
                    continue
                family_flags = None if selected_flags is None else selected_flags[family_mask]
                attacked_subset[family_mask] = self.adversary.perturb_tensor(
                    selected_clean[family_mask],
                    actor=actor,
                    critic=critic,
                    attack_family=family,
                    is_new_arrivals=family_flags,
                ).detach()
        else:
            family_fracs = {}
            single_family = attack_family
            if isinstance(single_family, (tuple, list)):
                single_family = single_family[0] if len(single_family) > 0 else None
            attacked_subset = self.adversary.perturb_tensor(
                selected_clean,
                actor=actor,
                critic=critic,
                attack_family=single_family,
                is_new_arrivals=selected_flags,
            ).detach()
        attacked[mask] = attacked_subset
        delta = attacked_subset - clean[mask]
        linf = torch.max(torch.abs(delta), dim=1).values
        l2 = torch.linalg.vector_norm(delta, ord=2, dim=1)
        stats = {
            'attacked_frac': float(mask.float().mean().detach().cpu().item()),
            'adv_linf_mean': float(linf.mean().detach().cpu().item()) if linf.numel() > 0 else 0.0,
            'adv_l2_mean': float(l2.mean().detach().cpu().item()) if l2.numel() > 0 else 0.0,
        }
        stats.update(family_fracs)
        return attacked.detach(), stats

    def select_action(
        self,
        obs,
        *,
        exploration_noise: float = 0.0,
        deterministic: bool = False,
        adversarial: bool = False,
        attack_prob: float | None = None,
        attack_family: str | None = None,
        is_new_arrival: bool = False,
    ) -> tuple[np.ndarray, dict[str, float]]:
        clean_obs = to_numpy_1d(obs)
        obs_t = torch.as_tensor(clean_obs, dtype=torch.float32, device=self.device)
        attacked_obs_t, attack_stats = self._attack_batch(
            obs_t.unsqueeze(0),
            actor=self.actor,
            critic=self.critic,
            attack_prob=attack_prob,
            adversarial=adversarial,
            attack_family=attack_family,
            is_new_arrivals=torch.as_tensor([float(bool(is_new_arrival))], dtype=torch.float32, device=self.device),
        )
        action = self.actor(attacked_obs_t).reshape(-1)
        if not deterministic and exploration_noise > 0.0:
            noise = torch.normal(mean=0.0, std=float(exploration_noise), size=action.shape, device=self.device)
            action = action + noise
        action = action.clamp(-1.0, 1.0).detach().cpu().numpy().astype(np.float32)
        return action, attack_stats

    def update(
        self,
        batch: dict[str, torch.Tensor],
        *,
        adversarial: bool = True,
        attack_prob: float | None = None,
        attack_family: str | None = None,
    ) -> dict[str, float]:
        current_is_new_arrivals = batch.get('is_new_arrivals')
        obs_adv, obs_stats = self._attack_batch(
            batch['observations'],
            actor=self.actor,
            critic=self.critic,
            attack_prob=attack_prob,
            adversarial=adversarial,
            attack_family=attack_family,
            is_new_arrivals=current_is_new_arrivals,
        )
        next_obs_adv, next_stats = self._attack_batch(
            batch['next_observations'],
            actor=self.actor_target,
            critic=self.critic_target,
            attack_prob=attack_prob,
            adversarial=adversarial,
            attack_family=attack_family,
            is_new_arrivals=None if current_is_new_arrivals is None else torch.zeros_like(current_is_new_arrivals),
        )
        critic_obs = obs_adv

        with torch.no_grad():
            next_actions = self.actor_target(next_obs_adv)
            q_target = batch['rewards'] + (1.0 - batch['dones']) * self.gamma * self.critic_target(
                next_obs_adv, next_actions
            ).reshape(-1)

        q_pred = self.critic(critic_obs, batch['actions']).reshape(-1)
        critic_loss = F.mse_loss(q_pred, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        preservation_weights = _clean_preservation_weights(
            batch['observations'],
            target_soc=self.risk_target_soc,
            scale=self.risk_weight_scale,
            max_weight=self.risk_weight_max,
        )
        actor_actions = self.actor(obs_adv)
        actor_clean_actions = self.actor(batch['observations'])
        actor_adv_policy_loss = -self.critic(obs_adv, actor_actions).mean()
        actor_clean_policy_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        if self.clean_policy_weight > 0.0:
            actor_clean_policy_loss = _weighted_mean(
                -self.critic(batch['observations'], actor_clean_actions),
                preservation_weights,
            )
        actor_policy_loss = actor_adv_policy_loss + self.clean_policy_weight * actor_clean_policy_loss
        actor_reg_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        if self.actor_reg_weight > 0.0 and adversarial and self.adversary is not None:
            actor_reg_loss = _weighted_mse(actor_actions, actor_clean_actions.detach(), preservation_weights)
        actor_anchor_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        actor_anchor_adv_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        actor_anchor_clean_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        if self.anchor_actor is not None and self.anchor_reg_weight > 0.0:
            with torch.no_grad():
                anchor_clean_actions = self.anchor_actor(batch['observations']).detach()
            actor_anchor_adv_loss = _weighted_mse(actor_actions, anchor_clean_actions, preservation_weights)
            if self.anchor_clean_weight > 0.0:
                actor_anchor_clean_loss = _weighted_mse(actor_clean_actions, anchor_clean_actions, preservation_weights)
            actor_anchor_loss = actor_anchor_adv_loss + self.anchor_clean_weight * actor_anchor_clean_loss
        actor_loss = actor_policy_loss + self.actor_reg_weight * actor_reg_loss + self.anchor_reg_weight * actor_anchor_loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        return {
            'actor_loss': float(actor_loss.detach().cpu().item()),
            'actor_policy_loss': float(actor_policy_loss.detach().cpu().item()),
            'actor_adv_policy_loss': float(actor_adv_policy_loss.detach().cpu().item()),
            'actor_clean_policy_loss': float(actor_clean_policy_loss.detach().cpu().item()),
            'actor_reg_loss': float(actor_reg_loss.detach().cpu().item()),
            'actor_anchor_loss': float(actor_anchor_loss.detach().cpu().item()),
            'actor_anchor_adv_loss': float(actor_anchor_adv_loss.detach().cpu().item()),
            'actor_anchor_clean_loss': float(actor_anchor_clean_loss.detach().cpu().item()),
            'critic_loss': float(critic_loss.detach().cpu().item()),
            'mean_q': float(q_pred.detach().mean().cpu().item()),
            'risk_weight_mean': 1.0 if preservation_weights is None else float(preservation_weights.detach().mean().cpu().item()),
            'risk_weight_max': 1.0 if preservation_weights is None else float(preservation_weights.detach().max().cpu().item()),
            'update_adv_frac': float(obs_stats['attacked_frac']),
            'update_adv_linf': float(obs_stats['adv_linf_mean']),
            'update_adv_l2': float(obs_stats['adv_l2_mean']),
            'target_adv_frac': float(next_stats['attacked_frac']),
            'target_adv_linf': float(next_stats['adv_linf_mean']),
            'target_adv_l2': float(next_stats['adv_l2_mean']),
        }


class SADDPG1DCrownAgent(SADDPGAgent):
    """SA-DDPG with CROWN smoothing and optional short-attack hard negatives."""

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
        actor_reg_weight: float = 0.3,
        directional_adversary: StateObservationAdversary | None = None,
        directional_attack_families: tuple[str, ...] = (),
        directional_reg_weight: float = 0.0,
        directional_top_fraction: float = 0.5,
        directional_weight_clip: float = 3.0,
        crown_split_dimensions: int = 2,
        robust_sarsa_reg_weight: float = 0.1,
        robust_sarsa_action_radius: float = 0.05,
        robust_sarsa_grid_size: int = 9,
    ) -> None:
        super().__init__(
            actor,
            device,
            gamma=gamma,
            tau=tau,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            adversary=directional_adversary,
            rollout_attack_prob=0.0,
            update_attack_prob=0.0,
            actor_reg_weight=actor_reg_weight,
        )
        if int(actor.fc_mu.out_features) != 1:
            raise ValueError('SADDPG1DCrownAgent requires action_dim=1.')
        self.obs_low_t = torch.as_tensor(obs_low, dtype=torch.float32, device=device).reshape(1, -1)
        self.obs_high_t = torch.as_tensor(obs_high, dtype=torch.float32, device=device).reshape(1, -1)
        self.max_epsilon = max(float(epsilon), 0.0)
        self.bound_epsilon = 0.0
        self.state_scope = canonical_attack_state_scope(state_scope)
        self.attack_indices = tuple(int(i) for i in attack_indices_for_state_scope(self.state_scope))
        self.directional_attack_families = tuple(
            canonical_sa_train_attack(name) for name in directional_attack_families
        )
        self.directional_reg_weight = max(float(directional_reg_weight), 0.0)
        self.directional_top_fraction = float(np.clip(float(directional_top_fraction), 0.0, 1.0))
        self.directional_weight_clip = max(float(directional_weight_clip), 1.0)
        self.crown_split_dimensions = max(int(crown_split_dimensions), 0)
        self.robust_sarsa_critic = Critic().to(device)
        self.robust_sarsa_critic_target = Critic().to(device)
        self.robust_sarsa_critic_target.load_state_dict(self.robust_sarsa_critic.state_dict())
        self.robust_sarsa_optimizer = torch.optim.Adam(
            self.robust_sarsa_critic.parameters(), lr=float(critic_lr)
        )
        self.robust_sarsa_reg_weight = max(float(robust_sarsa_reg_weight), 0.0)
        self.robust_sarsa_action_radius = max(float(robust_sarsa_action_radius), 0.0)
        self.robust_sarsa_grid_size = max(int(robust_sarsa_grid_size), 3)

    def initialize_robust_sarsa_from_critic(self) -> None:
        self.robust_sarsa_critic.load_state_dict(self.critic.state_dict())
        self.robust_sarsa_critic_target.load_state_dict(self.critic_target.state_dict())

    def set_bound_epsilon(self, epsilon: float) -> None:
        self.bound_epsilon = float(np.clip(float(epsilon), 0.0, self.max_epsilon))

    def reachable_action_bounds(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bound_fn = (
            actor_split_crown_action_bounds
            if self.crown_split_dimensions > 0
            else actor_crown_action_bounds
        )
        kwargs = {'split_dimensions': self.crown_split_dimensions} if self.crown_split_dimensions > 0 else {}
        return bound_fn(
            self.actor,
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
        adversarial: bool = False,
        attack_prob: float | None = None,
        attack_family: str | tuple[str, ...] | None = None,
    ) -> dict[str, float]:
        del adversarial, attack_prob
        observations = batch['observations']
        next_observations = batch['next_observations']

        with torch.no_grad():
            next_actions = self.actor_target(next_observations)
            q_target = batch['rewards'] + (1.0 - batch['dones']) * self.gamma * self.critic_target(
                next_observations,
                next_actions,
            ).reshape(-1)

        q_pred = self.critic(observations, batch['actions']).reshape(-1)
        critic_loss = F.mse_loss(q_pred, q_target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        with torch.no_grad():
            rs_next_actions = self.actor_target(next_observations)
            rs_target = batch['rewards'] + (1.0 - batch['dones']) * self.gamma * self.robust_sarsa_critic_target(
                next_observations,
                rs_next_actions,
            ).reshape(-1)
        rs_pred = self.robust_sarsa_critic(observations, batch['actions']).reshape(-1)
        robust_sarsa_td_loss = F.mse_loss(rs_pred, rs_target)
        offsets = torch.linspace(
            -self.robust_sarsa_action_radius,
            self.robust_sarsa_action_radius,
            self.robust_sarsa_grid_size,
            dtype=batch['actions'].dtype,
            device=batch['actions'].device,
        ).reshape(1, -1, 1)
        rs_action_grid = (batch['actions'].unsqueeze(1) + offsets).clamp(-1.0, 1.0)
        rs_batch, rs_count, _ = rs_action_grid.shape
        rs_repeated_obs = observations.unsqueeze(1).expand(-1, rs_count, -1)
        rs_grid_q = self.robust_sarsa_critic(
            rs_repeated_obs.reshape(rs_batch * rs_count, -1),
            rs_action_grid.reshape(rs_batch * rs_count, 1),
        ).reshape(rs_batch, rs_count)
        robust_sarsa_reg_loss = (rs_grid_q - rs_pred.unsqueeze(1)).pow(2).amax(dim=1).mean()
        robust_sarsa_loss = robust_sarsa_td_loss + self.robust_sarsa_reg_weight * robust_sarsa_reg_loss
        self.robust_sarsa_optimizer.zero_grad(set_to_none=True)
        robust_sarsa_loss.backward()
        self.robust_sarsa_optimizer.step()

        clean_actions = self.actor(observations)
        actor_policy_loss = -self.critic(observations, clean_actions).mean()
        action_lower, action_upper = self.reachable_action_bounds(observations)
        lower_deviation = (clean_actions - action_lower).pow(2).mean(dim=1)
        upper_deviation = (action_upper - clean_actions).pow(2).mean(dim=1)
        reg_per_sample = torch.maximum(lower_deviation, upper_deviation)
        actor_reg_loss = reg_per_sample.mean()

        directional_loss = clean_actions.new_tensor(0.0)
        directional_damage_mean = 0.0
        directional_damage_max = 0.0
        directional_selected_fraction = 0.0
        directional_candidate_count = 0
        families = self.directional_attack_families
        if attack_family is not None:
            raw_families = attack_family if isinstance(attack_family, (tuple, list)) else (attack_family,)
            families = tuple(canonical_sa_train_attack(name) for name in raw_families)
        if (
            self.directional_reg_weight > 0.0
            and self.directional_top_fraction > 0.0
            and self.adversary is not None
            and families
            and self.bound_epsilon > 0.0
        ):
            previous_epsilon = float(self.adversary.epsilon)
            self.adversary.epsilon = float(self.bound_epsilon)
            candidate_observations = []
            try:
                for family in families:
                    candidate_critic = (
                        self.robust_sarsa_critic_target
                        if family == 'q_function'
                        else self.critic_target
                    )
                    candidate_observations.append(
                        self.adversary.perturb_tensor(
                            observations,
                            actor=self.actor,
                            critic=candidate_critic,
                            attack_family=family,
                            is_new_arrivals=batch.get('is_new_arrivals'),
                        ).detach()
                    )
            finally:
                self.adversary.epsilon = previous_epsilon

            stacked_observations = torch.stack(candidate_observations, dim=1)
            batch_size, candidate_count, obs_dim = stacked_observations.shape
            flat_candidate_actions = self.actor(
                stacked_observations.reshape(batch_size * candidate_count, obs_dim)
            )
            candidate_actions = flat_candidate_actions.reshape(batch_size, candidate_count, 1)
            with torch.no_grad():
                clean_q = self.robust_sarsa_critic_target(
                    observations, clean_actions.detach()
                ).reshape(-1)
                repeated_clean = observations.unsqueeze(1).expand(-1, candidate_count, -1)
                candidate_q = self.robust_sarsa_critic_target(
                    repeated_clean.reshape(batch_size * candidate_count, obs_dim),
                    candidate_actions.detach().reshape(batch_size * candidate_count, 1),
                ).reshape(batch_size, candidate_count)
                q_damage = (clean_q.unsqueeze(1) - candidate_q).clamp_min(0.0)
                action_damage = (
                    candidate_actions.detach() - clean_actions.detach().unsqueeze(1)
                ).pow(2).mean(dim=2)
                damage_score = q_damage + 0.25 * action_damage
                selected_ids = torch.argmax(damage_score, dim=1)
                selected_damage = damage_score[
                    torch.arange(batch_size, device=observations.device), selected_ids
                ]
                selected_count = max(
                    1,
                    int(math.ceil(float(batch_size) * self.directional_top_fraction)),
                )
                hard_ids = torch.topk(selected_damage, k=selected_count, largest=True).indices
                hard_weights = selected_damage[hard_ids]
                hard_weights = hard_weights / hard_weights.mean().clamp_min(1e-6)
                hard_weights = hard_weights.clamp(max=self.directional_weight_clip)
            selected_actions = candidate_actions[
                torch.arange(batch_size, device=observations.device), selected_ids
            ]
            directional_per_sample = (selected_actions - clean_actions).pow(2).mean(dim=1)
            directional_loss = (hard_weights * directional_per_sample[hard_ids]).mean()
            directional_damage_mean = float(selected_damage.mean().cpu().item())
            directional_damage_max = float(selected_damage.max().cpu().item())
            directional_selected_fraction = float(selected_count) / float(batch_size)
            directional_candidate_count = int(candidate_count)

        actor_loss = (
            actor_policy_loss
            + self.actor_reg_weight * actor_reg_loss
            + self.directional_reg_weight * directional_loss
        )
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        self._soft_update(self.robust_sarsa_critic, self.robust_sarsa_critic_target)
        action_width = (action_upper - action_lower).detach().reshape(-1)
        return {
            'actor_loss': float(actor_loss.detach().cpu().item()),
            'actor_policy_loss': float(actor_policy_loss.detach().cpu().item()),
            'actor_adv_policy_loss': 0.0,
            'actor_clean_policy_loss': float(actor_policy_loss.detach().cpu().item()),
            'actor_reg_loss': float(actor_reg_loss.detach().cpu().item()),
            'sa_directional_reg_loss': float(directional_loss.detach().cpu().item()),
            'sa_directional_damage_mean': directional_damage_mean,
            'sa_directional_damage_max': directional_damage_max,
            'sa_directional_selected_fraction': directional_selected_fraction,
            'sa_directional_candidate_count': float(directional_candidate_count),
            'actor_anchor_loss': 0.0,
            'actor_anchor_adv_loss': 0.0,
            'actor_anchor_clean_loss': 0.0,
            'critic_loss': float(critic_loss.detach().cpu().item()),
            'robust_sarsa_loss': float(robust_sarsa_loss.detach().cpu().item()),
            'robust_sarsa_td_loss': float(robust_sarsa_td_loss.detach().cpu().item()),
            'robust_sarsa_reg_loss': float(robust_sarsa_reg_loss.detach().cpu().item()),
            'robust_sarsa_action_radius': float(self.robust_sarsa_action_radius),
            'mean_q': float(q_pred.detach().mean().cpu().item()),
            'risk_weight_mean': 1.0,
            'risk_weight_max': 1.0,
            'update_adv_frac': 0.0,
            'update_adv_linf': 0.0,
            'update_adv_l2': 0.0,
            'target_adv_frac': 0.0,
            'target_adv_linf': 0.0,
            'target_adv_l2': 0.0,
            'sa_bound_epsilon': float(self.bound_epsilon),
            'sa_reachable_action_width_mean': float(action_width.mean().cpu().item()),
            'sa_reachable_action_width_max': float(action_width.max().cpu().item()),
            'sa_stability_bound_mean': float(reg_per_sample.detach().mean().cpu().item()),
            'sa_paper_faithful_update': float(self.directional_reg_weight <= 0.0),
        }


def train_sa_ddpg_agent(
    arrivals,
    signals_path,
    device: torch.device,
    *,
    seed: int = 42,
    episodes: int = 20,
    buffer_size: int = 100000,
    batch_size: int = 256,
    learning_starts: int = 2500,
    exploration_noise: float = 1.0,
    gamma: float = 0.9,
    tau: float = 0.005,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    print_every: int = 1,
    init_actor_path=None,
    resume_bundle_path=None,
    reward_profile: RewardProfile = TRAIN_PROFILE,
    sa_train_attacks=None,
    sa_epsilon: float = 0.1,
    sa_alpha: float | None = None,
    sa_steps: int | None = None,
    sa_objective: str = 'q_function',
    sa_noise_std: float = 0.0,
    sa_rollout_attack_prob_start: float = 0.3,
    sa_rollout_attack_prob: float = 1.0,
    sa_update_attack_prob_start: float = 0.5,
    sa_update_attack_prob: float = 1.0,
    sa_curriculum_steps: int = 30000,
    sa_warmup_steps: int | None = None,
    sa_soc_new_threshold: float = 0.5,
    sa_soc_rollout_threshold: float = 0.3,
    sa_state_scope: str = 'all',
    sa_actor_reg_weight: float = 0.1,
    sa_mixed_update_attacks: bool = True,
    sa_anchor_actor_path=None,
    sa_anchor_reg_weight: float = 0.0,
    sa_anchor_clean_weight: float = 1.0,
    sa_clean_policy_weight: float = 0.0,
    sa_risk_weight_scale: float = 0.0,
    sa_risk_weight_max: float = 3.0,
    sa_risk_target_soc: float | None = None,
    sa_validation_every: int = 0,
    sa_validation_attacks=None,
    sa_validation_baseline_bundle_path=None,
    sa_validation_clean_drop_weight: float = 0.3,
    sa_validation_clean_drop_budget: float = 0.0,
    sa_validation_clean_drop_hard_cap: float = 0.0,
    sa_validation_clean_exit_weight: float = 0.0,
    sa_paper_mode: bool = False,
    sa_epsilon_schedule_steps: int = 60000,
    sa_directional_reg_weight: float = 0.0,
    sa_directional_top_fraction: float = 0.5,
    sa_directional_weight_clip: float = 3.0,
    sa_directional_update_interval: int = 2,
    sa_crown_split_dimensions: int = 2,
    sa_robust_sarsa_reg_weight: float = 0.1,
    sa_robust_sarsa_action_radius: float = 0.05,
    sa_robust_sarsa_grid_size: int = 9,
    checkpoint_every: int = 0,
    checkpoint_dir: str | Path | None = None,
    checkpoint_prefix: str = 'sa_ddpg',
    checkpoint_metadata: dict | None = None,
    episode_scenarios=None,
) -> tuple[SADDPGAgent, SADDPGTrainHistory]:
    set_seed(seed)
    scenarios = normalize_episode_scenarios(arrivals, signals_path, episode_scenarios)
    env = ChargingEnv(signals_path=scenarios[0].signals_path, reward_profile=reward_profile)
    if init_actor_path is not None and resume_bundle_path is not None:
        raise ValueError('train_sa_ddpg_agent supports either init_actor_path or resume_bundle_path, not both.')
    critic_state_dict = None
    if resume_bundle_path is not None:
        bundle = load_actor_critic_bundle(resume_bundle_path, device)
        if bundle.get('critic_state_dict') is None:
            raise ValueError(f'resume_bundle_path does not contain critic weights: {resume_bundle_path}')
        actor = Actor().to(device)
        actor.load_state_dict(bundle['actor_state_dict'])
        critic_state_dict = bundle['critic_state_dict']
    else:
        actor = load_actor_from_path(init_actor_path, device) if init_actor_path is not None else Actor().to(device)
    anchor_actor = None
    if sa_anchor_actor_path is not None and float(sa_anchor_reg_weight) > 0.0:
        anchor_actor = load_actor_from_path(sa_anchor_actor_path, device)
    max_duration = max_duration_across_scenarios(scenarios)
    obs_low, obs_high = env.observation_bounds(max_duration_of_stay=max_duration)
    if sa_paper_mode:
        directional_adversary = None
        directional_families: tuple[str, ...] = ()
        if float(sa_directional_reg_weight) > 0.0:
            directional_families = normalize_sa_train_attacks(sa_train_attacks)
            directional_adversary = StateObservationAdversary(
                device=device,
                epsilon=sa_epsilon,
                alpha=sa_alpha,
                steps=sa_steps,
                objective=sa_objective,
                noise_std=sa_noise_std,
                soc_new_threshold=sa_soc_new_threshold,
                soc_rollout_threshold=sa_soc_rollout_threshold,
                obs_low=obs_low,
                obs_high=obs_high,
                attack_state_scope=sa_state_scope,
            )
        agent = SADDPG1DCrownAgent(
            actor,
            device=device,
            obs_low=obs_low,
            obs_high=obs_high,
            epsilon=sa_epsilon,
            state_scope=sa_state_scope,
            gamma=gamma,
            tau=tau,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            actor_reg_weight=sa_actor_reg_weight,
            directional_adversary=directional_adversary,
            directional_attack_families=directional_families,
            directional_reg_weight=sa_directional_reg_weight,
            directional_top_fraction=sa_directional_top_fraction,
            directional_weight_clip=sa_directional_weight_clip,
            crown_split_dimensions=sa_crown_split_dimensions,
            robust_sarsa_reg_weight=sa_robust_sarsa_reg_weight,
            robust_sarsa_action_radius=sa_robust_sarsa_action_radius,
            robust_sarsa_grid_size=sa_robust_sarsa_grid_size,
        )
    else:
        adversary = StateObservationAdversary(
            device=device,
            epsilon=sa_epsilon,
            alpha=sa_alpha,
            steps=sa_steps,
            objective=sa_objective,
            noise_std=sa_noise_std,
            soc_new_threshold=sa_soc_new_threshold,
            soc_rollout_threshold=sa_soc_rollout_threshold,
            obs_low=obs_low,
            obs_high=obs_high,
            attack_state_scope=sa_state_scope,
        )
        agent = SADDPGAgent(
            actor,
            device=device,
            gamma=gamma,
            tau=tau,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            adversary=adversary,
            rollout_attack_prob=sa_rollout_attack_prob,
            update_attack_prob=sa_update_attack_prob,
            actor_reg_weight=sa_actor_reg_weight,
            anchor_actor=anchor_actor,
            anchor_reg_weight=sa_anchor_reg_weight,
            anchor_clean_weight=sa_anchor_clean_weight,
            clean_policy_weight=sa_clean_policy_weight,
            risk_weight_scale=sa_risk_weight_scale,
            risk_weight_max=sa_risk_weight_max,
            risk_target_soc=reward_profile.exit_target_min if sa_risk_target_soc is None else sa_risk_target_soc,
        )
    if critic_state_dict is not None:
        agent.critic.load_state_dict(critic_state_dict)
        agent.critic_target.load_state_dict(critic_state_dict)
    if isinstance(agent, SADDPG1DCrownAgent):
        agent.initialize_robust_sarsa_from_critic()
    attack_families = normalize_sa_train_attacks(sa_train_attacks)
    validation_attacks = normalize_sa_train_attacks(sa_validation_attacks or attack_families)
    buffer = SAReplayBuffer(buffer_size, env.obs_dim, env.action_dim, device)
    rows: list[dict] = []
    validation_rows: list[dict] = []
    current_noise = float(exploration_noise)
    total_steps = 0
    attack_warmup = int(learning_starts if sa_warmup_steps is None else sa_warmup_steps)
    curriculum_steps = int(sa_curriculum_steps)
    validation_every = max(int(sa_validation_every), 0)
    validation_clean_drop_weight = max(float(sa_validation_clean_drop_weight), 0.0)
    validation_clean_drop_budget = max(float(sa_validation_clean_drop_budget), 0.0)
    validation_clean_drop_hard_cap = max(float(sa_validation_clean_drop_hard_cap), 0.0)
    validation_clean_exit_weight = max(float(sa_validation_clean_exit_weight), 0.0)
    baseline_eval_actor: Actor | None = None
    baseline_eval_critic: Critic | None = None
    baseline_clean_reward: float | None = None
    baseline_clean_exit_vio: int | None = None
    baseline_attack_cache: dict[str, dict] = {}
    best_score = -math.inf
    best_validation: dict | None = None
    best_state: dict[str, dict[str, torch.Tensor]] | None = None
    train_started_at = time.perf_counter()
    checkpoint_every = max(int(checkpoint_every), 0)
    checkpoint_dir_path = None if checkpoint_dir is None else ensure_dir(Path(checkpoint_dir))
    checkpoint_prefix = str(checkpoint_prefix or 'sa_ddpg')
    checkpoint_metadata = dict(checkpoint_metadata or {})

    if validation_every > 0:
        if sa_validation_baseline_bundle_path is None:
            raise ValueError('SA-DDPG validation requires sa_validation_baseline_bundle_path for fair recovery scoring.')
        validation_bundle = load_actor_critic_bundle(sa_validation_baseline_bundle_path, device)
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
        if isinstance(agent, SADDPG1DCrownAgent):
            state['robust_sarsa_critic'] = _module_state_to_cpu(agent.robust_sarsa_critic)
            state['robust_sarsa_critic_target'] = _module_state_to_cpu(agent.robust_sarsa_critic_target)
        return state

    def _restore_agent_state(state: dict[str, dict[str, torch.Tensor]]) -> None:
        _load_module_state(agent.actor, state['actor'])
        _load_module_state(agent.critic, state['critic'])
        _load_module_state(agent.actor_target, state['actor_target'])
        _load_module_state(agent.critic_target, state['critic_target'])
        if isinstance(agent, SADDPG1DCrownAgent):
            if 'robust_sarsa_critic' in state and 'robust_sarsa_critic_target' in state:
                _load_module_state(agent.robust_sarsa_critic, state['robust_sarsa_critic'])
                _load_module_state(agent.robust_sarsa_critic_target, state['robust_sarsa_critic_target'])

    def _validation_attacker(policy_actor: Actor, policy_critic: Critic | None, family: str):
        # Training uses the internal SA family name `electhacker_o` to mean the
        # O-scenario ElectHacker rollout adversary.  The generic merged attack
        # builder, however, only accepts the canonical algorithm name
        # `electhacker`; the O scenario is handled by rollout state flags.
        canonical_family = canonical_sa_train_attack(family)
        attack_algorithm = 'electhacker' if canonical_family == 'electhacker_o' else canonical_family
        critic_for_attack = policy_critic if canonical_family == 'q_function' else None
        if canonical_family == 'q_function' and critic_for_attack is None:
            raise ValueError('SA-DDPG q_function validation requires a matching critic.')
        return build_state_attacker(
            policy_actor,
            device=device,
            algorithm=attack_algorithm,
            epsilon=sa_epsilon,
            alpha=sa_alpha,
            iters=sa_steps,
            seed=seed,
            obs_low=obs_low,
            obs_high=obs_high,
            critic=critic_for_attack,
            attack_state_scope=sa_state_scope,
        )

    def _run_validation(episode: int, *, initial: bool = False) -> dict:
        nonlocal baseline_clean_reward, baseline_clean_exit_vio
        if baseline_eval_actor is None:
            raise RuntimeError('SA-DDPG validation baseline actor is not initialized.')
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

        sa_clean = rollout_episode(
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
        sa_clean_reward = float(sa_clean['ep_reward'])
        sa_clean_exit_vio = int(sa_clean.get('exit_vio', 0))
        clean_drop = float(baseline_clean_reward - sa_clean_reward)
        clean_drop_ratio = 0.0 if abs(float(baseline_clean_reward)) < 1e-9 else clean_drop / abs(float(baseline_clean_reward))
        attack_rewards: dict[str, float] = {}
        baseline_attack_rewards: dict[str, float] = {}
        recovery_ratios: dict[str, float] = {}

        for family in validation_attacks:
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
            baseline_attack_reward = float(baseline_attack_cache[canonical_family]['ep_reward'])
            sa_attacker = _validation_attacker(agent.actor, agent.critic, canonical_family)
            sa_attack = rollout_episode(
                arrivals,
                agent.actor,
                signals_path,
                device,
                reward_profile,
                attack_enabled=True,
                attack_scenario='O',
                attacker=sa_attacker,
                route_mode='none',
                exploration_noise=0.0,
                attack_ratio=1.0,
                attack_scope='obs',
                detector_feature_mode='posterior',
            )
            sa_attack_reward = float(sa_attack['ep_reward'])
            attack_drop = float(baseline_clean_reward - baseline_attack_reward)
            recovery = 0.0 if abs(attack_drop) < 1e-9 else float((sa_attack_reward - baseline_attack_reward) / attack_drop)
            attack_rewards[canonical_family] = sa_attack_reward
            baseline_attack_rewards[canonical_family] = baseline_attack_reward
            recovery_ratios[canonical_family] = recovery

        recovery_values = list(recovery_ratios.values())
        avg_recovery = float(np.mean(recovery_values)) if recovery_values else 0.0
        min_recovery = float(np.min(recovery_values)) if recovery_values else 0.0
        if validation_clean_drop_budget > 0.0:
            clean_drop_penalty = max(clean_drop, 0.0) / validation_clean_drop_budget
        else:
            clean_drop_penalty = max(clean_drop_ratio, 0.0)
        clean_drop_hard_excess = (
            max(clean_drop - validation_clean_drop_hard_cap, 0.0) if validation_clean_drop_hard_cap > 0.0 else 0.0
        )
        clean_drop_hard_penalty = (
            clean_drop_hard_excess / max(validation_clean_drop_hard_cap, 1.0) if validation_clean_drop_hard_cap > 0.0 else 0.0
        )
        clean_exit_delta = max(float(sa_clean_exit_vio - int(baseline_clean_exit_vio or 0)), 0.0)
        clean_exit_penalty = clean_exit_delta / max(float(abs(int(baseline_clean_exit_vio or 0))), 1.0)
        score = float(
            min_recovery
            - validation_clean_drop_weight * clean_drop_penalty
            - validation_clean_exit_weight * clean_exit_penalty
        )
        if clean_drop_hard_excess > 0.0:
            score -= 1000.0 + clean_drop_hard_penalty
        row = {
            'episode': int(episode),
            'validation_initial': int(bool(initial)),
            'val_score': score,
            'val_min_recovery_ratio': min_recovery,
            'val_avg_recovery_ratio': avg_recovery,
            'val_clean_drop': clean_drop,
            'val_clean_drop_ratio': clean_drop_ratio,
            'val_clean_drop_penalty': float(clean_drop_penalty),
            'val_clean_drop_hard_cap': float(validation_clean_drop_hard_cap),
            'val_clean_drop_hard_excess': float(clean_drop_hard_excess),
            'val_clean_drop_hard_penalty': float(clean_drop_hard_penalty),
            'val_clean_drop_hard_ok': int(clean_drop_hard_excess <= 0.0),
            'val_clean_exit_vio': int(sa_clean_exit_vio),
            'val_baseline_clean_exit_vio': int(baseline_clean_exit_vio or 0),
            'val_clean_exit_penalty': float(clean_exit_penalty),
            'val_baseline_clean_reward': float(baseline_clean_reward),
            'val_clean_reward': sa_clean_reward,
        }
        for family in validation_attacks:
            canonical_family = canonical_sa_train_attack(family)
            row[f'val_{canonical_family}_baseline_attack_reward'] = float(baseline_attack_rewards[canonical_family])
            row[f'val_{canonical_family}_attack_reward'] = float(attack_rewards[canonical_family])
            row[f'val_{canonical_family}_recovery_ratio'] = float(recovery_ratios[canonical_family])
        return row

    if validation_every > 0:
        initial_validation = _run_validation(0, initial=True)
        validation_rows.append(initial_validation)
        best_score = float(initial_validation['val_score'])
        best_validation = dict(initial_validation)
        best_validation['best_source'] = 'initial'
        best_state = _snapshot_agent_state()
        print(
            f"[train-sa-ddpg][val] ep=000 score={initial_validation['val_score']:.4f} "
            f"min_recovery={initial_validation['val_min_recovery_ratio']:.4f} "
            f"clean_drop={initial_validation['val_clean_drop']:.2f} best=*"
        )

    for episode in range(1, episodes + 1):
        episode_scenario = scenario_for_episode(scenarios, episode)
        episode_arrivals = episode_scenario.arrivals
        env = ChargingEnv(signals_path=episode_scenario.signals_path, reward_profile=reward_profile)
        env.reset()
        idx = 0
        active: list[QueueItem] = []
        episode_attack_family = (
            'none_policy_smoothing'
            if sa_paper_mode
            else attack_families[(episode - 1) % len(attack_families)]
        )
        last_update = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'robust_sarsa_loss': 0.0,
            'robust_sarsa_td_loss': 0.0,
            'robust_sarsa_reg_loss': 0.0,
            'mean_q': 0.0,
            'actor_policy_loss': 0.0,
            'actor_adv_policy_loss': 0.0,
            'actor_clean_policy_loss': 0.0,
            'actor_reg_loss': 0.0,
            'actor_anchor_loss': 0.0,
            'actor_anchor_adv_loss': 0.0,
            'actor_anchor_clean_loss': 0.0,
            'risk_weight_mean': 1.0,
            'risk_weight_max': 1.0,
            'update_adv_frac': 0.0,
            'update_adv_linf': 0.0,
            'update_adv_l2': 0.0,
            'target_adv_frac': 0.0,
            'target_adv_linf': 0.0,
            'target_adv_l2': 0.0,
            'sa_bound_epsilon': 0.0,
            'sa_reachable_action_width_mean': 0.0,
            'sa_reachable_action_width_max': 0.0,
            'sa_stability_bound_mean': 0.0,
            'sa_paper_faithful_update': float(bool(sa_paper_mode)),
        }
        decision_count = 0
        rollout_attack_count = 0
        rollout_adv_linf_sum = 0.0
        rollout_adv_l2_sum = 0.0

        while env.t < env.horizon:
            rollout_attack_prob = scheduled_attack_probability(
                total_steps=total_steps,
                warmup_steps=attack_warmup,
                start_prob=sa_rollout_attack_prob_start,
                final_prob=sa_rollout_attack_prob,
                curriculum_steps=curriculum_steps,
            )
            update_attack_prob = scheduled_attack_probability(
                total_steps=total_steps,
                warmup_steps=attack_warmup,
                start_prob=sa_update_attack_prob_start,
                final_prob=sa_update_attack_prob,
                curriculum_steps=curriculum_steps,
            )
            if sa_paper_mode:
                rollout_attack_prob = 0.0
                update_attack_prob = 0.0
            attack_enabled = (not sa_paper_mode) and (rollout_attack_prob > 0.0 or update_attack_prob > 0.0)
            step_is_new_arrivals: list[bool] = []

            while idx < len(episode_arrivals) and int(episode_arrivals.loc[idx, 'Arrive_time']) == env.t:
                obs = env.build_initial_obs(int(episode_arrivals.loc[idx, 'Duration_of_stay']))
                action, attack_stats = agent.select_action(
                    obs,
                    exploration_noise=current_noise,
                    deterministic=False,
                    adversarial=attack_enabled,
                    attack_prob=rollout_attack_prob,
                    attack_family=None if sa_paper_mode else episode_attack_family,
                    is_new_arrival=True,
                )
                decision_count += 1
                if attack_stats['attacked_frac'] > 0.0:
                    rollout_attack_count += 1
                    rollout_adv_linf_sum += float(attack_stats['adv_linf_mean'])
                    rollout_adv_l2_sum += float(attack_stats['adv_l2_mean'])
                env.enqueue(obs, action, int(episode_arrivals.loc[idx, 'Station']))
                step_is_new_arrivals.append(True)
                idx += 1

            for item in active:
                action, attack_stats = agent.select_action(
                    item.obs,
                    exploration_noise=current_noise,
                    deterministic=False,
                    adversarial=attack_enabled,
                    attack_prob=rollout_attack_prob,
                    attack_family=None if sa_paper_mode else episode_attack_family,
                    is_new_arrival=False,
                )
                decision_count += 1
                if attack_stats['attacked_frac'] > 0.0:
                    rollout_attack_count += 1
                    rollout_adv_linf_sum += float(attack_stats['adv_linf_mean'])
                    rollout_adv_l2_sum += float(attack_stats['adv_l2_mean'])
                env.enqueue(item.obs, action, item.station)
                step_is_new_arrivals.append(False)

            transitions, active, metrics = env.step()
            if len(transitions) != len(step_is_new_arrivals):
                raise RuntimeError('SA-DDPG transition count does not match queued decision metadata.')
            for tr, is_new_arrival in zip(transitions, step_is_new_arrivals):
                buffer.add(tr.obs, tr.next_obs, tr.action, tr.reward, tr.done, is_new_arrival=is_new_arrival)
                total_steps += 1
                if buffer.size >= max(batch_size, learning_starts):
                    update_attack_prob = (
                        0.0
                        if sa_paper_mode
                        else scheduled_attack_probability(
                            total_steps=total_steps,
                            warmup_steps=attack_warmup,
                            start_prob=sa_update_attack_prob_start,
                            final_prob=sa_update_attack_prob,
                            curriculum_steps=curriculum_steps,
                        )
                    )
                    if sa_paper_mode:
                        if not isinstance(agent, SADDPG1DCrownAgent):
                            raise RuntimeError('SA-DDPG paper mode initialized the wrong agent type.')
                        robust_steps = max(int(total_steps) - int(attack_warmup), 0)
                        schedule_steps = max(int(sa_epsilon_schedule_steps), 1)
                        scheduled_epsilon = float(sa_epsilon) * min(float(robust_steps) / float(schedule_steps), 1.0)
                        agent.set_bound_epsilon(scheduled_epsilon)
                        run_directional = bool(
                            float(sa_directional_reg_weight) > 0.0
                            and robust_steps > 0
                            and total_steps % max(int(sa_directional_update_interval), 1) == 0
                        )
                        last_update = agent.update(
                            buffer.sample(batch_size),
                            attack_family=attack_families if run_directional else (),
                        )
                    else:
                        last_update = agent.update(
                            buffer.sample(batch_size),
                            adversarial=update_attack_prob > 0.0,
                            attack_prob=update_attack_prob,
                            attack_family=attack_families if bool(sa_mixed_update_attacks) else episode_attack_family,
                        )
            current_noise *= 0.9999 if current_noise > 0.1 else 0.999977

        routed_decisions = max(rollout_attack_count, 1)
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
            'actor_loss': float(last_update.get('actor_loss', 0.0)),
            'critic_loss': float(last_update.get('critic_loss', 0.0)),
            'robust_sarsa_loss': float(last_update.get('robust_sarsa_loss', 0.0)),
            'robust_sarsa_td_loss': float(last_update.get('robust_sarsa_td_loss', 0.0)),
            'robust_sarsa_reg_loss': float(last_update.get('robust_sarsa_reg_loss', 0.0)),
            'mean_q': float(last_update.get('mean_q', 0.0)),
            'actor_policy_loss': float(last_update.get('actor_policy_loss', 0.0)),
            'actor_adv_policy_loss': float(last_update.get('actor_adv_policy_loss', 0.0)),
            'actor_clean_policy_loss': float(last_update.get('actor_clean_policy_loss', 0.0)),
            'actor_reg_loss': float(last_update.get('actor_reg_loss', 0.0)),
            'actor_anchor_loss': float(last_update.get('actor_anchor_loss', 0.0)),
            'actor_anchor_adv_loss': float(last_update.get('actor_anchor_adv_loss', 0.0)),
            'actor_anchor_clean_loss': float(last_update.get('actor_anchor_clean_loss', 0.0)),
            'risk_weight_mean': float(last_update.get('risk_weight_mean', 1.0)),
            'risk_weight_max': float(last_update.get('risk_weight_max', 1.0)),
            'rollout_attack_rate': 0.0 if decision_count == 0 else float(rollout_attack_count) / float(decision_count),
            'rollout_adv_linf': 0.0 if rollout_attack_count == 0 else float(rollout_adv_linf_sum) / float(routed_decisions),
            'rollout_adv_l2': 0.0 if rollout_attack_count == 0 else float(rollout_adv_l2_sum) / float(routed_decisions),
            'update_adv_frac': float(last_update.get('update_adv_frac', 0.0)),
            'update_adv_linf': float(last_update.get('update_adv_linf', 0.0)),
            'update_adv_l2': float(last_update.get('update_adv_l2', 0.0)),
            'target_adv_frac': float(last_update.get('target_adv_frac', 0.0)),
            'target_adv_linf': float(last_update.get('target_adv_linf', 0.0)),
            'target_adv_l2': float(last_update.get('target_adv_l2', 0.0)),
            'sa_bound_epsilon': float(last_update.get('sa_bound_epsilon', 0.0)),
            'sa_reachable_action_width_mean': float(last_update.get('sa_reachable_action_width_mean', 0.0)),
            'sa_reachable_action_width_max': float(last_update.get('sa_reachable_action_width_max', 0.0)),
            'sa_stability_bound_mean': float(last_update.get('sa_stability_bound_mean', 0.0)),
            'sa_directional_reg_loss': float(last_update.get('sa_directional_reg_loss', 0.0)),
            'sa_directional_damage_mean': float(last_update.get('sa_directional_damage_mean', 0.0)),
            'sa_directional_damage_max': float(last_update.get('sa_directional_damage_max', 0.0)),
            'sa_directional_selected_fraction': float(last_update.get('sa_directional_selected_fraction', 0.0)),
            'sa_directional_candidate_count': float(last_update.get('sa_directional_candidate_count', 0.0)),
            'sa_paper_faithful_update': float(last_update.get('sa_paper_faithful_update', float(bool(sa_paper_mode)))),
            'train_attack_family': episode_attack_family,
            'update_attack_families': (
                (
                    'none_policy_smoothing'
                    if float(sa_directional_reg_weight) <= 0.0
                    else ','.join(attack_families)
                )
                if sa_paper_mode
                else ','.join(attack_families if bool(sa_mixed_update_attacks) else (episode_attack_family,))
            ),
            'sa_state_scope': canonical_attack_state_scope(sa_state_scope),
            'sa_actor_reg_weight': float(sa_actor_reg_weight),
            'sa_directional_reg_weight': float(sa_directional_reg_weight),
            'sa_directional_top_fraction': float(sa_directional_top_fraction),
            'sa_directional_weight_clip': float(sa_directional_weight_clip),
            'sa_directional_update_interval': int(max(sa_directional_update_interval, 1)),
            'sa_crown_split_dimensions': int(max(sa_crown_split_dimensions, 0)),
            'sa_robust_sarsa_reg_weight': float(sa_robust_sarsa_reg_weight),
            'sa_robust_sarsa_action_radius': float(sa_robust_sarsa_action_radius),
            'sa_robust_sarsa_grid_size': int(max(sa_robust_sarsa_grid_size, 3)),
            'sa_mixed_update_attacks': int(bool(sa_mixed_update_attacks)),
            'sa_anchor_reg_weight': float(sa_anchor_reg_weight),
            'sa_anchor_clean_weight': float(sa_anchor_clean_weight),
            'sa_clean_policy_weight': float(sa_clean_policy_weight),
            'sa_risk_weight_scale': float(sa_risk_weight_scale),
            'sa_risk_weight_max': float(sa_risk_weight_max),
            'sa_risk_target_soc': float(reward_profile.exit_target_min if sa_risk_target_soc is None else sa_risk_target_soc),
            'sa_training_mode': (
                'split_crown_plus_sgld_and_robust_sarsa_directional_smoothing'
                if sa_paper_mode and float(sa_directional_reg_weight) > 0.0
                else ('paper_crown_policy_smoothing' if sa_paper_mode else 'adversarial_training')
            ),
            'sa_epsilon_schedule_steps': int(sa_epsilon_schedule_steps if sa_paper_mode else 0),
            'rollout_attack_prob_target': float(rollout_attack_prob),
            'update_attack_prob_target': float(update_attack_prob),
            'total_steps': int(total_steps),
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
                    'val_clean_drop_penalty': float(validation_row['val_clean_drop_penalty']),
                    'val_clean_drop_hard_ok': int(validation_row['val_clean_drop_hard_ok']),
                    'val_clean_drop_hard_excess': float(validation_row['val_clean_drop_hard_excess']),
                    'val_clean_exit_vio': int(validation_row['val_clean_exit_vio']),
                    'val_clean_exit_penalty': float(validation_row['val_clean_exit_penalty']),
                    'val_is_best': int(bool(is_best)),
                }
            )
            print(
                f"[train-sa-ddpg][val] ep={episode:03d} score={validation_row['val_score']:.4f} "
                f"min_recovery={validation_row['val_min_recovery_ratio']:.4f} "
                f"avg_recovery={validation_row['val_avg_recovery_ratio']:.4f} "
                f"clean_drop={validation_row['val_clean_drop']:.2f} best={'*' if is_best else '-'}"
            )
        if checkpoint_every > 0 and checkpoint_dir_path is not None and (episode % checkpoint_every == 0 or episode == episodes):
            ckpt_path = checkpoint_dir_path / f"{checkpoint_prefix}_ep{int(episode):03d}_bundle.pt"
            save_sa_ddpg_bundle(
                agent,
                ckpt_path,
                metadata={
                    **checkpoint_metadata,
                    'checkpoint_episode': int(episode),
                    'episode': int(episode),
                    'total_steps': int(total_steps),
                    'cumulative_train_seconds': float(time.perf_counter() - train_started_at),
                    'checkpoint_kind': 'continuous_training_checkpoint',
                    'best_validation_so_far': best_validation,
                },
            )
            row['checkpoint_path'] = str(ckpt_path)
            row['cumulative_train_seconds'] = float(time.perf_counter() - train_started_at)
            print(f"[train-sa-ddpg][ckpt] saved ep={episode:03d}: {ckpt_path}")
        if episode == 1 or episode % print_every == 0 or episode == episodes:
            print(
                f"[train-sa-ddpg] ep={episode:03d}/{episodes} "
                f"reward={row['ep_reward']:.4f} exit={row['exit_vio']} running={row['run_vio']} "
                f"actor_loss={row['actor_loss']:.6f} critic_loss={row['critic_loss']:.6f} "
                f"attack={episode_attack_family} rollout_p={row['rollout_attack_prob_target']:.3f} "
                f"update_p={row['update_attack_prob_target']:.3f} rollout_attack_rate={row['rollout_attack_rate']:.3f}"
            )
    if best_state is not None:
        _restore_agent_state(best_state)
        agent.best_validation = best_validation
        print(
            f"[train-sa-ddpg][val] restored best checkpoint from {best_validation.get('best_source', 'unknown') if best_validation else 'unknown'} "
            f"score={best_score:.4f}"
        )
    else:
        agent.best_validation = None
    return agent, SADDPGTrainHistory(rows, validation_rows=validation_rows)


def train_sa_ddpg_paper_agent(
    *args,
    sa_actor_reg_weight: float = 0.3,
    sa_epsilon_schedule_steps: int = 60000,
    **kwargs,
) -> tuple[SADDPGAgent, SADDPGTrainHistory]:
    """Experiment baseline matching SA-DDPG's clean DDPG plus policy-smoothing objective."""
    return train_sa_ddpg_agent(
        *args,
        sa_paper_mode=True,
        sa_actor_reg_weight=sa_actor_reg_weight,
        sa_epsilon_schedule_steps=sa_epsilon_schedule_steps,
        sa_rollout_attack_prob_start=0.0,
        sa_rollout_attack_prob=0.0,
        sa_update_attack_prob_start=0.0,
        sa_update_attack_prob=0.0,
        **kwargs,
    )


def save_sa_ddpg_bundle(
    agent: SADDPGAgent,
    path: str | Path,
    *,
    metadata: dict | None = None,
) -> Path:
    out_path = Path(path)
    ensure_dir(out_path.parent)
    meta = dict(metadata or {})
    if isinstance(agent, SADDPG1DCrownAgent):
        meta.setdefault('source_variant', 'sa_ddpg_1d_crown')
        meta.setdefault('training_objective', 'clean_ddpg_plus_crown_policy_smoothing')
        meta.setdefault('action_dim', 1)
    torch.save(
        {
            'model_type': 'sa_ddpg_1d_crown_bundle' if isinstance(agent, SADDPG1DCrownAgent) else 'sa_ddpg_bundle',
            'actor_state_dict': agent.actor.state_dict(),
            'critic_state_dict': agent.critic.state_dict(),
            **(
                {'robust_sarsa_critic_state_dict': agent.robust_sarsa_critic.state_dict()}
                if isinstance(agent, SADDPG1DCrownAgent)
                else {}
            ),
            'metadata': meta,
        },
        out_path,
    )
    return out_path
