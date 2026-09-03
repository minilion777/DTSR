"""状态攻击器与 C/F/O 场景规则。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import hashlib

import numpy as np
import torch
import torch.nn as nn

from .merged_core import ATTACK_DEFAULTS, canonical_attack_algorithm, to_numpy_1d

AttackAlgorithm = Literal['electhacker', 'opposite_pgd', 'opposite_fgsm', 'q_function', 'critic_v', 'action', 'advpolicy', 'pgd', 'fgsm']
AttackScenario = Literal['C', 'F', 'O']
AttackScope = Literal['obs', 'vehicle', 'window']

LOCAL_ATTACK_IDX: tuple[int, ...] = (0, 1, 10)
GLOBAL_ATTACK_IDX: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9)
ALL_ATTACK_IDX: tuple[int, ...] = tuple(range(11))


def canonical_attack_state_scope(value: str | None) -> str:
    token = str(value or 'local').strip().lower().replace('-', '_')
    aliases = {
        'local_only': 'local',
        'local': 'local',
        'global_only': 'global',
        'global': 'global',
        'all_state': 'all',
        'all_states': 'all',
        'full': 'all',
        'local_global': 'all',
        'local_and_global': 'all',
        'all': 'all',
    }
    if token not in aliases:
        raise ValueError(f'Unsupported attack state scope: {value!r}')
    return aliases[token]


def attack_indices_for_state_scope(scope: str | None) -> tuple[int, ...]:
    token = canonical_attack_state_scope(scope)
    if token == 'local':
        return LOCAL_ATTACK_IDX
    if token == 'global':
        return GLOBAL_ATTACK_IDX
    return ALL_ATTACK_IDX


@dataclass
class AttackContext:
    """一次攻击决策需要用到的上下文。"""

    scenario: AttackScenario
    time_index: int
    raw_price: float
    station: int
    is_new_arrival: bool
    price_threshold: float = 400.0
    soc_new_threshold: float = 0.5
    soc_rollout_threshold: float = 0.3
    even_station_target: float = 1.0
    odd_station_target: float = -0.5


class PGDStateAttacker:
    """统一版状态攻击器。

    - electhacker: project-specific targeted attack;
    - opposite_pgd / opposite_fgsm: paper-style pointwise opposite attacks;
    - q_function: paper-style critic-guided pointwise attack;
    - critic_v: paper-style value-guided pointwise attack;
    - action: maximal action-difference attack;
    - advpolicy: learned adversarial policy attack.
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        device: torch.device,
        algorithm: AttackAlgorithm = 'electhacker',
        epsilon: float | None = None,
        alpha: float | None = None,
        iters: int | None = None,
        seed: int = 42,
        obs_low: np.ndarray | torch.Tensor | None = None,
        obs_high: np.ndarray | torch.Tensor | None = None,
        critic: torch.nn.Module | None = None,
        value_model: torch.nn.Module | None = None,
        adversary: torch.nn.Module | None = None,
        attack_state_scope: str = 'local',
        attack_indices: tuple[int, ...] | list[int] | None = None,
        adversary_temperature: float = 1.0,
        adversary_deterministic: bool = True,
        adversary_context_mode: str = 'none',
        adversary_station_count: int = 9,
    ) -> None:
        canonical_algorithm = canonical_attack_algorithm(str(algorithm))
        defaults = ATTACK_DEFAULTS[str(canonical_algorithm)]
        self.actor = actor.to(device).eval()
        self.device = device
        self.algorithm = canonical_algorithm
        self.critic = None if critic is None else critic.to(device).eval()
        self.value_model = None if value_model is None else value_model.to(device).eval()
        self.adversary = None if adversary is None else adversary.to(device).eval()
        self.epsilon = float(defaults.epsilon if epsilon is None else epsilon)
        self.alpha = float(defaults.alpha if alpha is None else alpha)
        self.iters = int(defaults.iters if iters is None else iters)
        self.seed = int(seed)
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(self.seed)
        self.adversary_temperature = max(float(adversary_temperature), 1e-6)
        self.adversary_deterministic = bool(adversary_deterministic)
        self.adversary_context_mode = str(adversary_context_mode or 'none').strip().lower()
        self.adversary_station_count = max(int(adversary_station_count), 1)
        self.obs_low = None if obs_low is None else torch.as_tensor(obs_low, dtype=torch.float32, device=self.device).reshape(1, -1)
        self.obs_high = None if obs_high is None else torch.as_tensor(obs_high, dtype=torch.float32, device=self.device).reshape(1, -1)
        if (self.obs_low is None) != (self.obs_high is None):
            raise ValueError('PGDStateAttacker requires both obs_low and obs_high or neither.')
        if self.obs_low is not None and self.obs_low.shape != self.obs_high.shape:
            raise ValueError('PGDStateAttacker obs_low/obs_high must share the same shape.')
        if self.algorithm == 'q_function' and self.critic is None:
            raise ValueError('q_function attack requires a critic model.')
        if self.algorithm == 'critic_v' and self.value_model is None:
            raise ValueError('critic_v attack requires a value model.')
        if self.algorithm == 'advpolicy' and self.adversary is None:
            raise ValueError('advpolicy attack requires an adversary policy.')
        self.attack_state_scope = canonical_attack_state_scope(attack_state_scope)
        self.local_attack_idx = tuple(int(v) for v in (attack_indices if attack_indices is not None else attack_indices_for_state_scope(self.attack_state_scope)))

    def reset(self) -> None:
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.seed)

    def clone(self) -> 'PGDStateAttacker':
        return PGDStateAttacker(
            self.actor,
            device=self.device,
            algorithm=self.algorithm,
            epsilon=self.epsilon,
            alpha=self.alpha,
            iters=self.iters,
            seed=self.seed,
            obs_low=None if self.obs_low is None else self.obs_low.detach().cpu().numpy().reshape(-1),
            obs_high=None if self.obs_high is None else self.obs_high.detach().cpu().numpy().reshape(-1),
            critic=self.critic,
            value_model=self.value_model,
            adversary=self.adversary,
            attack_state_scope=self.attack_state_scope,
            attack_indices=self.local_attack_idx,
            adversary_temperature=self.adversary_temperature,
            adversary_deterministic=self.adversary_deterministic,
            adversary_context_mode=self.adversary_context_mode,
            adversary_station_count=self.adversary_station_count,
        )

    def _local_attack_mask(self, obs: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(obs)
        mask[..., list(self.local_attack_idx)] = 1.0
        return mask

    def _project_obs(self, original: torch.Tensor, proposal: torch.Tensor) -> torch.Tensor:
        if self.obs_low is None or self.obs_high is None:
            return torch.clamp(proposal, min=0.0, max=1.0)
        if self.obs_low.shape[1] != original.shape[1]:
            raise ValueError(
                f'PGDStateAttacker obs bounds dim mismatch: bounds={int(self.obs_low.shape[1])} obs={int(original.shape[1])}'
            )
        return torch.maximum(torch.minimum(proposal, self.obs_high), self.obs_low)

    def attack(self, obs_batch: np.ndarray, target_actions: np.ndarray | None = None) -> np.ndarray:
        obs_t = torch.as_tensor(np.asarray(obs_batch, dtype=np.float32), device=self.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        if self.algorithm == 'q_function':
            adv = self._generate_q_function(obs_t)
            return adv.detach().cpu().numpy().astype(np.float32)
        if self.algorithm == 'critic_v':
            adv = self._generate_critic_value(obs_t)
            return adv.detach().cpu().numpy().astype(np.float32)
        if self.algorithm == 'action':
            adv = self._generate_action_mad(obs_t)
            return adv.detach().cpu().numpy().astype(np.float32)
        if self.algorithm == 'advpolicy':
            adv = self._generate_advpolicy(obs_t, contexts=None)
            return adv.detach().cpu().numpy().astype(np.float32)
        if target_actions is None:
            with torch.no_grad():
                ref = self._actor_mean_action(obs_t).detach()
            targeted = False
        else:
            ref = torch.as_tensor(np.asarray(target_actions, dtype=np.float32), device=self.device)
            if ref.ndim == 1:
                ref = ref.unsqueeze(1)
            targeted = True
        adv = self._generate_policy_alignment(obs_t, ref, targeted=targeted)
        return adv.detach().cpu().numpy().astype(np.float32)

    def get_adversarial_example(self, obs, target_action: float | None = None) -> np.ndarray:
        target = None if target_action is None else np.asarray([target_action], dtype=np.float32)
        return self.attack(to_numpy_1d(obs), target_actions=target).reshape(-1)

    def attack_with_context(self, obs_batch: np.ndarray, contexts: list[AttackContext]) -> np.ndarray:
        if self.algorithm != 'advpolicy':
            return self.attack(obs_batch)
        obs_t = torch.as_tensor(np.asarray(obs_batch, dtype=np.float32), device=self.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        adv = self._generate_advpolicy(obs_t, contexts=contexts)
        return adv.detach().cpu().numpy().astype(np.float32)

    def _random_start(self, original: torch.Tensor) -> torch.Tensor:
        noise = torch.empty_like(original).uniform_(-self.epsilon, self.epsilon, generator=self.generator)
        noise = noise * self._local_attack_mask(original)
        return self._project_obs(original, original + noise)

    def _randn_like(self, ref: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        return torch.randn(ref.shape, device=ref.device, dtype=ref.dtype, generator=self.generator) * float(scale)

    def _actor_action_stats(self, obs_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self.actor, 'mean_action') and hasattr(self.actor, '_distribution'):
            action_mean = self.actor.mean_action(obs_t)
            _, std, _ = self.actor._distribution(obs_t)
            return action_mean, std
        output = self.actor(obs_t)
        if isinstance(output, (tuple, list)) and len(output) >= 2:
            action_mean = output[0]
            action_std = output[1]
            return action_mean, action_std
        action_mean = output
        action_std = torch.ones_like(action_mean)
        return action_mean, action_std

    def _actor_mean_action(self, obs_t: torch.Tensor) -> torch.Tensor:
        if hasattr(self.actor, 'mean_action'):
            return self.actor.mean_action(obs_t)
        output = self.actor(obs_t)
        if isinstance(output, (tuple, list)) and len(output) >= 1:
            return output[0]
        return output

    def _generate_policy_alignment(self, obs_t: torch.Tensor, ref: torch.Tensor, targeted: bool) -> torch.Tensor:
        original = obs_t.detach().clone()
        image = self._random_start(original)
        loss_fn = nn.MSELoss()
        for _ in range(self.iters):
            image.requires_grad_(True)
            output = self._actor_mean_action(image)
            loss = loss_fn(output, ref)
            grad = torch.autograd.grad(loss, image, retain_graph=False, create_graph=False)[0]
            mask = self._local_attack_mask(original)
            step = (-self.alpha * grad.sign() if targeted else self.alpha * grad.sign()) * mask
            adv = image + step
            eta = torch.clamp(adv - original, min=-self.epsilon, max=self.epsilon) * mask
            image = self._project_obs(original, original + eta).detach()
        return image

    def _generate_q_function(self, obs_t: torch.Tensor) -> torch.Tensor:
        if self.critic is None:
            raise RuntimeError('q_function attack requires critic.')
        clean_obs = obs_t.detach().clone()
        image = self._random_start(clean_obs)
        for _ in range(self.iters):
            image.requires_grad_(True)
            adv_action = self._actor_mean_action(image)
            critic_obs = image if bool(getattr(self.critic, 'uses_state_value', False)) else clean_obs
            q_value = self.critic(critic_obs, adv_action).mean()
            grad = torch.autograd.grad(q_value, image, retain_graph=False, create_graph=False)[0]
            mask = self._local_attack_mask(clean_obs)
            adv = image - self.alpha * grad.sign() * mask
            eta = torch.clamp(adv - clean_obs, min=-self.epsilon, max=self.epsilon) * mask
            image = self._project_obs(clean_obs, clean_obs + eta).detach()
        return image

    def _generate_action_mad(self, obs_t: torch.Tensor) -> torch.Tensor:
        if self.iters <= 0 or self.epsilon <= 0.0:
            return obs_t.detach().clone()
        original = obs_t.detach().clone()
        mask = self._local_attack_mask(original)
        step_eps = float(self.alpha) if self.alpha > 0.0 else float(self.epsilon) / float(max(self.iters, 1))
        step_eps = max(step_eps, 1e-6)
        noise_factor = float(np.sqrt(2.0 * step_eps))
        states = original + self._randn_like(original, noise_factor).sign() * step_eps * mask
        eta = torch.clamp(states - original, min=-self.epsilon, max=self.epsilon) * mask
        states = self._project_obs(original, original + eta).detach()
        with torch.no_grad():
            old_action, old_stdev = self._actor_action_stats(original)
            old_action = old_action.detach()
            old_stdev = old_stdev.detach()
            old_stdev = old_stdev / old_stdev.mean().clamp_min(1e-6)
        for i in range(self.iters):
            states = states.clone().detach().requires_grad_()
            action_mean, _ = self._actor_action_stats(states)
            action_change = (action_mean - old_action) / old_stdev
            action_objective = (action_change * action_change).sum(dim=1)
            grad = torch.autograd.grad(action_objective.sum(), states, retain_graph=False, create_graph=False)[0]
            step_noise = self._randn_like(original, float(np.sqrt(2.0 * step_eps)) / float(i + 2))
            update = (grad + step_noise).sign() * step_eps * mask
            adv = states + update
            eta = torch.clamp(adv - original, min=-self.epsilon, max=self.epsilon) * mask
            states = self._project_obs(original, original + eta).detach()
        return states

    def _generate_critic_value(self, obs_t: torch.Tensor) -> torch.Tensor:
        if self.value_model is None:
            raise RuntimeError('critic_v attack requires value model.')
        clean_obs = obs_t.detach().clone()
        image = self._random_start(clean_obs)
        for _ in range(self.iters):
            image.requires_grad_(True)
            value = self.value_model(image).reshape(-1).mean()
            grad = torch.autograd.grad(value, image, retain_graph=False, create_graph=False)[0]
            mask = self._local_attack_mask(clean_obs)
            adv = image - self.alpha * grad.sign() * mask
            eta = torch.clamp(adv - clean_obs, min=-self.epsilon, max=self.epsilon) * mask
            image = self._project_obs(clean_obs, clean_obs + eta).detach()
        return image

    def _build_adversary_context(self, contexts: list[AttackContext] | None, batch_size: int) -> torch.Tensor | None:
        if self.adversary is None or contexts is None or self.adversary_context_mode == 'none':
            return None
        if len(contexts) != batch_size:
            raise ValueError('advpolicy attack context length mismatch.')
        if self.adversary_context_mode == 'arrival':
            return torch.as_tensor(
                [[float(bool(ctx.is_new_arrival))] for ctx in contexts],
                dtype=torch.float32,
                device=self.device,
            )
        if self.adversary_context_mode == 'station':
            station_ctx = torch.zeros((batch_size, self.adversary_station_count), dtype=torch.float32, device=self.device)
            for row_idx, ctx in enumerate(contexts):
                if 0 <= int(ctx.station) < self.adversary_station_count:
                    station_ctx[row_idx, int(ctx.station)] = 1.0
            return station_ctx
        if self.adversary_context_mode == 'station_is_new_arrival':
            station_ctx = torch.zeros((batch_size, self.adversary_station_count), dtype=torch.float32, device=self.device)
            arrival_ctx = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.device)
            for row_idx, ctx in enumerate(contexts):
                if 0 <= int(ctx.station) < self.adversary_station_count:
                    station_ctx[row_idx, int(ctx.station)] = 1.0
                arrival_ctx[row_idx, 0] = float(bool(ctx.is_new_arrival))
            return torch.cat([station_ctx, arrival_ctx], dim=1)
        return None

    def _generate_advpolicy(self, obs_t: torch.Tensor, contexts: list[AttackContext] | None) -> torch.Tensor:
        if self.adversary is None:
            raise RuntimeError('advpolicy attack requires adversary.')
        clean_obs = obs_t.detach().clone()
        context_t = self._build_adversary_context(contexts, int(clean_obs.shape[0]))
        with torch.no_grad():
            if hasattr(self.adversary, 'sample'):
                sample_out = self.adversary.sample(
                    clean_obs,
                    context=context_t,
                    deterministic=self.adversary_deterministic,
                    temperature=self.adversary_temperature,
                )
                adv_obs = sample_out[0] if isinstance(sample_out, tuple) else sample_out
            else:
                adv_obs = self.adversary(clean_obs, context=context_t)
        mask = self._local_attack_mask(clean_obs)
        delta = torch.clamp(adv_obs - clean_obs, min=-self.epsilon, max=self.epsilon) * mask
        return self._project_obs(clean_obs, clean_obs + delta)


def build_state_attacker(
    actor: torch.nn.Module,
    *,
    device: torch.device,
    algorithm: AttackAlgorithm = 'electhacker',
    epsilon: float | None = None,
    alpha: float | None = None,
    iters: int | None = None,
    seed: int = 42,
    obs_low: np.ndarray | torch.Tensor | None = None,
    obs_high: np.ndarray | torch.Tensor | None = None,
    critic: torch.nn.Module | None = None,
    value_model: torch.nn.Module | None = None,
    adversary: torch.nn.Module | None = None,
    attack_state_scope: str = 'local',
    attack_indices: tuple[int, ...] | list[int] | None = None,
    adversary_temperature: float = 1.0,
    adversary_deterministic: bool = True,
    adversary_context_mode: str = 'none',
    adversary_station_count: int = 9,
    signals_path=None,
    reward_profile=None,
):
    """根据 algorithm 构造攻击器。"""
    canonical_algorithm = canonical_attack_algorithm(str(algorithm))
    return PGDStateAttacker(
        actor,
        device=device,
        algorithm=canonical_algorithm,
        epsilon=epsilon,
        alpha=alpha,
        iters=iters,
        seed=seed,
        obs_low=obs_low,
        obs_high=obs_high,
        critic=critic,
        value_model=value_model,
        adversary=adversary,
        attack_state_scope=attack_state_scope,
        attack_indices=attack_indices,
        adversary_temperature=adversary_temperature,
        adversary_deterministic=adversary_deterministic,
        adversary_context_mode=adversary_context_mode,
        adversary_station_count=adversary_station_count,
    )


def scenario_target(context: AttackContext, obs: np.ndarray) -> float | None:
    """根据 C/F/O 规则给 targeted attack 生成目标动作。"""
    obs = to_numpy_1d(obs)
    if context.scenario == 'C':
        return -1.0 if context.raw_price < context.price_threshold else 1.0
    if context.scenario == 'F':
        if context.is_new_arrival:
            return None
        return context.even_station_target if context.station % 2 == 0 else context.odd_station_target
    if context.scenario == 'O':
        threshold = context.soc_new_threshold if context.is_new_arrival else context.soc_rollout_threshold
        return -1.0 if float(obs[0]) < threshold else 1.0
    raise ValueError(f'未知攻击场景: {context.scenario}')


def _bounded_ratio(attack_ratio: float) -> float:
    return float(np.clip(float(attack_ratio), 0.0, 1.0))


def _uniform_from_key(seed: int, parts: tuple[int, ...]) -> float:
    payload = '|'.join([str(int(seed)), *[str(int(v)) for v in parts]]).encode('utf-8')
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    value = int.from_bytes(digest, byteorder='little', signed=False)
    return float((value >> 11) / float(1 << 53))


def _build_attack_ratio_mask(
    contexts: list[AttackContext],
    *,
    attack_ratio: float,
    attack_scope: AttackScope,
    vehicle_ids: list[int] | np.ndarray | None,
    episode_index: int,
    seed: int,
) -> list[bool]:
    ratio = _bounded_ratio(attack_ratio)
    count = len(contexts)
    if ratio <= 0.0:
        return [False for _ in range(count)]
    if ratio >= 1.0:
        return [True for _ in range(count)]

    if vehicle_ids is None:
        ids = [int(i) for i in range(count)]
    else:
        ids = [int(v) for v in np.asarray(vehicle_ids, dtype=np.int64).reshape(-1)]
        if len(ids) != count:
            raise ValueError('vehicle_ids length must match contexts length.')

    scope = str(attack_scope)
    if scope not in {'obs', 'vehicle', 'window'}:
        raise ValueError(f'Unknown attack_scope: {attack_scope}')

    out: list[bool] = []
    for idx, (ctx, vehicle_id) in enumerate(zip(contexts, ids)):
        if scope == 'obs':
            parts = (int(episode_index), int(vehicle_id), int(ctx.time_index), int(idx))
        elif scope == 'vehicle':
            parts = (int(episode_index), int(vehicle_id))
        else:
            parts = (int(episode_index), int(ctx.time_index))
        out.append(_uniform_from_key(seed, parts) < ratio)
    return out


def attack_batch_by_context(
    attacker: PGDStateAttacker | None,
    obs_batch: list[np.ndarray],
    contexts: list[AttackContext],
    *,
    attack_ratio: float = 1.0,
    attack_scope: AttackScope = 'obs',
    vehicle_ids: list[int] | np.ndarray | None = None,
    episode_indices: list[int] | np.ndarray | None = None,
    episode_index: int = 0,
    seed: int = 42,
) -> tuple[list[np.ndarray], list[bool]]:
    """按上下文批量攻击一组观测。"""
    if attacker is None or not obs_batch:
        return [to_numpy_1d(obs) for obs in obs_batch], [False for _ in obs_batch]

    ratio_mask = _build_attack_ratio_mask(
        contexts,
        attack_ratio=float(attack_ratio),
        attack_scope=attack_scope,
        vehicle_ids=vehicle_ids,
        episode_index=int(episode_index),
        seed=int(seed),
    )
    if not any(ratio_mask):
        return [to_numpy_1d(obs) for obs in obs_batch], [False for _ in obs_batch]

    batch = np.stack([to_numpy_1d(obs) for obs in obs_batch], axis=0)
    if vehicle_ids is None:
        vehicle_id_list = [int(i) for i in range(len(obs_batch))]
    else:
        vehicle_id_list = [int(v) for v in np.asarray(vehicle_ids, dtype=np.int64).reshape(-1)]
        if len(vehicle_id_list) != len(obs_batch):
            raise ValueError('vehicle_ids length must match obs_batch length.')
    if episode_indices is None:
        episode_id_list = [int(episode_index) for _ in obs_batch]
    else:
        episode_id_list = [int(v) for v in np.asarray(episode_indices, dtype=np.int64).reshape(-1)]
        if len(episode_id_list) != len(obs_batch):
            raise ValueError('episode_indices length must match obs_batch length.')
    algorithm = getattr(attacker, 'algorithm', None)
    if hasattr(attacker, 'observe_batch'):
        skipped_indices = [i for i, should_attack in enumerate(ratio_mask) if not should_attack]
        if skipped_indices:
            attacker.observe_batch(
                batch[skipped_indices],
                contexts=[contexts[i] for i in skipped_indices],
                vehicle_ids=[vehicle_id_list[i] for i in skipped_indices],
                episode_indices=[episode_id_list[i] for i in skipped_indices],
            )

    if hasattr(attacker, 'attack_with_metadata'):
        out = [to_numpy_1d(obs).copy() for obs in obs_batch]
        attacked_flags = [False for _ in obs_batch]
        selected_indices = [i for i, should_attack in enumerate(ratio_mask) if should_attack]
        if not selected_indices:
            return out, attacked_flags
        adv = attacker.attack_with_metadata(
            batch[selected_indices],
            contexts=[contexts[i] for i in selected_indices],
            vehicle_ids=[vehicle_id_list[i] for i in selected_indices],
            episode_indices=[episode_id_list[i] for i in selected_indices],
        )
        for local_idx, global_idx in enumerate(selected_indices):
            out[global_idx] = adv[local_idx].reshape(-1)
            delta = to_numpy_1d(out[global_idx]) - to_numpy_1d(batch[global_idx])
            attacked_flags[global_idx] = bool(np.max(np.abs(delta)) > 1e-9)
        return out, attacked_flags

    if algorithm in ('opposite_pgd', 'opposite_fgsm', 'q_function', 'critic_v', 'action', 'advpolicy'):
        out = [to_numpy_1d(obs).copy() for obs in obs_batch]
        attacked_flags = [False for _ in obs_batch]
        selected_indices = [i for i, should_attack in enumerate(ratio_mask) if should_attack]
        if not selected_indices:
            return out, attacked_flags
        selected_batch = batch[selected_indices]
        if algorithm == 'advpolicy' and hasattr(attacker, 'attack_with_context'):
            selected_contexts = [contexts[i] for i in selected_indices]
            adv = attacker.attack_with_context(selected_batch, selected_contexts)
        else:
            adv = attacker.attack(selected_batch, target_actions=None)
        for local_idx, global_idx in enumerate(selected_indices):
            out[global_idx] = adv[local_idx].reshape(-1)
            attacked_flags[global_idx] = True
        return out, attacked_flags

    out = [to_numpy_1d(obs).copy() for obs in obs_batch]
    attacked_flags = [False for _ in obs_batch]
    groups: dict[float, list[int]] = {}
    for idx, (obs, ctx) in enumerate(zip(obs_batch, contexts)):
        if not ratio_mask[idx]:
            continue
        target = scenario_target(ctx, to_numpy_1d(obs))
        if target is None:
            continue
        groups.setdefault(float(target), []).append(idx)
        attacked_flags[idx] = True

    for target, indices in groups.items():
        sub_batch = np.stack([to_numpy_1d(obs_batch[i]) for i in indices], axis=0)
        target_actions = np.full((len(indices), 1), float(target), dtype=np.float32)
        adv = attacker.attack(sub_batch, target_actions=target_actions)
        for local_idx, global_idx in enumerate(indices):
            out[global_idx] = adv[local_idx].reshape(-1)

    return out, attacked_flags
