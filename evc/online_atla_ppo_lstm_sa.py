from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal

from .formal_experimental_long_horizon import build_formal_experimental_long_horizon_attacker
from .long_horizon_attacks import build_long_horizon_attacker, canonical_long_horizon_attack_name
from .merged_attacks import AttackContext, attack_indices_for_state_scope, canonical_attack_state_scope
from .merged_core import (
    ATTACK_DEFAULTS,
    Actor,
    ChargingEnv,
    Critic,
    QueueItem,
    RewardProfile,
    TRAIN_PROFILE,
    canonical_attack_algorithm,
    ensure_dir,
    normalize_result_frame,
    resolve_max_duration_of_stay,
    set_seed,
    to_numpy_1d,
)
from .merged_pipeline import summarize_metrics
from .multiday_schedule import max_duration_across_scenarios, normalize_episode_scenarios, scenario_for_episode
from .robust_bounds import observation_bounds_across_scenarios


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
TANH_EPS = 1e-6


@dataclass
class AtlaPpoLstmSaHistory:
    rows: list[dict]
    eval_rows: list[dict] | None = None


@dataclass
class PendingAgentStep:
    key: tuple[int, int]
    clean_obs: np.ndarray
    adv_obs: np.ndarray
    action: np.ndarray
    old_log_prob: float
    value: float
    station: int
    time_index: int


@dataclass
class PendingAdversaryStep:
    key: tuple[int, int]
    clean_obs: np.ndarray
    raw_delta: np.ndarray
    old_log_prob: float
    value: float
    station: int
    time_index: int


@dataclass
class AgentSession:
    episode_index: int
    vehicle_id: int
    clean_obs: list[np.ndarray] = field(default_factory=list)
    adv_obs: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    old_log_probs: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    next_clean_obs: list[np.ndarray] = field(default_factory=list)
    stations: list[int] = field(default_factory=list)
    time_indices: list[int] = field(default_factory=list)
    truncated: bool = False
    bootstrap_value: float = 0.0

    @property
    def length(self) -> int:
        return len(self.rewards)


@dataclass
class AdversarySession:
    episode_index: int
    vehicle_id: int
    clean_obs: list[np.ndarray] = field(default_factory=list)
    raw_deltas: list[np.ndarray] = field(default_factory=list)
    old_log_probs: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    next_clean_obs: list[np.ndarray] = field(default_factory=list)
    stations: list[int] = field(default_factory=list)
    time_indices: list[int] = field(default_factory=list)
    truncated: bool = False
    bootstrap_value: float = 0.0

    @property
    def length(self) -> int:
        return len(self.rewards)


@dataclass
class AgentRolloutBatch:
    sessions: list[AgentSession]
    day_summaries: list[dict]
    total_steps: int
    alignment_errors: list[str]


@dataclass
class AdversaryRolloutBatch:
    sessions: list[AdversarySession]
    day_summaries: list[dict]
    total_steps: int
    alignment_errors: list[str]


class AtlaLstmSquashedGaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int = 11, action_dim: int = 1, hidden_dim: int = 128, lstm_dim: int = 128) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.lstm_dim = int(lstm_dim)
        self.embed = nn.Linear(self.obs_dim, self.hidden_dim)
        self.lstm = nn.LSTM(self.hidden_dim, self.lstm_dim, num_layers=1, batch_first=True)
        self.mean = nn.Linear(self.lstm_dim, self.action_dim)
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.embed.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(self.embed.bias)
        nn.init.orthogonal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.zeros_(self.lstm.bias_ih_l0)
        nn.init.zeros_(self.lstm.bias_hh_l0)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)

    def _std(self) -> torch.Tensor:
        return torch.exp(torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX))

    def forward_sequence(self, obs_seq: torch.Tensor, hidden=None) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x = F.tanh(self.embed(obs_seq.float()))
        out, hidden = self.lstm(x, hidden)
        mean = self.mean(out)
        std = self._std().view(1, 1, -1).expand_as(mean)
        return mean, std, hidden

    def forward_step(self, obs: torch.Tensor, hidden=None) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if obs.ndim == 1:
            obs = obs.view(1, -1)
        mean, std, hidden = self.forward_sequence(obs.view(obs.shape[0], 1, -1), hidden)
        return mean[:, -1, :], std[:, -1, :], hidden

    def sample_step(self, obs: torch.Tensor, hidden=None, *, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        mean, std, hidden = self.forward_step(obs, hidden)
        if deterministic:
            raw = mean
        else:
            raw = Normal(mean, std).rsample()
        action = torch.tanh(raw)
        log_prob = self.log_prob_from_dist(mean, std, action)
        return action, log_prob, hidden

    @staticmethod
    def _atanh_action(action: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(action, -1.0 + TANH_EPS, 1.0 - TANH_EPS)
        return 0.5 * (torch.log1p(clipped) - torch.log1p(-clipped))

    def log_prob_from_dist(self, mean: torch.Tensor, std: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        raw = self._atanh_action(action)
        normal_log_prob = Normal(mean, std).log_prob(raw).sum(dim=-1)
        correction = torch.log(torch.clamp(1.0 - action.pow(2), min=TANH_EPS)).sum(dim=-1)
        return normal_log_prob - correction

    def sequence_log_prob_entropy(self, obs_seq: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std, _ = self.forward_sequence(obs_seq)
        log_prob = self.log_prob_from_dist(mean, std, actions)
        entropy = Normal(mean, std).entropy().sum(dim=-1)
        return log_prob, entropy, mean, std


class AtlaLstmValueNet(nn.Module):
    def __init__(self, obs_dim: int = 11, hidden_dim: int = 128, lstm_dim: int = 128) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.hidden_dim = int(hidden_dim)
        self.lstm_dim = int(lstm_dim)
        self.embed = nn.Linear(self.obs_dim, self.hidden_dim)
        self.lstm = nn.LSTM(self.hidden_dim, self.lstm_dim, num_layers=1, batch_first=True)
        self.value = nn.Linear(self.lstm_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.embed.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(self.embed.bias)
        nn.init.orthogonal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.zeros_(self.lstm.bias_ih_l0)
        nn.init.zeros_(self.lstm.bias_hh_l0)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.value.bias)

    def forward_sequence(self, obs_seq: torch.Tensor, hidden=None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x = F.tanh(self.embed(obs_seq.float()))
        out, hidden = self.lstm(x, hidden)
        return self.value(out).squeeze(-1), hidden

    def forward_step(self, obs: torch.Tensor, hidden=None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if obs.ndim == 1:
            obs = obs.view(1, -1)
        value, hidden = self.forward_sequence(obs.view(obs.shape[0], 1, -1), hidden)
        return value[:, -1], hidden


class AtlaObsMLPPolicy(nn.Module):
    def __init__(self, obs_dim: int = 11, delta_dim: int = 11, hidden_dim: int = 128) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.delta_dim = int(delta_dim)
        self.net = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.delta_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(self.delta_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.orthogonal_(last.weight, gain=0.01)

    def _std(self) -> torch.Tensor:
        return torch.exp(torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX))

    def dist(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.net(obs.float())
        std = self._std().view(1, -1).expand_as(mean)
        return mean, std

    def sample(
        self,
        obs: torch.Tensor,
        *,
        deterministic: bool = False,
        log_prob_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std = self.dist(obs)
        raw = mean if deterministic else Normal(mean, std).rsample()
        per_dim_log_prob = Normal(mean, std).log_prob(raw)
        if log_prob_mask is not None:
            per_dim_log_prob = per_dim_log_prob * log_prob_mask.view(1, -1)
        log_prob = per_dim_log_prob.sum(dim=-1)
        return raw, log_prob, mean, std

    def log_prob_entropy(
        self,
        obs: torch.Tensor,
        raw_delta: torch.Tensor,
        *,
        log_prob_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std = self.dist(obs)
        per_dim_log_prob = Normal(mean, std).log_prob(raw_delta)
        per_dim_entropy = Normal(mean, std).entropy()
        if log_prob_mask is not None:
            mask = log_prob_mask.view(1, -1)
            per_dim_log_prob = per_dim_log_prob * mask
            per_dim_entropy = per_dim_entropy * mask
        log_prob = per_dim_log_prob.sum(dim=-1)
        entropy = per_dim_entropy.sum(dim=-1)
        return log_prob, entropy


class AtlaObsMLPValueNet(nn.Module):
    def __init__(self, obs_dim: int = 11, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.orthogonal_(last.weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs.float()).squeeze(-1)


class AtlaPpoLstmSaAgent:
    def __init__(
        self,
        *,
        obs_dim: int = 11,
        action_dim: int = 1,
        hidden_dim: int = 128,
        lstm_dim: int = 128,
        adversary_hidden_dim: int = 128,
        device: torch.device,
    ) -> None:
        self.device = device
        self.policy = AtlaLstmSquashedGaussianPolicy(obs_dim, action_dim, hidden_dim, lstm_dim).to(device)
        self.value = AtlaLstmValueNet(obs_dim, hidden_dim, lstm_dim).to(device)
        self.adversary = AtlaObsMLPPolicy(obs_dim, obs_dim, adversary_hidden_dim).to(device)
        self.adversary_value = AtlaObsMLPValueNet(obs_dim, adversary_hidden_dim).to(device)


def _detach_hidden(hidden):
    if hidden is None:
        return None
    return tuple(x.detach() for x in hidden)


def _obs_bounds_tensor(obs_low, obs_high, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    low_t = torch.as_tensor(np.asarray(obs_low, dtype=np.float32), dtype=torch.float32, device=device).view(1, -1)
    high_t = torch.as_tensor(np.asarray(obs_high, dtype=np.float32), dtype=torch.float32, device=device).view(1, -1)
    return low_t, high_t


def _scope_mask(scope: str, obs_dim: int, device: torch.device) -> torch.Tensor:
    canonical = canonical_attack_state_scope(scope)
    mask = torch.zeros(obs_dim, dtype=torch.float32, device=device)
    mask[list(attack_indices_for_state_scope(canonical))] = 1.0
    return mask


def _perturb_from_raw(
    clean_obs: torch.Tensor,
    raw_delta: torch.Tensor,
    *,
    epsilon: float,
    mask: torch.Tensor,
    obs_low: torch.Tensor,
    obs_high: torch.Tensor,
) -> torch.Tensor:
    if clean_obs.ndim == 1:
        clean_obs = clean_obs.view(1, -1)
    if raw_delta.ndim == 1:
        raw_delta = raw_delta.view(1, -1)
    delta = torch.tanh(raw_delta) * float(epsilon) * mask.view(1, -1)
    clipped = torch.maximum(torch.minimum(clean_obs + delta, obs_high), obs_low)
    return clipped * mask.view(1, -1) + clean_obs * (1.0 - mask.view(1, -1))


def _random_attack_obs(
    clean_obs: np.ndarray,
    *,
    epsilon: float,
    mask_np: np.ndarray,
    obs_low: np.ndarray,
    obs_high: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    delta = rng.uniform(-float(epsilon), float(epsilon), size=clean_obs.shape).astype(np.float32) * mask_np
    clipped = np.clip(clean_obs + delta, obs_low, obs_high).astype(np.float32)
    return (clipped * mask_np + clean_obs * (1.0 - mask_np)).astype(np.float32)


def _session_dict_get(mapping: dict[tuple[int, int], AgentSession], key: tuple[int, int]) -> AgentSession:
    if key not in mapping:
        mapping[key] = AgentSession(episode_index=int(key[0]), vehicle_id=int(key[1]))
    return mapping[key]


def _adv_session_dict_get(mapping: dict[tuple[int, int], AdversarySession], key: tuple[int, int]) -> AdversarySession:
    if key not in mapping:
        mapping[key] = AdversarySession(episode_index=int(key[0]), vehicle_id=int(key[1]))
    return mapping[key]


def _mark_agent_truncations(
    sessions: dict[tuple[int, int], AgentSession],
    active_vehicle_ids: list[int],
    episode_index: int,
    policy_value: AtlaLstmValueNet,
    adversary: AtlaObsMLPPolicy,
    value_hiddens: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    epsilon: float,
    mask: torch.Tensor,
    obs_low_t: torch.Tensor,
    obs_high_t: torch.Tensor,
) -> None:
    for vehicle_id in active_vehicle_ids:
        key = (int(episode_index), int(vehicle_id))
        sess = sessions.get(key)
        if sess is None or sess.length == 0 or bool(sess.dones[-1]):
            continue
        next_obs_np = to_numpy_1d(sess.next_clean_obs[-1])
        next_obs_t = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device).view(1, -1)
        with torch.no_grad():
            raw_delta, _, _, _ = adversary.sample(next_obs_t, deterministic=False)
            next_adv_obs = _perturb_from_raw(next_obs_t, raw_delta, epsilon=epsilon, mask=mask, obs_low=obs_low_t, obs_high=obs_high_t)
            bootstrap, _ = policy_value.forward_step(next_adv_obs, _detach_hidden(value_hiddens.get(key)))
        sess.truncated = True
        sess.bootstrap_value = float(bootstrap.detach().cpu().item())


def _mark_adversary_truncations(
    sessions: dict[tuple[int, int], AdversarySession],
    active_vehicle_ids: list[int],
    episode_index: int,
    adversary_value: AtlaObsMLPValueNet,
    *,
    device: torch.device,
) -> None:
    for vehicle_id in active_vehicle_ids:
        key = (int(episode_index), int(vehicle_id))
        sess = sessions.get(key)
        if sess is None or sess.length == 0 or bool(sess.dones[-1]):
            continue
        next_obs_np = to_numpy_1d(sess.next_clean_obs[-1])
        next_obs_t = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device).view(1, -1)
        with torch.no_grad():
            bootstrap = adversary_value(next_obs_t)
        sess.truncated = True
        sess.bootstrap_value = float(bootstrap.detach().cpu().item())


def collect_agent_rollouts(
    arrivals: pd.DataFrame,
    signals_path,
    agent: AtlaPpoLstmSaAgent,
    *,
    reward_profile: RewardProfile = TRAIN_PROFILE,
    epsilon: float = 0.15,
    attack_state_scope: str = 'local',
    phase_steps: int = 2048,
    max_episodes: int | None = None,
    start_episode_index: int = 0,
    obs_low: np.ndarray | None = None,
    obs_high: np.ndarray | None = None,
) -> AgentRolloutBatch:
    device = agent.device
    env_for_bounds = ChargingEnv(signals_path=signals_path, reward_profile=reward_profile)
    if obs_low is None or obs_high is None:
        max_duration = resolve_max_duration_of_stay(arrivals)
        obs_low, obs_high = env_for_bounds.observation_bounds(max_duration_of_stay=max_duration)
    obs_low_t, obs_high_t = _obs_bounds_tensor(obs_low, obs_high, device)
    mask = _scope_mask(attack_state_scope, env_for_bounds.obs_dim, device)

    agent.policy.eval()
    agent.value.eval()
    agent.adversary.eval()
    sessions: dict[tuple[int, int], AgentSession] = {}
    day_summaries: list[dict] = []
    alignment_errors: list[str] = []
    total_steps = 0
    episode_index = int(start_episode_index)

    while total_steps < int(phase_steps) and (
        max_episodes is None or len(day_summaries) < int(max_episodes)
    ):
        env = ChargingEnv(signals_path=signals_path, reward_profile=reward_profile)
        env.reset()
        idx = 0
        active: list[QueueItem] = []
        active_vehicle_ids: list[int] = []
        policy_hiddens: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        value_hiddens: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

        while env.t < env.horizon:
            pending: list[PendingAgentStep] = []
            new_states: list[np.ndarray] = []
            new_stations: list[int] = []
            new_vehicle_ids: list[int] = []
            while idx < len(arrivals) and int(arrivals.loc[idx, 'Arrive_time']) == env.t:
                new_states.append(env.build_initial_obs(int(arrivals.loc[idx, 'Duration_of_stay'])))
                new_stations.append(int(arrivals.loc[idx, 'Station']))
                new_vehicle_ids.append(int(idx))
                idx += 1

            for clean_obs, station, vehicle_id in zip(new_states, new_stations, new_vehicle_ids):
                key = (episode_index, int(vehicle_id))
                clean_t = torch.as_tensor(to_numpy_1d(clean_obs), dtype=torch.float32, device=device).view(1, -1)
                with torch.no_grad():
                    raw_delta, _, _, _ = agent.adversary.sample(clean_t, deterministic=False)
                    adv_obs_t = _perturb_from_raw(clean_t, raw_delta, epsilon=epsilon, mask=mask, obs_low=obs_low_t, obs_high=obs_high_t)
                    action_t, log_prob_t, new_policy_hidden = agent.policy.sample_step(adv_obs_t, policy_hiddens.get(key), deterministic=False)
                    value_t, new_value_hidden = agent.value.forward_step(adv_obs_t, value_hiddens.get(key))
                policy_hiddens[key] = _detach_hidden(new_policy_hidden)
                value_hiddens[key] = _detach_hidden(new_value_hidden)
                action_np = action_t.detach().cpu().numpy().reshape(-1).astype(np.float32)
                adv_obs_np = adv_obs_t.detach().cpu().numpy().reshape(-1).astype(np.float32)
                pending.append(
                    PendingAgentStep(
                        key=key,
                        clean_obs=to_numpy_1d(clean_obs),
                        adv_obs=adv_obs_np,
                        action=action_np,
                        old_log_prob=float(log_prob_t.detach().cpu().item()),
                        value=float(value_t.detach().cpu().item()),
                        station=int(station),
                        time_index=int(env.t),
                    )
                )
                env.enqueue(clean_obs, action_np, int(station))

            if active:
                active_states = [item.obs for item in active]
                active_stations = [item.station for item in active]
                for clean_obs, station, vehicle_id in zip(active_states, active_stations, active_vehicle_ids):
                    key = (episode_index, int(vehicle_id))
                    clean_t = torch.as_tensor(to_numpy_1d(clean_obs), dtype=torch.float32, device=device).view(1, -1)
                    with torch.no_grad():
                        raw_delta, _, _, _ = agent.adversary.sample(clean_t, deterministic=False)
                        adv_obs_t = _perturb_from_raw(clean_t, raw_delta, epsilon=epsilon, mask=mask, obs_low=obs_low_t, obs_high=obs_high_t)
                        action_t, log_prob_t, new_policy_hidden = agent.policy.sample_step(adv_obs_t, policy_hiddens.get(key), deterministic=False)
                        value_t, new_value_hidden = agent.value.forward_step(adv_obs_t, value_hiddens.get(key))
                    policy_hiddens[key] = _detach_hidden(new_policy_hidden)
                    value_hiddens[key] = _detach_hidden(new_value_hidden)
                    action_np = action_t.detach().cpu().numpy().reshape(-1).astype(np.float32)
                    adv_obs_np = adv_obs_t.detach().cpu().numpy().reshape(-1).astype(np.float32)
                    pending.append(
                        PendingAgentStep(
                            key=key,
                            clean_obs=to_numpy_1d(clean_obs),
                            adv_obs=adv_obs_np,
                            action=action_np,
                            old_log_prob=float(log_prob_t.detach().cpu().item()),
                            value=float(value_t.detach().cpu().item()),
                            station=int(station),
                            time_index=int(env.t),
                        )
                    )
                    env.enqueue(clean_obs, action_np, int(station))

            step_vehicle_ids = new_vehicle_ids + active_vehicle_ids
            transitions, next_active, _ = env.step()
            if len(transitions) != len(pending) or len(step_vehicle_ids) != len(transitions):
                alignment_errors.append(
                    f'episode={episode_index} t={env.t - 1} pending={len(pending)} transitions={len(transitions)} ids={len(step_vehicle_ids)}'
                )
            for vehicle_id, pending_step, tr in zip(step_vehicle_ids, pending, transitions):
                if int(vehicle_id) != int(pending_step.key[1]):
                    alignment_errors.append(
                        f'episode={episode_index} t={pending_step.time_index} vehicle_id mismatch pending={pending_step.key[1]} transition={vehicle_id}'
                    )
                sess = _session_dict_get(sessions, pending_step.key)
                sess.clean_obs.append(pending_step.clean_obs.copy())
                sess.adv_obs.append(pending_step.adv_obs.copy())
                sess.actions.append(pending_step.action.copy())
                sess.old_log_probs.append(float(pending_step.old_log_prob))
                sess.rewards.append(float(tr.reward))
                sess.values.append(float(pending_step.value))
                sess.dones.append(bool(tr.done))
                sess.next_clean_obs.append(to_numpy_1d(tr.next_obs).copy())
                sess.stations.append(int(pending_step.station))
                sess.time_indices.append(int(pending_step.time_index))
                total_steps += 1
                if bool(tr.done):
                    policy_hiddens.pop(pending_step.key, None)
                    value_hiddens.pop(pending_step.key, None)

            active = next_active
            active_vehicle_ids = [vid for vid, tr in zip(step_vehicle_ids, transitions) if not bool(tr.done)]

        _mark_agent_truncations(
            sessions,
            active_vehicle_ids,
            episode_index,
            agent.value,
            agent.adversary,
            value_hiddens,
            device=device,
            epsilon=epsilon,
            mask=mask,
            obs_low_t=obs_low_t,
            obs_high_t=obs_high_t,
        )
        day_summary = summarize_metrics(env.metrics, 'agent_phase')
        day_summary['episode_index'] = int(episode_index)
        day_summary['truncated_sessions'] = int(sum(1 for vid in active_vehicle_ids if (episode_index, int(vid)) in sessions))
        day_summaries.append(day_summary)
        episode_index += 1

    ordered_sessions = [sess for _, sess in sorted(sessions.items(), key=lambda kv: kv[0]) if sess.length > 0]
    alignment_errors.extend(_validate_agent_sessions(ordered_sessions))
    return AgentRolloutBatch(ordered_sessions, day_summaries, int(total_steps), alignment_errors)


def collect_adversary_rollouts(
    arrivals: pd.DataFrame,
    signals_path,
    agent: AtlaPpoLstmSaAgent,
    *,
    reward_profile: RewardProfile = TRAIN_PROFILE,
    epsilon: float = 0.15,
    attack_state_scope: str = 'local',
    phase_steps: int = 2048,
    max_episodes: int | None = None,
    start_episode_index: int = 0,
    obs_low: np.ndarray | None = None,
    obs_high: np.ndarray | None = None,
    agent_deterministic: bool = False,
) -> AdversaryRolloutBatch:
    device = agent.device
    env_for_bounds = ChargingEnv(signals_path=signals_path, reward_profile=reward_profile)
    if obs_low is None or obs_high is None:
        max_duration = resolve_max_duration_of_stay(arrivals)
        obs_low, obs_high = env_for_bounds.observation_bounds(max_duration_of_stay=max_duration)
    obs_low_t, obs_high_t = _obs_bounds_tensor(obs_low, obs_high, device)
    mask = _scope_mask(attack_state_scope, env_for_bounds.obs_dim, device)

    agent.policy.eval()
    agent.adversary.train(False)
    agent.adversary_value.eval()
    sessions: dict[tuple[int, int], AdversarySession] = {}
    day_summaries: list[dict] = []
    alignment_errors: list[str] = []
    total_steps = 0
    episode_index = int(start_episode_index)

    while total_steps < int(phase_steps) and (
        max_episodes is None or len(day_summaries) < int(max_episodes)
    ):
        env = ChargingEnv(signals_path=signals_path, reward_profile=reward_profile)
        env.reset()
        idx = 0
        active: list[QueueItem] = []
        active_vehicle_ids: list[int] = []
        policy_hiddens: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

        while env.t < env.horizon:
            pending: list[PendingAdversaryStep] = []
            new_states: list[np.ndarray] = []
            new_stations: list[int] = []
            new_vehicle_ids: list[int] = []
            while idx < len(arrivals) and int(arrivals.loc[idx, 'Arrive_time']) == env.t:
                new_states.append(env.build_initial_obs(int(arrivals.loc[idx, 'Duration_of_stay'])))
                new_stations.append(int(arrivals.loc[idx, 'Station']))
                new_vehicle_ids.append(int(idx))
                idx += 1

            for clean_obs, station, vehicle_id in zip(new_states, new_stations, new_vehicle_ids):
                key = (episode_index, int(vehicle_id))
                clean_t = torch.as_tensor(to_numpy_1d(clean_obs), dtype=torch.float32, device=device).view(1, -1)
                with torch.no_grad():
                    raw_delta, adv_log_prob, _, _ = agent.adversary.sample(clean_t, deterministic=False, log_prob_mask=mask)
                    adv_value = agent.adversary_value(clean_t)
                    adv_obs_t = _perturb_from_raw(clean_t, raw_delta, epsilon=epsilon, mask=mask, obs_low=obs_low_t, obs_high=obs_high_t)
                    action_t, _, new_policy_hidden = agent.policy.sample_step(adv_obs_t, policy_hiddens.get(key), deterministic=agent_deterministic)
                policy_hiddens[key] = _detach_hidden(new_policy_hidden)
                action_np = action_t.detach().cpu().numpy().reshape(-1).astype(np.float32)
                pending.append(
                    PendingAdversaryStep(
                        key=key,
                        clean_obs=to_numpy_1d(clean_obs),
                        raw_delta=raw_delta.detach().cpu().numpy().reshape(-1).astype(np.float32),
                        old_log_prob=float(adv_log_prob.detach().cpu().item()),
                        value=float(adv_value.detach().cpu().item()),
                        station=int(station),
                        time_index=int(env.t),
                    )
                )
                env.enqueue(clean_obs, action_np, int(station))

            if active:
                active_states = [item.obs for item in active]
                active_stations = [item.station for item in active]
                for clean_obs, station, vehicle_id in zip(active_states, active_stations, active_vehicle_ids):
                    key = (episode_index, int(vehicle_id))
                    clean_t = torch.as_tensor(to_numpy_1d(clean_obs), dtype=torch.float32, device=device).view(1, -1)
                    with torch.no_grad():
                        raw_delta, adv_log_prob, _, _ = agent.adversary.sample(clean_t, deterministic=False, log_prob_mask=mask)
                        adv_value = agent.adversary_value(clean_t)
                        adv_obs_t = _perturb_from_raw(clean_t, raw_delta, epsilon=epsilon, mask=mask, obs_low=obs_low_t, obs_high=obs_high_t)
                        action_t, _, new_policy_hidden = agent.policy.sample_step(adv_obs_t, policy_hiddens.get(key), deterministic=agent_deterministic)
                    policy_hiddens[key] = _detach_hidden(new_policy_hidden)
                    action_np = action_t.detach().cpu().numpy().reshape(-1).astype(np.float32)
                    pending.append(
                        PendingAdversaryStep(
                            key=key,
                            clean_obs=to_numpy_1d(clean_obs),
                            raw_delta=raw_delta.detach().cpu().numpy().reshape(-1).astype(np.float32),
                            old_log_prob=float(adv_log_prob.detach().cpu().item()),
                            value=float(adv_value.detach().cpu().item()),
                            station=int(station),
                            time_index=int(env.t),
                        )
                    )
                    env.enqueue(clean_obs, action_np, int(station))

            step_vehicle_ids = new_vehicle_ids + active_vehicle_ids
            transitions, next_active, _ = env.step()
            if len(transitions) != len(pending) or len(step_vehicle_ids) != len(transitions):
                alignment_errors.append(
                    f'episode={episode_index} t={env.t - 1} pending={len(pending)} transitions={len(transitions)} ids={len(step_vehicle_ids)}'
                )
            for vehicle_id, pending_step, tr in zip(step_vehicle_ids, pending, transitions):
                if int(vehicle_id) != int(pending_step.key[1]):
                    alignment_errors.append(
                        f'episode={episode_index} t={pending_step.time_index} vehicle_id mismatch pending={pending_step.key[1]} transition={vehicle_id}'
                    )
                sess = _adv_session_dict_get(sessions, pending_step.key)
                sess.clean_obs.append(pending_step.clean_obs.copy())
                sess.raw_deltas.append(pending_step.raw_delta.copy())
                sess.old_log_probs.append(float(pending_step.old_log_prob))
                sess.rewards.append(float(-tr.reward))
                sess.values.append(float(pending_step.value))
                sess.dones.append(bool(tr.done))
                sess.next_clean_obs.append(to_numpy_1d(tr.next_obs).copy())
                sess.stations.append(int(pending_step.station))
                sess.time_indices.append(int(pending_step.time_index))
                total_steps += 1
                if bool(tr.done):
                    policy_hiddens.pop(pending_step.key, None)

            active = next_active
            active_vehicle_ids = [vid for vid, tr in zip(step_vehicle_ids, transitions) if not bool(tr.done)]

        _mark_adversary_truncations(
            sessions,
            active_vehicle_ids,
            episode_index,
            agent.adversary_value,
            device=device,
        )
        day_summary = summarize_metrics(env.metrics, 'adversary_phase')
        day_summary['episode_index'] = int(episode_index)
        day_summary['truncated_sessions'] = int(sum(1 for vid in active_vehicle_ids if (episode_index, int(vid)) in sessions))
        day_summaries.append(day_summary)
        episode_index += 1

    ordered_sessions = [sess for _, sess in sorted(sessions.items(), key=lambda kv: kv[0]) if sess.length > 0]
    alignment_errors.extend(_validate_adversary_sessions(ordered_sessions))
    return AdversaryRolloutBatch(ordered_sessions, day_summaries, int(total_steps), alignment_errors)


def _compute_gae_for_session(rewards: list[float], values: list[float], dones: list[bool], bootstrap_value: float, gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(rewards)
    advantages = np.zeros((n,), dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        if t == n - 1:
            next_nonterminal = 0.0 if bool(dones[t]) else 1.0
            next_value = float(bootstrap_value)
        else:
            next_nonterminal = 0.0 if bool(dones[t]) else 1.0
            next_value = float(values[t + 1])
        delta = float(rewards[t]) + float(gamma) * next_value * next_nonterminal - float(values[t])
        last_gae = delta + float(gamma) * float(gae_lambda) * next_nonterminal * last_gae
        advantages[t] = float(last_gae)
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns.astype(np.float32)


def _pad_agent_sessions(sessions: list[AgentSession], device: torch.device, gamma: float, gae_lambda: float) -> dict[str, torch.Tensor]:
    if not sessions:
        raise ValueError('Cannot pad an empty agent session batch.')
    max_len = max(sess.length for sess in sessions)
    batch = len(sessions)
    obs = np.zeros((batch, max_len, 11), dtype=np.float32)
    clean_obs = np.zeros((batch, max_len, 11), dtype=np.float32)
    actions = np.zeros((batch, max_len, 1), dtype=np.float32)
    rewards_pad = np.zeros((batch, max_len), dtype=np.float32)
    old_log_probs = np.zeros((batch, max_len), dtype=np.float32)
    values = np.zeros((batch, max_len), dtype=np.float32)
    advantages = np.zeros((batch, max_len), dtype=np.float32)
    returns = np.zeros((batch, max_len), dtype=np.float32)
    mask = np.zeros((batch, max_len), dtype=np.float32)
    dones = np.zeros((batch, max_len), dtype=np.float32)
    time_indices = np.full((batch, max_len), -1, dtype=np.int64)
    for b, sess in enumerate(sessions):
        adv, ret = _compute_gae_for_session(sess.rewards, sess.values, sess.dones, sess.bootstrap_value, gamma, gae_lambda)
        n = sess.length
        obs[b, :n] = np.asarray(sess.adv_obs, dtype=np.float32).reshape(n, 11)
        clean_obs[b, :n] = np.asarray(sess.clean_obs, dtype=np.float32).reshape(n, 11)
        actions[b, :n] = np.asarray(sess.actions, dtype=np.float32).reshape(n, 1)
        rewards_pad[b, :n] = np.asarray(sess.rewards, dtype=np.float32).reshape(n)
        old_log_probs[b, :n] = np.asarray(sess.old_log_probs, dtype=np.float32).reshape(n)
        values[b, :n] = np.asarray(sess.values, dtype=np.float32).reshape(n)
        advantages[b, :n] = adv
        returns[b, :n] = ret
        mask[b, :n] = 1.0
        dones[b, :n] = np.asarray(sess.dones, dtype=np.float32).reshape(n)
        time_indices[b, :n] = np.asarray(sess.time_indices, dtype=np.int64).reshape(n)
    adv_valid = advantages[mask > 0.5]
    adv_mean = float(np.mean(adv_valid)) if adv_valid.size else 0.0
    adv_std = float(np.std(adv_valid)) if adv_valid.size else 1.0
    advantages = (advantages - adv_mean) / max(adv_std, 1e-8)
    return {
        'obs': torch.as_tensor(obs, dtype=torch.float32, device=device),
        'clean_obs': torch.as_tensor(clean_obs, dtype=torch.float32, device=device),
        'actions': torch.as_tensor(actions, dtype=torch.float32, device=device),
        'rewards': torch.as_tensor(rewards_pad, dtype=torch.float32, device=device),
        'old_log_probs': torch.as_tensor(old_log_probs, dtype=torch.float32, device=device),
        'returns': torch.as_tensor(returns, dtype=torch.float32, device=device),
        'advantages': torch.as_tensor(advantages, dtype=torch.float32, device=device),
        'old_values': torch.as_tensor(values, dtype=torch.float32, device=device),
        'mask': torch.as_tensor(mask, dtype=torch.float32, device=device),
        'dones': torch.as_tensor(dones, dtype=torch.float32, device=device),
        'time_indices': torch.as_tensor(time_indices, dtype=torch.long, device=device),
    }


def _flatten_adversary_sessions(sessions: list[AdversarySession], device: torch.device, gamma: float, gae_lambda: float) -> dict[str, torch.Tensor]:
    clean_obs = []
    raw_deltas = []
    old_log_probs = []
    values = []
    advantages = []
    returns = []
    time_indices = []
    for sess in sessions:
        adv, ret = _compute_gae_for_session(sess.rewards, sess.values, sess.dones, sess.bootstrap_value, gamma, gae_lambda)
        clean_obs.extend(sess.clean_obs)
        raw_deltas.extend(sess.raw_deltas)
        old_log_probs.extend(sess.old_log_probs)
        values.extend(sess.values)
        advantages.extend(adv.tolist())
        returns.extend(ret.tolist())
        time_indices.extend(sess.time_indices)
    if not clean_obs:
        raise ValueError('Cannot flatten an empty adversary session batch.')
    adv_arr = np.asarray(advantages, dtype=np.float32)
    adv_arr = (adv_arr - float(adv_arr.mean())) / max(float(adv_arr.std()), 1e-8)
    return {
        'obs': torch.as_tensor(np.asarray(clean_obs, dtype=np.float32).reshape(-1, 11), dtype=torch.float32, device=device),
        'raw_deltas': torch.as_tensor(np.asarray(raw_deltas, dtype=np.float32).reshape(-1, 11), dtype=torch.float32, device=device),
        'old_log_probs': torch.as_tensor(np.asarray(old_log_probs, dtype=np.float32).reshape(-1), dtype=torch.float32, device=device),
        'returns': torch.as_tensor(np.asarray(returns, dtype=np.float32).reshape(-1), dtype=torch.float32, device=device),
        'advantages': torch.as_tensor(adv_arr.reshape(-1), dtype=torch.float32, device=device),
        'old_values': torch.as_tensor(np.asarray(values, dtype=np.float32).reshape(-1), dtype=torch.float32, device=device),
        'time_indices': torch.as_tensor(np.asarray(time_indices, dtype=np.int64).reshape(-1), dtype=torch.long, device=device),
    }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / torch.clamp(mask.sum(), min=1.0)


def _gaussian_kl(mean_p: torch.Tensor, std_p: torch.Tensor, mean_q: torch.Tensor, std_q: torch.Tensor) -> torch.Tensor:
    var_p = std_p.pow(2)
    var_q = std_q.pow(2)
    return 0.5 * (
        ((var_p + (mean_q - mean_p).pow(2)) / torch.clamp(var_q, min=1e-8)).sum(dim=-1)
        - mean_p.shape[-1]
        + (torch.log(torch.clamp(var_q, min=1e-8)) - torch.log(torch.clamp(var_p, min=1e-8))).sum(dim=-1)
    )


def _sa_reg_kl(
    policy: AtlaLstmSquashedGaussianPolicy,
    obs: torch.Tensor,
    mask: torch.Tensor,
    *,
    epsilon: float,
    steps: int,
    attack_mask: torch.Tensor,
    obs_low: torch.Tensor,
    obs_high: torch.Tensor,
) -> torch.Tensor:
    if float(epsilon) <= 0.0 or int(steps) <= 0:
        return obs.new_tensor(0.0)
    with torch.no_grad():
        base_mean, base_std, _ = policy.forward_sequence(obs)
        base_mean = base_mean.detach()
        base_std = base_std.detach()
    attack_mask_view = attack_mask.view(1, 1, -1)
    obs_low_view = obs_low.view(1, 1, -1)
    obs_high_view = obs_high.view(1, 1, -1)
    time_mask = mask.view(mask.shape[0], mask.shape[1], 1)
    step_eps = float(epsilon) / max(int(steps), 1)
    noise = torch.empty_like(obs).uniform_(-step_eps, step_eps) * attack_mask_view * time_mask
    var_obs = obs + noise
    var_obs = torch.maximum(torch.minimum(var_obs, obs + float(epsilon) * attack_mask_view), obs - float(epsilon) * attack_mask_view)
    var_obs = torch.maximum(torch.minimum(var_obs, obs_high_view), obs_low_view)
    var_obs = var_obs * attack_mask_view + obs * (1.0 - attack_mask_view)
    var_obs = (var_obs * time_mask + obs * (1.0 - time_mask)).detach()

    for i in range(int(steps)):
        var_obs = var_obs.detach().requires_grad_(True)
        mean_q, std_q, _ = policy.forward_sequence(var_obs)
        kl = _masked_mean(_gaussian_kl(base_mean, base_std, mean_q, std_q), mask)
        grad = torch.autograd.grad(kl, var_obs, retain_graph=False, create_graph=False)[0]
        beta = 1e-5
        noise_factor = math.sqrt(2.0 * step_eps * beta) / float(i + 2)
        update = (grad + torch.randn_like(var_obs) * noise_factor).sign() * step_eps
        var_obs = var_obs.detach() + update * attack_mask_view * time_mask
        var_obs = torch.maximum(torch.minimum(var_obs, obs + float(epsilon) * attack_mask_view), obs - float(epsilon) * attack_mask_view)
        var_obs = torch.maximum(torch.minimum(var_obs, obs_high_view), obs_low_view)
        var_obs = var_obs * attack_mask_view + obs * (1.0 - attack_mask_view)
        var_obs = var_obs * time_mask + obs * (1.0 - time_mask)

    mean_q, std_q, _ = policy.forward_sequence(var_obs.detach())
    return _masked_mean(_gaussian_kl(base_mean, base_std, mean_q, std_q), mask)


def distill_atla_lstm_policy(
    agent: AtlaPpoLstmSaAgent,
    teacher_actor: Actor,
    clean_inputs: np.ndarray,
    episode_indices: np.ndarray,
    vehicle_ids: np.ndarray,
    time_indices: np.ndarray,
    *,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 3e-4,
    max_sessions: int | None = None,
) -> dict[str, float]:
    """Warm-start the recurrent policy from the shared clean DDPG actor."""
    clean = np.asarray(clean_inputs, dtype=np.float32).reshape(-1, 11)
    episodes = np.asarray(episode_indices, dtype=np.int64).reshape(-1)
    vehicles = np.asarray(vehicle_ids, dtype=np.int64).reshape(-1)
    times = np.asarray(time_indices, dtype=np.int64).reshape(-1)
    if not (len(clean) == len(episodes) == len(vehicles) == len(times)):
        raise ValueError('ATLA distillation arrays must have the same length.')

    grouped: dict[tuple[int, int], list[int]] = {}
    for index, key in enumerate(zip(episodes.tolist(), vehicles.tolist())):
        grouped.setdefault((int(key[0]), int(key[1])), []).append(index)
    sequences: list[np.ndarray] = []
    for indices in grouped.values():
        ordered = sorted(indices, key=lambda idx: (int(times[idx]), int(idx)))
        if ordered:
            sequences.append(clean[np.asarray(ordered, dtype=np.int64)])
    if max_sessions is not None and len(sequences) > int(max_sessions):
        subset_rng = np.random.default_rng(42)
        selected = subset_rng.choice(len(sequences), size=int(max_sessions), replace=False)
        sequences = [sequences[int(index)] for index in selected]
    if not sequences or int(epochs) <= 0:
        return {'distill_epochs': 0.0, 'distill_sessions': float(len(sequences)), 'distill_loss': float('nan')}

    teacher = teacher_actor.to(agent.device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(agent.policy.parameters(), lr=float(lr), eps=1e-5)
    rng = np.random.default_rng(42)
    final_loss = float('nan')
    agent.policy.train()
    for _ in range(int(epochs)):
        order = rng.permutation(len(sequences))
        losses: list[float] = []
        for start in range(0, len(order), max(int(batch_size), 1)):
            selected = [sequences[int(i)] for i in order[start:start + max(int(batch_size), 1)]]
            max_len = max(len(sequence) for sequence in selected)
            obs = np.zeros((len(selected), max_len, 11), dtype=np.float32)
            mask = np.zeros((len(selected), max_len), dtype=np.float32)
            for row, sequence in enumerate(selected):
                obs[row, :len(sequence)] = sequence
                mask[row, :len(sequence)] = 1.0
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
            mask_t = torch.as_tensor(mask, dtype=torch.float32, device=agent.device)
            with torch.no_grad():
                teacher_actions = teacher(obs_t.reshape(-1, 11)).reshape(len(selected), max_len, -1)
            mean, _, _ = agent.policy.forward_sequence(obs_t)
            student_actions = torch.tanh(mean)
            per_step = torch.mean((student_actions - teacher_actions) ** 2, dim=-1)
            loss = _masked_mean(per_step, mask_t)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.policy.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        final_loss = float(np.mean(losses)) if losses else float('nan')
    agent.policy.eval()
    return {
        'distill_epochs': float(epochs),
        'distill_sessions': float(len(sequences)),
        'distill_loss': final_loss,
    }


def update_agent_ppo(
    agent: AtlaPpoLstmSaAgent,
    batch: AgentRolloutBatch,
    policy_optimizer: torch.optim.Optimizer,
    value_optimizer: torch.optim.Optimizer,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    ppo_epochs: int = 10,
    num_minibatches: int = 32,
    value_coef: float = 0.5,
    entropy_coeff: float = 3e-4,
    sa_reg_weight: float = 0.1,
    sa_reg_steps: int = 2,
    sa_reg_epsilon: float = 0.15,
    anchor_actor: Actor | None = None,
    anchor_reg_weight: float = 0.0,
    attack_state_scope: str = 'local',
    obs_low: np.ndarray | None = None,
    obs_high: np.ndarray | None = None,
    max_grad_norm: float = 0.5,
) -> dict:
    device = agent.device
    data = _pad_agent_sessions(batch.sessions, device, gamma, gae_lambda)
    if obs_low is None or obs_high is None:
        env = ChargingEnv()
        obs_low, obs_high = env.observation_bounds()
    obs_low_t = torch.as_tensor(np.asarray(obs_low, dtype=np.float32), dtype=torch.float32, device=device).view(1, -1)
    obs_high_t = torch.as_tensor(np.asarray(obs_high, dtype=np.float32), dtype=torch.float32, device=device).view(1, -1)
    attack_mask = _scope_mask(attack_state_scope, 11, device)

    agent.policy.train()
    agent.value.train()
    n_sessions = int(data['obs'].shape[0])
    minibatches = max(1, min(int(num_minibatches), n_sessions))
    stats: dict[str, float] = {}

    for _ in range(int(ppo_epochs)):
        indices = torch.randperm(n_sessions, device=device)
        for split in torch.chunk(indices, minibatches):
            obs = data['obs'][split]
            actions = data['actions'][split]
            old_log_probs = data['old_log_probs'][split]
            returns = data['returns'][split]
            advantages = data['advantages'][split]
            mask = data['mask'][split]

            new_log_probs, entropy, policy_mean, _ = agent.policy.sequence_log_prob_entropy(obs, actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps)) * advantages
            policy_loss = -_masked_mean(torch.minimum(unclipped, clipped), mask)
            entropy_loss = -float(entropy_coeff) * _masked_mean(entropy, mask)
            if float(sa_reg_weight) > 0.0 and int(sa_reg_steps) > 0:
                sa_kl = _sa_reg_kl(
                    agent.policy,
                    obs,
                    mask,
                    epsilon=float(sa_reg_epsilon),
                    steps=int(sa_reg_steps),
                    attack_mask=attack_mask,
                    obs_low=obs_low_t,
                    obs_high=obs_high_t,
                )
            else:
                sa_kl = obs.new_tensor(0.0)
            anchor_loss = obs.new_tensor(0.0)
            if anchor_actor is not None and float(anchor_reg_weight) > 0.0:
                clean_obs = data['clean_obs'][split]
                with torch.no_grad():
                    teacher_actions = anchor_actor(clean_obs.reshape(-1, 11)).reshape(
                        clean_obs.shape[0], clean_obs.shape[1], -1
                    )
                policy_actions = torch.tanh(policy_mean)
                anchor_loss = _masked_mean(
                    torch.mean((policy_actions - teacher_actions) ** 2, dim=-1),
                    mask,
                )
            value_pred, _ = agent.value.forward_sequence(obs)
            value_loss = 0.5 * _masked_mean((value_pred - returns).pow(2), mask)
            loss = (
                policy_loss
                + float(value_coef) * value_loss
                + entropy_loss
                + float(sa_reg_weight) * sa_kl
                + float(anchor_reg_weight) * anchor_loss
            )

            policy_optimizer.zero_grad(set_to_none=True)
            value_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm and float(max_grad_norm) > 0.0:
                nn.utils.clip_grad_norm_(list(agent.policy.parameters()) + list(agent.value.parameters()), float(max_grad_norm))
            policy_optimizer.step()
            value_optimizer.step()
            approx_kl = _masked_mean(old_log_probs - new_log_probs.detach(), mask)
            stats = {
                'agent_loss': float(loss.detach().cpu().item()),
                'agent_policy_loss': float(policy_loss.detach().cpu().item()),
                'agent_value_loss': float(value_loss.detach().cpu().item()),
                'agent_entropy': float(_masked_mean(entropy.detach(), mask).cpu().item()),
                'agent_sa_kl': float(sa_kl.detach().cpu().item()),
                'agent_anchor_loss': float(anchor_loss.detach().cpu().item()),
                'agent_approx_kl': float(approx_kl.detach().cpu().item()),
            }
    return stats


def update_adversary_ppo(
    agent: AtlaPpoLstmSaAgent,
    batch: AdversaryRolloutBatch,
    adversary_optimizer: torch.optim.Optimizer,
    adversary_value_optimizer: torch.optim.Optimizer,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    ppo_epochs: int = 10,
    num_minibatches: int = 32,
    value_coef: float = 0.5,
    entropy_coeff: float = 1e-4,
    attack_state_scope: str = 'all',
    max_grad_norm: float = 0.5,
    target_kl: float | None = None,
) -> dict:
    device = agent.device
    data = _flatten_adversary_sessions(batch.sessions, device, gamma, gae_lambda)
    attack_mask = _scope_mask(attack_state_scope, 11, device)
    agent.adversary.train()
    agent.adversary_value.train()
    n = int(data['obs'].shape[0])
    minibatches = max(1, min(int(num_minibatches), n))
    stats: dict[str, float] = {}
    epochs_run = 0
    early_stopped = False

    for epoch in range(int(ppo_epochs)):
        indices = torch.randperm(n, device=device)
        epoch_kls: list[float] = []
        for split in torch.chunk(indices, minibatches):
            obs = data['obs'][split]
            raw_deltas = data['raw_deltas'][split]
            old_log_probs = data['old_log_probs'][split]
            returns = data['returns'][split]
            advantages = data['advantages'][split]

            new_log_probs, entropy = agent.adversary.log_prob_entropy(obs, raw_deltas, log_prob_mask=attack_mask)
            log_ratio = new_log_probs - old_log_probs
            ratio = torch.exp(log_ratio)
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps)) * advantages
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_pred = agent.adversary_value(obs)
            value_loss = 0.5 * (value_pred - returns).pow(2).mean()
            entropy_loss = -float(entropy_coeff) * entropy.mean()
            loss = policy_loss + float(value_coef) * value_loss + entropy_loss

            adversary_optimizer.zero_grad(set_to_none=True)
            adversary_value_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm and float(max_grad_norm) > 0.0:
                nn.utils.clip_grad_norm_(list(agent.adversary.parameters()) + list(agent.adversary_value.parameters()), float(max_grad_norm))
            adversary_optimizer.step()
            adversary_value_optimizer.step()
            approx_kl = ((ratio.detach() - 1.0) - log_ratio.detach()).mean()
            epoch_kls.append(float(approx_kl.cpu().item()))
            stats = {
                'adversary_loss': float(loss.detach().cpu().item()),
                'adversary_policy_loss': float(policy_loss.detach().cpu().item()),
                'adversary_value_loss': float(value_loss.detach().cpu().item()),
                'adversary_entropy': float(entropy.detach().mean().cpu().item()),
                'adversary_approx_kl': float(approx_kl.detach().cpu().item()),
            }
        epochs_run = epoch + 1
        epoch_kl = float(np.mean(epoch_kls)) if epoch_kls else 0.0
        stats['adversary_approx_kl'] = epoch_kl
        if target_kl is not None and float(target_kl) > 0.0 and epoch_kl > float(target_kl):
            early_stopped = True
            break
    stats['adversary_ppo_epochs_run'] = int(epochs_run)
    stats['adversary_kl_early_stop'] = int(early_stopped)
    return stats


def _mean_day_metric(day_summaries: list[dict], key: str) -> float:
    vals = [float(row.get(key, 0.0)) for row in day_summaries]
    return float(np.mean(vals)) if vals else 0.0


def _sum_day_metric(day_summaries: list[dict], key: str) -> float:
    return float(sum(float(row.get(key, 0.0)) for row in day_summaries))


def _validate_agent_sessions(sessions: list[AgentSession]) -> list[str]:
    errors: list[str] = []
    for sess in sessions:
        n = sess.length
        fields = [
            len(sess.clean_obs),
            len(sess.adv_obs),
            len(sess.actions),
            len(sess.old_log_probs),
            len(sess.values),
            len(sess.dones),
            len(sess.next_clean_obs),
            len(sess.time_indices),
        ]
        if any(v != n for v in fields):
            errors.append(f'agent session length mismatch episode={sess.episode_index} vehicle={sess.vehicle_id} lengths={fields} rewards={n}')
        if any(int(b) <= int(a) for a, b in zip(sess.time_indices, sess.time_indices[1:])):
            errors.append(f'agent session time index is not strictly increasing episode={sess.episode_index} vehicle={sess.vehicle_id}')
        if sess.truncated and (n <= 0 or bool(sess.dones[-1])):
            errors.append(f'agent session marked truncated but terminal episode={sess.episode_index} vehicle={sess.vehicle_id}')
        if sess.truncated and not np.isfinite(float(sess.bootstrap_value)):
            errors.append(f'agent session truncated without finite bootstrap episode={sess.episode_index} vehicle={sess.vehicle_id}')
    return errors


def _validate_adversary_sessions(sessions: list[AdversarySession]) -> list[str]:
    errors: list[str] = []
    for sess in sessions:
        n = sess.length
        fields = [
            len(sess.clean_obs),
            len(sess.raw_deltas),
            len(sess.old_log_probs),
            len(sess.values),
            len(sess.dones),
            len(sess.next_clean_obs),
            len(sess.time_indices),
        ]
        if any(v != n for v in fields):
            errors.append(f'adversary session length mismatch episode={sess.episode_index} vehicle={sess.vehicle_id} lengths={fields} rewards={n}')
        if any(int(b) <= int(a) for a, b in zip(sess.time_indices, sess.time_indices[1:])):
            errors.append(f'adversary session time index is not strictly increasing episode={sess.episode_index} vehicle={sess.vehicle_id}')
        if sess.truncated and (n <= 0 or bool(sess.dones[-1])):
            errors.append(f'adversary session marked truncated but terminal episode={sess.episode_index} vehicle={sess.vehicle_id}')
        if sess.truncated and not np.isfinite(float(sess.bootstrap_value)):
            errors.append(f'adversary session truncated without finite bootstrap episode={sess.episode_index} vehicle={sess.vehicle_id}')
    return errors


def train_online_atla_ppo_lstm_sa_agent(
    arrivals: pd.DataFrame,
    signals_path,
    device: torch.device,
    *,
    seed: int = 42,
    outer_iters: int = 10,
    phase_steps: int = 2048,
    phase_episodes: int = 1,
    ppo_epochs: int = 10,
    num_minibatches: int = 32,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    adv_actor_lr: float = 1e-3,
    adv_critic_lr: float = 1e-5,
    entropy_coeff: float = 3e-4,
    adv_entropy_coeff: float = 1e-4,
    sa_reg_weight: float = 0.1,
    sa_reg_steps: int = 2,
    anchor_actor: Actor | None = None,
    anchor_reg_weight: float = 0.0,
    distill_bundle=None,
    distill_epochs: int = 0,
    distill_batch_size: int = 64,
    distill_max_sessions: int | None = None,
    epsilon: float = 0.15,
    attack_state_scope: str = 'local',
    reward_profile: RewardProfile = TRAIN_PROFILE,
    hidden_dim: int = 128,
    lstm_dim: int = 128,
    adversary_hidden_dim: int = 128,
    print_every: int = 1,
    validation_every: int = 0,
    checkpoint_every: int = 0,
    checkpoint_dir: str | Path | None = None,
    checkpoint_prefix: str = '',
    init_bundle_path: str | Path | None = None,
    iteration_offset: int = 0,
    checkpoint_metadata: dict | None = None,
    episode_scenarios=None,
) -> tuple[AtlaPpoLstmSaAgent, AtlaPpoLstmSaHistory]:
    set_seed(seed)
    train_started_at = time.perf_counter()
    canonical_scope = canonical_attack_state_scope(attack_state_scope)
    scenarios = normalize_episode_scenarios(arrivals, signals_path, episode_scenarios)
    bounds_env = ChargingEnv(signals_path=scenarios[0].signals_path, reward_profile=reward_profile)
    max_duration = max_duration_across_scenarios(scenarios)
    obs_low, obs_high = observation_bounds_across_scenarios(
        scenarios,
        reward_profile=reward_profile,
        max_duration_of_stay=max_duration,
    )

    if init_bundle_path is not None:
        agent = load_online_atla_ppo_lstm_sa_bundle(init_bundle_path, device)
    else:
        agent = AtlaPpoLstmSaAgent(
            obs_dim=bounds_env.obs_dim,
            action_dim=bounds_env.action_dim,
            hidden_dim=hidden_dim,
            lstm_dim=lstm_dim,
            adversary_hidden_dim=adversary_hidden_dim,
            device=device,
        )
    prior_train_seconds = float(
        (getattr(agent, 'metadata', {}) or {}).get(
            'cumulative_train_seconds',
            (getattr(agent, 'metadata', {}) or {}).get('training_seconds', 0.0),
        )
        if init_bundle_path is not None
        else 0.0
    )
    distillation_stats: dict[str, float] = {}
    if distill_bundle is not None and int(distill_epochs) > 0:
        if anchor_actor is None:
            raise ValueError('ATLA LSTM distillation requires anchor_actor.')
        required = ('clean_inputs', 'episode_indices', 'vehicle_ids', 'time_indices')
        if any(getattr(distill_bundle, name, None) is None for name in required):
            raise ValueError(f'ATLA distillation bundle requires {required}.')
        distillation_stats = distill_atla_lstm_policy(
            agent,
            anchor_actor,
            distill_bundle.clean_inputs,
            distill_bundle.episode_indices,
            distill_bundle.vehicle_ids,
            distill_bundle.time_indices,
            epochs=int(distill_epochs),
            batch_size=int(distill_batch_size),
            lr=float(actor_lr),
            max_sessions=distill_max_sessions,
        )
        print(
            f"[atla-ppo-lstm-sa][distill] sessions={int(distillation_stats['distill_sessions'])} "
            f"epochs={int(distillation_stats['distill_epochs'])} loss={distillation_stats['distill_loss']:.6f}",
            flush=True,
        )
    if anchor_actor is not None:
        anchor_actor = anchor_actor.to(device).eval()
        for parameter in anchor_actor.parameters():
            parameter.requires_grad_(False)
    policy_optimizer = torch.optim.Adam(agent.policy.parameters(), lr=float(actor_lr), eps=1e-5)
    value_optimizer = torch.optim.Adam(agent.value.parameters(), lr=float(critic_lr), eps=1e-5)
    adversary_optimizer = torch.optim.Adam(agent.adversary.parameters(), lr=float(adv_actor_lr), eps=1e-5)
    adversary_value_optimizer = torch.optim.Adam(agent.adversary_value.parameters(), lr=float(adv_critic_lr), eps=1e-5)
    optimizer_state = dict(getattr(agent, 'training_state', {}) or {})
    if init_bundle_path is not None:
        required_optimizers = (
            'policy_optimizer',
            'value_optimizer',
            'adversary_optimizer',
            'adversary_value_optimizer',
        )
        missing = [name for name in required_optimizers if name not in optimizer_state]
        if missing:
            raise ValueError(f'ATLA staged resume bundle lacks optimizer states: {missing}')
        policy_optimizer.load_state_dict(optimizer_state['policy_optimizer'])
        value_optimizer.load_state_dict(optimizer_state['value_optimizer'])
        adversary_optimizer.load_state_dict(optimizer_state['adversary_optimizer'])
        adversary_value_optimizer.load_state_dict(optimizer_state['adversary_value_optimizer'])

    rows: list[dict] = []
    next_episode_index = int(iteration_offset) * (1 if float(epsilon) <= 0.0 else 2)
    checkpoint_root = None if checkpoint_dir is None else ensure_dir(Path(checkpoint_dir))
    checkpoint_prefix = str(checkpoint_prefix or f'atla_ppo_lstm_sa_{canonical_scope}')
    checkpoint_metadata = dict(checkpoint_metadata or {})
    total_iteration_target = int(iteration_offset) + int(outer_iters)
    for local_iteration in range(1, int(outer_iters) + 1):
        iteration = int(iteration_offset) + int(local_iteration)
        episode_scenario = scenario_for_episode(scenarios, iteration)
        agent_batch = collect_agent_rollouts(
            episode_scenario.arrivals,
            episode_scenario.signals_path,
            agent,
            reward_profile=reward_profile,
            epsilon=epsilon,
            attack_state_scope=canonical_scope,
            phase_steps=phase_steps,
            max_episodes=phase_episodes,
            start_episode_index=next_episode_index,
            obs_low=obs_low,
            obs_high=obs_high,
        )
        next_episode_index += len(agent_batch.day_summaries)
        agent_update = update_agent_ppo(
            agent,
            agent_batch,
            policy_optimizer,
            value_optimizer,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_eps=clip_eps,
            ppo_epochs=ppo_epochs,
            num_minibatches=num_minibatches,
            entropy_coeff=entropy_coeff,
            sa_reg_weight=sa_reg_weight,
            sa_reg_steps=sa_reg_steps,
            sa_reg_epsilon=epsilon,
            anchor_actor=anchor_actor,
            anchor_reg_weight=anchor_reg_weight,
            attack_state_scope=canonical_scope,
            obs_low=obs_low,
            obs_high=obs_high,
        )

        skip_adversary_phase = float(epsilon) <= 0.0
        if skip_adversary_phase:
            adversary_batch = None
            adversary_update = {
                'adversary_loss': 0.0,
                'adversary_policy_loss': 0.0,
                'adversary_value_loss': 0.0,
                'adversary_entropy': 0.0,
                'adversary_approx_kl': 0.0,
            }
        else:
            adversary_batch = collect_adversary_rollouts(
                episode_scenario.arrivals,
                episode_scenario.signals_path,
                agent,
                reward_profile=reward_profile,
                epsilon=epsilon,
                attack_state_scope=canonical_scope,
                phase_steps=phase_steps,
                max_episodes=phase_episodes,
                start_episode_index=next_episode_index,
                obs_low=obs_low,
                obs_high=obs_high,
            )
            next_episode_index += len(adversary_batch.day_summaries)
            adversary_update = update_adversary_ppo(
                agent,
                adversary_batch,
                adversary_optimizer,
                adversary_value_optimizer,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_eps=clip_eps,
                ppo_epochs=ppo_epochs,
                num_minibatches=num_minibatches,
                entropy_coeff=adv_entropy_coeff,
                attack_state_scope=canonical_scope,
            )
        adversary_day_summaries = [] if adversary_batch is None else adversary_batch.day_summaries
        adversary_alignment_errors = [] if adversary_batch is None else adversary_batch.alignment_errors
        row = {
            'iteration': int(iteration),
            'scenario_id': str(episode_scenario.scenario_id),
            'agent_phase_steps': int(agent_batch.total_steps),
            'adversary_phase_steps': 0 if adversary_batch is None else int(adversary_batch.total_steps),
            'agent_days': int(len(agent_batch.day_summaries)),
            'adversary_days': int(len(adversary_day_summaries)),
            'agent_ep_reward': _mean_day_metric(agent_batch.day_summaries, 'ep_reward'),
            'adversary_ep_reward': _mean_day_metric(adversary_day_summaries, 'ep_reward'),
            'agent_exit_vio': _sum_day_metric(agent_batch.day_summaries, 'exit_vio'),
            'agent_run_vio': _sum_day_metric(agent_batch.day_summaries, 'run_vio'),
            'agent_truncated_sessions': _sum_day_metric(agent_batch.day_summaries, 'truncated_sessions'),
            'adversary_truncated_sessions': _sum_day_metric(adversary_day_summaries, 'truncated_sessions'),
            'alignment_errors': int(len(agent_batch.alignment_errors) + len(adversary_alignment_errors)),
            **agent_update,
            **adversary_update,
            'stage_training_seconds': float(time.perf_counter() - train_started_at),
            'cumulative_train_seconds': float(
                prior_train_seconds + time.perf_counter() - train_started_at
            ),
        }
        should_validate = (
            int(validation_every) > 0
            and (local_iteration == 1 or iteration % int(validation_every) == 0 or local_iteration == int(outer_iters))
        )
        if should_validate:
            clean_eval = run_atla_policy_episode(
                arrivals,
                signals_path,
                agent,
                reward_profile=reward_profile,
                attack_mode='none',
                epsilon=epsilon,
                attack_state_scope=canonical_scope,
                obs_low=obs_low,
                obs_high=obs_high,
                seed=seed + 1000 + iteration,
                label='validation_no_attack',
            )
            current_adv_eval = run_atla_policy_episode(
                arrivals,
                signals_path,
                agent,
                reward_profile=reward_profile,
                attack_mode='learned',
                adversary=agent.adversary,
                epsilon=epsilon,
                attack_state_scope=canonical_scope,
                obs_low=obs_low,
                obs_high=obs_high,
                seed=seed + 2000 + iteration,
                label='validation_current_adversary',
            )
            row.update(
                {
                    'validation_clean_reward': float(clean_eval.get('ep_reward', 0.0)),
                    'validation_clean_exit_vio': float(clean_eval.get('exit_vio', 0.0)),
                    'validation_clean_run_vio': float(clean_eval.get('run_vio', 0.0)),
                    'validation_clean_mean_final_soc': float(clean_eval.get('mean_fin_soc', 0.0)),
                    'validation_clean_dense_safety': float(clean_eval.get('ep_r4_dense', 0.0)),
                    'validation_current_adv_reward': float(current_adv_eval.get('ep_reward', 0.0)),
                    'validation_current_adv_exit_vio': float(current_adv_eval.get('exit_vio', 0.0)),
                    'validation_current_adv_mean_final_soc': float(current_adv_eval.get('mean_fin_soc', 0.0)),
                    'validation_current_adv_dense_safety': float(current_adv_eval.get('ep_r4_dense', 0.0)),
                }
            )
        rows.append(row)
        if local_iteration == 1 or iteration % max(int(print_every), 1) == 0 or local_iteration == int(outer_iters):
            validation_msg = ''
            if should_validate:
                validation_msg = (
                    f" clean={row['validation_clean_reward']:.2f}"
                    f" clean_exit={row['validation_clean_exit_vio']:.0f}"
                    f" curr_adv={row['validation_current_adv_reward']:.2f}"
                )
            print(
                f"[atla-ppo-lstm-sa] iter={iteration:03d}/{total_iteration_target} "
                f"agent_reward={row['agent_ep_reward']:.4f} adv_reward={row['adversary_ep_reward']:.4f} "
                f"sa_kl={row.get('agent_sa_kl', 0.0):.6f} align_err={row['alignment_errors']}{validation_msg}"
                ,
                flush=True,
            )
        if checkpoint_root is not None and int(checkpoint_every) > 0 and (
            iteration % int(checkpoint_every) == 0 or local_iteration == int(outer_iters)
        ):
            checkpoint_path = checkpoint_root / f'{checkpoint_prefix}_iter{iteration:03d}_bundle.pt'
            training_state = {
                'completed_iteration': int(iteration),
                'resume_from_iteration': int(iteration_offset),
                'stage_training_seconds': float(time.perf_counter() - train_started_at),
                'cumulative_train_seconds': float(
                    prior_train_seconds + time.perf_counter() - train_started_at
                ),
                'policy_optimizer': policy_optimizer.state_dict(),
                'value_optimizer': value_optimizer.state_dict(),
                'adversary_optimizer': adversary_optimizer.state_dict(),
                'adversary_value_optimizer': adversary_value_optimizer.state_dict(),
            }
            save_online_atla_ppo_lstm_sa_bundle(
                agent,
                checkpoint_path,
                metadata={
                    **checkpoint_metadata,
                    'algorithm': 'online_atla_ppo_lstm_sa',
                    'checkpoint_iteration': int(iteration),
                    'checkpoint_episode': int(iteration),
                    'iteration_offset': int(iteration_offset),
                    'attack_state_scope': canonical_scope,
                    'epsilon': float(epsilon),
                    'reward_profile': str(reward_profile.name),
                    'obs_low': np.asarray(obs_low, dtype=np.float32).tolist(),
                    'obs_high': np.asarray(obs_high, dtype=np.float32).tolist(),
                    'hidden_dim': int(hidden_dim),
                    'lstm_dim': int(lstm_dim),
                    'adversary_hidden_dim': int(adversary_hidden_dim),
                    'sa_reg_weight': float(sa_reg_weight),
                    'sa_reg_steps': int(sa_reg_steps),
                    'anchor_reg_weight': float(anchor_reg_weight),
                    'phase_episodes': int(phase_episodes),
                    'resume_from_iteration': int(iteration_offset),
                    'stage_training_seconds': float(time.perf_counter() - train_started_at),
                    'cumulative_train_seconds': float(
                        prior_train_seconds + time.perf_counter() - train_started_at
                    ),
                    **distillation_stats,
                },
                training_state=training_state,
            )

    metadata = {
        **checkpoint_metadata,
        'algorithm': 'online_atla_ppo_lstm_sa',
        'checkpoint_iteration': int(total_iteration_target),
        'iteration_offset': int(iteration_offset),
        'attack_state_scope': canonical_scope,
        'epsilon': float(epsilon),
        'reward_profile': str(reward_profile.name),
        'obs_low': np.asarray(obs_low, dtype=np.float32).tolist(),
        'obs_high': np.asarray(obs_high, dtype=np.float32).tolist(),
        'hidden_dim': int(hidden_dim),
        'lstm_dim': int(lstm_dim),
        'adversary_hidden_dim': int(adversary_hidden_dim),
        'sa_reg_weight': float(sa_reg_weight),
        'sa_reg_steps': int(sa_reg_steps),
        'anchor_reg_weight': float(anchor_reg_weight),
        'phase_episodes': int(phase_episodes),
        'resume_from_iteration': int(iteration_offset),
        'stage_training_seconds': float(time.perf_counter() - train_started_at),
        'cumulative_train_seconds': float(
            prior_train_seconds + time.perf_counter() - train_started_at
        ),
        'training_seconds': float(prior_train_seconds + time.perf_counter() - train_started_at),
        **distillation_stats,
    }
    agent.metadata = metadata
    agent.training_state = {
        'completed_iteration': int(total_iteration_target),
        'resume_from_iteration': int(iteration_offset),
        'stage_training_seconds': float(time.perf_counter() - train_started_at),
        'cumulative_train_seconds': float(
            prior_train_seconds + time.perf_counter() - train_started_at
        ),
        'policy_optimizer': policy_optimizer.state_dict(),
        'value_optimizer': value_optimizer.state_dict(),
        'adversary_optimizer': adversary_optimizer.state_dict(),
        'adversary_value_optimizer': adversary_value_optimizer.state_dict(),
    }
    return agent, AtlaPpoLstmSaHistory(rows)


def _attack_recurrent_observation(
    agent: AtlaPpoLstmSaAgent,
    clean_obs: np.ndarray,
    hidden,
    *,
    algorithm: str,
    critic: Critic | None,
    epsilon: float,
    alpha: float | None,
    iters: int | None,
    mask: torch.Tensor,
    obs_low_t: torch.Tensor,
    obs_high_t: torch.Tensor,
    generator: torch.Generator,
) -> np.ndarray:
    algorithm = canonical_attack_algorithm(algorithm)
    defaults = ATTACK_DEFAULTS[algorithm]
    attack_alpha = float(defaults.alpha if alpha is None else alpha)
    attack_iters = int(defaults.iters if iters is None else iters)
    clean_t = torch.as_tensor(to_numpy_1d(clean_obs), dtype=torch.float32, device=agent.device).view(1, -1)
    if float(epsilon) <= 0.0 or attack_iters <= 0:
        return clean_t.detach().cpu().numpy().reshape(-1).astype(np.float32)
    noise = torch.empty_like(clean_t).uniform_(-float(epsilon), float(epsilon), generator=generator)
    image = clean_t + noise * mask.view(1, -1)
    image = torch.maximum(torch.minimum(image, clean_t + float(epsilon) * mask.view(1, -1)), clean_t - float(epsilon) * mask.view(1, -1))
    image = torch.maximum(torch.minimum(image, obs_high_t), obs_low_t)
    image = image * mask.view(1, -1) + clean_t * (1.0 - mask.view(1, -1))
    with torch.no_grad():
        clean_mean, _, _ = agent.policy.forward_step(clean_t, _detach_hidden(hidden))
        clean_action = torch.tanh(clean_mean).detach()
    if algorithm == 'q_function' and critic is None:
        raise ValueError('Recurrent q_function evaluation requires a shared critic.')

    was_training = agent.policy.training
    agent.policy.train(True)
    try:
        for _ in range(attack_iters):
            image = image.detach().requires_grad_(True)
            mean, _, _ = agent.policy.forward_step(image, _detach_hidden(hidden))
            action = torch.tanh(mean)
            if algorithm == 'q_function':
                objective = critic(clean_t, action).mean()
                direction = -1.0
            elif algorithm in {'opposite_pgd', 'opposite_fgsm'}:
                objective = F.mse_loss(action, clean_action)
                direction = 1.0
            else:
                raise ValueError(f'Unsupported recurrent external attack: {algorithm}')
            gradient = torch.autograd.grad(objective, image, retain_graph=False, create_graph=False)[0]
            image = image.detach() + direction * attack_alpha * gradient.sign() * mask.view(1, -1)
            image = torch.maximum(torch.minimum(image, clean_t + float(epsilon) * mask.view(1, -1)), clean_t - float(epsilon) * mask.view(1, -1))
            image = torch.maximum(torch.minimum(image, obs_high_t), obs_low_t)
            image = image * mask.view(1, -1) + clean_t * (1.0 - mask.view(1, -1))
    finally:
        agent.policy.train(was_training)
    return image.detach().cpu().numpy().reshape(-1).astype(np.float32)


class _RecurrentPolicyActorAdapter(nn.Module):
    """Expose the LSTM policy as a pointwise actor at each vehicle's current history."""

    def __init__(self, policy: AtlaLstmSquashedGaussianPolicy) -> None:
        super().__init__()
        self.policy = policy
        self.hidden_by_key: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor] | None] = {}
        self.active_keys: list[tuple[int, int]] = []

    def set_hidden_context(self, hidden_by_key: dict) -> None:
        self.hidden_by_key = {
            (int(key[0]), int(key[1])): _detach_hidden(hidden)
            for key, hidden in hidden_by_key.items()
        }

    def prepare_attack_keys(self, keys) -> None:
        self.active_keys = [(int(key[0]), int(key[1])) for key in keys]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 1:
            obs = obs.view(1, -1)
        if len(self.active_keys) == 1 and obs.shape[0] > 1:
            keys = self.active_keys * int(obs.shape[0])
        elif len(self.active_keys) == int(obs.shape[0]):
            keys = self.active_keys
        else:
            raise RuntimeError(
                f'Recurrent attack key mismatch: keys={len(self.active_keys)} batch={int(obs.shape[0])}'
            )
        actions: list[torch.Tensor] = []
        with torch.backends.cudnn.flags(enabled=False):
            for row, key in zip(obs, keys):
                mean, _, _ = self.policy.forward_step(row.view(1, -1), self.hidden_by_key.get(key))
                actions.append(torch.tanh(mean))
        return torch.cat(actions, dim=0)


def run_atla_policy_episode(
    arrivals: pd.DataFrame,
    signals_path,
    agent: AtlaPpoLstmSaAgent,
    *,
    reward_profile: RewardProfile = TRAIN_PROFILE,
    attack_mode: str = 'none',
    adversary: AtlaObsMLPPolicy | None = None,
    critic: Critic | None = None,
    epsilon: float = 0.15,
    alpha: float | None = None,
    iters: int | None = None,
    attack_state_scope: str = 'local',
    obs_low: np.ndarray | None = None,
    obs_high: np.ndarray | None = None,
    seed: int = 42,
    label: str | None = None,
) -> dict:
    device = agent.device
    env = ChargingEnv(signals_path=signals_path, reward_profile=reward_profile)
    if obs_low is None or obs_high is None:
        max_duration = resolve_max_duration_of_stay(arrivals)
        obs_low, obs_high = env.observation_bounds(max_duration_of_stay=max_duration)
    obs_low_t, obs_high_t = _obs_bounds_tensor(obs_low, obs_high, device)
    mask = _scope_mask(attack_state_scope, env.obs_dim, device)
    mask_np = mask.detach().cpu().numpy().reshape(-1).astype(np.float32)
    rng = np.random.default_rng(int(seed))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    if critic is not None:
        critic = critic.to(device).eval()
    long_attack_mode = None
    if attack_mode in {'local_small_drift_q', 'local_deadline_drift_pgd'}:
        long_attack_mode = canonical_long_horizon_attack_name(attack_mode)
    recurrent_actor_adapter = None
    long_attacker = None
    if long_attack_mode is not None:
        recurrent_actor_adapter = _RecurrentPolicyActorAdapter(agent.policy).to(device).eval()
        if attack_mode in {'local_small_drift_q', 'local_deadline_drift_pgd'}:
            long_attacker = build_formal_experimental_long_horizon_attacker(
                attack_mode,
                actor=recurrent_actor_adapter,
                device=device,
                obs_low=np.asarray(obs_low, dtype=np.float32),
                obs_high=np.asarray(obs_high, dtype=np.float32),
                critic=critic,
                seed=seed,
            )
        else:
            long_attacker = build_long_horizon_attacker(
                long_attack_mode,
                actor=recurrent_actor_adapter,
                device=device,
                obs_low=np.asarray(obs_low, dtype=np.float32),
                obs_high=np.asarray(obs_high, dtype=np.float32),
                critic=critic,
                seed=seed,
            )
        long_attacker.reset()

    env.reset()
    agent.policy.eval()
    if adversary is not None:
        adversary.eval()
    idx = 0
    active: list[QueueItem] = []
    active_vehicle_ids: list[int] = []
    policy_hiddens: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    attack_obs_count = 0
    total_obs_count = 0

    def observed_obs(
        clean_obs: np.ndarray,
        hidden,
        *,
        vehicle_id: int,
        station: int,
        is_new_arrival: bool,
    ) -> np.ndarray:
        nonlocal attack_obs_count
        clean_np = to_numpy_1d(clean_obs)
        if attack_mode == 'none':
            return clean_np
        if attack_mode == 'random':
            attack_obs_count += 1
            return _random_attack_obs(clean_np, epsilon=epsilon, mask_np=mask_np, obs_low=np.asarray(obs_low), obs_high=np.asarray(obs_high), rng=rng)
        if attack_mode == 'learned':
            if adversary is None:
                raise ValueError('learned attack evaluation requires an adversary policy.')
            clean_t = torch.as_tensor(clean_np, dtype=torch.float32, device=device).view(1, -1)
            with torch.no_grad():
                raw_delta, _, _, _ = adversary.sample(clean_t, deterministic=True)
                adv_obs_t = _perturb_from_raw(clean_t, raw_delta, epsilon=epsilon, mask=mask, obs_low=obs_low_t, obs_high=obs_high_t)
            attack_obs_count += 1
            return adv_obs_t.detach().cpu().numpy().reshape(-1).astype(np.float32)
        if attack_mode in {'opposite_pgd', 'opposite_fgsm', 'q_function'}:
            attack_obs_count += 1
            return _attack_recurrent_observation(
                agent,
                clean_np,
                hidden,
                algorithm=attack_mode,
                critic=critic,
                epsilon=epsilon,
                alpha=alpha,
                iters=iters,
                mask=mask,
                obs_low_t=obs_low_t,
                obs_high_t=obs_high_t,
                generator=generator,
            )
        if long_attacker is not None and recurrent_actor_adapter is not None:
            key = (0, int(vehicle_id))
            recurrent_actor_adapter.set_hidden_context({key: hidden})
            context = AttackContext(
                scenario='O',
                time_index=int(env.t),
                raw_price=float(env.signals.price[env.t]),
                station=int(station),
                is_new_arrival=bool(is_new_arrival),
            )
            attack_obs_count += 1
            return long_attacker.attack_with_metadata(
                clean_np.reshape(1, -1),
                contexts=[context],
                vehicle_ids=[int(vehicle_id)],
                episode_indices=[0],
            )[0].astype(np.float32)
        raise ValueError(f'Unsupported attack_mode: {attack_mode}')

    while env.t < env.horizon:
        new_states: list[np.ndarray] = []
        new_stations: list[int] = []
        new_vehicle_ids: list[int] = []
        while idx < len(arrivals) and int(arrivals.loc[idx, 'Arrive_time']) == env.t:
            new_states.append(env.build_initial_obs(int(arrivals.loc[idx, 'Duration_of_stay'])))
            new_stations.append(int(arrivals.loc[idx, 'Station']))
            new_vehicle_ids.append(int(idx))
            idx += 1

        for clean_obs, station, vehicle_id in zip(new_states, new_stations, new_vehicle_ids):
            key = (0, int(vehicle_id))
            obs_np = observed_obs(
                clean_obs,
                policy_hiddens.get(key),
                vehicle_id=vehicle_id,
                station=station,
                is_new_arrival=True,
            )
            total_obs_count += 1
            obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device).view(1, -1)
            with torch.no_grad():
                action_t, _, hidden = agent.policy.sample_step(obs_t, policy_hiddens.get(key), deterministic=True)
            policy_hiddens[key] = _detach_hidden(hidden)
            env.enqueue(clean_obs, action_t.detach().cpu().numpy().reshape(-1).astype(np.float32), int(station))

        if active:
            active_states = [item.obs for item in active]
            active_stations = [item.station for item in active]
            for clean_obs, station, vehicle_id in zip(active_states, active_stations, active_vehicle_ids):
                key = (0, int(vehicle_id))
                obs_np = observed_obs(
                    clean_obs,
                    policy_hiddens.get(key),
                    vehicle_id=vehicle_id,
                    station=station,
                    is_new_arrival=False,
                )
                total_obs_count += 1
                obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device).view(1, -1)
                with torch.no_grad():
                    action_t, _, hidden = agent.policy.sample_step(obs_t, policy_hiddens.get(key), deterministic=True)
                policy_hiddens[key] = _detach_hidden(hidden)
                env.enqueue(clean_obs, action_t.detach().cpu().numpy().reshape(-1).astype(np.float32), int(station))

        step_vehicle_ids = new_vehicle_ids + active_vehicle_ids
        transitions, active, _ = env.step()
        for vehicle_id, tr in zip(step_vehicle_ids, transitions):
            if bool(tr.done):
                policy_hiddens.pop((0, int(vehicle_id)), None)
        active_vehicle_ids = [vid for vid, tr in zip(step_vehicle_ids, transitions) if not bool(tr.done)]

    out = summarize_metrics(env.metrics, label or attack_mode)
    out['attack_mode'] = str(attack_mode)
    out['attack_state_scope'] = 'local' if long_attacker is not None else str(canonical_attack_state_scope(attack_state_scope))
    out['epsilon'] = float(getattr(long_attacker, 'epsilon', epsilon))
    out['attack_obs_count'] = int(attack_obs_count)
    out['attack_obs_rate'] = 0.0 if total_obs_count <= 0 else float(attack_obs_count / total_obs_count)
    return out


def train_fresh_eval_adversary(
    arrivals: pd.DataFrame,
    signals_path,
    base_agent: AtlaPpoLstmSaAgent,
    *,
    reward_profile: RewardProfile = TRAIN_PROFILE,
    epsilon: float = 0.15,
    attack_state_scope: str = 'local',
    eval_adv_iters: int = 50,
    eval_adv_phase_steps: int = 2048,
    ppo_epochs: int = 10,
    num_minibatches: int = 32,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    adv_actor_lr: float = 1e-3,
    adv_critic_lr: float = 1e-5,
    adv_entropy_coeff: float = 1e-4,
    adversary_hidden_dim: int = 128,
    seed: int = 42,
    obs_low: np.ndarray | None = None,
    obs_high: np.ndarray | None = None,
) -> tuple[AtlaObsMLPPolicy, AtlaObsMLPValueNet, list[dict]]:
    set_seed(seed)
    fresh_agent = AtlaPpoLstmSaAgent(
        obs_dim=base_agent.policy.obs_dim,
        action_dim=base_agent.policy.action_dim,
        hidden_dim=base_agent.policy.hidden_dim,
        lstm_dim=base_agent.policy.lstm_dim,
        adversary_hidden_dim=adversary_hidden_dim,
        device=base_agent.device,
    )
    fresh_agent.policy.load_state_dict(base_agent.policy.state_dict())
    fresh_agent.value.load_state_dict(base_agent.value.state_dict())
    for param in fresh_agent.policy.parameters():
        param.requires_grad_(False)
    for param in fresh_agent.value.parameters():
        param.requires_grad_(False)
    adv_opt = torch.optim.Adam(fresh_agent.adversary.parameters(), lr=float(adv_actor_lr), eps=1e-5)
    adv_val_opt = torch.optim.Adam(fresh_agent.adversary_value.parameters(), lr=float(adv_critic_lr), eps=1e-5)
    rows: list[dict] = []
    next_episode_index = 0
    for iteration in range(1, int(eval_adv_iters) + 1):
        batch = collect_adversary_rollouts(
            arrivals,
            signals_path,
            fresh_agent,
            reward_profile=reward_profile,
            epsilon=epsilon,
            attack_state_scope=attack_state_scope,
            phase_steps=eval_adv_phase_steps,
            start_episode_index=next_episode_index,
            obs_low=obs_low,
            obs_high=obs_high,
            agent_deterministic=True,
        )
        next_episode_index += len(batch.day_summaries)
        update = update_adversary_ppo(
            fresh_agent,
            batch,
            adv_opt,
            adv_val_opt,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_eps=clip_eps,
            ppo_epochs=ppo_epochs,
            num_minibatches=num_minibatches,
            entropy_coeff=adv_entropy_coeff,
            attack_state_scope=attack_state_scope,
        )
        rows.append(
            {
                'iteration': int(iteration),
                'phase_steps': int(batch.total_steps),
                'eval_adv_ep_reward': _mean_day_metric(batch.day_summaries, 'ep_reward'),
                'alignment_errors': int(len(batch.alignment_errors)),
                **update,
            }
        )
    return fresh_agent.adversary, fresh_agent.adversary_value, rows


def evaluate_online_atla_ppo_lstm_sa_agent(
    arrivals: pd.DataFrame,
    signals_path,
    agent: AtlaPpoLstmSaAgent,
    *,
    reward_profile: RewardProfile = TRAIN_PROFILE,
    epsilon: float = 0.15,
    attack_state_scopes: list[str] | tuple[str, ...] = ('local', 'all'),
    seed: int = 42,
    eval_adv_iters: int = 50,
    eval_adv_phase_steps: int = 2048,
    ppo_epochs: int = 10,
    num_minibatches: int = 32,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    adv_actor_lr: float = 1e-3,
    adv_critic_lr: float = 1e-5,
    adv_entropy_coeff: float = 1e-4,
) -> tuple[pd.DataFrame, dict[str, list[dict]]]:
    env = ChargingEnv(signals_path=signals_path, reward_profile=reward_profile)
    max_duration = resolve_max_duration_of_stay(arrivals)
    obs_low, obs_high = env.observation_bounds(max_duration_of_stay=max_duration)
    rows: list[dict] = []
    attack_histories: dict[str, list[dict]] = {}
    clean = run_atla_policy_episode(
        arrivals,
        signals_path,
        agent,
        reward_profile=reward_profile,
        attack_mode='none',
        epsilon=epsilon,
        attack_state_scope='local',
        obs_low=obs_low,
        obs_high=obs_high,
        seed=seed,
        label='no_attack',
    )
    rows.append({**clean, 'scope': 'none', 'attack_mode': 'no_attack', 'attack_state_scope': 'none'})
    natural_reward = float(clean.get('ep_reward', 0.0))
    worst_reward = natural_reward

    for scope in attack_state_scopes:
        canonical_scope = canonical_attack_state_scope(scope)
        random_row = run_atla_policy_episode(
            arrivals,
            signals_path,
            agent,
            reward_profile=reward_profile,
            attack_mode='random',
            epsilon=epsilon,
            attack_state_scope=canonical_scope,
            obs_low=obs_low,
            obs_high=obs_high,
            seed=seed + 17,
            label=f'random_{canonical_scope}',
        )
        rows.append({**random_row, 'scope': canonical_scope, 'attack_mode': 'random_attack'})
        worst_reward = min(worst_reward, float(random_row.get('ep_reward', 0.0)))

        eval_adv, _, adv_history = train_fresh_eval_adversary(
            arrivals,
            signals_path,
            agent,
            reward_profile=reward_profile,
            epsilon=epsilon,
            attack_state_scope=canonical_scope,
            eval_adv_iters=eval_adv_iters,
            eval_adv_phase_steps=eval_adv_phase_steps,
            ppo_epochs=ppo_epochs,
            num_minibatches=num_minibatches,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_eps=clip_eps,
            adv_actor_lr=adv_actor_lr,
            adv_critic_lr=adv_critic_lr,
            adv_entropy_coeff=adv_entropy_coeff,
            seed=seed + 101,
            obs_low=obs_low,
            obs_high=obs_high,
        )
        attack_histories[canonical_scope] = adv_history
        learned_row = run_atla_policy_episode(
            arrivals,
            signals_path,
            agent,
            reward_profile=reward_profile,
            attack_mode='learned',
            adversary=eval_adv,
            epsilon=epsilon,
            attack_state_scope=canonical_scope,
            obs_low=obs_low,
            obs_high=obs_high,
            seed=seed + 31,
            label=f'learned_{canonical_scope}',
        )
        rows.append({**learned_row, 'scope': canonical_scope, 'attack_mode': 'learned_attack'})
        worst_reward = min(worst_reward, float(learned_row.get('ep_reward', 0.0)))

    summary_row = {
        'scope': 'all_evaluated',
        'attack_mode': 'summary',
        'Natural Reward': float(natural_reward),
        'Worst Attack Reward': float(worst_reward),
        'Robust Ratio': 0.0 if abs(natural_reward) < 1e-8 else float(worst_reward / natural_reward),
        'Total Charging Cost': float(clean.get('ep_r1', 0.0)),
        'Exit SOC Violation': float(clean.get('exit_vio', 0.0)),
        'Running SOC Violation': float(clean.get('run_vio', 0.0)),
        'Mean Final SOC': float(clean.get('mean_fin_soc', 0.0)),
        'Std Final SOC': float(clean.get('std_finl_soc', 0.0)),
    }
    rows.append(summary_row)
    return pd.DataFrame(rows), attack_histories


def save_online_atla_ppo_lstm_sa_bundle(
    agent: AtlaPpoLstmSaAgent,
    path: str | Path,
    *,
    metadata: dict | None = None,
    training_state: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(getattr(agent, 'metadata', {}) or {})
    if metadata:
        meta.update(metadata)
    torch.save(
        {
            'model_type': 'online_atla_ppo_lstm_sa_bundle',
            'policy_state_dict': agent.policy.state_dict(),
            'value_state_dict': agent.value.state_dict(),
            'adversary_state_dict': agent.adversary.state_dict(),
            'adversary_value_state_dict': agent.adversary_value.state_dict(),
            'metadata': meta,
            'training_state': dict(training_state if training_state is not None else getattr(agent, 'training_state', {}) or {}),
        },
        path,
    )
    return path


def load_online_atla_ppo_lstm_sa_bundle(path: str | Path, device: torch.device) -> AtlaPpoLstmSaAgent:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f'Invalid ATLA-PPO-LSTM-SA bundle: {path}')
    metadata = dict(payload.get('metadata') or {})
    agent = AtlaPpoLstmSaAgent(
        obs_dim=11,
        action_dim=1,
        hidden_dim=int(metadata.get('hidden_dim', 128)),
        lstm_dim=int(metadata.get('lstm_dim', 128)),
        adversary_hidden_dim=int(metadata.get('adversary_hidden_dim', 128)),
        device=device,
    )
    agent.policy.load_state_dict(payload['policy_state_dict'])
    agent.value.load_state_dict(payload['value_state_dict'])
    if payload.get('adversary_state_dict') is not None:
        agent.adversary.load_state_dict(payload['adversary_state_dict'])
    if payload.get('adversary_value_state_dict') is not None:
        agent.adversary_value.load_state_dict(payload['adversary_value_state_dict'])
    agent.metadata = metadata
    agent.training_state = dict(payload.get('training_state') or {})
    agent.policy.eval()
    agent.value.eval()
    agent.adversary.eval()
    agent.adversary_value.eval()
    return agent


def save_atla_ppo_lstm_sa_history(history: AtlaPpoLstmSaHistory, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalize_result_frame(pd.DataFrame(history.rows), rename_keys=False).to_csv(path, index=False, float_format='%.4f')
    return path


def save_atla_ppo_lstm_sa_eval(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    normalize_result_frame(df, rename_keys=False).to_csv(path, index=False, float_format='%.4f')
    return path
