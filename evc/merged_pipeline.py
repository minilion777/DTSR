"""Training, sampling, evaluation, and unified defense helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .merged_attacks import AttackContext, AttackScenario, AttackScope, LOCAL_ATTACK_IDX, PGDStateAttacker, attack_batch_by_context
from .merged_core import (
    TRAIN_PROFILE,
    Actor,
    ChargingEnv,
    DDPGAgent,
    EpisodeMetrics,
    QueueItem,
    ReplayBuffer,
    RewardProfile,
    ensure_arrivals,
    ensure_dir,
    gini,
    json_dump,
    load_actor_critic_bundle,
    load_arrivals,
    normalize_result_frame,
    normalize_result_object,
    save_actor,
    set_seed,
    to_numpy_1d,
)
from .defense import (
    _binary_operating_metrics_np,
    DAETrainResult,
    DetectorTrainResult,
    DenoisingAutoencoder,
    PosteriorBenefitMLPDetector,
    SequentialDAERuntime,
    canonical_state_scope,
    defended_indices_for_scope,
    dae_reconstruction_with_history,
    posterior_detector_probabilities,
    reconstruction_batch,
    train_dae,
    train_posterior_detector,
)


@dataclass
class TrainHistory:
    rows: list[dict]


@dataclass
class CleanTrajectoryBundle:
    clean_inputs: np.ndarray
    metadata: dict
    raw_prices: np.ndarray | None = None
    time_indices: np.ndarray | None = None
    stations: np.ndarray | None = None
    is_new_arrivals: np.ndarray | None = None
    vehicle_ids: np.ndarray | None = None
    episode_indices: np.ndarray | None = None


@dataclass
class PairDatasetBundle:
    adv_inputs: np.ndarray
    clean_inputs: np.ndarray
    metadata: dict
    clean_anchor_inputs: np.ndarray | None = None
    time_indices: np.ndarray | None = None
    stations: np.ndarray | None = None
    is_new_arrivals: np.ndarray | None = None
    vehicle_ids: np.ndarray | None = None
    episode_indices: np.ndarray | None = None
    attack_mask: np.ndarray | None = None


@dataclass
class EvaluationBundle:
    clean_summary: dict
    attack_summary: dict
    clean_dae_summary: dict | None
    attack_dae_summary: dict | None
    clean_dae_detector_summary: dict | None = None
    attack_dae_detector_summary: dict | None = None
    clean_dae_oracle_summary: dict | None = None
    attack_dae_oracle_summary: dict | None = None

    @property
    def defended_summary(self) -> dict | None:
        return self.attack_dae_summary


def iter_evaluation_summaries(bundle: EvaluationBundle) -> list[dict]:
    rows = [bundle.clean_summary, bundle.attack_summary]
    optional_rows = [
        bundle.clean_dae_summary,
        bundle.attack_dae_summary,
        bundle.clean_dae_detector_summary,
        bundle.attack_dae_detector_summary,
        bundle.clean_dae_oracle_summary,
        bundle.attack_dae_oracle_summary,
    ]
    rows.extend([row for row in optional_rows if row is not None])
    return rows


def get_arrivals(data_path, seed: int = 42, max_sessions: int | None = None) -> pd.DataFrame:
    arrivals = load_arrivals(ensure_arrivals(data_path, seed=seed))
    if max_sessions is not None:
        arrivals = arrivals.iloc[:max_sessions].copy().reset_index(drop=True)
    return arrivals


def summarize_metrics(metrics: EpisodeMetrics, label: str) -> dict:
    raw = {
        'label': label,
        'ep_reward': float(metrics.ep_reward),
        'ep_r1_cost_sum': float(metrics.ep_r1_cost_sum),
        'ep_r2_exit_penalty_sum': float(metrics.ep_r2_exit_penalty_sum),
        'ep_r3_running_penalty_sum': float(metrics.ep_r3_running_penalty_sum),
        'ep_r4_dense_safety_penalty_sum': float(metrics.ep_r4_dense_safety_penalty_sum),
        'exit_vio': int(metrics.exit_violation_count),
        'run_vio': int(metrics.running_violation_count),
        'total_transitions': int(metrics.total_transitions),
        'done_count': int(metrics.done_count),
        'gini_cost': float(gini(metrics.costlist)),
        'mean_final_soc': float(np.mean(metrics.final_soc_list) if metrics.final_soc_list else 0.0),
        'std_final_soc': float(np.std(metrics.final_soc_list) if metrics.final_soc_list else 0.0),
        'mean_abs_power': float(np.mean(np.abs(metrics.powercurve)) if metrics.powercurve else 0.0),
        'max_abs_power': float(np.max(np.abs(metrics.powercurve)) if metrics.powercurve else 0.0),
        'cost_count': int(len(metrics.costlist)),
        'final_soc_count': int(len(metrics.final_soc_list)),
        'powercurve': [float(x) for x in metrics.powercurve],
        'powerlist': [[float(v) for v in station_curve] for station_curve in metrics.powerlist],
        'costlist': [float(x) for x in metrics.costlist],
        'final_soc_list': [float(x) for x in metrics.final_soc_list],
    }
    return normalize_result_object(raw, rename_keys=True)


def save_evaluation_bundle(bundle: EvaluationBundle, save_dir: str | Path) -> Path:
    save_dir = ensure_dir(save_dir)
    rows = iter_evaluation_summaries(bundle)
    scalar_rows = [{k: v for k, v in row.items() if not isinstance(v, list)} for row in rows]
    normalize_result_frame(pd.DataFrame(scalar_rows), rename_keys=False).to_csv(save_dir / 'evaluation_metrics.csv', index=False, float_format='%.2f')
    return save_dir


def train_agent(
    arrivals: pd.DataFrame,
    signals_path,
    device: torch.device,
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
    freeze_actor: bool = False,
    reward_profile: RewardProfile = TRAIN_PROFILE,
) -> tuple[DDPGAgent, TrainHistory]:
    set_seed(seed)
    env = ChargingEnv(signals_path=signals_path, reward_profile=reward_profile)
    from .merged_core import load_actor_from_path
    if init_actor_path is not None and resume_bundle_path is not None:
        raise ValueError('train_agent supports either init_actor_path or resume_bundle_path, not both.')
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
    agent = DDPGAgent(actor, device=device, gamma=gamma, tau=tau, actor_lr=actor_lr, critic_lr=critic_lr)
    if critic_state_dict is not None:
        agent.critic.load_state_dict(critic_state_dict)
        agent.critic_target.load_state_dict(critic_state_dict)
    buffer = ReplayBuffer(buffer_size, env.obs_dim, env.action_dim, device)
    rows = []
    current_noise = float(exploration_noise)

    for episode in range(1, episodes + 1):
        env.reset()
        idx = 0
        active: list[QueueItem] = []
        last_update = {'actor_loss': 0.0, 'critic_loss': 0.0, 'mean_q': 0.0}
        while env.t < env.horizon:
            while idx < len(arrivals) and int(arrivals.loc[idx, 'Arrive_time']) == env.t:
                obs = env.build_initial_obs(int(arrivals.loc[idx, 'Duration_of_stay']))
                action = agent.act(obs, exploration_noise=current_noise, deterministic=False)
                env.enqueue(obs, action, int(arrivals.loc[idx, 'Station']))
                idx += 1
            for item in active:
                action = agent.act(item.obs, exploration_noise=current_noise, deterministic=False)
                env.enqueue(item.obs, action, item.station)
            transitions, active, metrics = env.step()
            for tr in transitions:
                buffer.add(tr.obs, tr.next_obs, tr.action, tr.reward, tr.done)
                if buffer.size >= max(batch_size, learning_starts):
                    last_update = agent.update(buffer.sample(batch_size), freeze_actor=freeze_actor)
            current_noise *= 0.9999 if current_noise > 0.1 else 0.999977
        row = {
            'episode': episode,
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
            'mean_q': float(last_update.get('mean_q', 0.0)),
            'freeze_actor': int(bool(freeze_actor)),
        }
        rows.append(row)
        if episode == 1 or episode % print_every == 0 or episode == episodes:
            print(f"[train-agent] ep={episode:03d}/{episodes} reward={row['ep_reward']:.4f} exit={row['exit_vio']} running={row['run_vio']} actor_loss={row['actor_loss']:.6f} critic_loss={row['critic_loss']:.6f}")
    return agent, TrainHistory(rows)


def save_train_history(history: TrainHistory, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalize_result_frame(pd.DataFrame(history.rows), rename_keys=False).to_csv(path, index=False, float_format='%.2f')
    return path


def _build_contexts(env: ChargingEnv, states: list[np.ndarray], stations: list[int], scenario: AttackScenario, is_new_arrival: bool, price_threshold: float, soc_new_threshold: float, soc_rollout_threshold: float, even_station_target: float, odd_station_target: float) -> list[AttackContext]:
    return [
        AttackContext(
            scenario=scenario,
            time_index=env.t,
            raw_price=float(env.signals.price[env.t]),
            station=int(station),
            is_new_arrival=is_new_arrival,
            price_threshold=price_threshold,
            soc_new_threshold=soc_new_threshold,
            soc_rollout_threshold=soc_rollout_threshold,
            even_station_target=even_station_target,
            odd_station_target=odd_station_target,
        )
        for station in stations
    ]


def _fresh_attacker(attacker: PGDStateAttacker | None) -> PGDStateAttacker | None:
    if attacker is None:
        return None
    if hasattr(attacker, 'clone'):
        return attacker.clone()
    return attacker


def _rollout_label(attack_enabled: bool, route_mode: str) -> str:
    mapping = {
        (False, 'none'): 'clean_no_dae',
        (True, 'none'): 'attack_no_dae',
        (False, 'always_dae'): 'clean_dae',
        (True, 'always_dae'): 'attack_dae',
        (False, 'detector'): 'clean_dae_detector',
        (True, 'detector'): 'attack_dae_detector',
        (False, 'oracle'): 'clean_dae_oracle',
        (True, 'oracle'): 'attack_dae_oracle',
    }
    return mapping[(bool(attack_enabled), str(route_mode))]


def evaluate_rollout_bundle(
    arrivals: pd.DataFrame,
    actor: Actor,
    signals_path,
    device: torch.device,
    attack_scenario: AttackScenario,
    attacker: PGDStateAttacker | None = None,
    defender: torch.nn.Module | None = None,
    detector_model: torch.nn.Module | None = None,
    detector_threshold: float | None = None,
    reward_profile: RewardProfile = TRAIN_PROFILE,
    exploration_noise: float = 0.0,
    price_threshold: float = 400.0,
    soc_new_threshold: float = 0.5,
    soc_rollout_threshold: float = 0.3,
    even_station_target: float = 1.0,
    odd_station_target: float = -0.5,
    attack_ratio: float = 1.0,
    attack_scope: AttackScope = 'obs',
    detector_feature_mode: str = 'sequence',
) -> EvaluationBundle:
    clean_summary = rollout_episode(
        arrivals,
        actor,
        signals_path,
        device,
        reward_profile,
        False,
        attack_scenario,
        None,
        None,
        None,
        'none',
        None,
        exploration_noise,
        price_threshold,
        soc_new_threshold,
        soc_rollout_threshold,
        even_station_target,
        odd_station_target,
        attack_ratio,
        attack_scope,
        detector_feature_mode,
    )
    attack_summary = rollout_episode(
        arrivals,
        actor,
        signals_path,
        device,
        reward_profile,
        True,
        attack_scenario,
        _fresh_attacker(attacker),
        None,
        None,
        'none',
        None,
        exploration_noise,
        price_threshold,
        soc_new_threshold,
        soc_rollout_threshold,
        even_station_target,
        odd_station_target,
        attack_ratio,
        attack_scope,
        detector_feature_mode,
    )
    clean_dae_summary = None
    attack_dae_summary = None
    clean_dae_detector_summary = None
    attack_dae_detector_summary = None
    clean_dae_oracle_summary = None
    attack_dae_oracle_summary = None

    if defender is not None:
        clean_dae_summary = rollout_episode(
            arrivals,
            actor,
            signals_path,
            device,
            reward_profile,
            False,
            attack_scenario,
            None,
            defender,
            None,
            'always_dae',
            None,
            exploration_noise,
            price_threshold,
            soc_new_threshold,
            soc_rollout_threshold,
            even_station_target,
            odd_station_target,
            attack_ratio,
            attack_scope,
            detector_feature_mode,
        )
        attack_dae_summary = rollout_episode(
            arrivals,
            actor,
            signals_path,
            device,
            reward_profile,
            True,
            attack_scenario,
            _fresh_attacker(attacker),
            defender,
            None,
            'always_dae',
            None,
            exploration_noise,
            price_threshold,
            soc_new_threshold,
            soc_rollout_threshold,
            even_station_target,
            odd_station_target,
            attack_ratio,
            attack_scope,
            detector_feature_mode,
        )
        clean_dae_oracle_summary = rollout_episode(
            arrivals,
            actor,
            signals_path,
            device,
            reward_profile,
            False,
            attack_scenario,
            None,
            defender,
            None,
            'oracle',
            None,
            exploration_noise,
            price_threshold,
            soc_new_threshold,
            soc_rollout_threshold,
            even_station_target,
            odd_station_target,
            attack_ratio,
            attack_scope,
            detector_feature_mode,
        )
        attack_dae_oracle_summary = rollout_episode(
            arrivals,
            actor,
            signals_path,
            device,
            reward_profile,
            True,
            attack_scenario,
            _fresh_attacker(attacker),
            defender,
            None,
            'oracle',
            None,
            exploration_noise,
            price_threshold,
            soc_new_threshold,
            soc_rollout_threshold,
            even_station_target,
            odd_station_target,
            attack_ratio,
            attack_scope,
            detector_feature_mode,
        )
        if detector_threshold is not None and detector_model is not None:
            clean_dae_detector_summary = rollout_episode(
                arrivals,
                actor,
                signals_path,
                device,
                reward_profile,
                False,
                attack_scenario,
                None,
                defender,
                detector_model,
                'detector',
                detector_threshold,
                exploration_noise,
                price_threshold,
                soc_new_threshold,
                soc_rollout_threshold,
                even_station_target,
                odd_station_target,
                attack_ratio,
                attack_scope,
                detector_feature_mode,
            )
            attack_dae_detector_summary = rollout_episode(
                arrivals,
                actor,
                signals_path,
                device,
                reward_profile,
                True,
                attack_scenario,
                _fresh_attacker(attacker),
                defender,
                detector_model,
                'detector',
                detector_threshold,
                exploration_noise,
                price_threshold,
                soc_new_threshold,
                soc_rollout_threshold,
                even_station_target,
                odd_station_target,
                attack_ratio,
                attack_scope,
                detector_feature_mode,
            )

    return EvaluationBundle(
        clean_summary,
        attack_summary,
        clean_dae_summary,
        attack_dae_summary,
        clean_dae_detector_summary,
        attack_dae_detector_summary,
        clean_dae_oracle_summary,
        attack_dae_oracle_summary,
    )

def save_clean_trajectory_dataset(bundle: CleanTrajectoryBundle, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'clean_inputs': bundle.clean_inputs,
        'metadata': bundle.metadata,
    }
    if bundle.raw_prices is not None:
        payload['raw_prices'] = np.asarray(bundle.raw_prices, dtype=np.float32).reshape(-1)
    if bundle.time_indices is not None:
        payload['time_indices'] = np.asarray(bundle.time_indices, dtype=np.int64).reshape(-1)
    if bundle.stations is not None:
        payload['stations'] = np.asarray(bundle.stations, dtype=np.int64).reshape(-1)
    if bundle.is_new_arrivals is not None:
        payload['is_new_arrivals'] = np.asarray(bundle.is_new_arrivals, dtype=np.int64).reshape(-1)
    if bundle.vehicle_ids is not None:
        payload['vehicle_ids'] = np.asarray(bundle.vehicle_ids, dtype=np.int64).reshape(-1)
    if bundle.episode_indices is not None:
        payload['episode_indices'] = np.asarray(bundle.episode_indices, dtype=np.int64).reshape(-1)
    np.savez_compressed(path, **payload)
    return path


def load_clean_trajectory_dataset(path: str | Path) -> CleanTrajectoryBundle:
    obj = np.load(Path(path), allow_pickle=True)
    try:
        metadata = dict(obj['metadata'].item() if 'metadata' in obj else {})
    except ModuleNotFoundError:
        # NumPy 2.x pickles reference numpy._core, which NumPy 1.x cannot
        # import. The numeric trajectory arrays remain portable and complete.
        metadata = {}
    clean_inputs = np.asarray(obj['clean_inputs'], dtype=np.float32)
    metadata['samples'] = int(clean_inputs.shape[0])
    raw_prices = None if 'raw_prices' not in obj else np.asarray(obj['raw_prices'], dtype=np.float32).reshape(-1)
    time_indices = None if 'time_indices' not in obj else np.asarray(obj['time_indices'], dtype=np.int64).reshape(-1)
    stations = None if 'stations' not in obj else np.asarray(obj['stations'], dtype=np.int64).reshape(-1)
    is_new_arrivals = None if 'is_new_arrivals' not in obj else np.asarray(obj['is_new_arrivals'], dtype=np.int64).reshape(-1)
    vehicle_ids = None if 'vehicle_ids' not in obj else np.asarray(obj['vehicle_ids'], dtype=np.int64).reshape(-1)
    episode_indices = None if 'episode_indices' not in obj else np.asarray(obj['episode_indices'], dtype=np.int64).reshape(-1)
    return CleanTrajectoryBundle(
        clean_inputs=clean_inputs,
        metadata=metadata,
        raw_prices=raw_prices,
        time_indices=time_indices,
        stations=stations,
        is_new_arrivals=is_new_arrivals,
        vehicle_ids=vehicle_ids,
        episode_indices=episode_indices,
    )


def collect_clean_trajectories(
    arrivals: pd.DataFrame,
    actor: Actor,
    signals_path,
    device: torch.device,
    reward_profile: RewardProfile = TRAIN_PROFILE,
    episodes: int = 1,
    max_samples: int | None = None,
) -> CleanTrajectoryBundle:
    actor = actor.to(device).eval()
    clean_list: list[np.ndarray] = []
    raw_prices: list[float] = []
    time_indices: list[int] = []
    stations: list[int] = []
    is_new_arrivals: list[int] = []
    vehicle_ids: list[int] = []
    episode_indices: list[int] = []

    for episode_idx in range(episodes):
        env = ChargingEnv(signals_path=signals_path, reward_profile=reward_profile)
        env.reset()
        idx = 0
        active: list[QueueItem] = []
        active_vehicle_ids: list[int] = []
        while env.t < env.horizon:
            new_states = []
            new_stations = []
            new_vehicle_ids = []
            while idx < len(arrivals) and int(arrivals.loc[idx, 'Arrive_time']) == env.t:
                new_states.append(env.build_initial_obs(int(arrivals.loc[idx, 'Duration_of_stay'])))
                new_stations.append(int(arrivals.loc[idx, 'Station']))
                new_vehicle_ids.append(int(idx))
                idx += 1
            if new_states:
                for clean_obs, station, vehicle_id in zip(new_states, new_stations, new_vehicle_ids):
                    clean_list.append(to_numpy_1d(clean_obs))
                    raw_prices.append(float(env.signals.norm_price[env.t]))
                    time_indices.append(int(env.t))
                    stations.append(int(station))
                    is_new_arrivals.append(1)
                    vehicle_ids.append(int(vehicle_id))
                    episode_indices.append(int(episode_idx))
                with torch.no_grad():
                    actions = actor(torch.as_tensor(np.asarray(new_states, dtype=np.float32), dtype=torch.float32, device=device)).detach().cpu().numpy()
                for clean_obs, action, station in zip(new_states, actions, new_stations):
                    env.enqueue(clean_obs, action, station)
            if max_samples is not None and len(clean_list) >= max_samples:
                break

            if active:
                active_states = [item.obs for item in active]
                active_stations = [item.station for item in active]
                for clean_obs, station, vehicle_id in zip(active_states, active_stations, active_vehicle_ids):
                    clean_list.append(to_numpy_1d(clean_obs))
                    raw_prices.append(float(env.signals.norm_price[env.t]))
                    time_indices.append(int(env.t))
                    stations.append(int(station))
                    is_new_arrivals.append(0)
                    vehicle_ids.append(int(vehicle_id))
                    episode_indices.append(int(episode_idx))
                with torch.no_grad():
                    actions = actor(torch.as_tensor(np.asarray(active_states, dtype=np.float32), dtype=torch.float32, device=device)).detach().cpu().numpy()
                for item, action in zip(active, actions):
                    env.enqueue(item.obs, action, item.station)
            if max_samples is not None and len(clean_list) >= max_samples:
                break
            step_vehicle_ids = new_vehicle_ids + active_vehicle_ids
            transitions, active, _ = env.step()
            active_vehicle_ids = [vid for vid, tr in zip(step_vehicle_ids, transitions) if not bool(tr.done)]
        if max_samples is not None and len(clean_list) >= max_samples:
            break

    clean_arr = np.asarray(clean_list, dtype=np.float32).reshape(-1, 11)
    raw_price_arr = np.asarray(raw_prices, dtype=np.float32).reshape(-1)
    time_arr = np.asarray(time_indices, dtype=np.int64).reshape(-1)
    station_arr = np.asarray(stations, dtype=np.int64).reshape(-1)
    is_new_arr = np.asarray(is_new_arrivals, dtype=np.int64).reshape(-1)
    vehicle_arr = np.asarray(vehicle_ids, dtype=np.int64).reshape(-1)
    episode_arr = np.asarray(episode_indices, dtype=np.int64).reshape(-1)
    if max_samples is not None:
        clean_arr = clean_arr[:max_samples]
        raw_price_arr = raw_price_arr[:max_samples]
        time_arr = time_arr[:max_samples]
        station_arr = station_arr[:max_samples]
        is_new_arr = is_new_arr[:max_samples]
        vehicle_arr = vehicle_arr[:max_samples]
        episode_arr = episode_arr[:max_samples]
    metadata = {
        'samples': int(clean_arr.shape[0]),
        'collection_mode': 'clean_rollout_dnormal',
        'reward_profile': str(reward_profile.name),
    }
    return CleanTrajectoryBundle(
        clean_inputs=clean_arr,
        metadata=metadata,
        raw_prices=raw_price_arr,
        time_indices=time_arr,
        stations=station_arr,
        is_new_arrivals=is_new_arr,
        vehicle_ids=vehicle_arr,
        episode_indices=episode_arr,
    )


def build_pair_dataset_from_clean_trajectories(
    dataset: CleanTrajectoryBundle,
    attacker: PGDStateAttacker,
    attack_scenario: AttackScenario,
    *,
    price_threshold: float = 400.0,
    soc_new_threshold: float = 0.5,
    soc_rollout_threshold: float = 0.3,
    even_station_target: float = 1.0,
    odd_station_target: float = -0.5,
    attack_ratio: float = 1.0,
    attack_scope: AttackScope = 'obs',
    chunk_size: int = 1024,
) -> PairDatasetBundle:
    clean_inputs = np.asarray(dataset.clean_inputs, dtype=np.float32).reshape(-1, 11)
    total = int(clean_inputs.shape[0])
    if total == 0:
        return PairDatasetBundle(
            adv_inputs=np.zeros((0, 11), dtype=np.float32),
            clean_inputs=np.zeros((0, 11), dtype=np.float32),
            metadata={
                'samples': 0,
                'source_samples': 0,
                'collection_mode': 'offline_attack_from_dnormal',
                'policy_input_mode': 'clean',
            },
        )
    if dataset.time_indices is None or dataset.stations is None or dataset.is_new_arrivals is None or dataset.vehicle_ids is None or dataset.episode_indices is None:
        raise ValueError('CleanTrajectoryBundle is missing metadata required for offline attack reconstruction.')
    if dataset.raw_prices is None:
        raise ValueError('CleanTrajectoryBundle is missing raw_prices required for ElectHacker contexts.')

    adv_arr = clean_inputs.copy()
    attack_mask = np.zeros((total,), dtype=np.int64)

    for start_idx in range(0, total, int(chunk_size)):
        end_idx = min(total, start_idx + int(chunk_size))
        obs_batch = [clean_inputs[i] for i in range(start_idx, end_idx)]
        contexts = [
            AttackContext(
                scenario=attack_scenario,
                time_index=int(dataset.time_indices[i]),
                raw_price=float(dataset.raw_prices[i]),
                station=int(dataset.stations[i]),
                is_new_arrival=bool(dataset.is_new_arrivals[i]),
                price_threshold=price_threshold,
                soc_new_threshold=soc_new_threshold,
                soc_rollout_threshold=soc_rollout_threshold,
                even_station_target=even_station_target,
                odd_station_target=odd_station_target,
            )
            for i in range(start_idx, end_idx)
        ]
        attacked_states, flags = attack_batch_by_context(
            attacker,
            obs_batch,
            contexts,
            attack_ratio=attack_ratio,
            attack_scope=attack_scope,
            vehicle_ids=np.asarray(dataset.vehicle_ids[start_idx:end_idx], dtype=np.int64),
            episode_indices=np.asarray(dataset.episode_indices[start_idx:end_idx], dtype=np.int64),
            episode_index=int(dataset.episode_indices[start_idx]) if end_idx > start_idx else 0,
            seed=int(getattr(attacker, 'seed', 42)),
        )
        adv_arr[start_idx:end_idx] = np.asarray(attacked_states, dtype=np.float32).reshape(-1, 11)
        attack_mask[start_idx:end_idx] = np.asarray(flags, dtype=np.int64).reshape(-1)

    metadata = {
        'samples': int(clean_inputs.shape[0]),
        'source_samples': int(total),
        'collection_mode': 'offline_attack_from_dnormal',
        'policy_input_mode': 'clean',
        'attack_ratio': float(np.clip(attack_ratio, 0.0, 1.0)),
        'attack_scope': str(attack_scope),
        'attacked_samples': int(attack_mask.sum()),
    }
    return PairDatasetBundle(
        adv_inputs=adv_arr,
        clean_inputs=clean_inputs,
        metadata=metadata,
        time_indices=np.asarray(dataset.time_indices, dtype=np.int64).reshape(-1),
        stations=np.asarray(dataset.stations, dtype=np.int64).reshape(-1),
        is_new_arrivals=np.asarray(dataset.is_new_arrivals, dtype=np.int64).reshape(-1),
        vehicle_ids=np.asarray(dataset.vehicle_ids, dtype=np.int64).reshape(-1),
        episode_indices=np.asarray(dataset.episode_indices, dtype=np.int64).reshape(-1),
        attack_mask=attack_mask,
    )


def save_pair_dataset(bundle: PairDatasetBundle, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'adv_inputs': bundle.adv_inputs,
        'clean_inputs': bundle.clean_inputs,
        'metadata': bundle.metadata,
    }
    if bundle.clean_anchor_inputs is not None:
        payload['clean_anchor_inputs'] = bundle.clean_anchor_inputs
    if bundle.time_indices is not None:
        payload['time_indices'] = np.asarray(bundle.time_indices, dtype=np.int64).reshape(-1)
    if bundle.stations is not None:
        payload['stations'] = np.asarray(bundle.stations, dtype=np.int64).reshape(-1)
    if bundle.is_new_arrivals is not None:
        payload['is_new_arrivals'] = np.asarray(bundle.is_new_arrivals, dtype=np.int64).reshape(-1)
    if bundle.vehicle_ids is not None:
        payload['vehicle_ids'] = np.asarray(bundle.vehicle_ids, dtype=np.int64).reshape(-1)
    if bundle.episode_indices is not None:
        payload['episode_indices'] = np.asarray(bundle.episode_indices, dtype=np.int64).reshape(-1)
    if bundle.attack_mask is not None:
        payload['attack_mask'] = np.asarray(bundle.attack_mask, dtype=np.int64).reshape(-1)
    np.savez_compressed(path, **payload)
    return path


def load_pair_dataset(path: str | Path) -> PairDatasetBundle:
    obj = np.load(Path(path), allow_pickle=True)
    metadata = dict(obj['metadata'].item() if 'metadata' in obj else {})
    clean_anchor_inputs = None
    time_indices = None
    stations = None
    is_new_arrivals = None
    vehicle_ids = None
    episode_indices = None
    attack_mask = None
    adv_inputs = np.asarray(obj['adv_inputs'], dtype=np.float32)
    clean_inputs = np.asarray(obj['clean_inputs'], dtype=np.float32)
    if 'clean_anchor_inputs' in obj:
        clean_anchor_inputs = np.asarray(obj['clean_anchor_inputs'], dtype=np.float32)
    if 'time_indices' in obj:
        time_indices = np.asarray(obj['time_indices'], dtype=np.int64).reshape(-1)
    if 'stations' in obj:
        stations = np.asarray(obj['stations'], dtype=np.int64).reshape(-1)
    if 'is_new_arrivals' in obj:
        is_new_arrivals = np.asarray(obj['is_new_arrivals'], dtype=np.int64).reshape(-1)
    if 'vehicle_ids' in obj:
        vehicle_ids = np.asarray(obj['vehicle_ids'], dtype=np.int64).reshape(-1)
    if 'episode_indices' in obj:
        episode_indices = np.asarray(obj['episode_indices'], dtype=np.int64).reshape(-1)
    if 'attack_mask' in obj:
        attack_mask = np.asarray(obj['attack_mask'], dtype=np.int64).reshape(-1)
    if attack_mask is None:
        attack_mask = (np.max(np.abs(adv_inputs - clean_inputs), axis=1) > 1e-8).astype(np.int64)
    metadata['samples'] = int(clean_inputs.shape[0])
    metadata['attacked_samples'] = int(np.asarray(attack_mask, dtype=np.int64).sum())
    return PairDatasetBundle(
        adv_inputs,
        clean_inputs,
        metadata,
        clean_anchor_inputs=clean_anchor_inputs,
        time_indices=time_indices,
        stations=stations,
        is_new_arrivals=is_new_arrivals,
        vehicle_ids=vehicle_ids,
        episode_indices=episode_indices,
        attack_mask=attack_mask,
    )


def train_dae_from_bundle(
    bundle: PairDatasetBundle,
    actor: Actor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    lambda_state: float = 1.0,
    lambda_identity: float = 1.0,
    validator: Callable[[torch.nn.Module], dict] | None = None,
    val_every: int = 1,
    select_by: str = 'reward_recovery',
    log_every: int = 1,
    seq_len: int = 8,
    hidden_dim: int = 128,
    latent_dim: int = 64,
    num_layers: int = 1,
    decoder_hidden_dim: int = 128,
    beta_kl: float = 1e-3,
    lambda_robust: float = 0.0,
    include_clean_sequences: bool = True,
    state_scope: str = 'local',
    progress_dir: str | Path | None = None,
    progress_prefix: str = 'dae',
) -> tuple[DenoisingAutoencoder, DAETrainResult]:
    return train_dae(
        bundle,
        actor,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        log_every=log_every,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
        decoder_hidden_dim=decoder_hidden_dim,
        beta_kl=beta_kl,
        lambda_recon=lambda_state,
        lambda_identity=lambda_identity,
        lambda_robust=lambda_robust,
        include_clean_sequences=include_clean_sequences,
        validator=validator,
        val_every=val_every,
        select_by=select_by,
        state_scope=state_scope,
        progress_dir=progress_dir,
        progress_prefix=progress_prefix,
    )


def evaluate_action_dataset(
    clean_inputs: np.ndarray,
    adv_inputs: np.ndarray,
    actor: Actor,
    device: torch.device,
    defender: torch.nn.Module | None = None,
    policy_actor: Actor | None = None,
    episode_indices: np.ndarray | None = None,
    vehicle_ids: np.ndarray | None = None,
    seq_len: int | None = None,
    batch_size: int = 1024,
) -> dict:
    actor = actor.to(device).eval()
    policy = actor if policy_actor is None else policy_actor.to(device).eval()
    clean_arr = np.asarray(clean_inputs, dtype=np.float32).reshape(-1, 11)
    adv_arr = np.asarray(adv_inputs, dtype=np.float32).reshape(-1, 11)
    clean_t = torch.as_tensor(clean_arr, dtype=torch.float32, device=device)
    adv_t = torch.as_tensor(adv_arr, dtype=torch.float32, device=device)
    with torch.no_grad():
        clean_act = actor(clean_t).detach().cpu().numpy().reshape(-1)
        adv_act = policy(adv_t).detach().cpu().numpy().reshape(-1)
        result = {
            'sample_count': int(clean_t.shape[0]),
            'clean_attack_action_mse': float(np.mean((clean_act - adv_act) ** 2)),
            'clean_attack_action_mae': float(np.mean(np.abs(clean_act - adv_act))),
            'attack_sign_flip_rate': float(np.mean(np.sign(clean_act) != np.sign(adv_act))),
        }
        if defender is not None:
            if isinstance(defender, DenoisingAutoencoder):
                rec_inputs = dae_reconstruction_with_history(
                    defender,
                    adv_arr,
                    device,
                    episode_indices=episode_indices,
                    vehicle_ids=vehicle_ids,
                    batch_size=batch_size,
                    seq_len=seq_len,
                )
            else:
                rec_inputs = defender(adv_t).detach().cpu().numpy().astype(np.float32)
            rec_t = torch.as_tensor(rec_inputs, dtype=torch.float32, device=device)
            rec_act = policy(rec_t).detach().cpu().numpy().reshape(-1)
            result.update(
                {
                    'clean_recovered_action_mse': float(np.mean((clean_act - rec_act) ** 2)),
                    'clean_recovered_action_mae': float(np.mean(np.abs(clean_act - rec_act))),
                    'recover_sign_flip_rate': float(np.mean(np.sign(clean_act) != np.sign(rec_act))),
                }
            )
    return normalize_result_object(result, rename_keys=False)


def _safe_recovery_ratio(clean: float, attack: float, defended: float | None) -> float | None:
    if defended is None:
        return None
    drop = float(clean) - float(attack)
    if abs(drop) < 1e-12:
        return None
    return (float(defended) - float(attack)) / drop


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    denom = float(denominator)
    if abs(denom) < 1e-12:
        return None
    return float(numerator) / denom


def _safe_retention_ratio(clean_baseline_reward: float, clean_reward: float) -> float | None:
    return _safe_ratio(clean_reward, clean_baseline_reward)


def _clean_reward_lower_bound(clean_reward: float, floor_ratio: float) -> float:
    floor_ratio = float(np.clip(floor_ratio, 0.0, 1.0))
    allowed_drop = abs(float(clean_reward)) * (1.0 - floor_ratio)
    return float(clean_reward) - allowed_drop


def _detector_threshold_candidates(scores: np.ndarray, grid_size: int) -> list[float]:
    arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return [float('inf')]
    quantiles = np.unique(np.quantile(arr, np.linspace(0.0, 1.0, max(int(grid_size), 5)))).astype(np.float32).tolist()
    quantiles.append(float(np.max(arr) + 1e-6))
    return [float(v) for v in quantiles]


def _cast_nullable_int_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].round().astype('Int64')
    return out


def build_paper_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in raw_df.iterrows():
        def _maybe_float(key: str) -> float | None:
            value = row.get(key)
            return None if pd.isna(value) else float(value)

        clean_no_dae = float(row['rollout_clean_reward'])
        attack_no_dae = float(row['rollout_attack_reward'])
        clean_dae = _maybe_float('rollout_clean_dae_reward')
        attack_dae = _maybe_float('rollout_attack_dae_reward')
        clean_dae_detector = _maybe_float('rollout_clean_dae_detector_reward')
        attack_dae_detector = _maybe_float('rollout_attack_dae_detector_reward')
        clean_dae_oracle = _maybe_float('rollout_clean_dae_oracle_reward')
        attack_dae_oracle = _maybe_float('rollout_attack_dae_oracle_reward')

        rows.append(
            {
                'attack': row['attack'],
                'epsilon': float(row['epsilon']),
                'clean_no_dae': clean_no_dae,
                'attack_no_dae': attack_no_dae,
                'clean_dae': clean_dae,
                'attack_dae': attack_dae,
                'clean_dae_detector': clean_dae_detector,
                'attack_dae_detector': attack_dae_detector,
                'clean_dae_oracle': clean_dae_oracle,
                'attack_dae_oracle': attack_dae_oracle,
                'reward_drop': clean_no_dae - attack_no_dae,
                'reward_recovery_dae': None if attack_dae is None else attack_dae - attack_no_dae,
                'reward_recovery_ratio_dae': _safe_recovery_ratio(clean_no_dae, attack_no_dae, attack_dae),
                'reward_recovery_detector': None if attack_dae_detector is None else attack_dae_detector - attack_no_dae,
                'reward_recovery_ratio_detector': _safe_recovery_ratio(clean_no_dae, attack_no_dae, attack_dae_detector),
                'reward_recovery_oracle': None if attack_dae_oracle is None else attack_dae_oracle - attack_no_dae,
                'reward_recovery_ratio_oracle': _safe_recovery_ratio(clean_no_dae, attack_no_dae, attack_dae_oracle),
                'ep_r1_attack_dae': _maybe_float('rollout_attack_dae_ep_r1'),
                'ep_r2_attack_dae': _maybe_float('rollout_attack_dae_ep_r2'),
                'ep_r3_attack_dae': _maybe_float('rollout_attack_dae_ep_r3'),
                'ep_r1_attack_dae_detector': _maybe_float('rollout_attack_dae_detector_ep_r1'),
                'ep_r2_attack_dae_detector': _maybe_float('rollout_attack_dae_detector_ep_r2'),
                'ep_r3_attack_dae_detector': _maybe_float('rollout_attack_dae_detector_ep_r3'),
                'ep_r1_attack_dae_oracle': _maybe_float('rollout_attack_dae_oracle_ep_r1'),
                'ep_r2_attack_dae_oracle': _maybe_float('rollout_attack_dae_oracle_ep_r2'),
                'ep_r3_attack_dae_oracle': _maybe_float('rollout_attack_dae_oracle_ep_r3'),
                'sample_count': int(row['action_sample_count']),
            }
        )

    order = [
        'attack', 'epsilon',
        'clean_no_dae', 'attack_no_dae',
        'clean_dae', 'attack_dae',
        'clean_dae_detector', 'attack_dae_detector',
        'clean_dae_oracle', 'attack_dae_oracle',
        'reward_drop',
        'reward_recovery_dae', 'reward_recovery_ratio_dae',
        'reward_recovery_detector', 'reward_recovery_ratio_detector',
        'reward_recovery_oracle', 'reward_recovery_ratio_oracle',
        'ep_r1_attack_dae', 'ep_r2_attack_dae', 'ep_r3_attack_dae',
        'ep_r1_attack_dae_detector', 'ep_r2_attack_dae_detector', 'ep_r3_attack_dae_detector',
        'ep_r1_attack_dae_oracle', 'ep_r2_attack_dae_oracle', 'ep_r3_attack_dae_oracle',
        'sample_count',
    ]
    out = normalize_result_frame(pd.DataFrame(rows)[order], rename_keys=False, digits=4)
    return _cast_nullable_int_columns(out, ['sample_count'])


def build_paper_display_table(paper_df: pd.DataFrame) -> pd.DataFrame:
    display = paper_df.copy()
    ratio_cols = [col for col in display.columns if str(col).endswith('_ratio') or '_ratio_' in str(col)]
    for col in ratio_cols:
        display[col] = display[col].apply(lambda x: None if pd.isna(x) else x * 100.0)
    display = normalize_result_frame(display, rename_keys=False, digits=2)
    return _cast_nullable_int_columns(display, ['sample_count'])


def _to_markdown_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    header = '| ' + ' | '.join(cols) + ' |'
    sep = '| ' + ' | '.join(['---'] * len(cols)) + ' |'
    body = []
    for _, row in df.iterrows():
        vals = []
        for v in row.tolist():
            if pd.isna(v):
                vals.append('')
            elif isinstance(v, float):
                vals.append(f'{v:.2f}')
            else:
                vals.append(str(v))
        body.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join([header, sep, *body]) + '\n'


def save_matrix_outputs(
    raw_df: pd.DataFrame,
    output_dir: str | Path,
    manifest_rows: pd.DataFrame | list[dict] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = ensure_dir(output_dir)
    raw_df = pd.DataFrame(raw_df) if not isinstance(raw_df, pd.DataFrame) else raw_df.copy()
    raw_out = normalize_result_frame(raw_df, rename_keys=False, digits=4)
    paper_df = build_paper_table(raw_out)
    display_df = build_paper_display_table(paper_df)

    raw_out.to_csv(output_dir / 'matrix_raw.csv', index=False, float_format='%.4f')
    paper_df.to_csv(output_dir / 'matrix_summary.csv', index=False, float_format='%.4f')
    paper_df.to_csv(output_dir / 'paper_table.csv', index=False, float_format='%.4f')
    (output_dir / 'paper_table.md').write_text(_to_markdown_table(display_df), encoding='utf-8')
    (output_dir / 'paper_table.tex').write_text(display_df.to_latex(index=False, escape=True, na_rep='', float_format=lambda x: f'{x:.2f}'), encoding='utf-8')

    if manifest_rows is not None:
        manifest_df = pd.DataFrame(manifest_rows)
        manifest_df = normalize_result_frame(manifest_df, rename_keys=False, digits=4)
        manifest_df.to_csv(output_dir / 'matrix_manifest.csv', index=False, float_format='%.4f')
        json_dump(manifest_df.to_dict(orient='records'), output_dir / 'matrix_manifest.json', normalize_numbers=True, rename_keys=False)

    return raw_out, paper_df


# === detector dataset and routing helpers ===
@dataclass
class DetectorDatasetBundle:
    clean_inputs: np.ndarray
    adv_inputs: np.ndarray
    metadata: dict
    time_indices: np.ndarray
    stations: np.ndarray
    is_new_arrivals: np.ndarray
    vehicle_ids: np.ndarray
    episode_indices: np.ndarray
    attack_mask: np.ndarray | None = None
    clean_refs: np.ndarray | None = None
    obs_inputs: np.ndarray | None = None
    rec_inputs: np.ndarray | None = None
    labels: np.ndarray | None = None
    benefit_scores: np.ndarray | None = None
    prev_obs_inputs: np.ndarray | None = None
    sample_weights: np.ndarray | None = None


@dataclass
class DetectorSelectionResult:
    threshold: float
    threshold_metric: str
    history_rows: list[dict]
    best_index: int
    clean_accuracy: float
    attack_precision: float
    attack_recall: float
    attack_f1: float
    false_negative_rate: float


def save_detector_dataset(bundle: DetectorDatasetBundle, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'clean_inputs': np.asarray(bundle.clean_inputs, dtype=np.float32),
        'adv_inputs': np.asarray(bundle.adv_inputs, dtype=np.float32),
        'time_indices': np.asarray(bundle.time_indices, dtype=np.int64),
        'stations': np.asarray(bundle.stations, dtype=np.int64),
        'is_new_arrivals': np.asarray(bundle.is_new_arrivals, dtype=np.int64),
        'vehicle_ids': np.asarray(bundle.vehicle_ids, dtype=np.int64),
        'episode_indices': np.asarray(bundle.episode_indices, dtype=np.int64),
        'metadata': bundle.metadata,
    }
    if bundle.attack_mask is not None:
        payload['attack_mask'] = np.asarray(bundle.attack_mask, dtype=np.int64)
    if bundle.clean_refs is not None:
        payload['clean_refs'] = np.asarray(bundle.clean_refs, dtype=np.float32)
    if bundle.obs_inputs is not None:
        payload['obs_inputs'] = np.asarray(bundle.obs_inputs, dtype=np.float32)
    if bundle.rec_inputs is not None:
        payload['rec_inputs'] = np.asarray(bundle.rec_inputs, dtype=np.float32)
    if bundle.labels is not None:
        payload['labels'] = np.asarray(bundle.labels, dtype=np.int64)
    if bundle.benefit_scores is not None:
        payload['benefit_scores'] = np.asarray(bundle.benefit_scores, dtype=np.float32)
    if bundle.prev_obs_inputs is not None:
        payload['prev_obs_inputs'] = np.asarray(bundle.prev_obs_inputs, dtype=np.float32)
    if bundle.sample_weights is not None:
        payload['sample_weights'] = np.asarray(bundle.sample_weights, dtype=np.float32)
    np.savez_compressed(path, **payload)
    return path


def load_detector_dataset(path: str | Path) -> DetectorDatasetBundle:
    obj = np.load(Path(path), allow_pickle=True)
    metadata = dict(obj['metadata'].item() if 'metadata' in obj else {})
    clean_inputs = np.asarray(obj['clean_inputs'], dtype=np.float32).reshape(-1, 11)
    adv_inputs = np.asarray(obj['adv_inputs'], dtype=np.float32).reshape(-1, 11)
    total = clean_inputs.shape[0]
    attack_mask = None if 'attack_mask' not in obj else np.asarray(obj['attack_mask'], dtype=np.int64).reshape(-1)
    if attack_mask is None:
        attack_mask = (np.max(np.abs(adv_inputs - clean_inputs), axis=1) > 1e-8).astype(np.int64)
    metadata['samples'] = int(total)
    metadata['attacked_samples'] = int(np.asarray(attack_mask, dtype=np.int64).sum())
    return DetectorDatasetBundle(
        clean_inputs=clean_inputs,
        adv_inputs=adv_inputs,
        metadata=metadata,
        time_indices=np.asarray(obj['time_indices'], dtype=np.int64).reshape(-1),
        stations=np.asarray(obj['stations'], dtype=np.int64).reshape(-1),
        is_new_arrivals=np.asarray(obj['is_new_arrivals'], dtype=np.int64).reshape(-1),
        vehicle_ids=np.asarray(obj['vehicle_ids'], dtype=np.int64).reshape(-1) if 'vehicle_ids' in obj else np.arange(total, dtype=np.int64),
        episode_indices=np.asarray(obj['episode_indices'], dtype=np.int64).reshape(-1) if 'episode_indices' in obj else np.zeros((total,), dtype=np.int64),
        attack_mask=attack_mask,
        clean_refs=None if 'clean_refs' not in obj else np.asarray(obj['clean_refs'], dtype=np.float32).reshape(-1, 11),
        obs_inputs=None if 'obs_inputs' not in obj else np.asarray(obj['obs_inputs'], dtype=np.float32).reshape(-1, 11),
        rec_inputs=None if 'rec_inputs' not in obj else np.asarray(obj['rec_inputs'], dtype=np.float32).reshape(-1, 11),
        labels=None if 'labels' not in obj else np.asarray(obj['labels'], dtype=np.int64).reshape(-1),
        benefit_scores=None if 'benefit_scores' not in obj else np.asarray(obj['benefit_scores'], dtype=np.float32).reshape(-1),
        prev_obs_inputs=None if 'prev_obs_inputs' not in obj else np.asarray(obj['prev_obs_inputs'], dtype=np.float32).reshape(-1, 11),
        sample_weights=None if 'sample_weights' not in obj else np.asarray(obj['sample_weights'], dtype=np.float32).reshape(-1),
    )


def train_detector_from_bundle(
    dataset: DetectorDatasetBundle,
    actor: Actor,
    defender: torch.nn.Module | None,
    device: torch.device,
    *,
    compare_actor: Actor | None = None,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    dropout: float = 0.1,
    val_ratio: float = 0.2,
    detector_temporal: bool = True,
    detector_feature_mode: str = 'sequence',
    seed: int = 42,
    latent_dim: int = 64,
    num_layers: int = 1,
    beta_kl: float = 1e-3,
    seq_len: int = 8,
    state_scope: str = 'local',
    progress_dir: str | Path | None = None,
    progress_prefix: str = 'detector',
    val_every: int = 1,
) -> tuple[torch.nn.Module, DetectorTrainResult]:
    detector_mode = str((dataset.metadata or {}).get('detector_mode', 'pre')).strip().lower()
    if detector_mode == 'posterior':
        if dataset.obs_inputs is None or dataset.rec_inputs is None or dataset.labels is None:
            raise ValueError('Posterior detector dataset requires obs_inputs, rec_inputs, and labels.')
        return train_posterior_detector(
            dataset.obs_inputs,
            dataset.rec_inputs,
            dataset.labels,
            actor,
            device,
            time_indices=dataset.time_indices,
            stations=dataset.stations,
            is_new_arrivals=dataset.is_new_arrivals,
            episode_indices=dataset.episode_indices,
            vehicle_ids=dataset.vehicle_ids,
            prev_obs_inputs=dataset.prev_obs_inputs,
            include_temporal=bool(detector_temporal),
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            hidden_dim=hidden_dim,
            dropout=dropout,
            val_ratio=val_ratio,
            seed=seed,
            sample_weights=dataset.sample_weights,
            progress_dir=progress_dir,
            progress_prefix=progress_prefix,
            val_every=val_every,
        )
    from .defense import train_sequence_detector
    return train_sequence_detector(dataset.clean_inputs, device, episode_indices=dataset.episode_indices, vehicle_ids=dataset.vehicle_ids, eval_clean_inputs=dataset.clean_inputs, eval_adv_inputs=dataset.adv_inputs, eval_episode_indices=dataset.episode_indices, eval_vehicle_ids=dataset.vehicle_ids, seq_len=seq_len, hidden_dim=hidden_dim, latent_dim=latent_dim, num_layers=num_layers, beta_kl=beta_kl, epochs=epochs, batch_size=batch_size, lr=lr, val_ratio=val_ratio, seed=seed, state_scope=state_scope, progress_dir=progress_dir, progress_prefix=progress_prefix)


def select_detector_threshold(dataset: DetectorDatasetBundle, detector_model: torch.nn.Module, arrivals: pd.DataFrame, actor: Actor, signals_path, device: torch.device, attack_scenario: AttackScenario, *, attacker: PGDStateAttacker | None, defender: torch.nn.Module, reward_profile: RewardProfile = TRAIN_PROFILE, grid_size: int = 31, clean_reward_floor_ratio: float = 0.99, detector_feature_mode: str = 'sequence', exploration_noise: float = 0.0, price_threshold: float = 400.0, soc_new_threshold: float = 0.5, soc_rollout_threshold: float = 0.3, even_station_target: float = 1.0, odd_station_target: float = -0.5, attack_ratio: float = 1.0, attack_scope: AttackScope = 'obs') -> DetectorSelectionResult:
    detector_mode = str((dataset.metadata or {}).get('detector_mode', 'pre')).strip().lower()
    if detector_mode == 'posterior':
        if not isinstance(detector_model, PosteriorBenefitMLPDetector):
            raise ValueError('Posterior detector threshold selection requires PosteriorBenefitMLPDetector.')
        if dataset.obs_inputs is None or dataset.rec_inputs is None:
            raise ValueError('Posterior detector dataset requires obs_inputs and rec_inputs.')
        probabilities = posterior_detector_probabilities(
            detector_model,
            dataset.obs_inputs,
            dataset.rec_inputs,
            actor,
            device,
            time_indices=dataset.time_indices,
            stations=dataset.stations,
            is_new_arrivals=dataset.is_new_arrivals,
            prev_obs_inputs=dataset.prev_obs_inputs,
            include_temporal=bool(getattr(detector_model, 'include_temporal', True)),
        )
        labels = None if dataset.labels is None else np.asarray(dataset.labels, dtype=np.int64).reshape(-1)
        candidates = _detector_threshold_candidates(probabilities, grid_size)
        clean_baseline = rollout_episode(
            arrivals,
            actor,
            signals_path,
            device,
            reward_profile,
            False,
            attack_scenario,
            None,
            None,
            None,
            'none',
            None,
            exploration_noise,
            price_threshold,
            soc_new_threshold,
            soc_rollout_threshold,
            even_station_target,
            odd_station_target,
            attack_ratio,
            attack_scope,
            detector_feature_mode,
        )
        attack_baseline = rollout_episode(
            arrivals,
            actor,
            signals_path,
            device,
            reward_profile,
            True,
            attack_scenario,
            _fresh_attacker(attacker),
            None,
            None,
            'none',
            None,
            exploration_noise,
            price_threshold,
            soc_new_threshold,
            soc_rollout_threshold,
            even_station_target,
            odd_station_target,
            attack_ratio,
            attack_scope,
            detector_feature_mode,
        )
        clean_floor = _clean_reward_lower_bound(float(clean_baseline['ep_reward']), clean_reward_floor_ratio)
        history_rows: list[dict] = []
        best_idx = -1
        best_key = None
        for idx, threshold in enumerate(candidates):
            benefit_metrics = None if labels is None else _binary_operating_metrics_np(labels, probabilities, threshold=float(threshold))
            clean_detector = rollout_episode(
                arrivals,
                actor,
                signals_path,
                device,
                reward_profile,
                False,
                attack_scenario,
                None,
                defender,
                detector_model,
                'detector',
                threshold,
                exploration_noise,
                price_threshold,
                soc_new_threshold,
                soc_rollout_threshold,
                even_station_target,
                odd_station_target,
                attack_ratio,
                attack_scope,
                detector_feature_mode,
            )
            attack_detector = rollout_episode(
                arrivals,
                actor,
                signals_path,
                device,
                reward_profile,
                True,
                attack_scenario,
                _fresh_attacker(attacker),
                defender,
                detector_model,
                'detector',
                threshold,
                exploration_noise,
                price_threshold,
                soc_new_threshold,
                soc_rollout_threshold,
                even_station_target,
                odd_station_target,
                attack_ratio,
                attack_scope,
                detector_feature_mode,
            )
            clean_reward = float(clean_detector['ep_reward'])
            attack_reward = float(attack_detector['ep_reward'])
            recovery = float(attack_reward - float(attack_baseline['ep_reward']))
            clean_reward_retention = _safe_retention_ratio(float(clean_baseline['ep_reward']), clean_reward)
            clean_ok = clean_reward >= clean_floor
            key = (
                1.0 if clean_ok else 0.0,
                float(recovery) if clean_ok else -float('inf'),
                float(clean_reward),
                -float(clean_detector.get('route_rate', 0.0)),
                float((benefit_metrics or {}).get('f1', 0.0)),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_idx = idx
            history_rows.append(
                {
                    'candidate_rank': int(idx),
                    'threshold': float(threshold),
                    'benefit_precision': None if benefit_metrics is None else float(benefit_metrics['precision']),
                    'benefit_recall': None if benefit_metrics is None else float(benefit_metrics['recall']),
                    'benefit_f1': None if benefit_metrics is None else float(benefit_metrics['f1']),
                    'clean_reward': clean_reward,
                    'attack_reward': attack_reward,
                    'reward_recovery': recovery,
                    'clean_reward_retention': clean_reward_retention,
                    'clean_reward_floor': float(clean_floor),
                    'clean_constraint_ok': bool(clean_ok),
                    'clean_rollout_route_rate': float(clean_detector.get('route_rate', 0.0)),
                    'attack_rollout_route_rate': float(attack_detector.get('route_rate', 0.0)),
                }
            )
        if best_idx < 0:
            return DetectorSelectionResult(float('inf'), 'posterior_probability', [], -1, 1.0, 0.0, 0.0, 0.0, 1.0)
        row = history_rows[best_idx]
        return DetectorSelectionResult(
            float(row['threshold']),
            'posterior_probability',
            history_rows,
            int(best_idx),
            float(1.0 if row['clean_constraint_ok'] else 0.0),
            0.0 if row['benefit_precision'] is None else float(row['benefit_precision']),
            0.0 if row['benefit_recall'] is None else float(row['benefit_recall']),
            0.0 if row['benefit_f1'] is None else float(row['benefit_f1']),
            0.0,
        )
    from .defense import detector_anomaly_scores, select_canomaly_from_scores
    seq_len = int(getattr(detector_model, 'seq_len', 1))
    clean_scores = detector_anomaly_scores(detector_model, dataset.clean_inputs, device, episode_indices=dataset.episode_indices, vehicle_ids=dataset.vehicle_ids, seq_len=seq_len)
    adv_scores = detector_anomaly_scores(detector_model, dataset.adv_inputs, device, episode_indices=dataset.episode_indices, vehicle_ids=dataset.vehicle_ids, seq_len=seq_len)
    threshold, rows, best_idx = select_canomaly_from_scores(clean_scores, adv_scores, grid_size=grid_size, min_clean_accuracy=clean_reward_floor_ratio)
    if best_idx < 0:
        return DetectorSelectionResult(float('inf'), 'linf_reconstruction', [], -1, 1.0, 0.0, 0.0, 0.0, 1.0)
    row = rows[best_idx]
    return DetectorSelectionResult(float(threshold), 'linf_reconstruction', rows, int(best_idx), float(row['clean_accuracy']), float(row['attack_precision']), float(row['attack_recall']), float(row['attack_f1']), float(row['false_negative_rate']))


def _route_policy_states(attacked_states: list[np.ndarray], attacked_flags: list[bool], defender: torch.nn.Module | None, detector_model, actor: Actor, device: torch.device, *, route_mode: str, detector_threshold: float | None, detector_feature_mode: str = 'sequence', time_indices: list[int] | None = None, stations: list[int] | None = None, is_new_arrivals: list[int] | None = None, prev_obs_refs: list[np.ndarray] | None = None, vehicle_ids: list[int] | None = None, episode_index: int = 0, dae_runtime: SequentialDAERuntime | None = None, detector_runtime=None) -> tuple[list[np.ndarray], list[bool], np.ndarray]:
    from .defense import DetectorGRUVAE, SequentialDetectorRuntime
    if route_mode == 'none':
        return [to_numpy_1d(s) for s in attacked_states], [False for _ in attacked_states], np.full((len(attacked_states),), np.nan, dtype=np.float32)
    if route_mode == 'always_dae':
        if defender is None:
            raise ValueError('route_mode=always_dae requires defender.')
        recovered = dae_runtime.reconstruct_batch(attacked_states, vehicle_ids=vehicle_ids, episode_index=episode_index) if dae_runtime is not None and vehicle_ids is not None else reconstruction_batch(defender, attacked_states, device)
        flags = [True for _ in attacked_states]
        return [recovered[i].reshape(-1) for i in range(recovered.shape[0])], flags, np.full((len(attacked_states),), np.nan, dtype=np.float32)
    if route_mode == 'oracle':
        if defender is None:
            raise ValueError('route_mode=oracle requires defender.')
        recovered = dae_runtime.reconstruct_batch(attacked_states, vehicle_ids=vehicle_ids, episode_index=episode_index) if dae_runtime is not None and vehicle_ids is not None else reconstruction_batch(defender, attacked_states, device)
        flags = [bool(flag) for flag in attacked_flags]
        routed = [recovered[i].reshape(-1) if flags[i] else to_numpy_1d(attacked_states[i]) for i in range(len(flags))]
        return routed, flags, np.full((len(flags),), np.nan, dtype=np.float32)
    if route_mode == 'detector':
        if detector_threshold is None:
            raise ValueError('route_mode=detector requires detector_threshold.')
        if detector_model is None:
            raise ValueError('route_mode=detector requires detector_model.')
        if isinstance(detector_model, PosteriorBenefitMLPDetector):
            if defender is None:
                raise ValueError('posterior detector routing requires defender.')
            recovered = dae_runtime.reconstruct_batch(attacked_states, vehicle_ids=vehicle_ids, episode_index=episode_index) if dae_runtime is not None and vehicle_ids is not None else reconstruction_batch(defender, attacked_states, device)
            scores = posterior_detector_probabilities(
                detector_model,
                attacked_states,
                recovered,
                actor,
                device,
                time_indices=time_indices,
                stations=stations,
                is_new_arrivals=is_new_arrivals,
                prev_obs_inputs=prev_obs_refs,
                include_temporal=bool(getattr(detector_model, 'include_temporal', True)),
            )
            flags = [bool(score >= float(detector_threshold)) for score in np.asarray(scores, dtype=np.float32).reshape(-1)]
            routed = [recovered[i].reshape(-1) if flags[i] else to_numpy_1d(attacked_states[i]) for i in range(len(flags))]
            return routed, flags, np.asarray(scores, dtype=np.float32).reshape(-1)
        if isinstance(detector_model, DetectorGRUVAE):
            if vehicle_ids is None:
                raise ValueError('sequence detector requires vehicle_ids.')
            if detector_runtime is None:
                detector_runtime = SequentialDetectorRuntime(detector_model, device)
            scores, flags = detector_runtime.score_batch(attacked_states, vehicle_ids=vehicle_ids, episode_index=episode_index, threshold=float(detector_threshold))
            if defender is None:
                routed = [to_numpy_1d(x) for x in attacked_states]
            else:
                recovered = dae_runtime.reconstruct_batch(attacked_states, vehicle_ids=vehicle_ids, episode_index=episode_index) if dae_runtime is not None else reconstruction_batch(defender, attacked_states, device)
                routed = [recovered[i].reshape(-1) if flags[i] else to_numpy_1d(attacked_states[i]) for i in range(len(flags))]
            return routed, flags, scores
        raise ValueError('route_mode=detector supports PosteriorBenefitMLPDetector or DetectorGRUVAE.')
    raise ValueError(f'Unknown route_mode: {route_mode}')


def rollout_episode(arrivals: pd.DataFrame, actor: Actor, signals_path, device: torch.device, reward_profile: RewardProfile, attack_enabled: bool = False, attack_scenario: AttackScenario = 'C', attacker: PGDStateAttacker | None = None, defender: torch.nn.Module | None = None, detector_model=None, route_mode: str = 'none', detector_threshold: float | None = None, exploration_noise: float = 0.0, price_threshold: float = 400.0, soc_new_threshold: float = 0.5, soc_rollout_threshold: float = 0.3, even_station_target: float = 1.0, odd_station_target: float = -0.5, attack_ratio: float = 1.0, attack_scope: AttackScope = 'obs', detector_feature_mode: str = 'sequence') -> dict:
    from .defense import DetectorGRUVAE, SequentialDetectorRuntime
    env = ChargingEnv(signals_path=signals_path, reward_profile=reward_profile)
    env.reset()
    actor = actor.to(device).eval()
    idx = 0
    active: list[QueueItem] = []
    active_vehicle_ids: list[int] = []
    route_count = 0
    route_total = 0
    attack_obs_count = 0
    delta_linf_sum = 0.0
    delta_l2_sum = 0.0
    delta_local_linf_sum = 0.0
    delta_local_l2_sum = 0.0
    delta_linf_max = 0.0
    delta_l2_max = 0.0
    delta_local_linf_max = 0.0
    delta_local_l2_max = 0.0
    attack_delta_count = 0
    attack_action_abs_diff_sum = 0.0
    attack_action_abs_diff_max = 0.0
    attack_clean_action_sum = 0.0
    attack_adv_action_sum = 0.0
    attack_action_count = 0
    prev_observed_obs_by_vehicle: dict[int, np.ndarray] = {}
    dae_runtime = None if defender is None else SequentialDAERuntime(defender, device)
    detector_runtime = None if detector_model is None or not isinstance(detector_model, DetectorGRUVAE) else SequentialDetectorRuntime(detector_model, device)

    def _lookup_prev_refs(vehicle_ids: list[int], observed_states: list[np.ndarray]) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for vehicle_id, observed_obs in zip(vehicle_ids, observed_states):
            observed_vec = to_numpy_1d(observed_obs)
            out.append(prev_observed_obs_by_vehicle.get(int(vehicle_id), observed_vec))
        return out

    def _update_prev_refs(vehicle_ids: list[int], observed_states: list[np.ndarray]) -> None:
        for vehicle_id, observed_obs in zip(vehicle_ids, observed_states):
            prev_observed_obs_by_vehicle[int(vehicle_id)] = to_numpy_1d(observed_obs)

    def _compute_actions(policy_states: list[np.ndarray]) -> np.ndarray:
        with torch.no_grad():
            state_t = torch.as_tensor(np.asarray(policy_states, dtype=np.float32), dtype=torch.float32, device=device)
            actions = actor(state_t).detach().cpu().numpy()
        if exploration_noise > 0.0:
            actions = actions + np.random.normal(0.0, exploration_noise, size=actions.shape)
        return np.clip(actions, -1.0, 1.0)

    def _record_attack_diagnostics(clean_states: list[np.ndarray], observed_states: list[np.ndarray], attacked_flags: list[bool]) -> None:
        nonlocal delta_linf_sum, delta_l2_sum, delta_local_linf_sum, delta_local_l2_sum
        nonlocal delta_linf_max, delta_l2_max, delta_local_linf_max, delta_local_l2_max, attack_delta_count
        nonlocal attack_action_abs_diff_sum, attack_action_abs_diff_max
        nonlocal attack_clean_action_sum, attack_adv_action_sum, attack_action_count
        selected = [i for i, flag in enumerate(attacked_flags) if bool(flag)]
        if not selected:
            return
        clean_batch = np.asarray([to_numpy_1d(clean_states[i]) for i in selected], dtype=np.float32)
        observed_batch = np.asarray([to_numpy_1d(observed_states[i]) for i in selected], dtype=np.float32)
        deltas = observed_batch - clean_batch
        for delta in deltas:
            local_delta = delta[list(LOCAL_ATTACK_IDX)]
            linf = float(np.max(np.abs(delta)))
            l2 = float(np.linalg.norm(delta, ord=2))
            local_linf = float(np.max(np.abs(local_delta)))
            local_l2 = float(np.linalg.norm(local_delta, ord=2))
            delta_linf_sum += linf
            delta_l2_sum += l2
            delta_local_linf_sum += local_linf
            delta_local_l2_sum += local_l2
            delta_linf_max = max(delta_linf_max, linf)
            delta_l2_max = max(delta_l2_max, l2)
            delta_local_linf_max = max(delta_local_linf_max, local_linf)
            delta_local_l2_max = max(delta_local_l2_max, local_l2)
            attack_delta_count += 1
        with torch.no_grad():
            clean_t = torch.as_tensor(clean_batch, dtype=torch.float32, device=device)
            observed_t = torch.as_tensor(observed_batch, dtype=torch.float32, device=device)
            clean_actions = actor(clean_t).detach().cpu().numpy().reshape(-1)
            adv_actions = actor(observed_t).detach().cpu().numpy().reshape(-1)
        diffs = np.abs(adv_actions - clean_actions)
        attack_action_abs_diff_sum += float(np.sum(diffs))
        attack_action_abs_diff_max = max(attack_action_abs_diff_max, float(np.max(diffs)))
        attack_clean_action_sum += float(np.sum(clean_actions))
        attack_adv_action_sum += float(np.sum(adv_actions))
        attack_action_count += int(len(selected))

    while env.t < env.horizon:
        new_states = []
        new_stations = []
        new_vehicle_ids = []
        while idx < len(arrivals) and int(arrivals.loc[idx, 'Arrive_time']) == env.t:
            new_states.append(env.build_initial_obs(int(arrivals.loc[idx, 'Duration_of_stay'])))
            new_stations.append(int(arrivals.loc[idx, 'Station']))
            new_vehicle_ids.append(int(idx))
            idx += 1
        if new_states:
            contexts = _build_contexts(env, new_states, new_stations, attack_scenario, True, price_threshold, soc_new_threshold, soc_rollout_threshold, even_station_target, odd_station_target)
            attacked_states, attacked_flags = attack_batch_by_context(attacker if attack_enabled else None, new_states, contexts, attack_ratio=attack_ratio, attack_scope=attack_scope, vehicle_ids=new_vehicle_ids, episode_index=0, seed=42 if attacker is None else int(getattr(attacker, 'seed', 42)))
            observed_states = attacked_states if attack_enabled else [to_numpy_1d(x) for x in new_states]
            _record_attack_diagnostics(new_states, observed_states, attacked_flags)
            prev_refs = _lookup_prev_refs(new_vehicle_ids, observed_states)
            policy_states, route_flags, _ = _route_policy_states(observed_states, attacked_flags, defender, detector_model, actor, device, route_mode=route_mode, detector_threshold=detector_threshold, detector_feature_mode=detector_feature_mode, time_indices=[env.t for _ in new_states], stations=new_stations, is_new_arrivals=[1 for _ in new_states], prev_obs_refs=prev_refs, vehicle_ids=new_vehicle_ids, episode_index=0, dae_runtime=dae_runtime, detector_runtime=detector_runtime)
            route_count += int(sum(route_flags))
            route_total += len(route_flags)
            attack_obs_count += int(sum(attacked_flags))
            actions = _compute_actions(policy_states)
            for clean_obs, action, station in zip(new_states, actions, new_stations):
                env.enqueue(clean_obs, action, station)
            _update_prev_refs(new_vehicle_ids, observed_states)

        if active:
            active_states = [item.obs for item in active]
            active_stations = [item.station for item in active]
            contexts = _build_contexts(env, active_states, active_stations, attack_scenario, False, price_threshold, soc_new_threshold, soc_rollout_threshold, even_station_target, odd_station_target)
            attacked_states, attacked_flags = attack_batch_by_context(attacker if attack_enabled else None, active_states, contexts, attack_ratio=attack_ratio, attack_scope=attack_scope, vehicle_ids=active_vehicle_ids, episode_index=0, seed=42 if attacker is None else int(getattr(attacker, 'seed', 42)))
            observed_states = attacked_states if attack_enabled else [to_numpy_1d(x) for x in active_states]
            _record_attack_diagnostics(active_states, observed_states, attacked_flags)
            prev_refs = _lookup_prev_refs(active_vehicle_ids, observed_states)
            policy_states, route_flags, _ = _route_policy_states(observed_states, attacked_flags, defender, detector_model, actor, device, route_mode=route_mode, detector_threshold=detector_threshold, detector_feature_mode=detector_feature_mode, time_indices=[env.t for _ in active_states], stations=active_stations, is_new_arrivals=[0 for _ in active_states], prev_obs_refs=prev_refs, vehicle_ids=active_vehicle_ids, episode_index=0, dae_runtime=dae_runtime, detector_runtime=detector_runtime)
            route_count += int(sum(route_flags))
            route_total += len(route_flags)
            attack_obs_count += int(sum(attacked_flags))
            actions = _compute_actions(policy_states)
            for item, action in zip(active, actions):
                env.enqueue(item.obs, action, item.station)
            _update_prev_refs(active_vehicle_ids, observed_states)

        step_vehicle_ids = new_vehicle_ids + active_vehicle_ids
        transitions, next_active, _ = env.step()
        active = next_active
        active_vehicle_ids = [vid for vid, tr in zip(step_vehicle_ids, transitions) if not bool(tr.done)]

    summary = summarize_metrics(env.metrics, _rollout_label(attack_enabled, route_mode))
    summary['route_count'] = int(route_count)
    summary['route_total'] = int(route_total)
    summary['route_rate'] = 0.0 if route_total == 0 else float(route_count / route_total)
    summary['attack_obs_count'] = int(attack_obs_count)
    summary['attack_obs_rate'] = 0.0 if route_total == 0 else float(attack_obs_count / route_total)
    summary['attack_ratio_target'] = float(np.clip(attack_ratio, 0.0, 1.0))
    summary['attack_scope'] = str(attack_scope)
    summary['attack_delta_count'] = int(attack_delta_count)
    summary['attack_delta_linf_mean'] = 0.0 if attack_delta_count == 0 else float(delta_linf_sum / attack_delta_count)
    summary['attack_delta_l2_mean'] = 0.0 if attack_delta_count == 0 else float(delta_l2_sum / attack_delta_count)
    summary['attack_delta_linf_max'] = float(delta_linf_max)
    summary['attack_delta_l2_max'] = float(delta_l2_max)
    summary['attack_delta_local_linf_mean'] = 0.0 if attack_delta_count == 0 else float(delta_local_linf_sum / attack_delta_count)
    summary['attack_delta_local_l2_mean'] = 0.0 if attack_delta_count == 0 else float(delta_local_l2_sum / attack_delta_count)
    summary['attack_delta_local_linf_max'] = float(delta_local_linf_max)
    summary['attack_delta_local_l2_max'] = float(delta_local_l2_max)
    summary['attack_action_abs_diff_mean'] = 0.0 if attack_action_count == 0 else float(attack_action_abs_diff_sum / attack_action_count)
    summary['attack_action_abs_diff_max'] = float(attack_action_abs_diff_max)
    summary['attack_clean_action_mean'] = 0.0 if attack_action_count == 0 else float(attack_clean_action_sum / attack_action_count)
    summary['attack_adv_action_mean'] = 0.0 if attack_action_count == 0 else float(attack_adv_action_sum / attack_action_count)
    return summary
