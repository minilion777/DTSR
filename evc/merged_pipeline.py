"""Dataset preparation and routing helpers required by the DTSR mainline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from .attacks import AttackContext, AttackScenario, AttackScope, PGDStateAttacker, attack_batch_by_context
from .defense import (
    DAETrainResult, DetectorTrainResult, DenoisingAutoencoder,
    PosteriorBenefitMLPDetector, SequentialDAERuntime,
    posterior_detector_probabilities, reconstruction_batch, train_dae,
    train_posterior_detector,
)
from .merged_core import (
    TRAIN_PROFILE, Actor, ChargingEnv, EpisodeMetrics, QueueItem,
    RewardProfile, gini, normalize_result_object, to_numpy_1d,
)


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


def summarize_metrics(metrics: EpisodeMetrics, label: str) -> dict:
    raw = {
        "label": label,
        "ep_reward": float(metrics.ep_reward),
        "ep_r1_cost_sum": float(metrics.ep_r1_cost_sum),
        "ep_r2_exit_penalty_sum": float(metrics.ep_r2_exit_penalty_sum),
        "ep_r3_running_penalty_sum": float(metrics.ep_r3_running_penalty_sum),
        "ep_r4_dense_safety_penalty_sum": float(metrics.ep_r4_dense_safety_penalty_sum),
        "exit_vio": int(metrics.exit_violation_count),
        "run_vio": int(metrics.running_violation_count),
        "total_transitions": int(metrics.total_transitions),
        "done_count": int(metrics.done_count),
        "gini_cost": float(gini(metrics.costlist)),
        "mean_final_soc": float(np.mean(metrics.final_soc_list) if metrics.final_soc_list else 0.0),
        "std_final_soc": float(np.std(metrics.final_soc_list) if metrics.final_soc_list else 0.0),
        "mean_abs_power": float(np.mean(np.abs(metrics.powercurve)) if metrics.powercurve else 0.0),
        "max_abs_power": float(np.max(np.abs(metrics.powercurve)) if metrics.powercurve else 0.0),
        "cost_count": int(len(metrics.costlist)),
        "final_soc_count": int(len(metrics.final_soc_list)),
        "powercurve": [float(x) for x in metrics.powercurve],
        "powerlist": [[float(v) for v in curve] for curve in metrics.powerlist],
        "costlist": [float(x) for x in metrics.costlist],
        "final_soc_list": [float(x) for x in metrics.final_soc_list],
    }
    return normalize_result_object(raw, rename_keys=True)


def _build_contexts(
    env: ChargingEnv, states: list[np.ndarray], stations: list[int],
    scenario: AttackScenario, is_new_arrival: bool, price_threshold: float,
    soc_new_threshold: float, soc_rollout_threshold: float,
    even_station_target: float, odd_station_target: float,
) -> list[AttackContext]:
    del states
    return [
        AttackContext(
            scenario=scenario, time_index=env.t,
            raw_price=float(env.signals.price[env.t]), station=int(station),
            is_new_arrival=is_new_arrival, price_threshold=price_threshold,
            soc_new_threshold=soc_new_threshold,
            soc_rollout_threshold=soc_rollout_threshold,
            even_station_target=even_station_target,
            odd_station_target=odd_station_target,
        )
        for station in stations
    ]


def _rollout_label(attack_enabled: bool, route_mode: str) -> str:
    mapping = {
        (False, "none"): "clean", (True, "none"): "attack",
        (False, "detector"): "clean_dtsr", (True, "detector"): "attack_dtsr",
    }
    try:
        return mapping[(bool(attack_enabled), str(route_mode))]
    except KeyError as exc:
        raise ValueError(f"Unsupported mainline route_mode: {route_mode!r}") from exc


def _route_policy_states(
    attacked_states: list[np.ndarray], attacked_flags: list[bool],
    defender: torch.nn.Module | None, detector_model, actor: Actor,
    device: torch.device, *, route_mode: str,
    detector_threshold: float | None, detector_feature_mode: str = "posterior",
    time_indices: list[int] | None = None, stations: list[int] | None = None,
    is_new_arrivals: list[int] | None = None,
    prev_obs_refs: list[np.ndarray] | None = None,
    vehicle_ids: list[int] | None = None, episode_index: int = 0,
    dae_runtime: SequentialDAERuntime | None = None, detector_runtime=None,
) -> tuple[list[np.ndarray], list[bool], np.ndarray]:
    del attacked_flags, detector_feature_mode, detector_runtime
    if route_mode == "none":
        return (
            [to_numpy_1d(state) for state in attacked_states],
            [False for _ in attacked_states],
            np.full((len(attacked_states),), np.nan, dtype=np.float32),
        )
    if route_mode != "detector":
        raise ValueError(f"Unsupported mainline route_mode: {route_mode!r}")
    if detector_threshold is None or detector_model is None or defender is None:
        raise ValueError("Detector routing requires DAE, DeT, and a threshold.")
    if not isinstance(detector_model, PosteriorBenefitMLPDetector):
        raise ValueError("Detector routing requires PosteriorBenefitMLPDetector.")
    recovered = (
        dae_runtime.reconstruct_batch(attacked_states, vehicle_ids=vehicle_ids, episode_index=episode_index)
        if dae_runtime is not None and vehicle_ids is not None
        else reconstruction_batch(defender, attacked_states, device)
    )
    scores = posterior_detector_probabilities(
        detector_model, attacked_states, recovered, actor, device,
        time_indices=time_indices, stations=stations,
        is_new_arrivals=is_new_arrivals, prev_obs_inputs=prev_obs_refs,
        include_temporal=bool(getattr(detector_model, "include_temporal", True)),
    )
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    flags = [bool(score >= float(detector_threshold)) for score in scores]
    routed = [
        recovered[index].reshape(-1) if flags[index] else to_numpy_1d(attacked_states[index])
        for index in range(len(flags))
    ]
    return routed, flags, scores


def save_clean_trajectory_dataset(bundle: CleanTrajectoryBundle, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"clean_inputs": bundle.clean_inputs, "metadata": bundle.metadata}
    for name, dtype in (
        ("raw_prices", np.float32), ("time_indices", np.int64),
        ("stations", np.int64), ("is_new_arrivals", np.int64),
        ("vehicle_ids", np.int64), ("episode_indices", np.int64),
    ):
        value = getattr(bundle, name)
        if value is not None:
            payload[name] = np.asarray(value, dtype=dtype).reshape(-1)
    np.savez_compressed(path, **payload)
    return path


def load_clean_trajectory_dataset(path: str | Path) -> CleanTrajectoryBundle:
    obj = np.load(Path(path), allow_pickle=True)
    try:
        metadata = dict(obj["metadata"].item() if "metadata" in obj else {})
    except ModuleNotFoundError:
        metadata = {}
    clean_inputs = np.asarray(obj["clean_inputs"], dtype=np.float32).reshape(-1, 11)
    metadata["samples"] = int(clean_inputs.shape[0])

    def optional(name: str, dtype):
        return None if name not in obj else np.asarray(obj[name], dtype=dtype).reshape(-1)

    return CleanTrajectoryBundle(
        clean_inputs, metadata, optional("raw_prices", np.float32),
        optional("time_indices", np.int64), optional("stations", np.int64),
        optional("is_new_arrivals", np.int64), optional("vehicle_ids", np.int64),
        optional("episode_indices", np.int64),
    )


def collect_clean_trajectories(
    arrivals: pd.DataFrame, actor: Actor, signals_path, device: torch.device,
    reward_profile: RewardProfile = TRAIN_PROFILE, episodes: int = 1,
    max_samples: int | None = None,
) -> CleanTrajectoryBundle:
    actor = actor.to(device).eval()
    clean, prices, times, stations = [], [], [], []
    is_new_list, vehicle_ids, episode_ids = [], [], []
    for episode_index in range(episodes):
        env = ChargingEnv(signals_path=signals_path, reward_profile=reward_profile)
        env.reset()
        arrival_index, active, active_ids = 0, [], []
        while env.t < env.horizon and (max_samples is None or len(clean) < max_samples):
            new_states, new_stations, new_ids = [], [], []
            while arrival_index < len(arrivals) and int(arrivals.loc[arrival_index, "Arrive_time"]) == env.t:
                new_states.append(env.build_initial_obs(int(arrivals.loc[arrival_index, "Duration_of_stay"])))
                new_stations.append(int(arrivals.loc[arrival_index, "Station"]))
                new_ids.append(int(arrival_index))
                arrival_index += 1
            states = new_states + [item.obs for item in active]
            state_stations = new_stations + [item.station for item in active]
            state_ids = new_ids + active_ids
            state_is_new = [1] * len(new_states) + [0] * len(active)
            take = len(states) if max_samples is None else min(len(states), max_samples - len(clean))
            for state, station, vehicle_id, is_new in zip(
                states[:take], state_stations[:take], state_ids[:take], state_is_new[:take]
            ):
                clean.append(to_numpy_1d(state))
                prices.append(float(env.signals.price[env.t]))
                times.append(int(env.t)); stations.append(int(station))
                is_new_list.append(int(is_new)); vehicle_ids.append(int(vehicle_id))
                episode_ids.append(int(episode_index))
            if states:
                with torch.no_grad():
                    actions = actor(torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device)).cpu().numpy()
                for state, action, station in zip(states, actions, state_stations):
                    env.enqueue(state, action, station)
            transitions, active, _ = env.step()
            active_ids = [vehicle_id for vehicle_id, transition in zip(state_ids, transitions) if not bool(transition.done)]
    clean_inputs = np.asarray(clean, dtype=np.float32).reshape(-1, 11)
    return CleanTrajectoryBundle(
        clean_inputs,
        {"samples": int(clean_inputs.shape[0]), "collection_mode": "clean_rollout_dnormal", "reward_profile": reward_profile.name},
        np.asarray(prices, dtype=np.float32), np.asarray(times, dtype=np.int64),
        np.asarray(stations, dtype=np.int64), np.asarray(is_new_list, dtype=np.int64),
        np.asarray(vehicle_ids, dtype=np.int64), np.asarray(episode_ids, dtype=np.int64),
    )


def build_pair_dataset_from_clean_trajectories(
    dataset: CleanTrajectoryBundle, attacker: PGDStateAttacker,
    attack_scenario: AttackScenario, *, price_threshold: float = 400.0,
    soc_new_threshold: float = 0.5, soc_rollout_threshold: float = 0.3,
    even_station_target: float = 1.0, odd_station_target: float = -0.5,
    attack_ratio: float = 1.0, attack_scope: AttackScope = "obs",
    chunk_size: int = 1024,
) -> PairDatasetBundle:
    clean_inputs = np.asarray(dataset.clean_inputs, dtype=np.float32).reshape(-1, 11)
    total = int(clean_inputs.shape[0])
    required = (dataset.raw_prices, dataset.time_indices, dataset.stations,
                dataset.is_new_arrivals, dataset.vehicle_ids, dataset.episode_indices)
    if any(value is None for value in required):
        raise ValueError("Clean trajectory metadata is incomplete.")
    adv_inputs, attack_mask = clean_inputs.copy(), np.zeros(total, dtype=np.int64)
    for start in range(0, total, int(chunk_size)):
        end = min(total, start + int(chunk_size))
        contexts = [
            AttackContext(
                attack_scenario, int(dataset.time_indices[index]), float(dataset.raw_prices[index]),
                int(dataset.stations[index]), bool(dataset.is_new_arrivals[index]),
                price_threshold, soc_new_threshold, soc_rollout_threshold,
                even_station_target, odd_station_target,
            )
            for index in range(start, end)
        ]
        attacked, flags = attack_batch_by_context(
            attacker, [clean_inputs[index] for index in range(start, end)], contexts,
            attack_ratio=attack_ratio, attack_scope=attack_scope,
            vehicle_ids=dataset.vehicle_ids[start:end],
            episode_indices=dataset.episode_indices[start:end],
            seed=int(getattr(attacker, "seed", 42)),
        )
        adv_inputs[start:end] = np.asarray(attacked, dtype=np.float32).reshape(-1, 11)
        attack_mask[start:end] = np.asarray(flags, dtype=np.int64)
    return PairDatasetBundle(
        adv_inputs, clean_inputs,
        {"samples": total, "collection_mode": "offline_attack_from_dnormal",
         "attack_ratio": float(np.clip(attack_ratio, 0.0, 1.0)),
         "attack_scope": str(attack_scope), "attacked_samples": int(attack_mask.sum())},
        time_indices=np.asarray(dataset.time_indices, dtype=np.int64),
        stations=np.asarray(dataset.stations, dtype=np.int64),
        is_new_arrivals=np.asarray(dataset.is_new_arrivals, dtype=np.int64),
        vehicle_ids=np.asarray(dataset.vehicle_ids, dtype=np.int64),
        episode_indices=np.asarray(dataset.episode_indices, dtype=np.int64),
        attack_mask=attack_mask,
    )


def save_pair_dataset(bundle: PairDatasetBundle, path: str | Path) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"adv_inputs": bundle.adv_inputs, "clean_inputs": bundle.clean_inputs, "metadata": bundle.metadata}
    for name, dtype in (
        ("clean_anchor_inputs", np.float32), ("time_indices", np.int64),
        ("stations", np.int64), ("is_new_arrivals", np.int64),
        ("vehicle_ids", np.int64), ("episode_indices", np.int64),
        ("attack_mask", np.int64),
    ):
        value = getattr(bundle, name)
        if value is not None:
            payload[name] = np.asarray(value, dtype=dtype)
    np.savez_compressed(path, **payload)
    return path


def train_dae_from_bundle(
    bundle: PairDatasetBundle, actor: Actor, device: torch.device,
    epochs: int, batch_size: int, lr: float, lambda_state: float = 1.0,
    lambda_identity: float = 1.0,
    validator: Callable[[torch.nn.Module], dict] | None = None,
    val_every: int = 1, select_by: str = "reward_recovery", log_every: int = 1,
    seq_len: int = 8, hidden_dim: int = 128, latent_dim: int = 64,
    num_layers: int = 1, decoder_hidden_dim: int = 128, beta_kl: float = 1e-3,
    lambda_robust: float = 0.0, include_clean_sequences: bool = True,
    state_scope: str = "local", progress_dir: str | Path | None = None,
    progress_prefix: str = "dae",
) -> tuple[DenoisingAutoencoder, DAETrainResult]:
    return train_dae(
        bundle, actor, device=device, epochs=epochs, batch_size=batch_size, lr=lr,
        log_every=log_every, seq_len=seq_len, hidden_dim=hidden_dim,
        latent_dim=latent_dim, num_layers=num_layers,
        decoder_hidden_dim=decoder_hidden_dim, beta_kl=beta_kl,
        lambda_recon=lambda_state, lambda_identity=lambda_identity,
        lambda_robust=lambda_robust, include_clean_sequences=include_clean_sequences,
        validator=validator, val_every=val_every, select_by=select_by,
        state_scope=state_scope, progress_dir=progress_dir, progress_prefix=progress_prefix,
    )


def train_detector_from_bundle(
    dataset: DetectorDatasetBundle, actor: Actor, defender: torch.nn.Module | None,
    device: torch.device, *, compare_actor: Actor | None = None, epochs: int = 30,
    batch_size: int = 256, lr: float = 1e-3, hidden_dim: int = 128,
    dropout: float = 0.1, val_ratio: float = 0.2, detector_temporal: bool = True,
    detector_feature_mode: str = "posterior", seed: int = 42,
    latent_dim: int = 64, num_layers: int = 1, beta_kl: float = 1e-3,
    seq_len: int = 8, state_scope: str = "local",
    progress_dir: str | Path | None = None, progress_prefix: str = "detector",
    val_every: int = 1,
) -> tuple[torch.nn.Module, DetectorTrainResult]:
    del defender, compare_actor, detector_feature_mode, latent_dim, num_layers, beta_kl, seq_len, state_scope
    if str((dataset.metadata or {}).get("detector_mode", "posterior")).lower() != "posterior":
        raise ValueError("Only the posterior-benefit detector dataset is supported.")
    if dataset.obs_inputs is None or dataset.rec_inputs is None or dataset.labels is None:
        raise ValueError("Posterior detector dataset requires obs_inputs, rec_inputs, and labels.")
    return train_posterior_detector(
        dataset.obs_inputs, dataset.rec_inputs, dataset.labels, actor, device,
        time_indices=dataset.time_indices, stations=dataset.stations,
        is_new_arrivals=dataset.is_new_arrivals,
        episode_indices=dataset.episode_indices, vehicle_ids=dataset.vehicle_ids,
        prev_obs_inputs=dataset.prev_obs_inputs, include_temporal=bool(detector_temporal),
        epochs=epochs, batch_size=batch_size, lr=lr, hidden_dim=hidden_dim,
        dropout=dropout, val_ratio=val_ratio, seed=seed,
        sample_weights=dataset.sample_weights, progress_dir=progress_dir,
        progress_prefix=progress_prefix, val_every=val_every,
    )
