from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .long_horizon_attacks import StatefulLongHorizonAttacker
from .merged_attacks import AttackContext, GLOBAL_ATTACK_IDX, LOCAL_ATTACK_IDX, PGDStateAttacker
from .merged_core import min_max_denormalization, normalize_scalar, to_numpy_1d


KNOWLEDGE_LEVELS = ("K0", "K1", "K2", "K3", "K4")
OBJECTIVES = ("deadline", "economic")


@dataclass(frozen=True)
class CEMMPCConfig:
    objective: str = "deadline"
    knowledge_level: str = "K4"
    horizon: int = 8
    samples: int = 64
    iterations: int = 3
    elite_frac: float = 0.125
    temporal_eta: float = 0.045
    total_l1_budget: float = 1.20
    discount: float = 0.97
    mean_momentum: float = 0.35
    min_std: float = 0.004
    max_std: float = 0.055
    terminal_soc_target: float = 0.90
    deadline_shortfall_weight: float = 18.0
    deadline_action_weight: float = 0.20
    economic_cost_weight: float = 9.0
    economic_soc_weight: float = 14.0
    route_penalty: float = 0.010
    belief_penalty: float = 0.015
    shield_penalty: float = 0.018

    @classmethod
    def from_overrides(cls, overrides: dict[str, Any] | None = None) -> "CEMMPCConfig":
        data = dict(overrides or {})
        if "knowledge" in data and "knowledge_level" not in data:
            data["knowledge_level"] = data.pop("knowledge")
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"Unsupported CEM-MPC attack override keys: {unknown}")
        cfg = cls(**data)
        objective = str(cfg.objective).strip().lower().replace("-", "_")
        if objective in {"deadline_denial", "deadline_drift"}:
            objective = "deadline"
        if objective in {"economic_shift", "price_shift"}:
            objective = "economic"
        knowledge = str(cfg.knowledge_level).strip().upper()
        if knowledge.startswith("LEVEL_"):
            knowledge = knowledge.replace("LEVEL_", "K", 1)
        if objective not in OBJECTIVES:
            raise ValueError(f"Unsupported CEM-MPC objective: {cfg.objective!r}")
        if knowledge not in KNOWLEDGE_LEVELS:
            raise ValueError(f"Unsupported CEM-MPC knowledge level: {cfg.knowledge_level!r}")
        if int(cfg.horizon) <= 0 or int(cfg.samples) <= 0 or int(cfg.iterations) <= 0:
            raise ValueError("CEM-MPC horizon, samples, and iterations must be positive.")
        if not (0.0 < float(cfg.elite_frac) <= 1.0):
            raise ValueError("CEM-MPC elite_frac must be in (0, 1].")
        if float(cfg.temporal_eta) <= 0.0 or float(cfg.total_l1_budget) <= 0.0:
            raise ValueError("CEM-MPC temporal_eta and total_l1_budget must be positive.")
        return cls(
            **{
                **cfg.__dict__,
                "objective": objective,
                "knowledge_level": knowledge,
                "horizon": int(cfg.horizon),
                "samples": int(cfg.samples),
                "iterations": int(cfg.iterations),
            }
        )


@dataclass
class _PipelineSnapshot:
    dae_runtime: Any
    belief: Any
    gate: Any
    prev_observed: dict[int, np.ndarray]
    prev_policy: dict[int, np.ndarray]
    prev_action: dict[int, np.ndarray]
    prev_time: dict[int, int]


class ModuleAwareCEMMPCAttacker(StatefulLongHorizonAttacker):
    """Module-aware receding-horizon search for Experiment 4.

    The attacker's knowledge level controls which defense modules are included
    in the planning model. Evaluation can still run against the complete DTSR
    stack through the normal rollout code.
    """

    def __init__(
        self,
        base_attacker: PGDStateAttacker,
        *,
        epsilon: float = 0.075,
        passive_decay: float = 0.98,
        attack_state_scope: str = "local",
        config: CEMMPCConfig | None = None,
    ) -> None:
        cfg = CEMMPCConfig.from_overrides(None if config is None else config.__dict__)
        super().__init__(
            base_attacker,
            name="module_aware_cem_mpc",
            attack_state_scope=attack_state_scope,
            epsilon=epsilon,
            passive_decay=passive_decay,
        )
        self.config = cfg
        self._target_cfg: dict[str, Any] | None = None
        self._target_ready = False
        self._shadow_dae_runtime = None
        self._shadow_belief = None
        self._shadow_gate = None
        self._shadow_env = None
        self._shadow_prev_observed: dict[int, np.ndarray] = {}
        self._shadow_prev_policy: dict[int, np.ndarray] = {}
        self._shadow_prev_action: dict[int, np.ndarray] = {}
        self._shadow_prev_time: dict[int, int] = {}
        self._budget_used_by_key: defaultdict[tuple[int, int], float] = defaultdict(float)
        self._warm_start_by_key: dict[tuple[int, int], np.ndarray] = {}
        self._candidate_log: list[dict[str, float | str]] = []

    def clone(self):
        cloned = ModuleAwareCEMMPCAttacker(
            self.base_attacker.clone(),
            epsilon=self.epsilon,
            passive_decay=self.passive_decay,
            attack_state_scope=self.attack_state_scope,
            config=self.config,
        )
        if self._target_cfg is not None:
            cloned.configure_target_defense(**self._target_cfg)
        return cloned

    def reset(self) -> None:
        super().reset()
        self._budget_used_by_key.clear()
        self._warm_start_by_key.clear()
        self._reset_shadow_pipeline()

    def _base_attack(self, obs_arr: np.ndarray, contexts: list[AttackContext], *, keys=None) -> np.ndarray:
        del contexts, keys
        return np.asarray(obs_arr, dtype=np.float32)

    def configure_target_defense(
        self,
        *,
        defender,
        detector_model,
        detector_threshold: float,
        shield_config,
        ug_bcr_config,
        reward_profile,
        signals_path,
        device,
        actor=None,
        repair_mode: str = "core_only",
        ug_bcr_v3_config=None,
    ) -> None:
        self._target_cfg = dict(
            defender=defender,
            detector_model=detector_model,
            detector_threshold=float(detector_threshold),
            shield_config=shield_config,
            ug_bcr_config=ug_bcr_config,
            ug_bcr_v3_config=ug_bcr_v3_config,
            reward_profile=reward_profile,
            signals_path=signals_path,
            device=device,
            actor=self.base_attacker.actor if actor is None else actor,
            repair_mode=str(repair_mode or "core_only").strip().lower().replace("-", "_"),
        )
        self._target_ready = True
        self._reset_shadow_pipeline()

    def _reset_shadow_pipeline(self) -> None:
        self._candidate_log = []
        self._shadow_prev_observed = {}
        self._shadow_prev_policy = {}
        self._shadow_prev_action = {}
        self._shadow_prev_time = {}
        if not self._target_ready or self._target_cfg is None:
            self._shadow_dae_runtime = None
            self._shadow_belief = None
            self._shadow_gate = None
            self._shadow_env = None
            return
        from .defense import SequentialDAERuntime
        from .merged_core import ChargingEnv
        from .ug_bcr import BeliefCoreEstimator, UrgencyGatedBeliefSelector
        from .ug_bcr_v3 import ContinuousScoreBeliefSelector

        cfg = self._target_cfg
        self._shadow_dae_runtime = SequentialDAERuntime(cfg["defender"], cfg["device"]) if cfg.get("defender") is not None else None
        v3_config = cfg.get("ug_bcr_v3_config")
        base_config = v3_config.base_v2 if v3_config is not None else cfg["ug_bcr_config"]
        self._shadow_belief = BeliefCoreEstimator(base_config.belief)
        self._shadow_gate = (
            ContinuousScoreBeliefSelector(v3_config.continuous_gate)
            if v3_config is not None
            else UrgencyGatedBeliefSelector(base_config.urgency_gate)
        )
        self._shadow_env = ChargingEnv(signals_path=cfg["signals_path"], reward_profile=cfg["reward_profile"])
        self._shadow_env.reset()

    def _shadow_ready(self) -> bool:
        return bool(self._target_ready and self._target_cfg is not None and self._shadow_env is not None)

    def _clone_dae_runtime(self):
        if self._shadow_dae_runtime is None:
            return None
        from .defense import SequentialDAERuntime

        rt = SequentialDAERuntime(self._shadow_dae_runtime.model, self._shadow_dae_runtime.device)
        rt.buffers = defaultdict(lambda: deque(maxlen=rt.seq_len))
        for key, buf in self._shadow_dae_runtime.buffers.items():
            rt.buffers[key] = deque([np.asarray(x, dtype=np.float32).copy() for x in buf], maxlen=rt.seq_len)
        return rt

    def _snapshot(self, *, commit: bool) -> _PipelineSnapshot:
        return _PipelineSnapshot(
            dae_runtime=self._shadow_dae_runtime if commit else self._clone_dae_runtime(),
            belief=self._shadow_belief if commit else deepcopy(self._shadow_belief),
            gate=self._shadow_gate if commit else deepcopy(self._shadow_gate),
            prev_observed=self._shadow_prev_observed if commit else {k: v.copy() for k, v in self._shadow_prev_observed.items()},
            prev_policy=self._shadow_prev_policy if commit else {k: v.copy() for k, v in self._shadow_prev_policy.items()},
            prev_action=self._shadow_prev_action if commit else {k: v.copy() for k, v in self._shadow_prev_action.items()},
            prev_time=self._shadow_prev_time if commit else dict(self._shadow_prev_time),
        )

    def _actor_action_on_state(self, state: np.ndarray) -> float:
        cfg = self._target_cfg or {}
        actor = cfg.get("actor", self.base_attacker.actor)
        device = cfg.get("device", getattr(self.base_attacker, "device", torch.device("cpu")))
        with torch.no_grad():
            st = torch.as_tensor(to_numpy_1d(state).astype(np.float32), dtype=torch.float32, device=device).reshape(1, -1)
            act = actor(st).reshape(-1)
        return float(np.clip(act.detach().cpu().numpy()[0], -1.0, 1.0))

    def _shape_delta(self, key: tuple[int, int], obs: np.ndarray, base_delta: np.ndarray, context: AttackContext) -> np.ndarray:
        del base_delta
        obs_vec = to_numpy_1d(obs).astype(np.float32)
        if not self._shadow_ready():
            return self._fallback_delta(key, obs_vec, context)
        if self._remaining_budget(key) <= 1e-8:
            return np.zeros_like(obs_vec, dtype=np.float32)
        best_seq, best_info = self._cem_search(key, obs_vec, context)
        first_delta = self._bounded_delta(best_seq[0])
        self._budget_used_by_key[key] += float(np.sum(np.abs(first_delta[list(self.attack_indices)])))
        self._warm_start_by_key[key] = self._shift_sequence(best_seq)
        commit_snapshot = self._snapshot(commit=True)
        commit_info = self._pipeline_step(
            key=key,
            clean_obs=obs_vec,
            delta=first_delta,
            context=context,
            snapshot=commit_snapshot,
            knowledge_level=self.config.knowledge_level,
            commit=True,
        )
        self._candidate_log.append(
            {
                "score": float(best_info.get("score", 0.0)),
                "objective": self.config.objective,
                "knowledge": self.config.knowledge_level,
                "predicted_return": float(best_info.get("predicted_return", 0.0)),
                "predicted_cost": float(best_info.get("predicted_cost", 0.0)),
                "predicted_final_soc": float(best_info.get("predicted_final_soc", obs_vec[0])),
                "route": float(commit_info.get("route_flag", 0.0)),
                "belief": float(commit_info.get("belief_flag", 0.0)),
                "temporal": float(commit_info.get("temporal", 0.0)),
                "action": float(commit_info.get("action", 0.0)),
                "budget_used": float(self._budget_used_by_key[key]),
                "budget_remaining": float(self._remaining_budget(key)),
                "horizon": float(self.config.horizon),
                "samples": float(self.config.samples),
            }
        )
        return first_delta

    def _fallback_delta(self, key: tuple[int, int], obs: np.ndarray, context: AttackContext) -> np.ndarray:
        del key
        if self.config.objective == "economic":
            delta = np.zeros_like(obs, dtype=np.float32)
            idx = np.asarray(GLOBAL_ATTACK_IDX, dtype=np.int64)
            signs = np.asarray((0.25, 0.20, 0.15, 0.65, 0.45, 0.15, -0.20, -0.35), dtype=np.float32)
            delta[idx] = signs * min(self.epsilon, self.config.temporal_eta)
            return self._bounded_delta(delta)
        return self._bounded_delta(self._undercharge_bias(obs, context, local_scale=0.025, time_scale=0.045, cost_scale=0.030, env_scale=0.0, price_scale=0.0))

    def _remaining_budget(self, key: tuple[int, int]) -> float:
        return max(0.0, float(self.config.total_l1_budget) - float(self._budget_used_by_key.get(key, 0.0)))

    def _rng_for(self, key: tuple[int, int]) -> np.random.Generator:
        step = self._step_count(key)
        seed = int(self.seed + 104729 * (int(key[0]) + 1) + 1009 * (int(key[1]) + 1) + 7919 * (step + 1))
        return np.random.default_rng(seed % (2**32 - 1))

    def _initial_mean(self, key: tuple[int, int], obs: np.ndarray, context: AttackContext) -> np.ndarray:
        prev = self._warm_start_by_key.get(key)
        if prev is not None and prev.shape == (self.config.horizon, obs.size):
            return self._project_sequence(key, prev)
        heuristic = np.zeros((self.config.horizon, obs.size), dtype=np.float32)
        base = self._fallback_delta(key, obs, context)
        for h in range(self.config.horizon):
            decay = float(0.92**h)
            heuristic[h] = base * decay
        return self._project_sequence(key, heuristic)

    def _shift_sequence(self, seq: np.ndarray) -> np.ndarray:
        arr = np.asarray(seq, dtype=np.float32).copy()
        if arr.shape[0] <= 1:
            return arr
        shifted = np.vstack([arr[1:], arr[-1:]])
        return shifted.astype(np.float32)

    def _project_sequence(self, key: tuple[int, int], seq: np.ndarray) -> np.ndarray:
        arr = np.asarray(seq, dtype=np.float32).reshape(int(self.config.horizon), -1).copy()
        out = np.zeros_like(arr, dtype=np.float32)
        prev = self._prev_delta(key)
        for h in range(arr.shape[0]):
            delta = self._bounded_delta(arr[h])
            delta = self._mask_delta(np.clip(delta, prev - float(self.config.temporal_eta), prev + float(self.config.temporal_eta)))
            out[h] = delta
            prev = delta
        total = float(np.sum(np.abs(out[:, list(self.attack_indices)])))
        remaining = self._remaining_budget(key)
        if total > remaining > 0.0:
            out *= float(remaining / max(total, 1e-8))
        if remaining <= 0.0:
            out *= 0.0
        return out.astype(np.float32)

    def _cem_search(self, key: tuple[int, int], obs: np.ndarray, context: AttackContext) -> tuple[np.ndarray, dict[str, float]]:
        rng = self._rng_for(key)
        cfg = self.config
        mean = self._initial_mean(key, obs, context)
        std = np.zeros_like(mean, dtype=np.float32)
        std[:, list(self.attack_indices)] = min(float(cfg.max_std), max(float(cfg.min_std), float(self.epsilon) * 0.55))
        elite_count = max(1, int(np.ceil(float(cfg.samples) * float(cfg.elite_frac))))
        best_seq = mean.copy()
        best_info: dict[str, float] = {"score": -float("inf")}
        for _ in range(int(cfg.iterations)):
            samples = rng.normal(loc=mean, scale=std, size=(int(cfg.samples),) + mean.shape).astype(np.float32)
            samples[0] = 0.0
            if int(cfg.samples) > 1:
                samples[1] = mean
            if int(cfg.samples) > 2:
                samples[2] = self._initial_mean(key, obs, context)
            scored: list[tuple[float, int, dict[str, float], np.ndarray]] = []
            for sample_idx in range(int(cfg.samples)):
                seq = self._project_sequence(key, samples[sample_idx])
                score, info = self._evaluate_sequence(key, obs, context, seq)
                scored.append((float(score), sample_idx, info, seq))
            scored.sort(key=lambda item: item[0], reverse=True)
            if scored[0][0] > float(best_info.get("score", -float("inf"))):
                best_seq = scored[0][3].copy()
                best_info = dict(scored[0][2])
            elites = np.asarray([item[3] for item in scored[:elite_count]], dtype=np.float32)
            elite_mean = np.mean(elites, axis=0)
            elite_std = np.std(elites, axis=0)
            mean = float(cfg.mean_momentum) * mean + (1.0 - float(cfg.mean_momentum)) * elite_mean
            std = np.clip(elite_std, float(cfg.min_std), float(cfg.max_std)).astype(np.float32)
            mask = np.zeros_like(std)
            mask[:, list(self.attack_indices)] = 1.0
            std *= mask
            mean = self._project_sequence(key, mean)
        return self._project_sequence(key, best_seq), best_info

    def _evaluate_sequence(self, key: tuple[int, int], obs: np.ndarray, context: AttackContext, seq: np.ndarray) -> tuple[float, dict[str, float]]:
        cfg = self.config
        snapshot = self._snapshot(commit=False)
        clean_obs = to_numpy_1d(obs).astype(np.float32)
        current_time = int(context.time_index)
        total_return = 0.0
        total_cost = 0.0
        total_action = 0.0
        route_sum = 0.0
        belief_sum = 0.0
        temporal_sum = 0.0
        done = False
        for h, delta in enumerate(seq):
            ctx = self._context_at(context, current_time, is_new=(h == 0 and bool(context.is_new_arrival)))
            info = self._pipeline_step(
                key=key,
                clean_obs=clean_obs,
                delta=delta,
                context=ctx,
                snapshot=snapshot,
                knowledge_level=cfg.knowledge_level,
                commit=False,
            )
            action = float(info["action"])
            reward, step_cost, next_obs, done = self._simulate_transition(clean_obs, action, current_time)
            weight = float(cfg.discount) ** h
            total_return += weight * float(reward)
            total_cost += weight * float(step_cost)
            total_action += weight * action
            route_sum += float(info.get("route_flag", 0.0))
            belief_sum += float(info.get("belief_flag", 0.0))
            temporal_sum += float(info.get("temporal", 0.0))
            clean_obs = next_obs
            current_time = min(current_time + 1, int(self._shadow_env.horizon) - 1)
            if done:
                break
        final_soc = float(clean_obs[0])
        remaining_slots = max(0.0, float(clean_obs[1]) * 12.0)
        max_step_soc = float(self._shadow_env.max_power * self._shadow_env.slice_hours / self._shadow_env.battery_capacity)
        reachable_soc = final_soc + remaining_slots * max_step_soc
        terminal_shortfall = max(0.0, float(cfg.terminal_soc_target) - final_soc)
        unreachable_shortfall = max(0.0, float(cfg.terminal_soc_target) - reachable_soc)
        if cfg.objective == "economic":
            score = (
                float(cfg.economic_cost_weight) * total_cost
                - float(cfg.economic_soc_weight) * (terminal_shortfall**2 + 2.0 * unreachable_shortfall**2)
                - float(cfg.route_penalty) * route_sum
                - float(cfg.belief_penalty) * belief_sum
                - float(cfg.shield_penalty) * temporal_sum
            )
        else:
            score = (
                -total_return
                - float(cfg.deadline_action_weight) * total_action
                + float(cfg.deadline_shortfall_weight) * terminal_shortfall**2
                + 0.5 * unreachable_shortfall
                - float(cfg.route_penalty) * route_sum
                - float(cfg.belief_penalty) * belief_sum
                - float(cfg.shield_penalty) * temporal_sum
            )
        info = {
            "score": float(score),
            "predicted_return": float(total_return),
            "predicted_cost": float(total_cost),
            "predicted_final_soc": float(final_soc),
            "route_sum": float(route_sum),
            "belief_sum": float(belief_sum),
            "temporal_sum": float(temporal_sum),
        }
        return float(score), info

    def _context_at(self, context: AttackContext, time_index: int, *, is_new: bool) -> AttackContext:
        price = float(self._shadow_env.signals.price[min(max(int(time_index), 0), int(self._shadow_env.horizon) - 1)])
        return AttackContext(
            scenario=context.scenario,
            time_index=int(time_index),
            raw_price=price,
            station=int(context.station),
            is_new_arrival=bool(is_new),
            price_threshold=float(context.price_threshold),
            soc_new_threshold=float(context.soc_new_threshold),
            soc_rollout_threshold=float(context.soc_rollout_threshold),
            even_station_target=float(context.even_station_target),
            odd_station_target=float(context.odd_station_target),
        )

    def _pipeline_step(
        self,
        *,
        key: tuple[int, int],
        clean_obs: np.ndarray,
        delta: np.ndarray,
        context: AttackContext,
        snapshot: _PipelineSnapshot,
        knowledge_level: str,
        commit: bool,
    ) -> dict[str, Any]:
        del commit
        cfg = self._target_cfg or {}
        vehicle_id = int(key[1])
        episode_id = int(key[0])
        if self._shadow_env is not None:
            self._shadow_env.t = int(context.time_index)
        obs_vec = to_numpy_1d(clean_obs).astype(np.float32)
        adv = self._project_obs(obs_vec, obs_vec + self._bounded_delta(delta)).astype(np.float32)
        policy_vec = adv.copy()
        route_flag = 0.0
        det_score = float("nan")
        belief_flag = 0.0
        temporal = 0.0
        level = str(knowledge_level).upper()
        if level in {"K1", "K2", "K3", "K4"} and cfg.get("defender") is not None:
            from .offline_dae_det_temporal_shield import _route_policy_states_core_only
            from .merged_pipeline import _route_policy_states

            route_mode = "oracle" if level == "K1" else "detector"
            if route_mode == "detector" and cfg.get("detector_model") is None:
                route_mode = "oracle"
            route_fn = _route_policy_states_core_only if cfg.get("repair_mode", "core_only") == "core_only" else _route_policy_states
            prev_ref = snapshot.prev_observed.get(vehicle_id, adv)
            policy_states, route_flags, det_scores = route_fn(
                [adv],
                [True],
                cfg.get("defender"),
                cfg.get("detector_model"),
                cfg.get("actor", self.base_attacker.actor),
                cfg.get("device", getattr(self.base_attacker, "device", torch.device("cpu"))),
                route_mode=route_mode,
                detector_threshold=float(cfg.get("detector_threshold", 0.5)),
                detector_feature_mode="posterior",
                time_indices=[int(context.time_index)],
                stations=[int(context.station)],
                is_new_arrivals=[int(bool(context.is_new_arrival))],
                prev_obs_refs=[prev_ref],
                vehicle_ids=[vehicle_id],
                episode_index=episode_id,
                dae_runtime=snapshot.dae_runtime,
            )
            policy_vec = to_numpy_1d(policy_states[0]).astype(np.float32)
            route_flag = float(bool(route_flags[0]))
            score_arr = np.asarray(det_scores).reshape(-1)
            det_score = float(score_arr[0]) if score_arr.size else float("nan")
        if level in {"K3", "K4"} and snapshot.belief is not None and snapshot.gate is not None:
            belief_states = snapshot.belief.repair_batch(
                [policy_vec],
                [vehicle_id],
                [int(bool(context.is_new_arrival))],
                np.asarray([det_score], dtype=np.float32),
                self._shadow_env,
            )
            selected_states, branches = snapshot.gate.select_batch(
                [policy_vec],
                belief_states,
                [vehicle_id],
                [int(bool(context.is_new_arrival))],
                snapshot.belief,
                cfg.get("shield_config"),
                self._shadow_env,
                cfg.get("reward_profile"),
                snapshot.prev_policy,
                snapshot.prev_action,
                snapshot.prev_time,
                detector_scores=[det_score],
                route_flags=[bool(route_flag)],
            )
            policy_vec = to_numpy_1d(selected_states[0]).astype(np.float32)
            belief_flag = float(str(branches[0]) == "belief")
        final_state = policy_vec.copy()
        if level == "K4" and cfg.get("shield_config") is not None:
            from .offline_dae_det_temporal_shield import LOCAL_SHIELD_INDICES, _shield_single_state

            corrected, flags = _shield_single_state(
                policy_vec,
                snapshot.prev_policy.get(vehicle_id),
                snapshot.prev_action.get(vehicle_id),
                snapshot.prev_time.get(vehicle_id),
                cfg.get("shield_config"),
                self._shadow_env,
                is_new_arrival=bool(context.is_new_arrival),
            )
            final_state = to_numpy_1d(corrected).astype(np.float32)
            temporal = float(np.max(np.abs(final_state[list(LOCAL_SHIELD_INDICES)] - policy_vec[list(LOCAL_SHIELD_INDICES)])))
            temporal += 0.01 * float(flags.get("soc", False) or flags.get("time", False) or flags.get("cost", False))
        action = self._actor_action_on_state(final_state)
        snapshot.prev_policy[vehicle_id] = final_state.copy()
        snapshot.prev_observed[vehicle_id] = adv.copy()
        snapshot.prev_action[vehicle_id] = np.asarray([action], dtype=np.float32)
        snapshot.prev_time[vehicle_id] = int(context.time_index)
        if snapshot.belief is not None and hasattr(snapshot.belief, "update_actions"):
            snapshot.belief.update_actions([vehicle_id], np.asarray([[action]], dtype=np.float32), int(context.time_index))
        return {
            "action": float(action),
            "route_flag": float(route_flag),
            "belief_flag": float(belief_flag),
            "temporal": float(temporal),
            "det_score": float(det_score),
            "final_state": final_state,
        }

    def _simulate_transition(self, obs: np.ndarray, action: float, time_index: int) -> tuple[float, float, np.ndarray, bool]:
        env = self._shadow_env
        reward_profile = env.reward_profile
        obs_vec = to_numpy_1d(obs).astype(np.float32)
        a = float(np.clip(action, -1.0, 1.0))
        soc = float(obs_vec[0])
        t_re = float(obs_vec[1])
        cost_norm = float(obs_vec[10])
        t = min(max(int(time_index), 0), int(env.horizon) - 1)
        new_soc = soc + a * float(env.max_power) * float(env.slice_hours) / float(env.battery_capacity)
        new_t_re = t_re - 1.0 / 12.0
        next_idx = min(t + 1, int(env.horizon) - 1)
        next_price = []
        for j in range(t + 1, t + 6):
            next_price.append(float(env.signals.norm_price[j] if j < env.horizon else env.signals.norm_price[-1]))
        step_cost = a * float(env.max_power) * float(env.slice_hours) * float(env.signals.price[t])
        ncost = normalize_scalar(
            step_cost,
            -float(env.signals.max_price) * float(env.slice_hours) * float(env.max_power) * 0.5,
            float(env.signals.max_price) * float(env.slice_hours) * float(env.max_power) * 0.5,
        )
        cost_upper = float(env.signals.max_price) * float(env.battery_capacity)
        cum_cost = float(min_max_denormalization(cost_norm, 0.0, cost_upper)) + step_cost
        new_cost_norm = normalize_scalar(cum_cost, 0.0, cost_upper)
        done = bool(new_t_re < 1e-8)
        if done:
            if float(reward_profile.exit_target_min) <= new_soc <= float(reward_profile.exit_target_max):
                p_soc = 0.0
            elif new_soc > float(reward_profile.running_soc_max):
                p_soc = 1.0 + new_soc - float(reward_profile.running_soc_max)
            else:
                p_soc = 1.0 + float(reward_profile.exit_target_min) - new_soc
        else:
            if float(reward_profile.running_soc_min) <= new_soc <= float(reward_profile.running_soc_max):
                p_soc = 0.0
            elif new_soc > float(reward_profile.running_soc_max):
                p_soc = 1.0 + new_soc - float(reward_profile.running_soc_max)
            else:
                p_soc = 1.0 - new_soc + float(reward_profile.running_soc_min)
        action_penalty = 0.0
        if reward_profile.action_penalty_threshold is not None and abs(a) > float(reward_profile.action_penalty_threshold):
            action_penalty = (abs(a) - float(reward_profile.action_penalty_threshold)) * float(reward_profile.action_penalty_scale)
        dense = env._dense_safety_penalty(soc, t_re, a)
        reward = (
            -float(reward_profile.reward_soc_weight) * p_soc
            - float(ncost)
            - float(action_penalty)
            - float(reward_profile.dense_safety_penalty_weight) * float(dense)
        )
        next_obs = np.asarray(
            [
                new_soc,
                new_t_re,
                float(env.signals.pv[next_idx]),
                float(env.signals.wt[next_idx]),
                float(env.signals.load[next_idx]),
                next_price[0],
                next_price[1],
                next_price[2],
                next_price[3],
                next_price[4],
                new_cost_norm,
            ],
            dtype=np.float32,
        )
        return float(reward), float(step_cost), next_obs, done


__all__ = ["CEMMPCConfig", "KNOWLEDGE_LEVELS", "ModuleAwareCEMMPCAttacker", "OBJECTIVES"]
