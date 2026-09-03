"""Causal, leak-free adaptive knowledge-ladder attacker.

This module is intentionally separate from ``module_aware_attacks.py``.  Its
contract is stricter: K0--K4 only add the declared defense knowledge and the
attacker never receives a test-day signal file, future environment states,
oracle routing labels, or post-rollout restart outcomes.
"""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .long_horizon_attacks import StatefulLongHorizonAttacker
from .merged_attacks import AttackContext, PGDStateAttacker
from .merged_core import to_numpy_1d


KNOWLEDGE_LEVELS = ("K0", "K1", "K2", "K3", "K4")


@dataclass(frozen=True)
class CausalKnowledgeConfig:
    knowledge_level: str = "K0"
    epsilon: float = 0.075
    temporal_eta: float = 0.045
    public_cost_upper_bound: float = 1.0

    def normalized(self) -> "CausalKnowledgeConfig":
        level = str(self.knowledge_level).upper().strip()
        if level not in KNOWLEDGE_LEVELS:
            raise ValueError(f"Unsupported causal knowledge level: {self.knowledge_level!r}")
        if self.epsilon <= 0.0 or self.temporal_eta <= 0.0:
            raise ValueError("epsilon and temporal_eta must be positive")
        if self.public_cost_upper_bound <= 0.0:
            raise ValueError("public_cost_upper_bound must be positive")
        return CausalKnowledgeConfig(
            knowledge_level=level,
            epsilon=float(self.epsilon),
            temporal_eta=float(self.temporal_eta),
            public_cost_upper_bound=float(self.public_cost_upper_bound),
        )


class _CausalPrices:
    """Price history visible to the attacker; missing timestamps use last seen price.

    Returning the last observed price is deliberately conservative.  In
    particular, this object has no reference to a test-day signal sequence.
    """

    def __init__(self) -> None:
        self._history: dict[int, float] = {}
        self._last = 0.0

    def observe(self, time_index: int, price: float) -> None:
        self._history[int(time_index)] = float(price)
        self._last = float(price)

    def __getitem__(self, time_index: int) -> float:
        index = int(time_index)
        if index in self._history:
            return self._history[index]
        earlier = [key for key in self._history if key <= index]
        return self._history[max(earlier)] if earlier else self._last


class _CausalSignals:
    def __init__(self) -> None:
        self.price = _CausalPrices()


class _CausalDefenseEnv:
    """Minimal causal protocol adapter required by UG-BCR and Temporal Shield."""

    def __init__(self, *, reward_profile, public_cost_upper_bound: float) -> None:
        self.reward_profile = reward_profile
        self.max_power = 0.07
        self.slice_hours = 0.25
        self.battery_capacity = 0.04992
        self.initial_soc = 0.0
        self.initial_cost_norm = 0.2
        self.horizon = 1_000_000
        self.t = 0
        self.signals = _CausalSignals()
        self._public_cost_upper_bound = float(public_cost_upper_bound)

    def observe_context(self, context: AttackContext) -> None:
        self.t = int(context.time_index)
        self.signals.price.observe(self.t, float(context.raw_price))

    def _cost_upper_bound(self) -> float:
        return self._public_cost_upper_bound


@dataclass
class _Snapshot:
    dae_runtime: Any
    belief: Any
    gate: Any
    prev_observed: dict[int, np.ndarray]
    prev_policy: dict[int, np.ndarray]
    prev_action: dict[int, np.ndarray]
    prev_time: dict[int, int]


class CausalKnowledgeLadderAttacker(StatefulLongHorizonAttacker):
    """Online adaptive attacker with an auditable K0--K4 disclosure boundary.

    K0 knows only the Actor. K1 adds DAE and uses *always-DAE* routing, never an
    attack-label oracle. K2 adds DET. K3 adds UG-BCR but passes ``None`` for the
    Shield configuration. Only K4 receives and applies the Temporal Shield.
    Candidate ranking is myopic and causal: it only evaluates the currently
    observed state and previous observed history, not simulated future signals.
    """

    def __init__(
        self,
        base_attacker: PGDStateAttacker,
        *,
        config: CausalKnowledgeConfig,
        attack_state_scope: str = "local",
    ) -> None:
        cfg = config.normalized()
        super().__init__(
            base_attacker,
            name="causal_knowledge_ladder",
            attack_state_scope=attack_state_scope,
            epsilon=cfg.epsilon,
            passive_decay=0.98,
        )
        self.config = cfg
        self._known: dict[str, Any] = {}
        self._ready = False
        self._dae_runtime = None
        self._belief = None
        self._gate = None
        self._env = None
        self._prev_observed: dict[int, np.ndarray] = {}
        self._prev_policy: dict[int, np.ndarray] = {}
        self._prev_action: dict[int, np.ndarray] = {}
        self._prev_time: dict[int, int] = {}
        self._candidate_log: list[dict[str, float | str]] = []

    @property
    def knowledge_level(self) -> str:
        return self.config.knowledge_level

    def configure_target_defense(
        self,
        *,
        defender=None,
        detector_model=None,
        detector_threshold: float | None = None,
        shield_config=None,
        ug_bcr_config=None,
        reward_profile,
        device,
        actor=None,
        repair_mode: str = "core_only",
        ug_bcr_v3_config=None,
    ) -> None:
        """Accept only components revealed at the selected knowledge level.

        Deliberately do not accept ``signals_path``.  Keeping that parameter out
        of this API makes future-test-signal leakage impossible by construction.
        """
        if ug_bcr_v3_config is not None:
            raise ValueError("The causal knowledge ladder currently supports the audited UG-BCR-v2 gate only.")
        level = self.knowledge_level
        actor = self.base_attacker.actor if actor is None else actor
        known: dict[str, Any] = {
            "actor": actor,
            "device": device,
            "reward_profile": reward_profile,
            "repair_mode": str(repair_mode).strip().lower().replace("-", "_"),
        }
        if level in {"K1", "K2", "K3", "K4"}:
            if defender is None:
                raise ValueError(f"{level} requires the disclosed DAE")
            known["defender"] = defender
        if level in {"K2", "K3", "K4"}:
            if detector_model is None or detector_threshold is None:
                raise ValueError(f"{level} requires the disclosed DET and threshold")
            known["detector_model"] = detector_model
            known["detector_threshold"] = float(detector_threshold)
        if level in {"K3", "K4"}:
            if ug_bcr_config is None:
                raise ValueError(f"{level} requires the disclosed UG-BCR configuration")
            known["ug_bcr_config"] = ug_bcr_config
        if level == "K4":
            if shield_config is None:
                raise ValueError("K4 requires the disclosed Temporal Shield configuration")
            known["shield_config"] = shield_config
        self._known = known
        self._ready = True
        self._reset_shadow_state()

    def _reset_shadow_state(self) -> None:
        self._prev_observed.clear()
        self._prev_policy.clear()
        self._prev_action.clear()
        self._prev_time.clear()
        self._candidate_log = []
        self._dae_runtime = None
        self._belief = None
        self._gate = None
        self._env = None
        if not self._ready:
            return
        from .defense import SequentialDAERuntime
        from .ug_bcr import BeliefCoreEstimator, UrgencyGatedBeliefSelector

        cfg = self._known
        self._env = _CausalDefenseEnv(
            reward_profile=cfg["reward_profile"],
            public_cost_upper_bound=self.config.public_cost_upper_bound,
        )
        if self.knowledge_level in {"K1", "K2", "K3", "K4"}:
            self._dae_runtime = SequentialDAERuntime(cfg["defender"], cfg["device"])
        if self.knowledge_level in {"K3", "K4"}:
            self._belief = BeliefCoreEstimator(cfg["ug_bcr_config"].belief)
            self._gate = UrgencyGatedBeliefSelector(cfg["ug_bcr_config"].urgency_gate)

    def reset(self) -> None:
        super().reset()
        self._reset_shadow_state()

    def clone(self):
        clone = CausalKnowledgeLadderAttacker(
            self.base_attacker.clone(), config=self.config, attack_state_scope=self.attack_state_scope
        )
        if self._ready:
            clone._known = dict(self._known)
            clone._ready = True
            clone._reset_shadow_state()
        return clone

    def _clone_dae_runtime(self):
        if self._dae_runtime is None:
            return None
        from .defense import SequentialDAERuntime

        clone = SequentialDAERuntime(self._dae_runtime.model, self._dae_runtime.device)
        clone.buffers = defaultdict(lambda: deque(maxlen=clone.seq_len))
        for key, values in self._dae_runtime.buffers.items():
            clone.buffers[key] = deque([np.asarray(value, dtype=np.float32).copy() for value in values], maxlen=clone.seq_len)
        return clone

    def _snapshot(self, *, commit: bool) -> _Snapshot:
        return _Snapshot(
            dae_runtime=self._dae_runtime if commit else self._clone_dae_runtime(),
            belief=self._belief if commit else deepcopy(self._belief),
            gate=self._gate if commit else deepcopy(self._gate),
            prev_observed=self._prev_observed if commit else {key: value.copy() for key, value in self._prev_observed.items()},
            prev_policy=self._prev_policy if commit else {key: value.copy() for key, value in self._prev_policy.items()},
            prev_action=self._prev_action if commit else {key: value.copy() for key, value in self._prev_action.items()},
            prev_time=self._prev_time if commit else dict(self._prev_time),
        )

    def _actor_action(self, state: np.ndarray) -> float:
        cfg = self._known
        with torch.no_grad():
            tensor = torch.as_tensor(to_numpy_1d(state), dtype=torch.float32, device=cfg["device"]).reshape(1, -1)
            action = cfg["actor"](tensor).reshape(-1)
        return float(np.clip(action.detach().cpu().numpy()[0], -1.0, 1.0))

    def _pipeline_eval(self, *, key: tuple[int, int], obs: np.ndarray, delta: np.ndarray, context: AttackContext, commit: bool) -> dict[str, float | np.ndarray]:
        if not self._ready or self._env is None:
            raise RuntimeError("Causal knowledge attacker must be configured before rollout")
        from .merged_pipeline import _route_policy_states
        from .offline_dae_det_temporal_shield import _route_policy_states_core_only, _shield_single_state

        cfg = self._known
        snapshot = self._snapshot(commit=commit)
        self._env.observe_context(context)
        vehicle_id, episode_id = int(key[1]), int(key[0])
        observed = to_numpy_1d(obs).astype(np.float32)
        adv = self._project_obs(observed, observed + self._mask_delta(self._bounded_delta(delta))).astype(np.float32)
        state = adv.copy()
        route_flag = belief_flag = temporal = 0.0
        detector_score = float("nan")
        level = self.knowledge_level
        if level in {"K1", "K2", "K3", "K4"}:
            route_fn = _route_policy_states_core_only if cfg["repair_mode"] == "core_only" else _route_policy_states
            # K1 knows the DAE-only system and therefore assumes deterministic
            # always-DAE routing. It never observes an oracle attacked flag.
            route_mode = "always_dae" if level == "K1" else "detector"
            policy_states, flags, scores = route_fn(
                [adv], [True], cfg["defender"], cfg.get("detector_model"), cfg["actor"], cfg["device"],
                route_mode=route_mode,
                detector_threshold=cfg.get("detector_threshold"),
                detector_feature_mode="posterior",
                time_indices=[int(context.time_index)], stations=[int(context.station)],
                is_new_arrivals=[int(bool(context.is_new_arrival))],
                prev_obs_refs=[snapshot.prev_observed.get(vehicle_id, adv)], vehicle_ids=[vehicle_id],
                episode_index=episode_id, dae_runtime=snapshot.dae_runtime,
            )
            state = to_numpy_1d(policy_states[0]).astype(np.float32)
            route_flag = float(bool(flags[0]))
            score_arr = np.asarray(scores).reshape(-1)
            detector_score = float(score_arr[0]) if score_arr.size else float("nan")
        if level in {"K3", "K4"}:
            belief_states = snapshot.belief.repair_batch(
                [state], [vehicle_id], [int(bool(context.is_new_arrival))], np.asarray([detector_score], dtype=np.float32), self._env
            )
            # The explicit None at K3 is the disclosure boundary: its gate cannot
            # call the Shield or inspect Shield thresholds before K4.
            shield_for_gate = cfg.get("shield_config") if level == "K4" else None
            selected, branches = snapshot.gate.select_batch(
                [state], belief_states, [vehicle_id], [int(bool(context.is_new_arrival))], snapshot.belief,
                shield_for_gate, self._env, cfg["reward_profile"], snapshot.prev_policy,
                snapshot.prev_action, snapshot.prev_time, detector_scores=[detector_score], route_flags=[bool(route_flag)],
            )
            state = to_numpy_1d(selected[0]).astype(np.float32)
            belief_flag = float(branches[0] == "belief")
        if level == "K4":
            corrected, _flags = _shield_single_state(
                state, snapshot.prev_policy.get(vehicle_id), snapshot.prev_action.get(vehicle_id),
                snapshot.prev_time.get(vehicle_id), cfg["shield_config"], self._env,
                is_new_arrival=bool(context.is_new_arrival),
            )
            final_state = to_numpy_1d(corrected).astype(np.float32)
            temporal = float(np.max(np.abs(final_state - state)))
        else:
            final_state = state
        action = self._actor_action(final_state)
        if commit:
            snapshot.prev_observed[vehicle_id] = adv.copy()
            snapshot.prev_policy[vehicle_id] = final_state.copy()
            snapshot.prev_action[vehicle_id] = np.asarray([action], dtype=np.float32)
            snapshot.prev_time[vehicle_id] = int(context.time_index)
            if snapshot.belief is not None:
                snapshot.belief.update_actions([vehicle_id], np.asarray([[action]], dtype=np.float32), int(context.time_index))
        return {"action": action, "route": route_flag, "belief": belief_flag, "temporal": temporal, "state": final_state}

    def _candidate_deltas(self, key: tuple[int, int], base_delta: np.ndarray) -> list[np.ndarray]:
        previous = self._prev_delta(key)
        base = self._mask_delta(self._bounded_delta(base_delta))
        smooth = self._mask_delta(np.clip(0.65 * previous + base, previous - self.config.temporal_eta, previous + self.config.temporal_eta))
        cautious = self._mask_delta(np.clip(0.5 * base, previous - self.config.temporal_eta, previous + self.config.temporal_eta))
        return [base, smooth, cautious, np.zeros_like(base, dtype=np.float32)]

    def _shape_delta(self, key: tuple[int, int], obs: np.ndarray, base_delta: np.ndarray, context: AttackContext) -> np.ndarray:
        if not self._ready:
            raise RuntimeError("Causal knowledge attacker was used before configure_target_defense()")
        best_delta = np.zeros_like(obs, dtype=np.float32)
        best_score = -float("inf")
        best_info: dict[str, float | np.ndarray] = {}
        for candidate in self._candidate_deltas(key, base_delta):
            info = self._pipeline_eval(key=key, obs=obs, delta=candidate, context=context, commit=False)
            norm = float(np.linalg.norm(candidate[list(self.attack_indices)], ord=2))
            # Penalize only mechanisms disclosed by the current K level.
            score = -float(info["action"]) - 0.02 * norm
            if self.knowledge_level in {"K2", "K3", "K4"}:
                score -= 0.035 * float(info["route"])
            if self.knowledge_level in {"K3", "K4"}:
                score -= 0.055 * float(info["belief"])
            if self.knowledge_level == "K4":
                score -= 0.080 * float(info["temporal"])
            if score > best_score:
                best_score, best_delta, best_info = score, candidate, info
        committed = self._pipeline_eval(key=key, obs=obs, delta=best_delta, context=context, commit=True)
        self._candidate_log.append({
            "knowledge": self.knowledge_level, "score": float(best_score), "action": float(committed["action"]),
            "route": float(committed["route"]), "belief": float(committed["belief"]), "temporal": float(committed["temporal"]),
        })
        return self._mask_delta(self._bounded_delta(best_delta))


__all__ = ["CausalKnowledgeConfig", "CausalKnowledgeLadderAttacker", "KNOWLEDGE_LEVELS"]
