from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .merged_core import ChargingEnv, RewardProfile, TRAIN_PROFILE


@dataclass(frozen=True)
class ExactStepRewardContext:
    price: np.ndarray
    norm_price: np.ndarray
    pv: np.ndarray
    wt: np.ndarray
    load: np.ndarray
    horizon: int
    max_price: float
    slice_hours: float
    max_power: float
    battery_capacity: float
    reward_profile: RewardProfile


def _normalized_step_cost(step_cost: np.ndarray, *, max_price: float, slice_hours: float, max_power: float) -> np.ndarray:
    lower = -float(max_price) * float(slice_hours) * float(max_power) * 0.5
    upper = float(max_price) * float(slice_hours) * float(max_power) * 0.5
    if np.isclose(lower, upper):
        return np.zeros_like(step_cost, dtype=np.float32)
    return ((step_cost - lower) / (upper - lower)).astype(np.float32)


def _cost_upper_bound(*, max_price: float, battery_capacity: float) -> float:
    return float(max_price) * float(battery_capacity)


def _normalize_cumulative_cost(cost: np.ndarray, *, max_price: float, battery_capacity: float) -> np.ndarray:
    upper = _cost_upper_bound(max_price=max_price, battery_capacity=battery_capacity)
    if np.isclose(upper, 0.0):
        return np.zeros_like(cost, dtype=np.float32)
    return (cost / upper).astype(np.float32)


def _dense_safety_penalty(
    soc: np.ndarray,
    t_re: np.ndarray,
    action: np.ndarray,
    *,
    target_soc: float,
    max_power: float,
    slice_hours: float,
    battery_capacity: float,
) -> np.ndarray:
    step_delta = max(float(max_power) * float(slice_hours) / float(battery_capacity), 1e-8)
    remaining_slots = np.maximum(t_re.astype(np.float32) * 12.0, 1.0)
    action_need = (float(target_soc) - soc.astype(np.float32)) / (remaining_slots * step_delta)
    return np.maximum(0.0, action_need - action.astype(np.float32)).astype(np.float32) ** 2


def build_exact_step_reward_context(
    signals_path,
    *,
    reward_profile: RewardProfile = TRAIN_PROFILE,
) -> ExactStepRewardContext:
    env = ChargingEnv(signals_path=signals_path, reward_profile=reward_profile)
    return ExactStepRewardContext(
        price=env.signals.price.astype(np.float32),
        norm_price=env.signals.norm_price.astype(np.float32),
        pv=env.signals.pv.astype(np.float32),
        wt=env.signals.wt.astype(np.float32),
        load=env.signals.load.astype(np.float32),
        horizon=int(env.horizon),
        max_price=float(env.signals.max_price),
        slice_hours=float(env.slice_hours),
        max_power=float(env.max_power),
        battery_capacity=float(env.battery_capacity),
        reward_profile=reward_profile,
    )


def exact_counterfactual_step_transitions(
    obs_inputs: np.ndarray | list[np.ndarray],
    actions: np.ndarray | list[float],
    *,
    time_indices: np.ndarray | list[int],
    signals_path=None,
    reward_profile: RewardProfile = TRAIN_PROFILE,
    context: ExactStepRewardContext | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs_arr = np.asarray(obs_inputs, dtype=np.float32).reshape(-1, 11)
    action_arr = np.asarray(actions, dtype=np.float32).reshape(-1)
    time_arr = np.asarray(time_indices, dtype=np.int64).reshape(-1)
    if obs_arr.shape[0] != action_arr.shape[0]:
        raise ValueError('obs_inputs and actions must have identical batch size.')
    if obs_arr.shape[0] != time_arr.shape[0]:
        raise ValueError('time_indices and obs_inputs must have identical batch size.')
    if obs_arr.shape[0] == 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 11), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=bool),
        )
    reward_ctx = context
    if reward_ctx is None:
        if signals_path is None:
            raise ValueError('exact_counterfactual_step_transitions requires either signals_path or a prebuilt context.')
        reward_ctx = build_exact_step_reward_context(signals_path, reward_profile=reward_profile)
    profile = reward_ctx.reward_profile
    time_clipped = np.clip(time_arr, 0, reward_ctx.horizon - 1)
    soc = obs_arr[:, 0]
    t_re = obs_arr[:, 1]
    cost_norm = obs_arr[:, 10]
    new_soc = soc + action_arr * reward_ctx.max_power * reward_ctx.slice_hours / reward_ctx.battery_capacity
    new_t_re = t_re - (1.0 / 12.0)
    done = new_t_re < 1e-8

    p_soc = np.zeros_like(new_soc, dtype=np.float32)
    if np.any(done):
        done_soc = new_soc[done]
        done_penalty = np.zeros_like(done_soc, dtype=np.float32)
        within_exit = (
            (done_soc >= float(profile.exit_target_min))
            & (done_soc <= float(profile.exit_target_max))
        )
        above_run = done_soc > float(profile.running_soc_max)
        below_exit = (~within_exit) & (~above_run)
        done_penalty[above_run] = 1.0 + done_soc[above_run] - float(profile.running_soc_max)
        done_penalty[below_exit] = 1.0 + float(profile.exit_target_min) - done_soc[below_exit]
        p_soc[done] = done_penalty
    if np.any(~done):
        run_soc = new_soc[~done]
        run_penalty = np.zeros_like(run_soc, dtype=np.float32)
        above_run = run_soc > float(profile.running_soc_max)
        below_run = run_soc < float(profile.running_soc_min)
        run_penalty[above_run] = 1.0 + run_soc[above_run] - float(profile.running_soc_max)
        run_penalty[below_run] = 1.0 - run_soc[below_run] + float(profile.running_soc_min)
        p_soc[~done] = run_penalty

    current_price = reward_ctx.price[time_clipped].astype(np.float32)
    step_cost = action_arr * reward_ctx.max_power * reward_ctx.slice_hours * current_price
    normalized_cost = _normalized_step_cost(
        step_cost,
        max_price=reward_ctx.max_price,
        slice_hours=reward_ctx.slice_hours,
        max_power=reward_ctx.max_power,
    )
    action_penalty = np.zeros_like(action_arr, dtype=np.float32)
    if profile.action_penalty_threshold is not None:
        over = np.maximum(np.abs(action_arr) - float(profile.action_penalty_threshold), 0.0)
        action_penalty = over * float(profile.action_penalty_scale)
    dense_safety_penalty = np.zeros_like(action_arr, dtype=np.float32)
    if float(profile.dense_safety_penalty_weight) > 0.0:
        dense_target = (
            float(profile.exit_target_min)
            if profile.dense_safety_target_soc is None
            else float(profile.dense_safety_target_soc)
        )
        dense_safety_penalty = _dense_safety_penalty(
            soc,
            t_re,
            action_arr,
            target_soc=dense_target,
            max_power=reward_ctx.max_power,
            slice_hours=reward_ctx.slice_hours,
            battery_capacity=reward_ctx.battery_capacity,
        )
    reward = (
        -float(profile.reward_soc_weight) * p_soc
        - normalized_cost
        - action_penalty
        - float(profile.dense_safety_penalty_weight) * dense_safety_penalty
    )

    next_idx = np.clip(time_clipped + 1, 0, reward_ctx.horizon - 1)
    price_offsets = np.arange(1, 6, dtype=np.int64).reshape(1, -1)
    future_idx = np.clip(time_clipped[:, None] + price_offsets, 0, reward_ctx.horizon - 1)
    cumulative_cost = cost_norm.astype(np.float32) * _cost_upper_bound(
        max_price=reward_ctx.max_price,
        battery_capacity=reward_ctx.battery_capacity,
    ) + step_cost.astype(np.float32)
    next_obs = np.stack(
        [
            new_soc.astype(np.float32),
            new_t_re.astype(np.float32),
            reward_ctx.pv[next_idx].astype(np.float32),
            reward_ctx.wt[next_idx].astype(np.float32),
            reward_ctx.load[next_idx].astype(np.float32),
            reward_ctx.norm_price[future_idx[:, 0]].astype(np.float32),
            reward_ctx.norm_price[future_idx[:, 1]].astype(np.float32),
            reward_ctx.norm_price[future_idx[:, 2]].astype(np.float32),
            reward_ctx.norm_price[future_idx[:, 3]].astype(np.float32),
            reward_ctx.norm_price[future_idx[:, 4]].astype(np.float32),
            _normalize_cumulative_cost(
                cumulative_cost,
                max_price=reward_ctx.max_price,
                battery_capacity=reward_ctx.battery_capacity,
            ),
        ],
        axis=1,
    ).astype(np.float32)
    return reward.astype(np.float32), next_obs, next_idx.astype(np.int64), done.astype(bool)


def exact_counterfactual_step_rewards(
    obs_inputs: np.ndarray | list[np.ndarray],
    actions: np.ndarray | list[float],
    *,
    time_indices: np.ndarray | list[int],
    signals_path=None,
    reward_profile: RewardProfile = TRAIN_PROFILE,
    context: ExactStepRewardContext | None = None,
) -> np.ndarray:
    reward, _, _, _ = exact_counterfactual_step_transitions(
        obs_inputs,
        actions,
        time_indices=time_indices,
        signals_path=signals_path,
        reward_profile=reward_profile,
        context=context,
    )
    return reward.astype(np.float32)
