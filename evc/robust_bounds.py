from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .merged_core import Actor, ChargingEnv, RewardProfile


def observation_bounds_across_scenarios(
    scenarios,
    *,
    reward_profile: RewardProfile,
    max_duration_of_stay: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Conservative feature bounds covering every training scenario."""
    lows: list[np.ndarray] = []
    highs: list[np.ndarray] = []
    for scenario in scenarios:
        env = ChargingEnv(signals_path=scenario.signals_path, reward_profile=reward_profile)
        low, high = env.observation_bounds(max_duration_of_stay=max_duration_of_stay)
        lows.append(np.asarray(low, dtype=np.float32))
        highs.append(np.asarray(high, dtype=np.float32))
    if not lows:
        raise ValueError('At least one scenario is required to build observation bounds.')
    global_low = np.min(np.stack(lows, axis=0), axis=0).astype(np.float32)
    global_high = np.max(np.stack(highs, axis=0), axis=0).astype(np.float32)
    if np.any(global_low > global_high):
        raise RuntimeError('Invalid global observation bounds.')
    return global_low, global_high


def _linear_interval(
    lower: torch.Tensor,
    upper: torch.Tensor,
    layer: torch.nn.Linear,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight_pos = torch.clamp(layer.weight, min=0.0)
    weight_neg = torch.clamp(layer.weight, max=0.0)
    lower_out = F.linear(lower, weight_pos, layer.bias) + F.linear(upper, weight_neg, None)
    upper_out = F.linear(upper, weight_pos, layer.bias) + F.linear(lower, weight_neg, None)
    return lower_out, upper_out


def _bounded_actor_inputs(
    actor: Actor,
    observations: torch.Tensor,
    *,
    epsilon: float,
    obs_low: torch.Tensor,
    obs_high: torch.Tensor,
    attack_indices: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(actor.fc_mu.out_features) != 1:
        raise ValueError('Scalar CROWN bounds require action_dim=1.')
    clean = observations.float()
    if clean.ndim == 1:
        clean = clean.unsqueeze(0)
    low = obs_low.to(device=clean.device, dtype=clean.dtype)
    high = obs_high.to(device=clean.device, dtype=clean.dtype)
    if low.numel() == clean.shape[1] and high.numel() == clean.shape[1]:
        low = low.reshape(1, -1)
        high = high.reshape(1, -1)
    elif low.shape != clean.shape or high.shape != clean.shape:
        raise ValueError('Observation bounds do not match the actor input dimension.')

    mask = torch.zeros_like(clean)
    if attack_indices:
        mask[:, list(attack_indices)] = 1.0
    radius = max(float(epsilon), 0.0) * mask
    return torch.maximum(clean - radius, low), torch.minimum(clean + radius, high)


def actor_ibp_action_bounds(
    actor: Actor,
    observations: torch.Tensor,
    *,
    epsilon: float,
    obs_low: torch.Tensor,
    obs_high: torch.Tensor,
    attack_indices: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Conservative IBP reachable-action interval for the project's scalar actor."""
    lower, upper = _bounded_actor_inputs(
        actor,
        observations,
        epsilon=epsilon,
        obs_low=obs_low,
        obs_high=obs_high,
        attack_indices=attack_indices,
    )
    lower, upper = _linear_interval(lower, upper, actor.fc1)
    lower, upper = F.relu(lower), F.relu(upper)
    lower, upper = _linear_interval(lower, upper, actor.fc2)
    lower, upper = F.relu(lower), F.relu(upper)
    lower, upper = _linear_interval(lower, upper, actor.fc_mu)
    lower, upper = torch.tanh(lower), torch.tanh(upper)
    scaled_lower = lower * actor.action_scale + actor.action_bias
    scaled_upper = upper * actor.action_scale + actor.action_bias
    return torch.minimum(scaled_lower, scaled_upper), torch.maximum(scaled_lower, scaled_upper)


def _crown_relu_backward(
    coefficients: torch.Tensor,
    pre_lower: torch.Tensor,
    pre_upper: torch.Tensor,
    *,
    upper_bound: bool,
    lower_slope_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    positive = pre_lower >= 0.0
    negative = pre_upper <= 0.0
    unstable = ~(positive | negative)
    denominator = (pre_upper - pre_lower).clamp_min(1e-12)
    upper_slope = torch.where(unstable, pre_upper / denominator, positive.to(pre_lower.dtype))
    upper_intercept = torch.where(unstable, -pre_lower * upper_slope, torch.zeros_like(pre_lower))
    lower_slope = torch.where(
        unstable,
        (pre_upper >= -pre_lower).to(pre_lower.dtype),
        positive.to(pre_lower.dtype),
    )
    if lower_slope_override is not None:
        optimized = lower_slope_override.to(device=pre_lower.device, dtype=pre_lower.dtype).clamp(0.0, 1.0)
        lower_slope = torch.where(unstable, optimized, lower_slope)
    use_upper_line = coefficients >= 0.0 if upper_bound else coefficients < 0.0
    selected_slope = torch.where(use_upper_line, upper_slope, lower_slope)
    selected_intercept = torch.where(use_upper_line, upper_intercept, torch.zeros_like(upper_intercept))
    intercept_contribution = (coefficients * selected_intercept).sum(dim=1)
    return coefficients * selected_slope, intercept_contribution


def _actor_crown_preactivation_bound(
    actor: Actor,
    input_lower: torch.Tensor,
    input_upper: torch.Tensor,
    *,
    upper_bound: bool,
    first_lower_slope: torch.Tensor | None = None,
    second_lower_slope: torch.Tensor | None = None,
) -> torch.Tensor:
    first_lower, first_upper = _linear_interval(input_lower, input_upper, actor.fc1)
    first_relu_lower, first_relu_upper = F.relu(first_lower), F.relu(first_upper)
    second_lower, second_upper = _linear_interval(first_relu_lower, first_relu_upper, actor.fc2)

    batch_size = input_lower.shape[0]
    coefficients = actor.fc_mu.weight[0].reshape(1, -1).expand(batch_size, -1)
    bias = actor.fc_mu.bias[0].expand(batch_size)
    coefficients, contribution = _crown_relu_backward(
        coefficients,
        second_lower,
        second_upper,
        upper_bound=upper_bound,
        lower_slope_override=second_lower_slope,
    )
    bias = bias + contribution + torch.matmul(coefficients, actor.fc2.bias)
    coefficients = torch.matmul(coefficients, actor.fc2.weight)
    coefficients, contribution = _crown_relu_backward(
        coefficients,
        first_lower,
        first_upper,
        upper_bound=upper_bound,
        lower_slope_override=first_lower_slope,
    )
    bias = bias + contribution + torch.matmul(coefficients, actor.fc1.bias)
    coefficients = torch.matmul(coefficients, actor.fc1.weight)

    if upper_bound:
        return bias + (
            torch.clamp(coefficients, min=0.0) * input_upper
            + torch.clamp(coefficients, max=0.0) * input_lower
        ).sum(dim=1)
    return bias + (
        torch.clamp(coefficients, min=0.0) * input_lower
        + torch.clamp(coefficients, max=0.0) * input_upper
    ).sum(dim=1)


def actor_crown_action_bounds(
    actor: Actor,
    observations: torch.Tensor,
    *,
    epsilon: float,
    obs_low: torch.Tensor,
    obs_high: torch.Tensor,
    attack_indices: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """CROWN linear-relaxation reachable interval for a one-dimensional action."""
    input_lower, input_upper = _bounded_actor_inputs(
        actor,
        observations,
        epsilon=epsilon,
        obs_low=obs_low,
        obs_high=obs_high,
        attack_indices=attack_indices,
    )
    pre_lower = _actor_crown_preactivation_bound(
        actor,
        input_lower,
        input_upper,
        upper_bound=False,
    ).reshape(-1, 1)
    pre_upper = _actor_crown_preactivation_bound(
        actor,
        input_lower,
        input_upper,
        upper_bound=True,
    ).reshape(-1, 1)
    lower = torch.tanh(torch.minimum(pre_lower, pre_upper))
    upper = torch.tanh(torch.maximum(pre_lower, pre_upper))
    scaled_lower = lower * actor.action_scale + actor.action_bias
    scaled_upper = upper * actor.action_scale + actor.action_bias
    return torch.minimum(scaled_lower, scaled_upper), torch.maximum(scaled_lower, scaled_upper)


def _optimized_crown_preactivation_bound(
    actor: Actor,
    input_lower: torch.Tensor,
    input_upper: torch.Tensor,
    *,
    upper_bound: bool,
    optimization_steps: int,
) -> torch.Tensor:
    """Optimize valid unstable-ReLU lower slopes before the final CROWN pass."""
    batch_size = input_lower.shape[0]
    hidden_size = actor.fc1.out_features
    first_alpha = torch.full(
        (batch_size, hidden_size), 0.5, dtype=input_lower.dtype, device=input_lower.device
    )
    second_alpha = torch.full_like(first_alpha, 0.5)
    direction = -1.0 if upper_bound else 1.0
    for _ in range(max(int(optimization_steps), 0)):
        first_alpha = first_alpha.detach().requires_grad_(True)
        second_alpha = second_alpha.detach().requires_grad_(True)
        bound = _actor_crown_preactivation_bound(
            actor,
            input_lower,
            input_upper,
            upper_bound=upper_bound,
            first_lower_slope=first_alpha,
            second_lower_slope=second_alpha,
        )
        first_grad, second_grad = torch.autograd.grad(
            bound.mean(),
            (first_alpha, second_alpha),
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        if first_grad is None:
            first_grad = torch.zeros_like(first_alpha)
        if second_grad is None:
            second_grad = torch.zeros_like(second_alpha)
        first_alpha = (first_alpha + direction * 0.25 * first_grad.sign()).clamp(0.0, 1.0)
        second_alpha = (second_alpha + direction * 0.25 * second_grad.sign()).clamp(0.0, 1.0)
    return _actor_crown_preactivation_bound(
        actor,
        input_lower,
        input_upper,
        upper_bound=upper_bound,
        first_lower_slope=first_alpha.detach(),
        second_lower_slope=second_alpha.detach(),
    )


def actor_optimized_crown_action_bounds(
    actor: Actor,
    observations: torch.Tensor,
    *,
    epsilon: float,
    obs_low: torch.Tensor,
    obs_high: torch.Tensor,
    attack_indices: tuple[int, ...],
    optimization_steps: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Alpha-optimized CROWN interval for the scalar actor."""
    input_lower, input_upper = _bounded_actor_inputs(
        actor,
        observations,
        epsilon=epsilon,
        obs_low=obs_low,
        obs_high=obs_high,
        attack_indices=attack_indices,
    )
    if float(epsilon) <= 0.0 or int(optimization_steps) <= 0:
        return actor_crown_action_bounds(
            actor,
            observations,
            epsilon=epsilon,
            obs_low=obs_low,
            obs_high=obs_high,
            attack_indices=attack_indices,
        )
    with torch.enable_grad():
        pre_lower = _optimized_crown_preactivation_bound(
            actor,
            input_lower,
            input_upper,
            upper_bound=False,
            optimization_steps=optimization_steps,
        ).reshape(-1, 1)
        pre_upper = _optimized_crown_preactivation_bound(
            actor,
            input_lower,
            input_upper,
            upper_bound=True,
            optimization_steps=optimization_steps,
        ).reshape(-1, 1)
    lower = torch.tanh(torch.minimum(pre_lower, pre_upper))
    upper = torch.tanh(torch.maximum(pre_lower, pre_upper))
    scaled_lower = lower * actor.action_scale + actor.action_bias
    scaled_upper = upper * actor.action_scale + actor.action_bias
    return torch.minimum(scaled_lower, scaled_upper), torch.maximum(scaled_lower, scaled_upper)


def actor_split_crown_action_bounds(
    actor: Actor,
    observations: torch.Tensor,
    *,
    epsilon: float,
    obs_low: torch.Tensor,
    obs_high: torch.Tensor,
    attack_indices: tuple[int, ...],
    split_dimensions: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Certified CROWN bounds after splitting influential input dimensions.

    Every split box is a subset of the original perturbation box and their
    union covers it exactly. Taking the outer minimum/maximum therefore keeps
    the WocaR reachable-action superset while reducing relaxation looseness.
    """
    input_lower, input_upper = _bounded_actor_inputs(
        actor,
        observations,
        epsilon=epsilon,
        obs_low=obs_low,
        obs_high=obs_high,
        attack_indices=attack_indices,
    )
    requested = max(int(split_dimensions), 0)
    available = [int(index) for index in attack_indices if 0 <= int(index) < input_lower.shape[1]]
    if requested <= 0 or not available or float(epsilon) <= 0.0:
        return actor_crown_action_bounds(
            actor,
            observations,
            epsilon=epsilon,
            obs_low=obs_low,
            obs_high=obs_high,
            attack_indices=attack_indices,
        )

    with torch.no_grad():
        first_layer_influence = actor.fc1.weight.detach().abs().sum(dim=0)
        widths = (input_upper - input_lower).detach().mean(dim=0)
        scores = first_layer_influence * widths
        available_t = torch.as_tensor(available, dtype=torch.long, device=scores.device)
        count = min(requested, len(available))
        selected_local = torch.topk(scores[available_t], k=count, largest=True).indices
        selected = [available[int(index)] for index in selected_local.cpu().tolist()]

    branch_lowers = [input_lower]
    branch_uppers = [input_upper]
    for feature_index in selected:
        next_lowers: list[torch.Tensor] = []
        next_uppers: list[torch.Tensor] = []
        for branch_lower, branch_upper in zip(branch_lowers, branch_uppers):
            midpoint = 0.5 * (branch_lower[:, feature_index] + branch_upper[:, feature_index])
            lower_half_upper = branch_upper.clone()
            lower_half_upper[:, feature_index] = midpoint
            upper_half_lower = branch_lower.clone()
            upper_half_lower[:, feature_index] = midpoint
            next_lowers.extend((branch_lower, upper_half_lower))
            next_uppers.extend((lower_half_upper, branch_upper))
        branch_lowers, branch_uppers = next_lowers, next_uppers

    pre_lowers: list[torch.Tensor] = []
    pre_uppers: list[torch.Tensor] = []
    for branch_lower, branch_upper in zip(branch_lowers, branch_uppers):
        pre_lowers.append(
            _actor_crown_preactivation_bound(
                actor,
                branch_lower,
                branch_upper,
                upper_bound=False,
            ).reshape(-1, 1)
        )
        pre_uppers.append(
            _actor_crown_preactivation_bound(
                actor,
                branch_lower,
                branch_upper,
                upper_bound=True,
            ).reshape(-1, 1)
        )

    pre_lower = torch.stack(pre_lowers, dim=1).amin(dim=1)
    pre_upper = torch.stack(pre_uppers, dim=1).amax(dim=1)
    lower = torch.tanh(torch.minimum(pre_lower, pre_upper))
    upper = torch.tanh(torch.maximum(pre_lower, pre_upper))
    scaled_lower = lower * actor.action_scale + actor.action_bias
    scaled_upper = upper * actor.action_scale + actor.action_bias
    return torch.minimum(scaled_lower, scaled_upper), torch.maximum(scaled_lower, scaled_upper)


def scalar_action_stability_loss(
    clean_actions: torch.Tensor,
    action_lower: torch.Tensor,
    action_upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample upper bound on max |pi(s') - pi(s)| for scalar actions."""
    lower_deviation = torch.abs(clean_actions - action_lower).mean(dim=1)
    upper_deviation = torch.abs(action_upper - clean_actions).mean(dim=1)
    per_sample = torch.maximum(lower_deviation, upper_deviation)
    return per_sample.mean(), per_sample
