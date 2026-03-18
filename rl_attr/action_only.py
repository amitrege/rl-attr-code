from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Hashable

from .core import (
    FiniteAdaptiveModel,
    History,
    TargetFn,
    compute_interventional_effect,
    compute_psi,
    one_coordinate_weights,
)

Context = Hashable
Action = Hashable
Reward = Hashable
Interaction = tuple[Context, Action, Reward]
ContextDistFn = Callable[[int, History], dict[Context, float]]
RewardDistFn = Callable[[int, History, Context, Action], dict[Reward, float]]
PolicyFn = Callable[[Any, Context], dict[Action, float]]
UpdateFn = Callable[[int, Any, Interaction, float], Any]


@dataclass(frozen=True)
class ActionOnlyFiniteBandit:
    horizon: int
    initial_state: Any
    context_dist: ContextDistFn
    reward_dist: RewardDistFn
    policy: PolicyFn
    update: UpdateFn

    def to_model(self) -> FiniteAdaptiveModel:
        return FiniteAdaptiveModel(
            horizon=self.horizon,
            initial_state=self.initial_state,
            kernel=self.kernel,
            update=self.update,
        )

    def kernel(self, round_index: int, state: Any, history: History) -> dict[Interaction, float]:
        dist: dict[Interaction, float] = {}
        for context, context_mass in self.context_dist(round_index, history).items():
            for action, action_mass in self.policy(state, context).items():
                for reward, reward_mass in self.reward_dist(round_index, history, context, action).items():
                    dist[(context, action, reward)] = context_mass * action_mass * reward_mass
        return dist

    def policy_probability(self, state: Any, context: Context, action: Action) -> float:
        return self.policy(state, context)[action]


def identified_psi_from_baseline(
    model: ActionOnlyFiniteBandit,
    prefix: History,
    time_index: int,
    epsilon: float,
    target: TargetFn,
) -> float:
    base_model = model.to_model()
    baseline_outcomes = base_model.enumerate_histories()
    baseline_prefix_mass = sum(
        outcome.probability
        for outcome in baseline_outcomes
        if outcome.history[: len(prefix)] == prefix
    )
    if baseline_prefix_mass == 0.0:
        raise ValueError("prefix has zero baseline probability")

    perturbed_weights = one_coordinate_weights(model.horizon, time_index, epsilon)
    total = 0.0
    for outcome in baseline_outcomes:
        if outcome.history[: len(prefix)] != prefix:
            continue
        conditional_mass = outcome.probability / baseline_prefix_mass
        perturbed_states = base_model.replay_states_for_history(outcome.history, perturbed_weights)
        ratio = 1.0
        for round_index in range(time_index + 1, model.horizon + 1):
            context, action, _ = outcome.history[round_index - 1]
            baseline_state = outcome.states[round_index - 1]
            perturbed_state = perturbed_states[round_index - 1]
            numerator = model.policy_probability(perturbed_state, context, action)
            denominator = model.policy_probability(baseline_state, context, action)
            ratio *= numerator / denominator
        total += conditional_mass * ratio * target(perturbed_states[-1])
    return total


def identified_interventional_effect_from_baseline(
    model: ActionOnlyFiniteBandit,
    prefix: History,
    time_index: int,
    epsilon: float,
    target: TargetFn,
) -> float:
    return identified_psi_from_baseline(model, prefix, time_index, epsilon, target) - identified_psi_from_baseline(
        model, prefix, time_index, 0.0, target
    )


def identified_interventional_influence_from_baseline(
    model: ActionOnlyFiniteBandit,
    prefix: History,
    time_index: int,
    target: TargetFn,
    step: float = 1e-6,
) -> float:
    return (
        identified_psi_from_baseline(model, prefix, time_index, step, target)
        - identified_psi_from_baseline(model, prefix, time_index, -step, target)
    ) / (2.0 * step)


def direct_vs_identified_effect(
    model: ActionOnlyFiniteBandit,
    prefix: History,
    time_index: int,
    epsilon: float,
    target: TargetFn,
) -> tuple[float, float]:
    finite_model = model.to_model()
    direct = compute_interventional_effect(finite_model, prefix, time_index, epsilon, target)
    identified = identified_interventional_effect_from_baseline(model, prefix, time_index, epsilon, target)
    return direct, identified


def direct_vs_identified_psi(
    model: ActionOnlyFiniteBandit,
    prefix: History,
    time_index: int,
    epsilon: float,
    target: TargetFn,
) -> tuple[float, float]:
    finite_model = model.to_model()
    direct = compute_psi(finite_model, prefix, time_index, epsilon, target)
    identified = identified_psi_from_baseline(model, prefix, time_index, epsilon, target)
    return direct, identified
