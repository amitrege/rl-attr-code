from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from random import Random
from typing import Any, Callable, Hashable

Interaction = Hashable
History = tuple[Interaction, ...]
Distribution = dict[Interaction, float]
TargetFn = Callable[[Any], float]
KernelFn = Callable[[int, Any, History], Distribution]
UpdateFn = Callable[[int, Any, Interaction, float], Any]


@dataclass(frozen=True)
class HistoryOutcome:
    history: History
    probability: float
    states: tuple[Any, ...]

    @property
    def terminal_state(self) -> Any:
        return self.states[-1]


@dataclass(frozen=True)
class GapReport:
    prefix: History
    time_index: int
    epsilon: float
    psi_epsilon: float
    psi_zero: float
    interventional_effect: float
    expected_replay_effect: float
    replay_intervention_gap: float


@dataclass(frozen=True)
class ComparisonReport:
    prefix: History
    time_index: int
    epsilon: float
    local_effect: float
    expected_replay_effect: float
    interventional_effect: float
    replay_intervention_gap: float


class FiniteAdaptiveModel:
    def __init__(
        self,
        horizon: int,
        initial_state: Any,
        kernel: KernelFn,
        update: UpdateFn,
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.horizon = horizon
        self.initial_state = initial_state
        self.kernel = kernel
        self.update = update

    def enumerate_histories(self, weights: tuple[float, ...] | None = None) -> list[HistoryOutcome]:
        weights = _normalize_weights(self.horizon, weights)
        outcomes: list[HistoryOutcome] = []

        def recurse(round_index: int, history: History, state: Any, probability: float, states: tuple[Any, ...]) -> None:
            if round_index > self.horizon:
                outcomes.append(
                    HistoryOutcome(
                        history=history,
                        probability=probability,
                        states=states,
                    )
                )
                return

            dist = self.kernel(round_index, state, history)
            _validate_distribution(dist)
            for interaction, mass in dist.items():
                if mass == 0.0:
                    continue
                next_state = self.update(round_index, state, interaction, weights[round_index - 1])
                recurse(
                    round_index + 1,
                    history + (interaction,),
                    next_state,
                    probability * mass,
                    states + (next_state,),
                )

        recurse(1, tuple(), self.initial_state, 1.0, (self.initial_state,))
        return outcomes

    def replay_states_for_history(
        self,
        history: History,
        weights: tuple[float, ...] | None = None,
    ) -> tuple[Any, ...]:
        weights = _normalize_weights(len(history), weights, allow_shorter=True)
        state = self.initial_state
        states = [state]
        for round_index, interaction in enumerate(history, start=1):
            state = self.update(round_index, state, interaction, weights[round_index - 1])
            states.append(state)
        return tuple(states)

    def replay_terminal_state(
        self,
        history: History,
        weights: tuple[float, ...] | None = None,
    ) -> Any:
        return self.replay_states_for_history(history, weights)[-1]

    def simulate_run(
        self,
        rng: Random | None = None,
        weights: tuple[float, ...] | None = None,
    ) -> HistoryOutcome:
        rng = rng or Random()
        weights = _normalize_weights(self.horizon, weights)
        state = self.initial_state
        history: list[Interaction] = []
        states = [state]
        probability = 1.0
        for round_index in range(1, self.horizon + 1):
            dist = self.kernel(round_index, state, tuple(history))
            _validate_distribution(dist)
            interaction = _sample_from_distribution(dist, rng)
            probability *= dist[interaction]
            history.append(interaction)
            state = self.update(round_index, state, interaction, weights[round_index - 1])
            states.append(state)
        return HistoryOutcome(tuple(history), probability, tuple(states))


def one_coordinate_weights(horizon: int, time_index: int, epsilon: float) -> tuple[float, ...]:
    if not 1 <= time_index <= horizon:
        raise ValueError("time_index must lie in [1, horizon]")
    weights = [1.0] * horizon
    weights[time_index - 1] = 1.0 + epsilon
    return tuple(weights)


def compute_psi(
    model: FiniteAdaptiveModel,
    prefix: History,
    time_index: int,
    epsilon: float,
    target: TargetFn,
) -> float:
    if len(prefix) != time_index:
        raise ValueError("prefix length must match time_index")
    outcomes = model.enumerate_histories(one_coordinate_weights(model.horizon, time_index, epsilon))
    conditional = _conditional_outcomes(outcomes, prefix)
    return sum(outcome.probability * target(outcome.terminal_state) for outcome in conditional)


def compute_k_prefix_psi(
    model: FiniteAdaptiveModel,
    prefix: History,
    time_index: int,
    condition_index: int,
    epsilon: float,
    target: TargetFn,
) -> float:
    if not 0 <= condition_index <= model.horizon:
        raise ValueError("condition_index must lie in [0, horizon]")
    if len(prefix) != condition_index:
        raise ValueError("prefix length must match condition_index")
    if condition_index == model.horizon:
        terminal_state = model.replay_terminal_state(
            prefix,
            one_coordinate_weights(model.horizon, time_index, epsilon),
        )
        return target(terminal_state)
    outcomes = model.enumerate_histories(one_coordinate_weights(model.horizon, time_index, epsilon))
    conditional = _conditional_outcomes(outcomes, prefix)
    return sum(outcome.probability * target(outcome.terminal_state) for outcome in conditional)


def compute_interventional_effect(
    model: FiniteAdaptiveModel,
    prefix: History,
    time_index: int,
    epsilon: float,
    target: TargetFn,
) -> float:
    psi_epsilon = compute_psi(model, prefix, time_index, epsilon, target)
    psi_zero = compute_psi(model, prefix, time_index, 0.0, target)
    return psi_epsilon - psi_zero


def compute_k_prefix_effect(
    model: FiniteAdaptiveModel,
    prefix: History,
    time_index: int,
    condition_index: int,
    epsilon: float,
    target: TargetFn,
) -> float:
    return compute_k_prefix_psi(model, prefix, time_index, condition_index, epsilon, target) - compute_k_prefix_psi(
        model,
        prefix,
        time_index,
        condition_index,
        0.0,
        target,
    )


def compute_interventional_influence(
    model: FiniteAdaptiveModel,
    prefix: History,
    time_index: int,
    target: TargetFn,
    step: float = 1e-6,
) -> float:
    return _central_difference(
        lambda epsilon: compute_psi(model, prefix, time_index, epsilon, target),
        step,
    )


def compute_k_prefix_influence(
    model: FiniteAdaptiveModel,
    prefix: History,
    time_index: int,
    condition_index: int,
    target: TargetFn,
    step: float = 1e-6,
) -> float:
    return _central_difference(
        lambda epsilon: compute_k_prefix_psi(model, prefix, time_index, condition_index, epsilon, target),
        step,
    )


def compute_replay_effect_on_log(
    model: FiniteAdaptiveModel,
    history: History,
    time_index: int,
    epsilon: float,
    target: TargetFn,
) -> float:
    if len(history) != model.horizon:
        raise ValueError("history length must match the model horizon")
    perturbed = model.replay_terminal_state(history, one_coordinate_weights(model.horizon, time_index, epsilon))
    baseline = model.replay_terminal_state(history)
    return target(perturbed) - target(baseline)


def compute_replay_influence_on_log(
    model: FiniteAdaptiveModel,
    history: History,
    time_index: int,
    target: TargetFn,
    step: float = 1e-6,
) -> float:
    return _central_difference(
        lambda epsilon: target(
            model.replay_terminal_state(history, one_coordinate_weights(model.horizon, time_index, epsilon))
        ),
        step,
    )


def compute_expected_replay_effect(
    model: FiniteAdaptiveModel,
    prefix: History,
    time_index: int,
    epsilon: float,
    target: TargetFn,
) -> float:
    if len(prefix) != time_index:
        raise ValueError("prefix length must match time_index")
    baseline_outcomes = model.enumerate_histories()
    conditional = _conditional_outcomes(baseline_outcomes, prefix)
    return sum(
        outcome.probability * compute_replay_effect_on_log(model, outcome.history, time_index, epsilon, target)
        for outcome in conditional
    )


def compute_expected_replay_influence(
    model: FiniteAdaptiveModel,
    prefix: History,
    time_index: int,
    target: TargetFn,
    step: float = 1e-6,
) -> float:
    return _central_difference(
        lambda epsilon: compute_expected_replay_effect(model, prefix, time_index, epsilon, target),
        step,
    )


def compute_local_next_state_effect(
    model: FiniteAdaptiveModel,
    prefix: History,
    time_index: int,
    epsilon: float,
    target: TargetFn,
) -> float:
    if len(prefix) != time_index:
        raise ValueError("prefix length must match time_index")
    perturbed_states = model.replay_states_for_history(prefix, one_coordinate_weights(time_index, time_index, epsilon))
    baseline_states = model.replay_states_for_history(prefix)
    return target(perturbed_states[-1]) - target(baseline_states[-1])


def build_gap_report(
    model: FiniteAdaptiveModel,
    prefix: History,
    time_index: int,
    epsilon: float,
    target: TargetFn,
) -> GapReport:
    psi_epsilon = compute_psi(model, prefix, time_index, epsilon, target)
    psi_zero = compute_psi(model, prefix, time_index, 0.0, target)
    interventional_effect = psi_epsilon - psi_zero
    expected_replay_effect = compute_expected_replay_effect(model, prefix, time_index, epsilon, target)
    return GapReport(
        prefix=prefix,
        time_index=time_index,
        epsilon=epsilon,
        psi_epsilon=psi_epsilon,
        psi_zero=psi_zero,
        interventional_effect=interventional_effect,
        expected_replay_effect=expected_replay_effect,
        replay_intervention_gap=interventional_effect - expected_replay_effect,
    )


def conditioning_ladder_table(
    model: FiniteAdaptiveModel,
    history: History,
    time_index: int,
    target: TargetFn,
    *,
    epsilon: float = 0.0,
    step: float = 1e-6,
) -> list[dict[str, Any]]:
    if len(history) != model.horizon:
        raise ValueError("history length must match the model horizon")
    rows: list[dict[str, Any]] = []
    for condition_index in range(model.horizon + 1):
        prefix = history[:condition_index]
        rows.append(
            {
                "condition_index": condition_index,
                "prefix": prefix,
                "psi": compute_k_prefix_psi(model, prefix, time_index, condition_index, epsilon, target),
                "effect": compute_k_prefix_effect(model, prefix, time_index, condition_index, epsilon, target),
                "influence": compute_k_prefix_influence(
                    model,
                    prefix,
                    time_index,
                    condition_index,
                    target,
                    step=step,
                ),
            }
        )
    return rows


def _conditional_outcomes(outcomes: list[HistoryOutcome], prefix: History) -> list[HistoryOutcome]:
    matching = [outcome for outcome in outcomes if outcome.history[: len(prefix)] == prefix]
    total_probability = sum(outcome.probability for outcome in matching)
    if isclose(total_probability, 0.0):
        raise ValueError("prefix has zero probability under the specified model")
    return [
        HistoryOutcome(
            history=outcome.history,
            probability=outcome.probability / total_probability,
            states=outcome.states,
        )
        for outcome in matching
    ]


def _normalize_weights(
    horizon: int,
    weights: tuple[float, ...] | None,
    *,
    allow_shorter: bool = False,
) -> tuple[float, ...]:
    if weights is None:
        return (1.0,) * horizon
    if allow_shorter and len(weights) != horizon:
        raise ValueError("weights must have the same length as the replayed history")
    if not allow_shorter and len(weights) != horizon:
        raise ValueError("weights must have length equal to horizon")
    return tuple(weights)


def _validate_distribution(dist: Distribution) -> None:
    total_probability = sum(dist.values())
    if not isclose(total_probability, 1.0, abs_tol=1e-12):
        raise ValueError(f"distribution must sum to 1, got {total_probability}")
    for interaction, mass in dist.items():
        if mass < 0.0:
            raise ValueError(f"distribution contains negative mass for {interaction!r}")


def _sample_from_distribution(dist: Distribution, rng: Random) -> Interaction:
    threshold = rng.random()
    running = 0.0
    last_key: Interaction | None = None
    for interaction, mass in dist.items():
        last_key = interaction
        running += mass
        if threshold <= running:
            return interaction
    if last_key is None:
        raise ValueError("cannot sample from an empty distribution")
    return last_key


def _central_difference(fn: Callable[[float], float], step: float) -> float:
    if step <= 0.0:
        raise ValueError("step must be positive")
    return (fn(step) - fn(-step)) / (2.0 * step)
