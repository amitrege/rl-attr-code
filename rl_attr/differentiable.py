from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .core import FiniteAdaptiveModel, History, HistoryOutcome, TargetFn

Vector = float | tuple[float, ...]
Matrix = float | tuple[tuple[float, ...], ...]
UpdateJacobianFn = Callable[[int, Any, Any, float], Matrix]
UpdateWeightGradFn = Callable[[int, Any, Any, float], Vector]
KernelGradFn = Callable[[int, Any, History, Any], Vector]
TargetGradFn = Callable[[Any], Vector]


@dataclass(frozen=True)
class RecursionBundle:
    prefix: History
    time_index: int
    baseline_outcomes: tuple[HistoryOutcome, ...]
    prefix_mass: float
    state_map: dict[History, Any]
    gamma_by_round: dict[int, dict[History, Vector]]
    value_by_round: dict[int, dict[History, float]]
    backward_by_round: dict[int, dict[History, float]]
    xi_by_round: dict[int, dict[History, float]]


class DifferentiableFiniteAdaptiveModel(FiniteAdaptiveModel):
    def __init__(
        self,
        horizon: int,
        initial_state: Any,
        kernel,
        update,
        *,
        update_jacobian: UpdateJacobianFn,
        update_weight_grad: UpdateWeightGradFn,
        kernel_grad: KernelGradFn,
    ) -> None:
        super().__init__(horizon=horizon, initial_state=initial_state, kernel=kernel, update=update)
        self.update_jacobian = update_jacobian
        self.update_weight_grad = update_weight_grad
        self.kernel_grad = kernel_grad


def build_recursion_bundle(
    model: DifferentiableFiniteAdaptiveModel,
    prefix: History,
    time_index: int,
    target: TargetFn,
    target_grad: TargetGradFn,
) -> RecursionBundle:
    if len(prefix) != time_index:
        raise ValueError("prefix length must match time_index")

    baseline_outcomes = tuple(_baseline_outcomes_extending(model, prefix))
    prefix_mass = sum(outcome.probability for outcome in baseline_outcomes)
    state_map = _state_map(baseline_outcomes)
    gamma_by_round = _build_gamma_by_round(model, prefix, time_index, baseline_outcomes, state_map)

    value_by_round: dict[int, dict[History, float]] = {
        model.horizon + 1: {
            outcome.history: target(outcome.terminal_state)
            for outcome in baseline_outcomes
        }
    }
    for round_index in range(model.horizon, time_index, -1):
        prefixes = _prefixes_of_length(baseline_outcomes, round_index - 1)
        current: dict[History, float] = {}
        for history_prefix in prefixes:
            state = state_map[history_prefix]
            total = 0.0
            for interaction, mass in model.kernel(round_index, state, history_prefix).items():
                next_prefix = history_prefix + (interaction,)
                if next_prefix not in value_by_round[round_index + 1]:
                    continue
                total += mass * value_by_round[round_index + 1][next_prefix]
            current[history_prefix] = total
        value_by_round[round_index] = current

    backward_by_round: dict[int, dict[History, float]] = {
        model.horizon + 1: {
            outcome.history: _dot(target_grad(outcome.terminal_state), gamma_by_round[model.horizon + 1][outcome.history])
            for outcome in baseline_outcomes
        }
    }
    xi_by_round: dict[int, dict[History, float]] = {}
    for round_index in range(model.horizon, time_index, -1):
        prefixes = _prefixes_of_length(baseline_outcomes, round_index - 1)
        current: dict[History, float] = {}
        xi_here: dict[History, float] = {}
        for history_prefix in prefixes:
            state = state_map[history_prefix]
            gamma = gamma_by_round[round_index][history_prefix]
            replay_term = 0.0
            law_term = 0.0
            xi_value = 0.0
            baseline_value = value_by_round[round_index][history_prefix]
            for interaction, mass in model.kernel(round_index, state, history_prefix).items():
                next_prefix = history_prefix + (interaction,)
                if next_prefix not in backward_by_round[round_index + 1]:
                    continue
                next_value = value_by_round[round_index + 1][next_prefix]
                kernel_derivative = _dot(model.kernel_grad(round_index, state, history_prefix, interaction), gamma)
                replay_term += mass * backward_by_round[round_index + 1][next_prefix]
                law_term += kernel_derivative * next_value
                xi_value += kernel_derivative * (next_value - baseline_value)
            current[history_prefix] = replay_term + law_term
            xi_here[history_prefix] = xi_value
        backward_by_round[round_index] = current
        xi_by_round[round_index] = xi_here

    return RecursionBundle(
        prefix=prefix,
        time_index=time_index,
        baseline_outcomes=baseline_outcomes,
        prefix_mass=prefix_mass,
        state_map=state_map,
        gamma_by_round=gamma_by_round,
        value_by_round=value_by_round,
        backward_by_round=backward_by_round,
        xi_by_round=xi_by_round,
    )


def compute_model_based_interventional_influence(
    model: DifferentiableFiniteAdaptiveModel,
    prefix: History,
    time_index: int,
    target: TargetFn,
    target_grad: TargetGradFn,
) -> float:
    bundle = build_recursion_bundle(model, prefix, time_index, target, target_grad)
    return bundle.backward_by_round[time_index + 1][prefix]


def compute_future_law_score_on_log(
    model: DifferentiableFiniteAdaptiveModel,
    history: History,
    time_index: int,
    *,
    bundle: RecursionBundle | None = None,
) -> float:
    if len(history) != model.horizon:
        raise ValueError("history length must match the model horizon")
    prefix = history[:time_index]
    if bundle is None:
        bundle = build_recursion_bundle(model, prefix, time_index, lambda _: 0.0, lambda _: 0.0)
    total = 0.0
    for round_index in range(time_index + 1, model.horizon + 1):
        history_prefix = history[: round_index - 1]
        state = bundle.state_map[history_prefix]
        interaction = history[round_index - 1]
        mass = model.kernel(round_index, state, history_prefix)[interaction]
        if mass == 0.0:
            continue
        total += _dot(model.kernel_grad(round_index, state, history_prefix, interaction), bundle.gamma_by_round[round_index][history_prefix]) / mass
    return total


def compute_score_representation_gap(
    model: DifferentiableFiniteAdaptiveModel,
    prefix: History,
    time_index: int,
    target: TargetFn,
    target_grad: TargetGradFn,
) -> float:
    bundle = build_recursion_bundle(model, prefix, time_index, target, target_grad)
    baseline_state = bundle.state_map[prefix]
    total = 0.0
    for outcome in bundle.baseline_outcomes:
        conditional_probability = outcome.probability / bundle.prefix_mass
        score = compute_future_law_score_on_log(
            model,
            outcome.history,
            time_index,
            bundle=bundle,
        )
        total += conditional_probability * (target(outcome.terminal_state) - target(baseline_state)) * score
    return total


def conditioned_stagewise_gap_terms(bundle: RecursionBundle) -> dict[int, float]:
    terms: dict[int, float] = {}
    for round_index, xi_map in bundle.xi_by_round.items():
        prefix_masses: dict[History, float] = {}
        for outcome in bundle.baseline_outcomes:
            history_prefix = outcome.history[: round_index - 1]
            prefix_masses[history_prefix] = prefix_masses.get(history_prefix, 0.0) + outcome.probability / bundle.prefix_mass
        terms[round_index] = sum(prefix_masses[history_prefix] * value for history_prefix, value in xi_map.items())
    return terms


def _baseline_outcomes_extending(
    model: FiniteAdaptiveModel,
    prefix: History,
) -> list[HistoryOutcome]:
    outcomes = [outcome for outcome in model.enumerate_histories() if outcome.history[: len(prefix)] == prefix]
    if not outcomes:
        raise ValueError("prefix has zero baseline probability")
    return outcomes


def _state_map(outcomes: tuple[HistoryOutcome, ...]) -> dict[History, Any]:
    mapping: dict[History, Any] = {}
    for outcome in outcomes:
        for index, state in enumerate(outcome.states):
            history_prefix = outcome.history[:index]
            mapping[history_prefix] = state
    return mapping


def _prefixes_of_length(
    outcomes: tuple[HistoryOutcome, ...],
    length: int,
) -> tuple[History, ...]:
    return tuple(sorted({outcome.history[:length] for outcome in outcomes}))


def _dot(left: Vector, right: Vector) -> float:
    if isinstance(left, tuple):
        if not isinstance(right, tuple) or len(left) != len(right):
            raise TypeError("vector dimensions do not match")
        return sum(a * b for a, b in zip(left, right))
    if isinstance(right, tuple):
        raise TypeError("vector dimensions do not match")
    return float(left) * float(right)


def _matvec(matrix: Matrix, vector: Vector) -> Vector:
    if isinstance(matrix, tuple):
        if not isinstance(vector, tuple):
            raise TypeError("matrix-vector dimensions do not match")
        return tuple(sum(entry * value for entry, value in zip(row, vector)) for row in matrix)
    if isinstance(vector, tuple):
        raise TypeError("matrix-vector dimensions do not match")
    return float(matrix) * float(vector)


def _build_gamma_by_round(
    model: DifferentiableFiniteAdaptiveModel,
    prefix: History,
    time_index: int,
    baseline_outcomes: tuple[HistoryOutcome, ...],
    state_map: dict[History, Any],
) -> dict[int, dict[History, Vector]]:
    previous_prefix = prefix[:-1]
    gamma_by_round: dict[int, dict[History, Vector]] = {
        time_index + 1: {
            prefix: model.update_weight_grad(
                time_index,
                state_map[previous_prefix],
                prefix[-1],
                1.0,
            )
        }
    }
    for round_index in range(time_index + 1, model.horizon + 1):
        current = gamma_by_round[round_index]
        next_map: dict[History, Vector] = {}
        supported_prefixes = set(_prefixes_of_length(baseline_outcomes, round_index))
        for history_prefix, gamma in current.items():
            state = state_map[history_prefix]
            for interaction, mass in model.kernel(round_index, state, history_prefix).items():
                next_prefix = history_prefix + (interaction,)
                if mass == 0.0 or next_prefix not in supported_prefixes:
                    continue
                next_map[next_prefix] = _matvec(
                    model.update_jacobian(round_index, state, interaction, 1.0),
                    gamma,
                )
        gamma_by_round[round_index + 1] = next_map
    return gamma_by_round
