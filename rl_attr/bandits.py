from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Callable

from .core import (
    ComparisonReport,
    FiniteAdaptiveModel,
    History,
    build_gap_report,
    compute_expected_replay_influence,
    compute_interventional_influence,
    compute_local_next_state_effect,
    one_coordinate_weights,
)
from .differentiable import DifferentiableFiniteAdaptiveModel


def sigma(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


def logit(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie in (0, 1)")
    return log(probability / (1.0 - probability))


def target_better_arm_probability(policy_value: float) -> float:
    return policy_value


@dataclass(frozen=True)
class TwoArmedBernoulliBandit:
    q: float
    mu0: float
    mu1: float
    etas: tuple[float, ...]

    def __post_init__(self) -> None:
        if not 0.0 < self.q < 1.0:
            raise ValueError("q must lie in (0, 1)")
        if not 0.0 <= self.mu0 <= 1.0:
            raise ValueError("mu0 must lie in [0, 1]")
        if not 0.0 <= self.mu1 <= 1.0:
            raise ValueError("mu1 must lie in [0, 1]")
        if not self.etas:
            raise ValueError("etas must be non-empty")
        if any(eta <= 0.0 for eta in self.etas):
            raise ValueError("all step sizes must be positive")

    @property
    def horizon(self) -> int:
        return len(self.etas)

    @property
    def interaction_space(self) -> tuple[tuple[int, int], ...]:
        return ((1, 1), (1, 0), (0, 1), (0, 0))

    def to_model(self) -> FiniteAdaptiveModel:
        return FiniteAdaptiveModel(
            horizon=self.horizon,
            initial_state=self.q,
            kernel=self.kernel,
            update=self.update,
        )

    def to_differentiable_model(self) -> DifferentiableFiniteAdaptiveModel:
        return DifferentiableFiniteAdaptiveModel(
            horizon=self.horizon,
            initial_state=self.q,
            kernel=self.kernel,
            update=self.update,
            update_jacobian=self.update_jacobian,
            update_weight_grad=self.update_weight_grad,
            kernel_grad=self.kernel_grad,
        )

    def kernel(self, round_index: int, state: float, history: History) -> dict[tuple[int, int], float]:
        del round_index, history
        return {
            (1, 1): state * self.mu1,
            (1, 0): state * (1.0 - self.mu1),
            (0, 1): (1.0 - state) * self.mu0,
            (0, 0): (1.0 - state) * (1.0 - self.mu0),
        }

    def update(self, round_index: int, state: float, interaction: tuple[int, int], weight: float) -> float:
        action, reward = interaction
        return mirror_descent_update(state, action, reward, self.etas[round_index - 1], weight)

    def update_jacobian(self, round_index: int, state: float, interaction: tuple[int, int], weight: float) -> float:
        action, reward = interaction
        eta = self.etas[round_index - 1]
        if reward == 0:
            return 1.0
        if action == 1:
            updated = mirror_descent_update(state, action, reward, eta, weight)
            slope = (1.0 / (state * (1.0 - state))) - (eta * weight / (state**2))
            return updated * (1.0 - updated) * slope
        updated = mirror_descent_update(state, action, reward, eta, weight)
        slope = (1.0 / (state * (1.0 - state))) - (eta * weight / ((1.0 - state) ** 2))
        return updated * (1.0 - updated) * slope

    def update_weight_grad(self, round_index: int, state: float, interaction: tuple[int, int], weight: float) -> float:
        action, reward = interaction
        eta = self.etas[round_index - 1]
        if reward == 0:
            return 0.0
        updated = mirror_descent_update(state, action, reward, eta, weight)
        if action == 1:
            return updated * (1.0 - updated) * (eta / state)
        return updated * (1.0 - updated) * (-eta / (1.0 - state))

    def kernel_grad(
        self,
        round_index: int,
        state: float,
        history: History,
        interaction: tuple[int, int],
    ) -> float:
        del round_index, state, history
        action, reward = interaction
        if action == 1 and reward == 1:
            return self.mu1
        if action == 1 and reward == 0:
            return 1.0 - self.mu1
        if action == 0 and reward == 1:
            return -self.mu0
        return -(1.0 - self.mu0)


def mirror_descent_update(
    probability: float,
    action: int,
    reward: int,
    eta: float,
    weight: float = 1.0,
) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("policy value must stay in (0, 1)")
    gain_one = reward / probability if action == 1 else 0.0
    gain_zero = reward / (1.0 - probability) if action == 0 else 0.0
    numerator = probability * exp(eta * weight * gain_one)
    denominator = numerator + (1.0 - probability) * exp(eta * weight * gain_zero)
    return numerator / denominator


def two_step_positive_prefix() -> History:
    return ((1, 1),)


def p2_for_epsilon(bandit: TwoArmedBernoulliBandit, epsilon: float) -> float:
    if bandit.horizon != 2:
        raise ValueError("closed-form two-step formulas require horizon 2")
    return sigma(logit(bandit.q) + bandit.etas[0] * (1.0 + epsilon) / bandit.q)


def c_constant(bandit: TwoArmedBernoulliBandit) -> float:
    p = p2_for_epsilon(bandit, 0.0)
    return (bandit.etas[0] / bandit.q) * p * (1.0 - p)


def f_eta(probability: float, eta: float) -> float:
    return sigma(logit(probability) + eta / probability)


def f_eta_prime(probability: float, eta: float) -> float:
    value = f_eta(probability, eta)
    slope = (1.0 / (probability * (1.0 - probability))) - (eta / (probability**2))
    return value * (1.0 - value) * slope


def g_eta(probability: float, eta: float) -> float:
    return sigma(logit(probability) - eta / (1.0 - probability))


def g_eta_prime(probability: float, eta: float) -> float:
    value = g_eta(probability, eta)
    slope = (1.0 / (probability * (1.0 - probability))) - (eta / ((1.0 - probability) ** 2))
    return value * (1.0 - value) * slope


def G_mu_eta(probability: float, mu0: float, mu1: float, eta: float) -> float:
    return (
        probability * (mu1 * f_eta(probability, eta) + (1.0 - mu1) * probability)
        + (1.0 - probability) * (mu0 * g_eta(probability, eta) + (1.0 - mu0) * probability)
    )


def G_mu_eta_prime(probability: float, mu0: float, mu1: float, eta: float) -> float:
    a = mu1 * f_eta(probability, eta) + (1.0 - mu1) * probability
    b = mu0 * g_eta(probability, eta) + (1.0 - mu0) * probability
    a_prime = mu1 * f_eta_prime(probability, eta) + (1.0 - mu1)
    b_prime = mu0 * g_eta_prime(probability, eta) + (1.0 - mu0)
    return a - b + probability * a_prime + (1.0 - probability) * b_prime


def R_mu_eta(probability: float, mu0: float, mu1: float, eta: float) -> float:
    return (
        probability * (mu1 * f_eta_prime(probability, eta) + (1.0 - mu1))
        + (1.0 - probability) * (mu0 * g_eta_prime(probability, eta) + (1.0 - mu0))
    )


def closed_form_interventional_influence(bandit: TwoArmedBernoulliBandit) -> float:
    if bandit.horizon != 2:
        raise ValueError("closed-form two-step formulas require horizon 2")
    p = p2_for_epsilon(bandit, 0.0)
    return c_constant(bandit) * G_mu_eta_prime(p, bandit.mu0, bandit.mu1, bandit.etas[1])


def closed_form_expected_replay_influence(bandit: TwoArmedBernoulliBandit) -> float:
    if bandit.horizon != 2:
        raise ValueError("closed-form two-step formulas require horizon 2")
    p = p2_for_epsilon(bandit, 0.0)
    return c_constant(bandit) * R_mu_eta(p, bandit.mu0, bandit.mu1, bandit.etas[1])


def compare_local_replay_interventional(
    bandit: TwoArmedBernoulliBandit,
    epsilon: float = 1e-6,
    local_target: Callable[[float], float] = target_better_arm_probability,
    global_target: Callable[[float], float] = target_better_arm_probability,
) -> ComparisonReport:
    model = bandit.to_model()
    prefix = two_step_positive_prefix()
    time_index = 1
    local_effect = compute_local_next_state_effect(model, prefix, time_index, epsilon, local_target)
    gap_report = build_gap_report(model, prefix, time_index, epsilon, global_target)
    return ComparisonReport(
        prefix=prefix,
        time_index=time_index,
        epsilon=epsilon,
        local_effect=local_effect,
        expected_replay_effect=gap_report.expected_replay_effect,
        interventional_effect=gap_report.interventional_effect,
        replay_intervention_gap=gap_report.replay_intervention_gap,
    )


def numeric_two_step_reports(
    bandit: TwoArmedBernoulliBandit,
    step: float = 1e-6,
) -> dict[str, float]:
    model = bandit.to_model()
    prefix = two_step_positive_prefix()
    local_influence = (
        compute_local_next_state_effect(model, prefix, 1, step, target_better_arm_probability)
        - compute_local_next_state_effect(model, prefix, 1, -step, target_better_arm_probability)
    ) / (2.0 * step)
    return {
        "local_influence": local_influence,
        "expected_replay_influence": compute_expected_replay_influence(
            model, prefix, 1, target_better_arm_probability, step=step
        ),
        "interventional_influence": compute_interventional_influence(
            model, prefix, 1, target_better_arm_probability, step=step
        ),
    }


def find_strong_separation_example(eta2: float = 2.0) -> TwoArmedBernoulliBandit:
    q = 0.25
    eta1 = 0.25 * log(3.0)
    for index in range(2000):
        mu0 = 0.95 + (0.049999 * index / 1999.0)
        candidate = TwoArmedBernoulliBandit(q=q, mu0=mu0, mu1=1.0, etas=(eta1, eta2))
        replay = closed_form_expected_replay_influence(candidate)
        intervention = closed_form_interventional_influence(candidate)
        if replay < 0.0 < intervention:
            return candidate
    raise RuntimeError("failed to find a strong-separation example on the preset grid")


def replay_sensitivity(
    bandit: TwoArmedBernoulliBandit,
    prefix: History,
    time_index: int,
    state_index: int,
    step: float = 1e-6,
) -> float:
    model = bandit.to_model()
    plus = model.replay_states_for_history(prefix, one_coordinate_weights(len(prefix), time_index, step))[state_index - 1]
    minus = model.replay_states_for_history(prefix, one_coordinate_weights(len(prefix), time_index, -step))[state_index - 1]
    return (plus - minus) / (2.0 * step)


def prefix_value_map(
    bandit: TwoArmedBernoulliBandit,
    target: Callable[[float], float] = target_better_arm_probability,
) -> dict[History, float]:
    outcomes = bandit.to_model().enumerate_histories()
    values: dict[History, float] = {}
    for prefix_length in range(bandit.horizon + 1):
        prefixes = {outcome.history[:prefix_length] for outcome in outcomes}
        for prefix in prefixes:
            matching = [outcome for outcome in outcomes if outcome.history[:prefix_length] == prefix]
            total = sum(outcome.probability for outcome in matching)
            values[prefix] = sum(outcome.probability * target(outcome.terminal_state) for outcome in matching) / total
    return values


def stagewise_gap_terms(
    bandit: TwoArmedBernoulliBandit,
    prefix: History,
    time_index: int,
    target: Callable[[float], float] = target_better_arm_probability,
    step: float = 1e-6,
) -> dict[int, float]:
    model = bandit.to_model()
    outcomes = model.enumerate_histories()
    baseline_conditioned = [
        outcome for outcome in outcomes if outcome.history[: len(prefix)] == prefix
    ]
    prefix_prob = sum(outcome.probability for outcome in baseline_conditioned)
    if prefix_prob == 0.0:
        raise ValueError("prefix has zero probability")
    value_map = prefix_value_map(bandit, target)
    terms: dict[int, float] = {}
    for round_index in range(time_index + 1, bandit.horizon + 1):
        prefix_terms: dict[History, float] = {}
        round_prefixes = {
            outcome.history[: round_index - 1]
            for outcome in outcomes
            if outcome.history[: len(prefix)] == prefix
        }
        for round_prefix in round_prefixes:
            states = model.replay_states_for_history(round_prefix)
            policy_value = states[-1]
            sensitivity = replay_sensitivity(bandit, round_prefix, time_index, round_index, step=step)
            v_here = value_map[round_prefix]
            xi = 0.0
            for action, reward in bandit.interaction_space:
                reward_probability = _reward_probability(bandit, action, reward)
                derivative = reward_probability if action == 1 else -reward_probability
                next_prefix = round_prefix + ((action, reward),)
                xi += derivative * sensitivity * (value_map[next_prefix] - v_here)
            prefix_terms[round_prefix] = xi
        expected_term = 0.0
        for round_prefix, xi in prefix_terms.items():
            probability = sum(
                outcome.probability
                for outcome in outcomes
                if outcome.history[: round_index - 1] == round_prefix
            )
            expected_term += (probability / prefix_prob) * xi
        terms[round_index] = expected_term
    return terms


def score_representation_gap(
    bandit: TwoArmedBernoulliBandit,
    prefix: History,
    time_index: int,
    target: Callable[[float], float] = target_better_arm_probability,
    step: float = 1e-6,
) -> float:
    model = bandit.to_model()
    baseline = model.enumerate_histories()
    conditioned = [outcome for outcome in baseline if outcome.history[: len(prefix)] == prefix]
    prefix_prob = sum(outcome.probability for outcome in conditioned)
    if prefix_prob == 0.0:
        raise ValueError("prefix has zero probability")
    baseline_next_state = model.replay_states_for_history(prefix)[-1]
    total = 0.0
    for outcome in conditioned:
        conditional_probability = outcome.probability / prefix_prob
        score = future_law_score_on_log(bandit, outcome.history, time_index, step=step)
        total += conditional_probability * (target(outcome.terminal_state) - target(baseline_next_state)) * score
    return total


def future_law_score_on_log(
    bandit: TwoArmedBernoulliBandit,
    history: History,
    time_index: int,
    step: float = 1e-6,
) -> float:
    model = bandit.to_model()
    baseline_states = model.replay_states_for_history(history)
    total = 0.0
    for round_index in range(time_index + 1, len(history) + 1):
        round_prefix = history[: round_index - 1]
        sensitivity = replay_sensitivity(bandit, round_prefix, time_index, round_index, step=step)
        policy_value = baseline_states[round_index - 1]
        action, _ = history[round_index - 1]
        action_score = 1.0 / policy_value if action == 1 else -1.0 / (1.0 - policy_value)
        total += action_score * sensitivity
    return total


def _reward_probability(bandit: TwoArmedBernoulliBandit, action: int, reward: int) -> float:
    if action == 1:
        return bandit.mu1 if reward == 1 else 1.0 - bandit.mu1
    return bandit.mu0 if reward == 1 else 1.0 - bandit.mu0
