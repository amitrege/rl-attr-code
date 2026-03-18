from __future__ import annotations

from .action_only import ActionOnlyFiniteBandit
from .bandits import TwoArmedBernoulliBandit, sigma
from .core import FiniteAdaptiveModel, History


def make_exogenous_binary_model(horizon: int = 3) -> FiniteAdaptiveModel:
    def kernel(round_index: int, state: int, history: History) -> dict[int, float]:
        del round_index, state, history
        return {0: 0.45, 1: 0.55}

    def update(round_index: int, state: int, interaction: int, weight: float) -> int:
        del round_index
        return state + int(weight * interaction)

    return FiniteAdaptiveModel(
        horizon=horizon,
        initial_state=0,
        kernel=kernel,
        update=update,
    )


def make_action_only_example() -> ActionOnlyFiniteBandit:
    def context_dist(round_index: int, history: History) -> dict[int, float]:
        del history
        if round_index == 1:
            return {0: 0.45, 1: 0.55}
        if round_index == 2:
            return {0: 0.5, 1: 0.5}
        return {0: 0.3, 1: 0.7}

    def reward_dist(round_index: int, history: History, context: int, action: int) -> dict[int, float]:
        del round_index, history
        reward_one_probability = 0.8 if action == context else 0.25
        return {1: reward_one_probability, 0: 1.0 - reward_one_probability}

    def policy(theta: float, context: int) -> dict[int, float]:
        probability = sigma(theta + (0.4 if context == 1 else -0.3))
        return {1: probability, 0: 1.0 - probability}

    def update(round_index: int, theta: float, interaction: tuple[int, int, int], weight: float) -> float:
        del round_index
        context, action, reward = interaction
        signed_action = 1.0 if action == 1 else -1.0
        context_offset = 0.15 if context == 1 else -0.05
        return theta + weight * reward * (signed_action + context_offset)

    return ActionOnlyFiniteBandit(
        horizon=3,
        initial_state=0.1,
        context_dist=context_dist,
        reward_dist=reward_dist,
        policy=policy,
        update=update,
    )


def make_reference_bandit() -> TwoArmedBernoulliBandit:
    return TwoArmedBernoulliBandit(
        q=0.4,
        mu0=0.2,
        mu1=0.85,
        etas=(0.2, 0.15, 0.1),
    )


def make_two_step_bandit_strong_separation() -> TwoArmedBernoulliBandit:
    return TwoArmedBernoulliBandit(
        q=0.25,
        mu0=0.98,
        mu1=1.0,
        etas=(0.25 * __import__("math").log(3.0), 2.0),
    )


def collect_prefixes(model: FiniteAdaptiveModel, time_index: int) -> list[History]:
    prefixes = {outcome.history[:time_index] for outcome in model.enumerate_histories()}
    return sorted(prefixes)


def prefix_to_str(prefix: History) -> str:
    if not prefix:
        return "∅"
    return " | ".join(str(item) for item in prefix)

