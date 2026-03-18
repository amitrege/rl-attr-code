from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from .action_only import (
    ActionOnlyFiniteBandit,
    identified_interventional_effect_from_baseline,
    identified_psi_from_baseline,
)
from .bandits import (
    TwoArmedBernoulliBandit,
    compare_local_replay_interventional,
    score_representation_gap,
    stagewise_gap_terms,
    target_better_arm_probability,
    two_step_positive_prefix,
)
from .core import (
    FiniteAdaptiveModel,
    History,
    TargetFn,
    compute_expected_replay_effect,
    compute_expected_replay_influence,
    compute_interventional_effect,
    compute_interventional_influence,
    compute_local_next_state_effect,
    compute_psi,
    compute_replay_influence_on_log,
)
from .examples import collect_prefixes, prefix_to_str


def effect_curve_over_epsilon(
    model: FiniteAdaptiveModel,
    prefix: History,
    time_index: int,
    epsilons: Sequence[float],
    target: TargetFn,
    *,
    include_local: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for epsilon in epsilons:
        row = {
            "epsilon": float(epsilon),
            "interventional_effect": compute_interventional_effect(model, prefix, time_index, float(epsilon), target),
            "expected_replay_effect": compute_expected_replay_effect(model, prefix, time_index, float(epsilon), target),
        }
        if include_local:
            row["local_effect"] = compute_local_next_state_effect(model, prefix, time_index, float(epsilon), target)
        rows.append(row)
    return pd.DataFrame(rows)


def prefix_gap_table(
    model: FiniteAdaptiveModel,
    time_index: int,
    target: TargetFn,
    *,
    step: float = 1e-6,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for prefix in collect_prefixes(model, time_index):
        interventional = compute_interventional_influence(model, prefix, time_index, target, step=step)
        replay = compute_expected_replay_influence(model, prefix, time_index, target, step=step)
        rows.append(
            {
                "time_index": time_index,
                "prefix": prefix_to_str(prefix),
                "interventional_influence": interventional,
                "expected_replay_influence": replay,
                "gap": interventional - replay,
            }
        )
    return pd.DataFrame(rows).sort_values("gap", ascending=False).reset_index(drop=True)


def full_history_table(
    model: FiniteAdaptiveModel,
    prefix: History,
    time_index: int,
    target: TargetFn,
    *,
    step: float = 1e-6,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    outcomes = model.enumerate_histories()
    prefix_mass = sum(outcome.probability for outcome in outcomes if outcome.history[: len(prefix)] == prefix)
    for outcome in outcomes:
        if outcome.history[: len(prefix)] != prefix:
            continue
        conditional_probability = outcome.probability / prefix_mass
        replay = compute_replay_influence_on_log(model, outcome.history, time_index, target, step=step)
        rows.append(
            {
                "history": prefix_to_str(outcome.history),
                "conditional_probability": conditional_probability,
                "replay_influence": replay,
            }
        )
    return pd.DataFrame(rows).sort_values("conditional_probability", ascending=False).reset_index(drop=True)


def bandit_mu0_sweep(
    q: float,
    mu0_values: Sequence[float],
    mu1: float,
    eta1: float,
    eta2: float,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for mu0 in mu0_values:
        bandit = TwoArmedBernoulliBandit(q=q, mu0=float(mu0), mu1=mu1, etas=(eta1, eta2))
        model = bandit.to_model()
        prefix = two_step_positive_prefix()
        rows.append(
            {
                "mu0": float(mu0),
                "interventional_influence": compute_interventional_influence(
                    model, prefix, 1, target_better_arm_probability
                ),
                "expected_replay_influence": compute_expected_replay_influence(
                    model, prefix, 1, target_better_arm_probability
                ),
                "local_influence": (
                    compute_local_next_state_effect(model, prefix, 1, 1e-6, target_better_arm_probability)
                    - compute_local_next_state_effect(model, prefix, 1, -1e-6, target_better_arm_probability)
                )
                / (2.0e-6),
            }
        )
    return pd.DataFrame(rows)


def bandit_gap_scaling_sweep(
    q: float,
    mu0: float,
    mu1: float,
    eta1: float,
    future_scales: Sequence[float],
    future_horizon: int = 4,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    prefix = ((1, 1),)
    for scale in future_scales:
        etas = (eta1,) + tuple(max(float(scale) * base, 1e-9) for base in np.linspace(0.04, 0.12, future_horizon - 1))
        bandit = TwoArmedBernoulliBandit(q=q, mu0=mu0, mu1=mu1, etas=etas)
        model = bandit.to_model()
        interventional = compute_interventional_influence(model, prefix, 1, target_better_arm_probability)
        replay = compute_expected_replay_influence(model, prefix, 1, target_better_arm_probability)
        rows.append(
            {
                "future_scale": float(scale),
                "interventional_influence": interventional,
                "expected_replay_influence": replay,
                "gap_abs": abs(interventional - replay),
                "replay_abs": abs(replay),
            }
        )
    return pd.DataFrame(rows)


def bandit_stagewise_gap_table(
    bandit: TwoArmedBernoulliBandit,
    prefix: History,
    time_index: int,
) -> pd.DataFrame:
    terms = stagewise_gap_terms(bandit, prefix, time_index, target_better_arm_probability)
    return pd.DataFrame(
        [{"future_round": round_index, "stagewise_gap_term": value} for round_index, value in terms.items()]
    )


def bandit_score_gap_summary(
    bandit: TwoArmedBernoulliBandit,
    prefix: History,
    time_index: int,
) -> pd.DataFrame:
    model = bandit.to_model()
    interventional = compute_interventional_influence(model, prefix, time_index, target_better_arm_probability)
    replay = compute_expected_replay_influence(model, prefix, time_index, target_better_arm_probability)
    return pd.DataFrame(
        [
            {
                "quantity": "interventional influence",
                "value": interventional,
            },
            {
                "quantity": "expected replay influence",
                "value": replay,
            },
            {
                "quantity": "gap from score representation",
                "value": score_representation_gap(bandit, prefix, time_index, target_better_arm_probability),
            },
            {
                "quantity": "direct gap",
                "value": interventional - replay,
            },
        ]
    )


def identification_curve(
    model: ActionOnlyFiniteBandit,
    prefix: History,
    time_index: int,
    epsilons: Sequence[float],
    target: TargetFn,
) -> pd.DataFrame:
    finite_model = model.to_model()
    rows: list[dict[str, float]] = []
    for epsilon in epsilons:
        epsilon = float(epsilon)
        rows.append(
            {
                "epsilon": epsilon,
                "direct_psi": compute_psi(finite_model, prefix, time_index, epsilon, target),
                "identified_psi": identified_psi_from_baseline(model, prefix, time_index, epsilon, target),
                "direct_effect": compute_interventional_effect(finite_model, prefix, time_index, epsilon, target),
                "identified_effect": identified_interventional_effect_from_baseline(
                    model, prefix, time_index, epsilon, target
                ),
            }
        )
    return pd.DataFrame(rows)


def notebook_comparison_summary(bandit: TwoArmedBernoulliBandit, epsilon: float = 0.05) -> pd.DataFrame:
    report = compare_local_replay_interventional(
        bandit,
        epsilon=epsilon,
        local_target=target_better_arm_probability,
        global_target=target_better_arm_probability,
    )
    return pd.DataFrame(
        [
            {"target": "local", "effect": report.local_effect},
            {"target": "expected replay", "effect": report.expected_replay_effect},
            {"target": "interventional", "effect": report.interventional_effect},
            {"target": "gap", "effect": report.replay_intervention_gap},
        ]
    )
