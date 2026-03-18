from __future__ import annotations

from math import isclose
from typing import Any

from .action_only import (
    identified_interventional_effect_from_baseline,
    identified_interventional_influence_from_baseline,
    identified_psi_from_baseline,
)
from .bandits import TwoArmedBernoulliBandit, sigma, target_better_arm_probability
from .core import (
    History,
    compute_expected_replay_influence,
    compute_interventional_influence,
    compute_k_prefix_influence,
    compute_psi,
    compute_replay_influence_on_log,
    conditioning_ladder_table,
)
from .differentiable import (
    DifferentiableFiniteAdaptiveModel,
    build_recursion_bundle,
    compute_model_based_interventional_influence,
    compute_score_representation_gap,
    conditioned_stagewise_gap_terms,
)
from .examples import collect_prefixes, make_action_only_example, make_reference_bandit, prefix_to_str

STRICT_TOL = 1e-9
NUMERIC_TOL = 1e-6
WITNESS_PREFIX: History = ("z1*",)


def make_replay_oracle_witness(gamma: float) -> DifferentiableFiniteAdaptiveModel:
    def kernel(round_index: int, state: float, history: History) -> dict[Any, float]:
        if round_index == 1:
            return {"z1*": 1.0}
        del history
        probability_one = sigma(gamma * state)
        return {1: probability_one, 0: 1.0 - probability_one}

    def update(round_index: int, state: float, interaction: Any, weight: float) -> float:
        if round_index == 1:
            del state, interaction
            return weight - 1.0
        del state, weight
        return float(interaction)

    def update_jacobian(round_index: int, state: float, interaction: Any, weight: float) -> float:
        del round_index, state, interaction, weight
        return 0.0

    def update_weight_grad(round_index: int, state: float, interaction: Any, weight: float) -> float:
        del state, interaction, weight
        return 1.0 if round_index == 1 else 0.0

    def kernel_grad(round_index: int, state: float, history: History, interaction: Any) -> float:
        del history
        if round_index == 1:
            return 0.0
        slope = gamma * sigma(gamma * state) * (1.0 - sigma(gamma * state))
        return slope if interaction == 1 else -slope

    return DifferentiableFiniteAdaptiveModel(
        horizon=2,
        initial_state=0.0,
        kernel=kernel,
        update=update,
        update_jacobian=update_jacobian,
        update_weight_grad=update_weight_grad,
        kernel_grad=kernel_grad,
    )


def make_reward_dependence_witness(gamma: float) -> DifferentiableFiniteAdaptiveModel:
    return make_replay_oracle_witness(gamma)


def make_context_dependence_witness(gamma: float) -> DifferentiableFiniteAdaptiveModel:
    return make_replay_oracle_witness(gamma)


def replay_oracle_insufficiency_report(
    alpha: float = 1.0,
    beta: float = 3.0,
    epsilons: tuple[float, ...] = (-0.75, -0.4, -0.1, 0.0, 0.1, 0.4, 0.75),
    *,
    step: float = 1e-6,
    tol: float = STRICT_TOL,
) -> dict[str, Any]:
    model_alpha = make_replay_oracle_witness(alpha)
    model_beta = make_replay_oracle_witness(beta)
    target = float

    future_law_alpha = _continuation_law(model_alpha, WITNESS_PREFIX)
    future_law_beta = _continuation_law(model_beta, WITNESS_PREFIX)
    baseline_rows = [
        {
            "continuation": continuation[0],
            "alpha_probability": future_law_alpha[continuation],
            "beta_probability": future_law_beta[continuation],
            "abs_diff": abs(future_law_alpha[continuation] - future_law_beta[continuation]),
        }
        for continuation in sorted(future_law_alpha)
    ]
    response_rows: list[dict[str, float | int]] = []
    replay_curve_max_abs_diff = 0.0
    for continuation in (0, 1):
        history = WITNESS_PREFIX + (continuation,)
        for epsilon in epsilons:
            alpha_value = target(model_alpha.replay_terminal_state(history, (1.0 + float(epsilon), 1.0)))
            beta_value = target(model_beta.replay_terminal_state(history, (1.0 + float(epsilon), 1.0)))
            replay_curve_max_abs_diff = max(replay_curve_max_abs_diff, abs(alpha_value - beta_value))
            response_rows.append(
                {
                    "continuation": continuation,
                    "epsilon": float(epsilon),
                    "alpha_value": alpha_value,
                    "beta_value": beta_value,
                    "abs_diff": abs(alpha_value - beta_value),
                }
            )

    psi_rows: list[dict[str, float]] = []
    psi_curve_max_abs_diff = 0.0
    closed_form_error = 0.0
    for epsilon in epsilons:
        alpha_psi = compute_psi(model_alpha, WITNESS_PREFIX, 1, float(epsilon), target)
        beta_psi = compute_psi(model_beta, WITNESS_PREFIX, 1, float(epsilon), target)
        alpha_closed = sigma(alpha * float(epsilon))
        beta_closed = sigma(beta * float(epsilon))
        psi_curve_max_abs_diff = max(psi_curve_max_abs_diff, abs(alpha_psi - beta_psi))
        closed_form_error = max(
            closed_form_error,
            abs(alpha_psi - alpha_closed),
            abs(beta_psi - beta_closed),
        )
        psi_rows.append(
            {
                "epsilon": float(epsilon),
                "alpha_psi": alpha_psi,
                "beta_psi": beta_psi,
                "alpha_closed_form": alpha_closed,
                "beta_closed_form": beta_closed,
                "abs_diff": abs(alpha_psi - beta_psi),
            }
        )

    influence_alpha = compute_interventional_influence(model_alpha, WITNESS_PREFIX, 1, target, step=step)
    influence_beta = compute_interventional_influence(model_beta, WITNESS_PREFIX, 1, target, step=step)
    replay_alpha = compute_expected_replay_influence(model_alpha, WITNESS_PREFIX, 1, target, step=step)
    replay_beta = compute_expected_replay_influence(model_beta, WITNESS_PREFIX, 1, target, step=step)
    baseline_future_law_max_abs_diff = max(row["abs_diff"] for row in baseline_rows)
    replay_oracle_max_abs_diff = max(baseline_future_law_max_abs_diff, replay_curve_max_abs_diff)
    metrics = {
        "baseline_future_law_max_abs_diff": baseline_future_law_max_abs_diff,
        "replay_curve_max_abs_diff": replay_curve_max_abs_diff,
        "replay_oracle_max_abs_diff": replay_oracle_max_abs_diff,
        "psi_curve_max_abs_diff": psi_curve_max_abs_diff,
        "closed_form_max_abs_error": closed_form_error,
        "alpha_replay_influence": replay_alpha,
        "beta_replay_influence": replay_beta,
        "alpha_interventional_influence": influence_alpha,
        "beta_interventional_influence": influence_beta,
        "alpha_closed_form_influence": alpha / 4.0,
        "beta_closed_form_influence": beta / 4.0,
    }
    status = "PASS"
    if replay_oracle_max_abs_diff > tol or closed_form_error > tol:
        status = "FAIL"
    if psi_curve_max_abs_diff <= 1e-4:
        status = "FAIL"
    return {
        "name": "replay_oracle_insufficiency",
        "status": status,
        "parameters": {"alpha": alpha, "beta": beta, "epsilons": list(epsilons)},
        "metrics": metrics,
        "baseline_future_law": baseline_rows,
        "replay_response_curves": response_rows,
        "interventional_curve": psi_rows,
    }


def recursion_validation_report(
    bandit: TwoArmedBernoulliBandit | None = None,
    *,
    step: float = 1e-6,
    tol: float = 5e-5,
) -> dict[str, Any]:
    bandit = bandit or make_reference_bandit()
    model = bandit.to_model()
    differentiable_model = bandit.to_differentiable_model()
    rows: list[dict[str, float | str | int]] = []
    max_interventional_error = 0.0
    max_gap_error = 0.0
    max_score_error = 0.0
    for time_index in range(1, bandit.horizon):
        for prefix in collect_prefixes(model, time_index):
            recursion = compute_model_based_interventional_influence(
                differentiable_model,
                prefix,
                time_index,
                target_better_arm_probability,
                _unit_target_grad,
            )
            brute = compute_interventional_influence(
                model,
                prefix,
                time_index,
                target_better_arm_probability,
                step=step,
            )
            replay = compute_expected_replay_influence(
                model,
                prefix,
                time_index,
                target_better_arm_probability,
                step=step,
            )
            bundle = build_recursion_bundle(
                differentiable_model,
                prefix,
                time_index,
                target_better_arm_probability,
                _unit_target_grad,
            )
            stagewise_gap = sum(conditioned_stagewise_gap_terms(bundle).values())
            direct_gap = brute - replay
            score_gap = compute_score_representation_gap(
                differentiable_model,
                prefix,
                time_index,
                target_better_arm_probability,
                _unit_target_grad,
            )
            interventional_error = abs(recursion - brute)
            gap_error = abs(stagewise_gap - direct_gap)
            score_error = abs(score_gap - direct_gap)
            max_interventional_error = max(max_interventional_error, interventional_error)
            max_gap_error = max(max_gap_error, gap_error)
            max_score_error = max(max_score_error, score_error)
            rows.append(
                {
                    "time_index": time_index,
                    "prefix": prefix_to_str(prefix),
                    "recursion_interventional": recursion,
                    "brute_interventional": brute,
                    "expected_replay": replay,
                    "direct_gap": direct_gap,
                    "stagewise_gap_sum": stagewise_gap,
                    "score_gap": score_gap,
                    "interventional_abs_error": interventional_error,
                    "stagewise_abs_error": gap_error,
                    "score_abs_error": score_error,
                }
            )
    status = "PASS" if max(max_interventional_error, max_gap_error, max_score_error) <= tol else "FAIL"
    return {
        "name": "recursion_validation",
        "status": status,
        "bandit": {
            "q": bandit.q,
            "mu0": bandit.mu0,
            "mu1": bandit.mu1,
            "etas": list(bandit.etas),
        },
        "metrics": {
            "max_interventional_abs_error": max_interventional_error,
            "max_stagewise_abs_error": max_gap_error,
            "max_score_abs_error": max_score_error,
            "tolerance": tol,
        },
        "rows": rows,
    }


def identification_frontier_report(
    epsilons: tuple[float, ...] = (-0.4, -0.2, 0.0, 0.2, 0.4),
    gamma_pair: tuple[float, float] = (1.0, 3.0),
    *,
    step: float = 1e-6,
    tol: float = STRICT_TOL,
) -> dict[str, Any]:
    action_model = make_action_only_example()
    finite_model = action_model.to_model()
    action_prefix = ((1, 1, 1),)
    action_rows: list[dict[str, float]] = []
    max_psi_error = 0.0
    max_effect_error = 0.0
    for epsilon in epsilons:
        epsilon = float(epsilon)
        direct_psi = compute_psi(finite_model, action_prefix, 1, epsilon, float)
        identified_psi = identified_psi_from_baseline(action_model, action_prefix, 1, epsilon, float)
        direct_effect = direct_psi - compute_psi(finite_model, action_prefix, 1, 0.0, float)
        identified_effect = identified_interventional_effect_from_baseline(action_model, action_prefix, 1, epsilon, float)
        max_psi_error = max(max_psi_error, abs(direct_psi - identified_psi))
        max_effect_error = max(max_effect_error, abs(direct_effect - identified_effect))
        action_rows.append(
            {
                "epsilon": epsilon,
                "direct_psi": direct_psi,
                "identified_psi": identified_psi,
                "direct_effect": direct_effect,
                "identified_effect": identified_effect,
                "psi_abs_error": abs(direct_psi - identified_psi),
                "effect_abs_error": abs(direct_effect - identified_effect),
            }
        )
    direct_influence = compute_interventional_influence(finite_model, action_prefix, 1, float, step=step)
    identified_influence = identified_interventional_influence_from_baseline(action_model, action_prefix, 1, float, step=step)
    action_status = "PASS" if max(max_psi_error, max_effect_error, abs(direct_influence - identified_influence)) <= 1e-6 else "FAIL"

    reward_report = _baseline_law_obstruction_report(
        make_reward_dependence_witness,
        gamma_pair[0],
        gamma_pair[1],
        epsilons,
        step=step,
        tol=tol,
    )
    context_report = _baseline_law_obstruction_report(
        make_context_dependence_witness,
        gamma_pair[0],
        gamma_pair[1],
        epsilons,
        step=step,
        tol=tol,
    )
    overall_status = "PASS" if action_status == "PASS" and reward_report["status"] == "PASS" and context_report["status"] == "PASS" else "FAIL"
    return {
        "name": "identification_frontier",
        "status": overall_status,
        "action_only": {
            "status": action_status,
            "metrics": {
                "max_psi_abs_error": max_psi_error,
                "max_effect_abs_error": max_effect_error,
                "influence_abs_error": abs(direct_influence - identified_influence),
            },
            "rows": action_rows,
        },
        "reward_dependence": reward_report,
        "context_dependence": context_report,
    }


def conditioning_ladder_report(
    *,
    epsilon: float = 0.15,
    step: float = 1e-6,
    tol: float = 1e-6,
) -> dict[str, Any]:
    bandit = _make_conditioning_ladder_bandit()
    model = bandit.to_model()
    time_index, history = _select_conditioning_history(model)
    ladder_rows = conditioning_ladder_table(
        model,
        history,
        time_index,
        target_better_arm_probability,
        epsilon=epsilon,
        step=step,
    )

    below_t_errors: list[float] = []
    averaging_rows: list[dict[str, float | int | str]] = []
    for condition_index in range(time_index):
        prefix = history[:condition_index]
        averaged = _conditional_average_of_occurrence_level_influence(
            model,
            prefix,
            condition_index,
            time_index,
            step=step,
        )
        ladder_value = compute_k_prefix_influence(
            model,
            prefix,
            time_index,
            condition_index,
            target_better_arm_probability,
            step=step,
        )
        error = abs(ladder_value - averaged)
        below_t_errors.append(error)
        averaging_rows.append(
            {
                "condition_index": condition_index,
                "prefix": prefix_to_str(prefix),
                "ladder_influence": ladder_value,
                "averaged_occurrence_level": averaged,
                "abs_error": error,
            }
        )

    main_target = compute_interventional_influence(
        model,
        history[:time_index],
        time_index,
        target_better_arm_probability,
        step=step,
    )
    full_conditioned = compute_k_prefix_influence(
        model,
        history,
        time_index,
        model.horizon,
        target_better_arm_probability,
        step=step,
    )
    replay = compute_replay_influence_on_log(
        model,
        history,
        time_index,
        target_better_arm_probability,
        step=step,
    )
    influence_values = [row["influence"] for row in ladder_rows]
    min_pairwise_gap = min(
        abs(left - right)
        for index, left in enumerate(influence_values)
        for right in influence_values[index + 1 :]
    )
    status = "PASS"
    if max(below_t_errors or [0.0]) > tol:
        status = "FAIL"
    if abs(ladder_rows[time_index]["influence"] - main_target) > tol:
        status = "FAIL"
    if abs(full_conditioned - replay) > tol:
        status = "FAIL"
    if min_pairwise_gap <= 1e-4:
        status = "WARN"
    return {
        "name": "conditioning_ladder",
        "status": status,
        "bandit": {
            "q": bandit.q,
            "mu0": bandit.mu0,
            "mu1": bandit.mu1,
            "etas": list(bandit.etas),
        },
        "selected_history": prefix_to_str(history),
        "time_index": time_index,
        "metrics": {
            "max_below_t_abs_error": max(below_t_errors or [0.0]),
            "main_target_abs_error": abs(ladder_rows[time_index]["influence"] - main_target),
            "full_conditioning_replay_abs_error": abs(full_conditioned - replay),
            "min_pairwise_gap": min_pairwise_gap,
        },
        "ladder_rows": [
            {
                "condition_index": row["condition_index"],
                "prefix": prefix_to_str(row["prefix"]),
                "psi": row["psi"],
                "effect": row["effect"],
                "influence": row["influence"],
            }
            for row in ladder_rows
        ],
        "averaging_rows": averaging_rows,
    }


def paper_claim_check_report() -> dict[str, Any]:
    replay_oracle = replay_oracle_insufficiency_report()
    recursion = recursion_validation_report()
    identification = identification_frontier_report()
    conditioning = conditioning_ladder_report()
    statuses = [replay_oracle["status"], recursion["status"], identification["status"], conditioning["status"]]
    overall = "PASS" if all(status == "PASS" for status in statuses) else "WARN" if "WARN" in statuses and "FAIL" not in statuses else "FAIL"
    return {
        "status": overall,
        "replay_oracle_insufficiency": replay_oracle,
        "recursion_validation": recursion,
        "identification_frontier": identification,
        "conditioning_ladder": conditioning,
    }


def _baseline_law_obstruction_report(
    model_factory,
    alpha: float,
    beta: float,
    epsilons: tuple[float, ...],
    *,
    step: float,
    tol: float,
) -> dict[str, Any]:
    model_alpha = model_factory(alpha)
    model_beta = model_factory(beta)
    law_alpha = _full_history_law(model_alpha)
    law_beta = _full_history_law(model_beta)
    law_rows = [
        {
            "history": prefix_to_str(history),
            "alpha_probability": law_alpha[history],
            "beta_probability": law_beta[history],
            "abs_diff": abs(law_alpha[history] - law_beta[history]),
        }
        for history in sorted(law_alpha)
    ]
    psi_rows: list[dict[str, float]] = []
    max_psi_diff = 0.0
    for epsilon in epsilons:
        alpha_psi = compute_psi(model_alpha, WITNESS_PREFIX, 1, float(epsilon), float)
        beta_psi = compute_psi(model_beta, WITNESS_PREFIX, 1, float(epsilon), float)
        max_psi_diff = max(max_psi_diff, abs(alpha_psi - beta_psi))
        psi_rows.append(
            {
                "epsilon": float(epsilon),
                "alpha_psi": alpha_psi,
                "beta_psi": beta_psi,
                "abs_diff": abs(alpha_psi - beta_psi),
            }
        )
    influence_alpha = compute_interventional_influence(model_alpha, WITNESS_PREFIX, 1, float, step=step)
    influence_beta = compute_interventional_influence(model_beta, WITNESS_PREFIX, 1, float, step=step)
    max_law_diff = max(row["abs_diff"] for row in law_rows)
    status = "PASS" if max_law_diff <= tol and max_psi_diff > 1e-4 and not isclose(influence_alpha, influence_beta, abs_tol=1e-4) else "FAIL"
    return {
        "status": status,
        "metrics": {
            "baseline_law_max_abs_diff": max_law_diff,
            "psi_curve_max_abs_diff": max_psi_diff,
            "alpha_interventional_influence": influence_alpha,
            "beta_interventional_influence": influence_beta,
        },
        "baseline_law_rows": law_rows,
        "psi_rows": psi_rows,
    }


def _make_conditioning_ladder_bandit() -> TwoArmedBernoulliBandit:
    return TwoArmedBernoulliBandit(
        q=0.3,
        mu0=0.75,
        mu1=0.95,
        etas=(0.5, 1.1, 0.7),
    )


def _select_conditioning_history(model) -> tuple[int, History]:
    best: tuple[int, History] | None = None
    best_score = (-1, -1.0)
    for time_index in range(1, model.horizon):
        for outcome in model.enumerate_histories():
            history = outcome.history
            influences = [
                compute_k_prefix_influence(
                    model,
                    history[:condition_index],
                    time_index,
                    condition_index,
                    target_better_arm_probability,
                )
                for condition_index in range(model.horizon + 1)
            ]
            distinct_count = _distinct_count(influences)
            spread = max(influences) - min(influences)
            score = (distinct_count, spread)
            if score > best_score:
                best = (time_index, history)
                best_score = score
    if best is None:
        raise RuntimeError("failed to select a conditioning-ladder history")
    return best


def _distinct_count(values: list[float], tol: float = 1e-4) -> int:
    representatives: list[float] = []
    for value in values:
        if not any(abs(value - representative) <= tol for representative in representatives):
            representatives.append(value)
    return len(representatives)


def _conditional_average_of_occurrence_level_influence(
    model,
    prefix: History,
    condition_index: int,
    time_index: int,
    *,
    step: float,
) -> float:
    outcomes = [outcome for outcome in model.enumerate_histories() if outcome.history[:condition_index] == prefix]
    total_probability = sum(outcome.probability for outcome in outcomes)
    if total_probability == 0.0:
        raise ValueError("conditioning prefix has zero probability")
    t_prefixes = sorted({outcome.history[:time_index] for outcome in outcomes})
    total = 0.0
    for t_prefix in t_prefixes:
        probability = sum(outcome.probability for outcome in outcomes if outcome.history[:time_index] == t_prefix) / total_probability
        total += probability * compute_interventional_influence(
            model,
            t_prefix,
            time_index,
            target_better_arm_probability,
            step=step,
        )
    return total


def _continuation_law(model, prefix: History) -> dict[History, float]:
    outcomes = [outcome for outcome in model.enumerate_histories() if outcome.history[: len(prefix)] == prefix]
    total_probability = sum(outcome.probability for outcome in outcomes)
    law: dict[History, float] = {}
    for outcome in outcomes:
        continuation = outcome.history[len(prefix) :]
        law[continuation] = law.get(continuation, 0.0) + outcome.probability / total_probability
    return law


def _full_history_law(model) -> dict[History, float]:
    return {outcome.history: outcome.probability for outcome in model.enumerate_histories()}


def _unit_target_grad(state: float) -> float:
    del state
    return 1.0
