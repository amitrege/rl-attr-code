from __future__ import annotations

import unittest
from math import isclose

from rl_attr.action_only import (
    ActionOnlyFiniteBandit,
    direct_vs_identified_effect,
    direct_vs_identified_psi,
    identified_interventional_influence_from_baseline,
)
from rl_attr.bandits import (
    TwoArmedBernoulliBandit,
    closed_form_expected_replay_influence,
    closed_form_interventional_influence,
    compare_local_replay_interventional,
    find_strong_separation_example,
    score_representation_gap,
    sigma,
    stagewise_gap_terms,
    target_better_arm_probability,
    two_step_positive_prefix,
)
from rl_attr.core import (
    FiniteAdaptiveModel,
    compute_expected_replay_effect,
    compute_expected_replay_influence,
    compute_interventional_effect,
    compute_interventional_influence,
    compute_replay_effect_on_log,
    one_coordinate_weights,
)


class FoundationsTest(unittest.TestCase):
    def test_prefix_invariance_under_round_one_perturbation(self) -> None:
        bandit = TwoArmedBernoulliBandit(q=0.4, mu0=0.2, mu1=0.9, etas=(0.3, 0.2))
        model = bandit.to_model()
        baseline = model.enumerate_histories()
        perturbed = model.enumerate_histories(one_coordinate_weights(model.horizon, 1, -0.75))

        def prefix_masses(outcomes: list, prefix_length: int) -> dict:
            masses: dict = {}
            for outcome in outcomes:
                prefix = outcome.history[:prefix_length]
                masses[prefix] = masses.get(prefix, 0.0) + outcome.probability
            return masses

        baseline_masses = prefix_masses(baseline, 1)
        perturbed_masses = prefix_masses(perturbed, 1)
        self.assertEqual(set(baseline_masses), set(perturbed_masses))
        for key, value in baseline_masses.items():
            self.assertTrue(isclose(value, perturbed_masses[key], abs_tol=1e-12))

    def test_full_conditioning_collapses_to_replay(self) -> None:
        bandit = TwoArmedBernoulliBandit(q=0.35, mu0=0.1, mu1=0.8, etas=(0.2, 0.15))
        model = bandit.to_model()
        history = ((1, 1), (0, 1))
        epsilon = -0.4
        interventional = compute_interventional_effect(
            model,
            history,
            2,
            epsilon,
            target_better_arm_probability,
        )
        replay = compute_replay_effect_on_log(
            model,
            history,
            2,
            epsilon,
            target_better_arm_probability,
        )
        self.assertTrue(isclose(interventional, replay, abs_tol=1e-12))

    def test_exogenous_future_has_zero_gap(self) -> None:
        def kernel(round_index: int, state: int, history: tuple[int, ...]) -> dict[int, float]:
            del round_index, state, history
            return {0: 0.5, 1: 0.5}

        def update(round_index: int, state: int, interaction: int, weight: float) -> int:
            del round_index
            return state + int(weight * interaction)

        model = FiniteAdaptiveModel(horizon=3, initial_state=0, kernel=kernel, update=update)
        prefix = (1,)
        epsilon = -0.5
        interventional = compute_interventional_effect(model, prefix, 1, epsilon, float)
        replay = compute_expected_replay_effect(model, prefix, 1, epsilon, float)
        self.assertTrue(isclose(interventional, replay, abs_tol=1e-12))

    def test_closed_form_two_step_formulas_match_numeric(self) -> None:
        bandit = TwoArmedBernoulliBandit(q=0.25, mu0=0.6, mu1=1.0, etas=(0.25, 1.3))
        model = bandit.to_model()
        prefix = two_step_positive_prefix()
        numeric_intervention = compute_interventional_influence(
            model,
            prefix,
            1,
            target_better_arm_probability,
            step=1e-7,
        )
        numeric_replay = compute_expected_replay_influence(
            model,
            prefix,
            1,
            target_better_arm_probability,
            step=1e-7,
        )
        self.assertTrue(
            isclose(
                numeric_intervention,
                closed_form_interventional_influence(bandit),
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
        )
        self.assertTrue(
            isclose(
                numeric_replay,
                closed_form_expected_replay_influence(bandit),
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
        )

    def test_strong_separation_example_has_opposite_signs(self) -> None:
        bandit = find_strong_separation_example()
        replay = closed_form_expected_replay_influence(bandit)
        intervention = closed_form_interventional_influence(bandit)
        self.assertLess(replay, 0.0)
        self.assertGreater(intervention, 0.0)

    def test_score_representation_matches_gap(self) -> None:
        bandit = TwoArmedBernoulliBandit(q=0.4, mu0=0.2, mu1=0.85, etas=(0.2, 0.15, 0.1))
        model = bandit.to_model()
        prefix = ((1, 1),)
        gap = compute_interventional_influence(
            model,
            prefix,
            1,
            target_better_arm_probability,
            step=1e-6,
        ) - compute_expected_replay_influence(
            model,
            prefix,
            1,
            target_better_arm_probability,
            step=1e-6,
        )
        represented = score_representation_gap(
            bandit,
            prefix,
            1,
            target_better_arm_probability,
            step=1e-6,
        )
        self.assertTrue(isclose(gap, represented, rel_tol=1e-4, abs_tol=1e-4))

    def test_stagewise_decomposition_matches_gap(self) -> None:
        bandit = TwoArmedBernoulliBandit(q=0.4, mu0=0.2, mu1=0.85, etas=(0.2, 0.15, 0.1))
        model = bandit.to_model()
        prefix = ((1, 1),)
        gap = compute_interventional_influence(
            model,
            prefix,
            1,
            target_better_arm_probability,
            step=1e-6,
        ) - compute_expected_replay_influence(
            model,
            prefix,
            1,
            target_better_arm_probability,
            step=1e-6,
        )
        terms = stagewise_gap_terms(
            bandit,
            prefix,
            1,
            target_better_arm_probability,
            step=1e-6,
        )
        self.assertTrue(isclose(gap, sum(terms.values()), rel_tol=1e-4, abs_tol=1e-4))

    def test_compare_local_replay_interventional_separates_targets(self) -> None:
        bandit = TwoArmedBernoulliBandit(q=0.25, mu0=0.98, mu1=1.0, etas=(0.25, 2.0))
        report = compare_local_replay_interventional(bandit, epsilon=1e-4)
        self.assertFalse(isclose(report.local_effect, report.expected_replay_effect, abs_tol=1e-10))
        self.assertFalse(isclose(report.expected_replay_effect, report.interventional_effect, abs_tol=1e-10))

    def test_action_only_identified_psi_and_effect_match_direct(self) -> None:
        model = _make_action_only_example()
        prefix = ((1, 1, 1),)
        psi_direct, psi_identified = direct_vs_identified_psi(
            model,
            prefix,
            1,
            epsilon=0.2,
            target=lambda theta: theta,
        )
        effect_direct, effect_identified = direct_vs_identified_effect(
            model,
            prefix,
            1,
            epsilon=0.2,
            target=lambda theta: theta,
        )
        self.assertTrue(isclose(psi_direct, psi_identified, rel_tol=1e-9, abs_tol=1e-9))
        self.assertTrue(isclose(effect_direct, effect_identified, rel_tol=1e-9, abs_tol=1e-9))

    def test_action_only_identified_influence_matches_direct(self) -> None:
        model = _make_action_only_example()
        prefix = ((1, 1, 1),)
        direct = compute_interventional_influence(
            model.to_model(),
            prefix,
            1,
            target=lambda theta: theta,
            step=1e-6,
        )
        identified = identified_interventional_influence_from_baseline(
            model,
            prefix,
            1,
            target=lambda theta: theta,
            step=1e-6,
        )
        self.assertTrue(isclose(direct, identified, rel_tol=1e-5, abs_tol=1e-5))


def _make_action_only_example() -> ActionOnlyFiniteBandit:
    def context_dist(round_index: int, history: tuple) -> dict[int, float]:
        del history
        if round_index == 1:
            return {0: 0.45, 1: 0.55}
        if round_index == 2:
            return {0: 0.5, 1: 0.5}
        return {0: 0.3, 1: 0.7}

    def reward_dist(round_index: int, history: tuple, context: int, action: int) -> dict[int, float]:
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


if __name__ == "__main__":
    unittest.main()
