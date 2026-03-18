from __future__ import annotations

import unittest
from math import isclose

from rl_attr.theorem_checks import (
    conditioning_ladder_report,
    identification_frontier_report,
    paper_claim_check_report,
    recursion_validation_report,
    replay_oracle_insufficiency_report,
)


class TheoremClaimChecksTest(unittest.TestCase):
    def test_replay_oracle_insufficiency_witness_passes(self) -> None:
        report = replay_oracle_insufficiency_report(alpha=1.0, beta=3.0)
        self.assertEqual(report["status"], "PASS")
        self.assertLessEqual(report["metrics"]["replay_oracle_max_abs_diff"], 1e-9)
        self.assertGreater(report["metrics"]["psi_curve_max_abs_diff"], 1e-4)
        self.assertTrue(isclose(report["metrics"]["alpha_interventional_influence"], 0.25, rel_tol=1e-5, abs_tol=1e-5))
        self.assertTrue(isclose(report["metrics"]["beta_interventional_influence"], 0.75, rel_tol=1e-5, abs_tol=1e-5))

    def test_recursion_validation_passes(self) -> None:
        report = recursion_validation_report()
        self.assertEqual(report["status"], "PASS")
        self.assertLessEqual(report["metrics"]["max_interventional_abs_error"], report["metrics"]["tolerance"])
        self.assertLessEqual(report["metrics"]["max_stagewise_abs_error"], report["metrics"]["tolerance"])
        self.assertLessEqual(report["metrics"]["max_score_abs_error"], report["metrics"]["tolerance"])

    def test_identification_frontier_passes(self) -> None:
        report = identification_frontier_report()
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["action_only"]["status"], "PASS")
        self.assertEqual(report["reward_dependence"]["status"], "PASS")
        self.assertEqual(report["context_dependence"]["status"], "PASS")
        self.assertLessEqual(report["action_only"]["metrics"]["max_psi_abs_error"], 1e-9)
        self.assertLessEqual(report["action_only"]["metrics"]["max_effect_abs_error"], 1e-9)

    def test_conditioning_ladder_passes(self) -> None:
        report = conditioning_ladder_report()
        self.assertEqual(report["status"], "PASS")
        self.assertLessEqual(report["metrics"]["max_below_t_abs_error"], 1e-6)
        self.assertLessEqual(report["metrics"]["main_target_abs_error"], 1e-6)
        self.assertLessEqual(report["metrics"]["full_conditioning_replay_abs_error"], 1e-6)
        self.assertGreater(report["metrics"]["min_pairwise_gap"], 1e-4)

    def test_overall_claim_report_passes(self) -> None:
        report = paper_claim_check_report()
        self.assertEqual(report["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
