from __future__ import annotations

import importlib.util
from math import isclose
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from rl_attr.approx.common import (
    ApproxCurriculumManifest,
    LookaheadSpec,
    TrainOccurrenceRef,
    sign_agreement,
    spearman_rank_correlation,
    top_k_overlap,
)
from rl_attr.approx.sweep import alignment_metrics_from_rows, summarize_sweep_runs


class ApproxBridgeCommonTest(unittest.TestCase):
    def test_manifest_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            manifest = ApproxCurriculumManifest(
                root_dir=tempdir,
                env_name="CartPole-v1",
                total_rollouts=2,
                steps_per_rollout=32,
                minibatch_size=16,
                rollout_seeds=(17, 1017),
                evaluation_seeds=(100, 101),
                initial_checkpoint_path=str(Path(tempdir) / "policy_initial.pt"),
                rollout_buffer_paths=(
                    str(Path(tempdir) / "buffer_0.pkl"),
                    str(Path(tempdir) / "buffer_1.pkl"),
                ),
                rollout_end_checkpoint_paths=(
                    str(Path(tempdir) / "policy_after_rollout_0.pt"),
                    str(Path(tempdir) / "policy_after_rollout_1.pt"),
                ),
                update_checkpoint_paths=(
                    str(Path(tempdir) / "policy_grad_1.pt"),
                    str(Path(tempdir) / "policy_grad_2.pt"),
                ),
                trainer_config={"env_name": "CartPole-v1", "observation_dim": 4, "action_dim": 2},
            )
            manifest.save()
            loaded = ApproxCurriculumManifest.load(manifest.manifest_path)
            self.assertEqual(manifest, loaded)

    def test_rank_and_sign_metrics(self) -> None:
        left = [3.0, 2.0, -1.0, 0.5]
        right = [30.0, 20.0, -5.0, 4.0]
        self.assertTrue(isclose(spearman_rank_correlation(left, right), 1.0, abs_tol=1e-12))
        self.assertTrue(isclose(sign_agreement(left, right), 1.0, abs_tol=1e-12))
        self.assertTrue(isclose(top_k_overlap(left, right, 2), 1.0, abs_tol=1e-12))

    def test_occurrence_and_lookahead_serialization(self) -> None:
        occurrence = TrainOccurrenceRef(rollout_index=1, row_index=7)
        lookahead = LookaheadSpec(rollout_index=1, horizon=2, target_rollout_index=2, evaluation_episodes=8)
        self.assertEqual(occurrence, TrainOccurrenceRef.from_dict(occurrence.to_dict()))
        self.assertEqual(lookahead, LookaheadSpec.from_dict(lookahead.to_dict()))

    def test_alignment_metrics_flip_removal_effects_to_helpfulness(self) -> None:
        rows = [
            {
                "rollout_index": 0,
                "row_index": 0,
                "local_snapshot_tracin": 3.0,
                "nonlocal_replay_tracin": 1.0,
                "exact_replay_loo": -30.0,
                "recollection_effect": -6.0,
            },
            {
                "rollout_index": 0,
                "row_index": 1,
                "local_snapshot_tracin": 2.0,
                "nonlocal_replay_tracin": 2.0,
                "exact_replay_loo": -20.0,
                "recollection_effect": -4.0,
            },
            {
                "rollout_index": 0,
                "row_index": 2,
                "local_snapshot_tracin": -1.0,
                "nonlocal_replay_tracin": 3.0,
                "exact_replay_loo": 10.0,
                "recollection_effect": 2.0,
            },
        ]
        metrics = alignment_metrics_from_rows(rows)
        self.assertTrue(
            isclose(metrics["local_vs_replay_helpfulness_spearman"], 1.0, abs_tol=1e-12)
        )
        self.assertTrue(
            isclose(metrics["local_vs_recollection_helpfulness_spearman"], 1.0, abs_tol=1e-12)
        )
        self.assertTrue(
            isclose(metrics["replay_vs_recollection_helpfulness_spearman"], 1.0, abs_tol=1e-12)
        )

    def test_summarize_sweep_runs_reports_horizon1_identity(self) -> None:
        run_rows = [
            {
                "seed": 11,
                "rollout_index": 0,
                "horizon": 1,
                "steps_per_rollout": 32,
                "evaluation_episodes": 8,
                "local_vs_replay_helpfulness_spearman": 0.5,
                "nonlocal_vs_replay_helpfulness_spearman": 0.5,
                "local_vs_recollection_helpfulness_spearman": 0.2,
                "nonlocal_vs_recollection_helpfulness_spearman": 0.2,
                "local_vs_replay_helpfulness_sign_agreement": 0.7,
                "nonlocal_vs_replay_helpfulness_sign_agreement": 0.7,
                "local_vs_recollection_helpfulness_sign_agreement": 0.6,
                "nonlocal_vs_recollection_helpfulness_sign_agreement": 0.6,
                "local_vs_replay_helpfulness_topk_overlap": 0.5,
                "nonlocal_vs_replay_helpfulness_topk_overlap": 0.5,
                "local_vs_recollection_helpfulness_topk_overlap": 0.5,
                "nonlocal_vs_recollection_helpfulness_topk_overlap": 0.5,
                "replay_vs_recollection_helpfulness_spearman": 0.4,
                "replay_vs_recollection_helpfulness_sign_agreement": 0.5,
                "replay_vs_recollection_helpfulness_topk_overlap": 0.5,
                "local_vs_nonlocal_spearman": 1.0,
                "local_nonlocal_max_abs_diff": 0.0,
            },
            {
                "seed": 13,
                "rollout_index": 0,
                "horizon": 2,
                "steps_per_rollout": 32,
                "evaluation_episodes": 8,
                "local_vs_replay_helpfulness_spearman": 0.2,
                "nonlocal_vs_replay_helpfulness_spearman": 0.4,
                "local_vs_recollection_helpfulness_spearman": -0.1,
                "nonlocal_vs_recollection_helpfulness_spearman": 0.3,
                "local_vs_replay_helpfulness_sign_agreement": 0.6,
                "nonlocal_vs_replay_helpfulness_sign_agreement": 0.7,
                "local_vs_recollection_helpfulness_sign_agreement": 0.4,
                "nonlocal_vs_recollection_helpfulness_sign_agreement": 0.6,
                "local_vs_replay_helpfulness_topk_overlap": 0.2,
                "nonlocal_vs_replay_helpfulness_topk_overlap": 0.4,
                "local_vs_recollection_helpfulness_topk_overlap": 0.1,
                "nonlocal_vs_recollection_helpfulness_topk_overlap": 0.3,
                "replay_vs_recollection_helpfulness_spearman": 0.3,
                "replay_vs_recollection_helpfulness_sign_agreement": 0.4,
                "replay_vs_recollection_helpfulness_topk_overlap": 0.2,
                "local_vs_nonlocal_spearman": 0.6,
                "local_nonlocal_max_abs_diff": 0.4,
            },
        ]
        summary = summarize_sweep_runs(run_rows)
        self.assertTrue(
            isclose(summary["verdict_inputs"]["horizon1_local_nonlocal_max_abs_diff"], 0.0, abs_tol=1e-12)
        )
        self.assertTrue(
            isclose(summary["overall"]["nonlocal_beats_local_replay_helpfulness_win_rate"], 0.5, abs_tol=1e-12)
        )

    def test_tracin_uses_rollout_start_for_source_and_rollout_end_for_target(self) -> None:
        import rl_attr.approx.tracin as tracin

        manifest = ApproxCurriculumManifest(
            root_dir="/tmp/approx",
            env_name="CartPole-v1",
            total_rollouts=3,
            steps_per_rollout=4,
            minibatch_size=2,
            rollout_seeds=(17, 1017, 2017),
            evaluation_seeds=(100, 101),
            initial_checkpoint_path="/tmp/approx/policy_initial.pt",
            rollout_buffer_paths=(
                "/tmp/approx/buffer_0.pkl",
                "/tmp/approx/buffer_1.pkl",
                "/tmp/approx/buffer_2.pkl",
            ),
            rollout_end_checkpoint_paths=(
                "/tmp/approx/policy_after_rollout_0.pt",
                "/tmp/approx/policy_after_rollout_1.pt",
                "/tmp/approx/policy_after_rollout_2.pt",
            ),
            update_checkpoint_paths=(),
            trainer_config={
                "env_name": "CartPole-v1",
                "total_rollouts": 3,
                "steps_per_rollout": 4,
                "minibatch_size": 2,
                "hidden_size": 8,
                "learning_rate": 0.5,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "value_coef": 0.5,
                "entropy_coef": 0.01,
                "max_grad_norm": 0.5,
                "seed": 17,
                "evaluation_episodes": 2,
                "observation_dim": 4,
                "action_dim": 2,
            },
        )
        lookahead = LookaheadSpec(rollout_index=1, horizon=2, target_rollout_index=2, evaluation_episodes=2)
        fake_buffer = {"actions": [0, 1]}

        class _FakeDot:
            def __init__(self, value: float) -> None:
                self._value = value

            def item(self) -> float:
                return self._value

        class _FakeTorch:
            @staticmethod
            def dot(left, right):
                return _FakeDot(sum(x * y for x, y in zip(left, right)))

        with (
            mock.patch.object(tracin, "_require_approx_dependencies", return_value=None),
            mock.patch.object(tracin, "_config_from_manifest") as config_from_manifest,
            mock.patch.object(tracin, "_load_policy_from_checkpoint", side_effect=["source", "target"]) as loader,
            mock.patch.object(tracin, "load_buffer", side_effect=[fake_buffer, fake_buffer]),
            mock.patch.object(tracin, "_utility_gradient", return_value=[2.0, 1.0]),
            mock.patch.object(tracin, "_row_training_gradient", return_value=[3.0, 4.0]),
            mock.patch.object(tracin, "torch", _FakeTorch),
        ):
            config_from_manifest.return_value = type("Config", (), {"learning_rate": 0.5})()
            scores = tracin.compute_nonlocal_replay_tracin(manifest, lookahead)

        self.assertEqual(len(scores), 2)
        self.assertEqual(
            loader.call_args_list,
            [
                mock.call(manifest.rollout_start_checkpoint_path(lookahead.rollout_index), config_from_manifest.return_value),
                mock.call(manifest.rollout_end_checkpoint_path(lookahead.target_rollout_index), config_from_manifest.return_value),
            ],
        )


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None and importlib.util.find_spec("gymnasium") is not None,
    "approximation bridge integration test requires optional torch and gymnasium dependencies",
)
class ApproxBridgeIntegrationTest(unittest.TestCase):
    def test_collect_and_compare_smoke(self) -> None:
        from rl_attr.approx import LookaheadSpec, PpoLiteConfig, collect_cached_curriculum, compare_occurrence_scores

        with tempfile.TemporaryDirectory() as tempdir:
            manifest = collect_cached_curriculum(
                Path(tempdir) / "curriculum",
                PpoLiteConfig(
                    total_rollouts=2,
                    steps_per_rollout=32,
                    minibatch_size=16,
                    evaluation_episodes=4,
                    hidden_size=32,
                ),
            )
            report = compare_occurrence_scores(
                manifest,
                LookaheadSpec(rollout_index=0, horizon=2, target_rollout_index=1, evaluation_episodes=4),
            )
            self.assertEqual(report["num_occurrences"], 32)
            self.assertIn("local_vs_replay_spearman", report["metrics"])
            self.assertEqual(len(report["rows"]), 32)


if __name__ == "__main__":
    unittest.main()
