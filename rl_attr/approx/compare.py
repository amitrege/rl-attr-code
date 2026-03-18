from __future__ import annotations

from typing import Any

from .common import (
    ApproxCurriculumManifest,
    LookaheadSpec,
    OccurrenceScoreRow,
    TrainOccurrenceRef,
    rows_to_table,
    sign_agreement,
    spearman_rank_correlation,
    top_k_overlap,
)
from .ppo_lite import (
    _config_from_manifest,
    _load_policy_from_checkpoint,
    _recollection_counterfactual_return,
    _replay_loo_counterfactual_utility,
    compute_buffer_surrogate_utility,
    evaluate_policy_return,
    load_buffer,
)
from .tracin import compute_local_snapshot_tracin, compute_nonlocal_replay_tracin


def compare_occurrence_scores(
    manifest: ApproxCurriculumManifest,
    lookahead: LookaheadSpec,
) -> dict[str, Any]:
    config = _config_from_manifest(manifest)
    local_scores = compute_local_snapshot_tracin(manifest, lookahead)
    nonlocal_scores = compute_nonlocal_replay_tracin(manifest, lookahead)
    buffer = load_buffer(manifest.rollout_buffer_path(lookahead.rollout_index))
    target_buffer = load_buffer(manifest.rollout_buffer_path(lookahead.target_rollout_index))
    baseline_replay_utility, baseline_recollection_return, evaluation_seeds = _baseline_targets(
        manifest=manifest,
        lookahead=lookahead,
        config=config,
        target_buffer=target_buffer,
    )

    rows: list[OccurrenceScoreRow] = []
    for row_index in range(len(buffer["actions"])):
        occurrence = TrainOccurrenceRef(rollout_index=lookahead.rollout_index, row_index=row_index)
        replay_counterfactual_utility = _replay_loo_counterfactual_utility(
            manifest=manifest,
            occurrence=occurrence,
            lookahead=lookahead,
            config=config,
            target_buffer=target_buffer,
        )
        replay_effect = replay_counterfactual_utility - baseline_replay_utility
        recollection_counterfactual_return = _recollection_counterfactual_return(
            manifest=manifest,
            occurrence=occurrence,
            lookahead=lookahead,
            config=config,
            evaluation_seeds=evaluation_seeds,
        )
        recollection_effect = recollection_counterfactual_return - baseline_recollection_return
        rows.append(
            OccurrenceScoreRow(
                occurrence=occurrence,
                local_snapshot_tracin=local_scores[occurrence],
                nonlocal_replay_tracin=nonlocal_scores[occurrence],
                exact_replay_loo=replay_effect,
                recollection_effect=recollection_effect,
            )
        )

    local_values = [row.local_snapshot_tracin for row in rows]
    nonlocal_values = [row.nonlocal_replay_tracin for row in rows]
    replay_values = [row.exact_replay_loo for row in rows]
    recollection_values = [row.recollection_effect for row in rows]
    k = max(1, len(rows) // 10)
    return {
        "lookahead": lookahead.to_dict(),
        "num_occurrences": len(rows),
        "metrics": {
            "local_vs_replay_spearman": spearman_rank_correlation(local_values, replay_values),
            "nonlocal_vs_replay_spearman": spearman_rank_correlation(nonlocal_values, replay_values),
            "local_vs_recollection_spearman": spearman_rank_correlation(local_values, recollection_values),
            "nonlocal_vs_recollection_spearman": spearman_rank_correlation(nonlocal_values, recollection_values),
            "local_vs_replay_sign_agreement": sign_agreement(local_values, replay_values),
            "nonlocal_vs_replay_sign_agreement": sign_agreement(nonlocal_values, replay_values),
            "local_vs_recollection_sign_agreement": sign_agreement(local_values, recollection_values),
            "nonlocal_vs_recollection_sign_agreement": sign_agreement(nonlocal_values, recollection_values),
            "local_vs_replay_topk_overlap": top_k_overlap(local_values, replay_values, k),
            "nonlocal_vs_replay_topk_overlap": top_k_overlap(nonlocal_values, replay_values, k),
            "local_vs_recollection_topk_overlap": top_k_overlap(local_values, recollection_values, k),
            "nonlocal_vs_recollection_topk_overlap": top_k_overlap(nonlocal_values, recollection_values, k),
        },
        "rows": rows_to_table(rows),
    }


def _baseline_targets(
    *,
    manifest: ApproxCurriculumManifest,
    lookahead: LookaheadSpec,
    config,
    target_buffer: dict[str, Any],
) -> tuple[float, float, tuple[int, ...]]:
    baseline_model = _load_policy_from_checkpoint(
        manifest.rollout_end_checkpoint_path(lookahead.target_rollout_index),
        config,
    )
    replay_score = compute_buffer_surrogate_utility(baseline_model, target_buffer)
    evaluation_seeds = tuple(manifest.evaluation_seeds[: lookahead.evaluation_episodes])
    recollection_score = evaluate_policy_return(
        baseline_model,
        config,
        evaluation_seeds,
    )
    return replay_score, recollection_score, evaluation_seeds
