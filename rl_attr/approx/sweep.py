from __future__ import annotations

from statistics import mean, median
from typing import Any

from .common import sign_agreement, spearman_rank_correlation, top_k_overlap


SELECTED_SWEEP_METRICS = (
    "local_vs_replay_helpfulness_spearman",
    "nonlocal_vs_replay_helpfulness_spearman",
    "local_vs_recollection_helpfulness_spearman",
    "nonlocal_vs_recollection_helpfulness_spearman",
    "local_vs_replay_helpfulness_sign_agreement",
    "nonlocal_vs_replay_helpfulness_sign_agreement",
    "local_vs_recollection_helpfulness_sign_agreement",
    "nonlocal_vs_recollection_helpfulness_sign_agreement",
    "local_vs_replay_helpfulness_topk_overlap",
    "nonlocal_vs_replay_helpfulness_topk_overlap",
    "local_vs_recollection_helpfulness_topk_overlap",
    "nonlocal_vs_recollection_helpfulness_topk_overlap",
    "replay_vs_recollection_helpfulness_spearman",
    "replay_vs_recollection_helpfulness_sign_agreement",
    "replay_vs_recollection_helpfulness_topk_overlap",
    "local_vs_nonlocal_spearman",
    "local_nonlocal_max_abs_diff",
)


def alignment_metrics_from_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("rows must be non-empty")

    local_values = [float(row["local_snapshot_tracin"]) for row in rows]
    nonlocal_values = [float(row["nonlocal_replay_tracin"]) for row in rows]
    replay_removal_values = [float(row["exact_replay_loo"]) for row in rows]
    recollection_removal_values = [float(row["recollection_effect"]) for row in rows]

    replay_helpfulness_values = [-value for value in replay_removal_values]
    recollection_helpfulness_values = [-value for value in recollection_removal_values]
    k = max(1, len(rows) // 10)

    return {
        "local_vs_replay_helpfulness_spearman": spearman_rank_correlation(local_values, replay_helpfulness_values),
        "nonlocal_vs_replay_helpfulness_spearman": spearman_rank_correlation(nonlocal_values, replay_helpfulness_values),
        "local_vs_recollection_helpfulness_spearman": spearman_rank_correlation(local_values, recollection_helpfulness_values),
        "nonlocal_vs_recollection_helpfulness_spearman": spearman_rank_correlation(nonlocal_values, recollection_helpfulness_values),
        "local_vs_replay_helpfulness_sign_agreement": sign_agreement(local_values, replay_helpfulness_values),
        "nonlocal_vs_replay_helpfulness_sign_agreement": sign_agreement(nonlocal_values, replay_helpfulness_values),
        "local_vs_recollection_helpfulness_sign_agreement": sign_agreement(local_values, recollection_helpfulness_values),
        "nonlocal_vs_recollection_helpfulness_sign_agreement": sign_agreement(nonlocal_values, recollection_helpfulness_values),
        "local_vs_replay_helpfulness_topk_overlap": top_k_overlap(local_values, replay_helpfulness_values, k),
        "nonlocal_vs_replay_helpfulness_topk_overlap": top_k_overlap(nonlocal_values, replay_helpfulness_values, k),
        "local_vs_recollection_helpfulness_topk_overlap": top_k_overlap(local_values, recollection_helpfulness_values, k),
        "nonlocal_vs_recollection_helpfulness_topk_overlap": top_k_overlap(nonlocal_values, recollection_helpfulness_values, k),
        "replay_vs_recollection_helpfulness_spearman": spearman_rank_correlation(
            replay_helpfulness_values,
            recollection_helpfulness_values,
        ),
        "replay_vs_recollection_helpfulness_sign_agreement": sign_agreement(
            replay_helpfulness_values,
            recollection_helpfulness_values,
        ),
        "replay_vs_recollection_helpfulness_topk_overlap": top_k_overlap(
            replay_helpfulness_values,
            recollection_helpfulness_values,
            k,
        ),
        "local_vs_nonlocal_spearman": spearman_rank_correlation(local_values, nonlocal_values),
        "local_nonlocal_max_abs_diff": max(
            abs(left - right) for left, right in zip(local_values, nonlocal_values)
        ),
    }


def summarize_sweep_runs(run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not run_rows:
        raise ValueError("run_rows must be non-empty")

    overall = _summarize_group(run_rows)
    horizon_identity_rows = [row for row in run_rows if int(row["horizon"]) == 1]
    nonlocal_replay_wins = _win_flags(
        run_rows,
        "local_vs_replay_helpfulness_spearman",
        "nonlocal_vs_replay_helpfulness_spearman",
    )
    nonlocal_recollection_wins = _win_flags(
        run_rows,
        "local_vs_recollection_helpfulness_spearman",
        "nonlocal_vs_recollection_helpfulness_spearman",
    )
    positive_proxy_replay_rows = _best_proxy_positive_flags(
        run_rows,
        "local_vs_replay_helpfulness_spearman",
        "nonlocal_vs_replay_helpfulness_spearman",
    )
    positive_proxy_recollection_rows = _best_proxy_positive_flags(
        run_rows,
        "local_vs_recollection_helpfulness_spearman",
        "nonlocal_vs_recollection_helpfulness_spearman",
    )
    replay_gap_rows = [
        1.0 if float(row["replay_vs_recollection_helpfulness_spearman"]) < 0.9 else 0.0
        for row in run_rows
    ]

    return {
        "overall": overall,
        "by_horizon": summarize_by_keys(run_rows, ["horizon"]),
        "by_rollout_index": summarize_by_keys(run_rows, ["rollout_index"]),
        "by_steps_per_rollout": summarize_by_keys(run_rows, ["steps_per_rollout"]),
        "by_evaluation_episodes": summarize_by_keys(run_rows, ["evaluation_episodes"]),
        "by_setting": summarize_by_keys(
            run_rows,
            ["rollout_index", "horizon", "steps_per_rollout", "evaluation_episodes"],
        ),
        "verdict_inputs": {
            "num_runs": len(run_rows),
            "horizon1_local_nonlocal_max_abs_diff": max(
                float(row["local_nonlocal_max_abs_diff"]) for row in horizon_identity_rows
            )
            if horizon_identity_rows
            else 0.0,
            "nonlocal_beats_local_replay_helpfulness_win_rate": mean(nonlocal_replay_wins),
            "nonlocal_beats_local_recollection_helpfulness_win_rate": mean(nonlocal_recollection_wins),
            "best_proxy_positive_replay_helpfulness_rate": mean(positive_proxy_replay_rows),
            "best_proxy_positive_recollection_helpfulness_rate": mean(positive_proxy_recollection_rows),
            "replay_recollection_gap_rate_below_point9": mean(replay_gap_rows),
        },
    }


def summarize_by_keys(run_rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in run_rows:
        group_key = tuple(row[key] for key in keys)
        groups.setdefault(group_key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for group_key in sorted(groups):
        payload = {key: value for key, value in zip(keys, group_key)}
        payload.update(_summarize_group(groups[group_key]))
        summaries.append(payload)
    return summaries


def _summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"num_runs": len(rows)}
    for metric_name in SELECTED_SWEEP_METRICS:
        values = [float(row[metric_name]) for row in rows]
        summary[f"median_{metric_name}"] = median(values)
        summary[f"mean_{metric_name}"] = mean(values)

    summary["nonlocal_beats_local_replay_helpfulness_win_rate"] = mean(
        _win_flags(
            rows,
            "local_vs_replay_helpfulness_spearman",
            "nonlocal_vs_replay_helpfulness_spearman",
        )
    )
    summary["nonlocal_beats_local_recollection_helpfulness_win_rate"] = mean(
        _win_flags(
            rows,
            "local_vs_recollection_helpfulness_spearman",
            "nonlocal_vs_recollection_helpfulness_spearman",
        )
    )
    summary["best_proxy_replay_helpfulness_median"] = median(
        max(
            float(row["local_vs_replay_helpfulness_spearman"]),
            float(row["nonlocal_vs_replay_helpfulness_spearman"]),
        )
        for row in rows
    )
    summary["best_proxy_recollection_helpfulness_median"] = median(
        max(
            float(row["local_vs_recollection_helpfulness_spearman"]),
            float(row["nonlocal_vs_recollection_helpfulness_spearman"]),
        )
        for row in rows
    )
    return summary


def _win_flags(rows: list[dict[str, Any]], left_key: str, right_key: str) -> list[float]:
    return [
        1.0 if float(row[right_key]) > float(row[left_key]) else 0.0
        for row in rows
    ]


def _best_proxy_positive_flags(
    rows: list[dict[str, Any]],
    left_key: str,
    right_key: str,
) -> list[float]:
    return [
        1.0 if max(float(row[left_key]), float(row[right_key])) > 0.0 else 0.0
        for row in rows
    ]
