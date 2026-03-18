from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from statistics import median

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_attr.approx import (
    LookaheadSpec,
    PpoLiteConfig,
    alignment_metrics_from_rows,
    collect_cached_curriculum,
    compare_occurrence_scores,
    summarize_sweep_runs,
)
from rl_attr.plotting import configure_notebook_style, finalise_axes, save_figure_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi-seed approximation bridge sweep.")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "approx_bridge_sweep")
    parser.add_argument("--seeds", type=str, default="11,13,17,19,23,29,31,37")
    parser.add_argument("--steps-per-rollout", type=str, default="32,64")
    parser.add_argument("--rollout-indices", type=str, default="0,1")
    parser.add_argument("--horizons", type=str, default="1,2,3")
    parser.add_argument("--evaluation-episodes", type=str, default="8,16")
    parser.add_argument("--total-rollouts", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=32)
    return parser.parse_args()


def parse_int_list(raw_value: str) -> list[int]:
    values = [piece.strip() for piece in raw_value.split(",")]
    parsed = [int(piece) for piece in values if piece]
    if not parsed:
        raise ValueError("expected at least one integer value")
    return parsed


def main() -> int:
    import matplotlib.pyplot as plt
    import numpy as np

    args = parse_args()
    seeds = parse_int_list(args.seeds)
    step_sizes = parse_int_list(args.steps_per_rollout)
    rollout_indices = parse_int_list(args.rollout_indices)
    horizons = parse_int_list(args.horizons)
    evaluation_episode_counts = parse_int_list(args.evaluation_episodes)

    output_root = args.output_root
    curricula_dir = output_root / "curricula"
    results_dir = output_root / "results"
    figures_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)

    total_runs = _count_valid_runs(
        seeds=seeds,
        step_sizes=step_sizes,
        rollout_indices=rollout_indices,
        horizons=horizons,
        evaluation_episode_counts=evaluation_episode_counts,
        total_rollouts=args.total_rollouts,
    )

    run_rows: list[dict[str, float | int]] = []
    completed_runs = 0
    max_eval_episodes = max(evaluation_episode_counts)

    for steps_per_rollout in step_sizes:
        minibatch_size = max(16, steps_per_rollout // 2)
        for seed in seeds:
            manifest_dir = curricula_dir / f"seed_{seed}_steps_{steps_per_rollout}"
            config = PpoLiteConfig(
                total_rollouts=args.total_rollouts,
                steps_per_rollout=steps_per_rollout,
                minibatch_size=minibatch_size,
                hidden_size=args.hidden_size,
                evaluation_episodes=max_eval_episodes,
                seed=seed,
            )
            manifest = collect_cached_curriculum(manifest_dir, config)

            for rollout_index in rollout_indices:
                for horizon in horizons:
                    target_rollout_index = rollout_index + horizon - 1
                    if target_rollout_index >= args.total_rollouts:
                        continue
                    for evaluation_episodes in evaluation_episode_counts:
                        lookahead = LookaheadSpec(
                            rollout_index=rollout_index,
                            horizon=horizon,
                            target_rollout_index=target_rollout_index,
                            evaluation_episodes=evaluation_episodes,
                        )
                        start_time = time.perf_counter()
                        report = compare_occurrence_scores(manifest, lookahead)
                        aligned_metrics = alignment_metrics_from_rows(report["rows"])
                        elapsed_seconds = time.perf_counter() - start_time

                        run_row = {
                            "seed": seed,
                            "steps_per_rollout": steps_per_rollout,
                            "minibatch_size": minibatch_size,
                            "rollout_index": rollout_index,
                            "horizon": horizon,
                            "target_rollout_index": target_rollout_index,
                            "evaluation_episodes": evaluation_episodes,
                            "num_occurrences": int(report["num_occurrences"]),
                            "elapsed_seconds": elapsed_seconds,
                        }
                        for key, value in report["metrics"].items():
                            run_row[key] = float(value)
                        for key, value in aligned_metrics.items():
                            run_row[key] = float(value)
                        run_rows.append(run_row)

                        completed_runs += 1
                        print(
                            (
                                f"[{completed_runs}/{total_runs}] "
                                f"seed={seed} steps={steps_per_rollout} rollout={rollout_index} "
                                f"horizon={horizon} eval={evaluation_episodes} "
                                f"local_replay_help={run_row['local_vs_replay_helpfulness_spearman']:.3f} "
                                f"nonlocal_replay_help={run_row['nonlocal_vs_replay_helpfulness_spearman']:.3f} "
                                f"local_recollect_help={run_row['local_vs_recollection_helpfulness_spearman']:.3f} "
                                f"nonlocal_recollect_help={run_row['nonlocal_vs_recollection_helpfulness_spearman']:.3f} "
                                f"replay_recollect={run_row['replay_vs_recollection_helpfulness_spearman']:.3f} "
                                f"time={elapsed_seconds:.2f}s"
                            ),
                            flush=True,
                        )

    summary = summarize_sweep_runs(run_rows)

    _write_csv(results_dir / "run_metrics.csv", run_rows)
    _write_csv(results_dir / "summary_by_horizon.csv", summary["by_horizon"])
    _write_csv(results_dir / "summary_by_rollout_index.csv", summary["by_rollout_index"])
    _write_csv(results_dir / "summary_by_steps_per_rollout.csv", summary["by_steps_per_rollout"])
    _write_csv(results_dir / "summary_by_evaluation_episodes.csv", summary["by_evaluation_episodes"])
    _write_csv(results_dir / "summary_by_setting.csv", summary["by_setting"])
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    configure_notebook_style()
    _plot_proxy_medians_by_horizon(run_rows, figures_dir, target="replay", np_module=np)
    _plot_proxy_medians_by_horizon(run_rows, figures_dir, target="recollection", np_module=np)
    _plot_gap_boxplots(run_rows, figures_dir)
    _plot_overall_medians(summary, figures_dir)

    print()
    print("Sweep summary:")
    overall = summary["overall"]
    verdict_inputs = summary["verdict_inputs"]
    print(
        f"  median local vs replay helpfulness spearman: "
        f"{overall['median_local_vs_replay_helpfulness_spearman']:.6f}"
    )
    print(
        f"  median nonlocal vs replay helpfulness spearman: "
        f"{overall['median_nonlocal_vs_replay_helpfulness_spearman']:.6f}"
    )
    print(
        f"  median local vs recollection helpfulness spearman: "
        f"{overall['median_local_vs_recollection_helpfulness_spearman']:.6f}"
    )
    print(
        f"  median nonlocal vs recollection helpfulness spearman: "
        f"{overall['median_nonlocal_vs_recollection_helpfulness_spearman']:.6f}"
    )
    print(
        f"  median replay vs recollection helpfulness spearman: "
        f"{overall['median_replay_vs_recollection_helpfulness_spearman']:.6f}"
    )
    print(
        f"  nonlocal beats local on replay helpfulness win rate: "
        f"{verdict_inputs['nonlocal_beats_local_replay_helpfulness_win_rate']:.6f}"
    )
    print(
        f"  nonlocal beats local on recollection helpfulness win rate: "
        f"{verdict_inputs['nonlocal_beats_local_recollection_helpfulness_win_rate']:.6f}"
    )
    print(
        f"  horizon-1 local/nonlocal max abs diff: "
        f"{verdict_inputs['horizon1_local_nonlocal_max_abs_diff']:.12f}"
    )
    print(
        f"  best-proxy positive replay helpfulness rate: "
        f"{verdict_inputs['best_proxy_positive_replay_helpfulness_rate']:.6f}"
    )
    print(
        f"  best-proxy positive recollection helpfulness rate: "
        f"{verdict_inputs['best_proxy_positive_recollection_helpfulness_rate']:.6f}"
    )
    print(
        f"  replay-recollection gap rate (<0.9 helpfulness spearman): "
        f"{verdict_inputs['replay_recollection_gap_rate_below_point9']:.6f}"
    )
    return 0


def _count_valid_runs(
    *,
    seeds: list[int],
    step_sizes: list[int],
    rollout_indices: list[int],
    horizons: list[int],
    evaluation_episode_counts: list[int],
    total_rollouts: int,
) -> int:
    count = 0
    for _seed in seeds:
        for _steps in step_sizes:
            for rollout_index in rollout_indices:
                for horizon in horizons:
                    if rollout_index + horizon - 1 >= total_rollouts:
                        continue
                    for _evaluation_episodes in evaluation_episode_counts:
                        count += 1
    return count


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_proxy_medians_by_horizon(
    run_rows: list[dict[str, float | int]],
    figures_dir: Path,
    *,
    target: str,
    np_module,
) -> None:
    import matplotlib.pyplot as plt

    rollout_indices = sorted({int(row["rollout_index"]) for row in run_rows})
    horizons = sorted({int(row["horizon"]) for row in run_rows})
    figure_width = max(7.0, 5.4 * len(rollout_indices))
    fig, axes = plt.subplots(1, len(rollout_indices), figsize=(figure_width, 4.8), sharey=True)
    if len(rollout_indices) == 1:
        axes = [axes]

    for axis, rollout_index in zip(axes, rollout_indices):
        local_medians = []
        local_low = []
        local_high = []
        nonlocal_medians = []
        nonlocal_low = []
        nonlocal_high = []
        for horizon in horizons:
            group_rows = [
                row
                for row in run_rows
                if int(row["rollout_index"]) == rollout_index and int(row["horizon"]) == horizon
            ]
            local_values = [
                float(row[f"local_vs_{target}_helpfulness_spearman"])
                for row in group_rows
            ]
            nonlocal_values = [
                float(row[f"nonlocal_vs_{target}_helpfulness_spearman"])
                for row in group_rows
            ]
            local_medians.append(median(local_values))
            nonlocal_medians.append(median(nonlocal_values))
            local_low.append(float(np_module.percentile(local_values, 25)))
            local_high.append(float(np_module.percentile(local_values, 75)))
            nonlocal_low.append(float(np_module.percentile(nonlocal_values, 25)))
            nonlocal_high.append(float(np_module.percentile(nonlocal_values, 75)))

        axis.plot(horizons, local_medians, marker="o", linewidth=2.2, color="#1d4e89", label="Local snapshot")
        axis.fill_between(horizons, local_low, local_high, color="#1d4e89", alpha=0.16)
        axis.plot(horizons, nonlocal_medians, marker="s", linewidth=2.2, color="#d95f02", label="Non-local replay")
        axis.fill_between(horizons, nonlocal_low, nonlocal_high, color="#d95f02", alpha=0.16)
        axis.set_title(f"rollout index {rollout_index}")
        axis.set_xlabel("horizon")
        axis.set_xticks(horizons)
        finalise_axes(axis, yzero=True)
    axes[0].set_ylabel("helpfulness Spearman")
    axes[0].legend(loc="best")
    fig.suptitle(f"Median proxy alignment vs {target}", y=1.02)
    fig.tight_layout()
    save_figure_bundle(fig, figures_dir / f"proxy_alignment_vs_{target}_by_horizon")
    plt.close(fig)


def _plot_gap_boxplots(run_rows: list[dict[str, float | int]], figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    horizons = sorted({int(row["horizon"]) for row in run_rows})
    replay_vs_recollection = [
        [
            float(row["replay_vs_recollection_helpfulness_spearman"])
            for row in run_rows
            if int(row["horizon"]) == horizon
        ]
        for horizon in horizons
    ]
    best_proxy_vs_recollection = [
        [
            max(
                float(row["local_vs_recollection_helpfulness_spearman"]),
                float(row["nonlocal_vs_recollection_helpfulness_spearman"]),
            )
            for row in run_rows
            if int(row["horizon"]) == horizon
        ]
        for horizon in horizons
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    axes[0].boxplot(replay_vs_recollection, positions=horizons, widths=0.55, patch_artist=True, boxprops={"facecolor": "#84dcc6"})
    axes[0].set_title("Replay vs recollection")
    axes[0].set_xlabel("horizon")
    axes[0].set_ylabel("helpfulness Spearman")
    axes[0].set_xticks(horizons)
    finalise_axes(axes[0], yzero=True)

    axes[1].boxplot(best_proxy_vs_recollection, positions=horizons, widths=0.55, patch_artist=True, boxprops={"facecolor": "#f6bd60"})
    axes[1].set_title("Best proxy vs recollection")
    axes[1].set_xlabel("horizon")
    axes[1].set_xticks(horizons)
    finalise_axes(axes[1], yzero=True)
    fig.tight_layout()
    save_figure_bundle(fig, figures_dir / "replay_recollection_gap_boxplots")
    plt.close(fig)


def _plot_overall_medians(summary: dict, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    overall = summary["overall"]
    metric_labels = [
        "Local vs replay",
        "Non-local vs replay",
        "Local vs recollection",
        "Non-local vs recollection",
        "Replay vs recollection",
    ]
    metric_values = [
        float(overall["median_local_vs_replay_helpfulness_spearman"]),
        float(overall["median_nonlocal_vs_replay_helpfulness_spearman"]),
        float(overall["median_local_vs_recollection_helpfulness_spearman"]),
        float(overall["median_nonlocal_vs_recollection_helpfulness_spearman"]),
        float(overall["median_replay_vs_recollection_helpfulness_spearman"]),
    ]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.barh(
        metric_labels,
        metric_values,
        color=["#1d4e89", "#d95f02", "#1d4e89", "#d95f02", "#0b6e69"],
    )
    ax.set_title("Overall median helpfulness Spearman")
    ax.set_xlabel("median value")
    finalise_axes(ax, xzero=True)
    fig.tight_layout()
    save_figure_bundle(fig, figures_dir / "overall_helpfulness_medians")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
