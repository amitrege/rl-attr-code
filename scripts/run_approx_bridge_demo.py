from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_attr.approx import (
    LookaheadSpec,
    PpoLiteConfig,
    collect_cached_curriculum,
    compare_occurrence_scores,
)
from rl_attr.plotting import configure_notebook_style, finalise_axes, save_figure_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PPO-lite approximation bridge demo.")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "approx_bridge_demo")
    parser.add_argument("--rollout-index", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--total-rollouts", type=int, default=4)
    parser.add_argument("--steps-per-rollout", type=int, default=128)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--evaluation-episodes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> int:
    import matplotlib.pyplot as plt
    import pandas as pd

    args = parse_args()
    config = PpoLiteConfig(
        total_rollouts=args.total_rollouts,
        steps_per_rollout=args.steps_per_rollout,
        minibatch_size=args.minibatch_size,
        evaluation_episodes=args.evaluation_episodes,
        seed=args.seed,
    )
    curriculum_dir = args.output_root / "curriculum"
    results_dir = args.output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    manifest = collect_cached_curriculum(curriculum_dir, config)
    target_rollout_index = args.rollout_index + args.horizon - 1
    lookahead = LookaheadSpec(
        rollout_index=args.rollout_index,
        horizon=args.horizon,
        target_rollout_index=target_rollout_index,
        evaluation_episodes=args.evaluation_episodes,
    )
    report = compare_occurrence_scores(manifest, lookahead)

    report_path = results_dir / "comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = results_dir / "occurrence_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["rows"][0].keys()))
        writer.writeheader()
        writer.writerows(report["rows"])
    metrics_path = results_dir / "metric_summary.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in report["metrics"].items():
            writer.writerow({"metric": key, "value": value})

    configure_notebook_style()
    rows_df = pd.DataFrame(report["rows"])
    metrics_df = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in report["metrics"].items()]
    )
    figures_dir = results_dir / "figures"
    _plot_scatter_grid(rows_df, figures_dir)
    _plot_sorted_profiles(rows_df, figures_dir)
    _plot_metric_summary(metrics_df, figures_dir)

    print(f"Wrote {report_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {metrics_path}")
    print("Metrics:")
    for key, value in report["metrics"].items():
        print(f"  {key}: {value:.6f}")
    return 0


def _plot_scatter_grid(rows_df, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    panels = [
        (
            "local_snapshot_tracin",
            "exact_replay_loo",
            "Local snapshot vs replay-LOO",
            "#1d4e89",
        ),
        (
            "nonlocal_replay_tracin",
            "exact_replay_loo",
            "Non-local replay vs replay-LOO",
            "#d95f02",
        ),
        (
            "local_snapshot_tracin",
            "recollection_effect",
            "Local snapshot vs recollection",
            "#0b6e69",
        ),
        (
            "nonlocal_replay_tracin",
            "recollection_effect",
            "Non-local replay vs recollection",
            "#b23a48",
        ),
    ]
    for axis, (x_col, y_col, title, color) in zip(axes.flat, panels):
        axis.scatter(rows_df[x_col], rows_df[y_col], s=32, alpha=0.85, color=color, edgecolor="white", linewidth=0.5)
        axis.set_title(title)
        axis.set_xlabel(x_col.replace("_", " "))
        axis.set_ylabel(y_col.replace("_", " "))
        finalise_axes(axis, xzero=True, yzero=True)
    fig.tight_layout()
    save_figure_bundle(fig, figures_dir / "score_scatter_grid")
    plt.close(fig)


def _plot_sorted_profiles(rows_df, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=False)
    replay_sorted = rows_df.sort_values("exact_replay_loo").reset_index(drop=True)
    recollection_sorted = rows_df.sort_values("recollection_effect").reset_index(drop=True)

    for axis, sorted_df, title, anchor in [
        (axes[0], replay_sorted, "Scores ordered by replay-LOO", "exact_replay_loo"),
        (axes[1], recollection_sorted, "Scores ordered by recollection", "recollection_effect"),
    ]:
        x_values = list(range(len(sorted_df)))
        axis.plot(x_values, sorted_df["local_snapshot_tracin"], linewidth=2.0, label="Local snapshot", color="#1d4e89")
        axis.plot(x_values, sorted_df["nonlocal_replay_tracin"], linewidth=2.0, label="Non-local replay", color="#d95f02")
        axis.plot(x_values, sorted_df[anchor], linewidth=2.4, label=anchor.replace("_", " "), color="#0b6e69")
        axis.set_title(title)
        axis.set_xlabel("occurrence rank")
        axis.set_ylabel("score")
        finalise_axes(axis, yzero=True)
    axes[0].legend(loc="best")
    fig.tight_layout()
    save_figure_bundle(fig, figures_dir / "sorted_score_profiles")
    plt.close(fig)


def _plot_metric_summary(metrics_df, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.barh(metrics_df["metric"], metrics_df["value"], color=["#1d4e89" if "local" in name else "#d95f02" for name in metrics_df["metric"]])
    ax.set_title("Approximation-bridge comparison metrics")
    ax.set_xlabel("value")
    finalise_axes(ax, xzero=True)
    fig.tight_layout()
    save_figure_bundle(fig, figures_dir / "metric_summary")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
