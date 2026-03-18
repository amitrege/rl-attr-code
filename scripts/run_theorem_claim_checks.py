from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_attr.theorem_checks import paper_claim_check_report


def main() -> int:
    report = paper_claim_check_report()
    print(f"Overall status: {report['status']}")
    print()
    _print_replay_oracle(report["replay_oracle_insufficiency"])
    _print_recursion(report["recursion_validation"])
    _print_identification(report["identification_frontier"])
    _print_conditioning(report["conditioning_ladder"])
    return 0 if report["status"] == "PASS" else 1


def _print_replay_oracle(report) -> None:
    print("== Replay-Oracle Insufficiency ==")
    print(f"Status: {report['status']}")
    for key, value in report["metrics"].items():
        print(f"{key}: {value:.12g}")
    print("Baseline future law:")
    for row in report["baseline_future_law"]:
        print(
            f"  continuation={row['continuation']} alpha={row['alpha_probability']:.12g} "
            f"beta={row['beta_probability']:.12g} abs_diff={row['abs_diff']:.12g}"
        )
    print("Interventional curve:")
    for row in report["interventional_curve"]:
        print(
            f"  epsilon={row['epsilon']:+.3f} alpha_psi={row['alpha_psi']:.12g} "
            f"beta_psi={row['beta_psi']:.12g} abs_diff={row['abs_diff']:.12g}"
        )
    print()


def _print_recursion(report) -> None:
    print("== Recursion Validation ==")
    print(f"Status: {report['status']}")
    for key, value in report["metrics"].items():
        if isinstance(value, float):
            print(f"{key}: {value:.12g}")
        else:
            print(f"{key}: {value}")
    print("Per-prefix rows:")
    for row in report["rows"]:
        print(
            f"  t={row['time_index']} prefix={row['prefix']} recursion={row['recursion_interventional']:.12g} "
            f"brute={row['brute_interventional']:.12g} replay={row['expected_replay']:.12g} "
            f"gap={row['direct_gap']:.12g} stagewise_err={row['stagewise_abs_error']:.12g} "
            f"score_err={row['score_abs_error']:.12g}"
        )
    print()


def _print_identification(report) -> None:
    print("== Identification Frontier ==")
    print(f"Status: {report['status']}")
    print(f"Action-only status: {report['action_only']['status']}")
    for key, value in report["action_only"]["metrics"].items():
        print(f"{key}: {value:.12g}")
    print("Reward dependence:")
    print(f"  status: {report['reward_dependence']['status']}")
    for key, value in report["reward_dependence"]["metrics"].items():
        print(f"  {key}: {value:.12g}")
    print("Context dependence:")
    print(f"  status: {report['context_dependence']['status']}")
    for key, value in report["context_dependence"]["metrics"].items():
        print(f"  {key}: {value:.12g}")
    print()


def _print_conditioning(report) -> None:
    print("== Conditioning Ladder ==")
    print(f"Status: {report['status']}")
    print(f"time_index: {report['time_index']}")
    print(f"selected_history: {report['selected_history']}")
    for key, value in report["metrics"].items():
        print(f"{key}: {value:.12g}")
    print("Ladder rows:")
    for row in report["ladder_rows"]:
        print(
            f"  k={row['condition_index']} prefix={row['prefix']} "
            f"psi={row['psi']:.12g} effect={row['effect']:.12g} influence={row['influence']:.12g}"
        )
    print("Averaging checks:")
    for row in report["averaging_rows"]:
        print(
            f"  k={row['condition_index']} prefix={row['prefix']} ladder={row['ladder_influence']:.12g} "
            f"avg_occurrence={row['averaged_occurrence_level']:.12g} abs_error={row['abs_error']:.12g}"
        )
    print()


if __name__ == "__main__":
    raise SystemExit(main())
