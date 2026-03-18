from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class TrainOccurrenceRef:
    rollout_index: int
    row_index: int

    def to_dict(self) -> dict[str, int]:
        return {"rollout_index": self.rollout_index, "row_index": self.row_index}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainOccurrenceRef:
        return cls(
            rollout_index=int(payload["rollout_index"]),
            row_index=int(payload["row_index"]),
        )


@dataclass(frozen=True)
class LookaheadSpec:
    rollout_index: int
    horizon: int
    target_rollout_index: int
    evaluation_episodes: int = 16

    def to_dict(self) -> dict[str, int]:
        return {
            "rollout_index": self.rollout_index,
            "horizon": self.horizon,
            "target_rollout_index": self.target_rollout_index,
            "evaluation_episodes": self.evaluation_episodes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LookaheadSpec:
        return cls(
            rollout_index=int(payload["rollout_index"]),
            horizon=int(payload["horizon"]),
            target_rollout_index=int(payload["target_rollout_index"]),
            evaluation_episodes=int(payload.get("evaluation_episodes", 16)),
        )


@dataclass(frozen=True)
class ReplayLooResult:
    occurrence: TrainOccurrenceRef
    lookahead: LookaheadSpec
    baseline_utility: float
    counterfactual_utility: float
    effect_of_removal: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "occurrence": self.occurrence.to_dict(),
            "lookahead": self.lookahead.to_dict(),
            "baseline_utility": self.baseline_utility,
            "counterfactual_utility": self.counterfactual_utility,
            "effect_of_removal": self.effect_of_removal,
        }


@dataclass(frozen=True)
class OccurrenceScoreRow:
    occurrence: TrainOccurrenceRef
    local_snapshot_tracin: float
    nonlocal_replay_tracin: float
    exact_replay_loo: float
    recollection_effect: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "occurrence": self.occurrence.to_dict(),
            "local_snapshot_tracin": self.local_snapshot_tracin,
            "nonlocal_replay_tracin": self.nonlocal_replay_tracin,
            "exact_replay_loo": self.exact_replay_loo,
            "recollection_effect": self.recollection_effect,
        }


@dataclass(frozen=True)
class ApproxCurriculumManifest:
    root_dir: str
    env_name: str
    total_rollouts: int
    steps_per_rollout: int
    minibatch_size: int
    rollout_seeds: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    initial_checkpoint_path: str
    rollout_buffer_paths: tuple[str, ...]
    rollout_end_checkpoint_paths: tuple[str, ...]
    update_checkpoint_paths: tuple[str, ...]
    trainer_config: dict[str, Any]

    @property
    def root_path(self) -> Path:
        return Path(self.root_dir)

    @property
    def manifest_path(self) -> Path:
        return self.root_path / "manifest.json"

    def rollout_buffer_path(self, rollout_index: int) -> Path:
        return Path(self.rollout_buffer_paths[rollout_index])

    def rollout_end_checkpoint_path(self, rollout_index: int) -> Path:
        return Path(self.rollout_end_checkpoint_paths[rollout_index])

    def rollout_start_checkpoint_path(self, rollout_index: int) -> Path:
        if rollout_index == 0:
            return Path(self.initial_checkpoint_path)
        return self.rollout_end_checkpoint_path(rollout_index - 1)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rollout_seeds"] = list(self.rollout_seeds)
        payload["evaluation_seeds"] = list(self.evaluation_seeds)
        payload["rollout_buffer_paths"] = list(self.rollout_buffer_paths)
        payload["rollout_end_checkpoint_paths"] = list(self.rollout_end_checkpoint_paths)
        payload["update_checkpoint_paths"] = list(self.update_checkpoint_paths)
        return payload

    def save(self) -> None:
        self.manifest_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ApproxCurriculumManifest:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            root_dir=payload["root_dir"],
            env_name=payload["env_name"],
            total_rollouts=int(payload["total_rollouts"]),
            steps_per_rollout=int(payload["steps_per_rollout"]),
            minibatch_size=int(payload["minibatch_size"]),
            rollout_seeds=tuple(int(value) for value in payload["rollout_seeds"]),
            evaluation_seeds=tuple(int(value) for value in payload["evaluation_seeds"]),
            initial_checkpoint_path=payload["initial_checkpoint_path"],
            rollout_buffer_paths=tuple(payload["rollout_buffer_paths"]),
            rollout_end_checkpoint_paths=tuple(payload["rollout_end_checkpoint_paths"]),
            update_checkpoint_paths=tuple(payload["update_checkpoint_paths"]),
            trainer_config=dict(payload["trainer_config"]),
        )


def rows_to_table(rows: list[OccurrenceScoreRow]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for row in rows:
        table.append(
            {
                "rollout_index": row.occurrence.rollout_index,
                "row_index": row.occurrence.row_index,
                "local_snapshot_tracin": row.local_snapshot_tracin,
                "nonlocal_replay_tracin": row.nonlocal_replay_tracin,
                "exact_replay_loo": row.exact_replay_loo,
                "recollection_effect": row.recollection_effect,
            }
        )
    return table


def spearman_rank_correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("rank-correlation inputs must have the same length")
    if len(left) < 2:
        return 0.0
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    return pearson_correlation(left_ranks, right_ranks)


def pearson_correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("correlation inputs must have the same length")
    if len(left) < 2:
        return 0.0
    mean_left = mean(left)
    mean_right = mean(right)
    cov = sum((x - mean_left) * (y - mean_right) for x, y in zip(left, right))
    var_left = sum((x - mean_left) ** 2 for x in left)
    var_right = sum((y - mean_right) ** 2 for y in right)
    if var_left == 0.0 or var_right == 0.0:
        return 0.0
    return cov / (var_left**0.5 * var_right**0.5)


def sign_agreement(left: list[float], right: list[float], *, tol: float = 1e-12) -> float:
    if len(left) != len(right):
        raise ValueError("sign-agreement inputs must have the same length")
    if not left:
        return 0.0
    matches = 0
    for x, y in zip(left, right):
        if _sign(x, tol=tol) == _sign(y, tol=tol):
            matches += 1
    return matches / len(left)


def top_k_overlap(left: list[float], right: list[float], k: int) -> float:
    if len(left) != len(right):
        raise ValueError("top-k inputs must have the same length")
    if k <= 0:
        raise ValueError("k must be positive")
    if not left:
        return 0.0
    k = min(k, len(left))
    left_top = set(sorted(range(len(left)), key=lambda index: left[index], reverse=True)[:k])
    right_top = set(sorted(range(len(right)), key=lambda index: right[index], reverse=True)[:k])
    return len(left_top & right_top) / k


def _average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        for index in range(start, end):
            ranks[indexed[index][0]] = rank
        start = end
    return ranks


def _sign(value: float, *, tol: float) -> int:
    if abs(value) <= tol:
        return 0
    return 1 if value > 0 else -1
