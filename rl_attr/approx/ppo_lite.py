from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import pickle
from typing import Any

from .common import ApproxCurriculumManifest, LookaheadSpec, ReplayLooResult, TrainOccurrenceRef

try:
    import gymnasium as gym
    import numpy as np
    import torch
    from torch import nn
    from torch.distributions import Categorical
except ModuleNotFoundError:  # pragma: no cover - exercised via guarded imports
    gym = None
    np = None
    torch = None
    nn = None
    Categorical = None


@dataclass(frozen=True)
class PpoLiteConfig:
    env_name: str = "CartPole-v1"
    total_rollouts: int = 4
    steps_per_rollout: int = 128
    minibatch_size: int = 32
    hidden_size: int = 64
    learning_rate: float = 3e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    seed: int = 17
    evaluation_episodes: int = 16
    observation_dim: int = 0
    action_dim: int = 0


class ApproxDependencyError(RuntimeError):
    pass


if nn is not None:
    class PolicyValueNet(nn.Module):  # type: ignore[misc]
        def __init__(self, observation_dim: int, action_dim: int, hidden_size: int) -> None:
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(observation_dim, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
            )
            self.policy_head = nn.Linear(hidden_size, action_dim)
            self.value_head = nn.Linear(hidden_size, 1)

        def forward(self, observations):
            hidden = self.backbone(observations)
            return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)
else:  # pragma: no cover - exercised only when optional deps are absent
    class PolicyValueNet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            _require_approx_dependencies()


def collect_cached_curriculum(
    output_dir: str | Path,
    config: PpoLiteConfig | None = None,
) -> ApproxCurriculumManifest:
    _require_approx_dependencies()
    config = config or PpoLiteConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    env = gym.make(config.env_name)
    observation_shape = env.observation_space.shape
    if observation_shape is None or len(observation_shape) != 1:
        raise ValueError("PPO-lite bridge currently expects 1D vector observations")
    if not hasattr(env.action_space, "n"):
        raise ValueError("PPO-lite bridge currently expects a discrete action space")
    observation_dim = int(observation_shape[0])
    action_dim = int(env.action_space.n)
    env.close()
    config = PpoLiteConfig(**{**asdict(config), "observation_dim": observation_dim, "action_dim": action_dim})

    model = PolicyValueNet(observation_dim, action_dim, config.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    rollout_seeds = tuple(config.seed + 1000 * index for index in range(config.total_rollouts))
    evaluation_seeds = tuple(config.seed + 100_000 + index for index in range(config.evaluation_episodes))
    initial_checkpoint_path = output_path / "policy_initial.pt"
    _save_checkpoint(model, initial_checkpoint_path)

    buffer_paths: list[str] = []
    rollout_end_paths: list[str] = []
    update_paths: list[str] = []
    global_update_index = 0
    for rollout_index, rollout_seed in enumerate(rollout_seeds):
        buffer = _collect_rollout(config, model, rollout_seed)
        buffer["rollout_index"] = rollout_index
        buffer_path = output_path / f"buffer_{rollout_index}.pkl"
        with buffer_path.open("wb") as handle:
            pickle.dump(buffer, handle)
        buffer_paths.append(str(buffer_path))

        global_update_index, new_update_paths = _train_on_buffer(
            model=model,
            optimizer=optimizer,
            buffer=buffer,
            config=config,
            global_update_index=global_update_index,
            save_dir=output_path,
        )
        update_paths.extend(str(path) for path in new_update_paths)
        rollout_end_path = output_path / f"policy_after_rollout_{rollout_index}.pt"
        _save_checkpoint(model, rollout_end_path)
        rollout_end_paths.append(str(rollout_end_path))

    manifest = ApproxCurriculumManifest(
        root_dir=str(output_path),
        env_name=config.env_name,
        total_rollouts=config.total_rollouts,
        steps_per_rollout=config.steps_per_rollout,
        minibatch_size=config.minibatch_size,
        rollout_seeds=rollout_seeds,
        evaluation_seeds=evaluation_seeds,
        initial_checkpoint_path=str(initial_checkpoint_path),
        rollout_buffer_paths=tuple(buffer_paths),
        rollout_end_checkpoint_paths=tuple(rollout_end_paths),
        update_checkpoint_paths=tuple(update_paths),
        trainer_config={
            **asdict(config),
            "observation_dim": observation_dim,
            "action_dim": action_dim,
        },
    )
    manifest.save()
    return manifest


def compute_exact_replay_loo(
    manifest: ApproxCurriculumManifest,
    occurrence: TrainOccurrenceRef,
    lookahead: LookaheadSpec,
) -> ReplayLooResult:
    config = _config_from_manifest(manifest)
    baseline_model = _load_policy_from_checkpoint(manifest.rollout_end_checkpoint_path(lookahead.target_rollout_index), config)
    target_buffer = load_buffer(manifest.rollout_buffer_path(lookahead.target_rollout_index))
    baseline_utility = compute_buffer_surrogate_utility(baseline_model, target_buffer)

    counterfactual_utility = _replay_loo_counterfactual_utility(
        manifest=manifest,
        occurrence=occurrence,
        lookahead=lookahead,
        config=config,
        target_buffer=target_buffer,
    )
    return ReplayLooResult(
        occurrence=occurrence,
        lookahead=lookahead,
        baseline_utility=baseline_utility,
        counterfactual_utility=counterfactual_utility,
        effect_of_removal=counterfactual_utility - baseline_utility,
    )


def compute_recollected_occurrence_effect(
    manifest: ApproxCurriculumManifest,
    occurrence: TrainOccurrenceRef,
    lookahead: LookaheadSpec,
) -> float:
    config = _config_from_manifest(manifest)
    baseline_model = _load_policy_from_checkpoint(manifest.rollout_end_checkpoint_path(lookahead.target_rollout_index), config)
    baseline_return = evaluate_policy_return(
        baseline_model,
        config,
        manifest.evaluation_seeds[: lookahead.evaluation_episodes],
    )

    counterfactual_return = _recollection_counterfactual_return(
        manifest=manifest,
        occurrence=occurrence,
        lookahead=lookahead,
        config=config,
        evaluation_seeds=manifest.evaluation_seeds[: lookahead.evaluation_episodes],
    )
    return counterfactual_return - baseline_return


def compute_buffer_surrogate_utility(model: PolicyValueNet, buffer: dict[str, Any]) -> float:
    observations = _tensor(buffer["observations"])
    actions = _tensor(buffer["actions"], dtype=torch.long)
    advantages = _tensor(buffer["advantages"])
    with torch.no_grad():
        logits, _ = model(observations)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
    return float(torch.mean(advantages * log_prob).item())


def evaluate_policy_return(
    model: PolicyValueNet,
    config: PpoLiteConfig,
    evaluation_seeds: tuple[int, ...] | list[int],
) -> float:
    env = gym.make(config.env_name)
    episode_returns: list[float] = []
    for seed in evaluation_seeds:
        observation, _ = env.reset(seed=int(seed))
        terminated = False
        truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            observation_tensor = _tensor(observation[None, :])
            with torch.no_grad():
                logits, _ = model(observation_tensor)
                action = int(torch.argmax(logits, dim=-1).item())
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
        episode_returns.append(total_reward)
    env.close()
    return float(sum(episode_returns) / len(episode_returns))


def load_buffer(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def list_occurrences(manifest: ApproxCurriculumManifest, rollout_index: int) -> list[TrainOccurrenceRef]:
    buffer = load_buffer(manifest.rollout_buffer_path(rollout_index))
    return [TrainOccurrenceRef(rollout_index=rollout_index, row_index=index) for index in range(len(buffer["actions"]))]


def _replay_loo_counterfactual_utility(
    manifest: ApproxCurriculumManifest,
    occurrence: TrainOccurrenceRef,
    lookahead: LookaheadSpec,
    config: PpoLiteConfig,
    target_buffer: dict[str, Any] | None = None,
) -> float:
    model = _load_policy_from_checkpoint(manifest.rollout_start_checkpoint_path(lookahead.rollout_index), config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    for rollout_index in range(lookahead.rollout_index, lookahead.target_rollout_index + 1):
        buffer = load_buffer(manifest.rollout_buffer_path(rollout_index))
        removed_row_index = occurrence.row_index if rollout_index == occurrence.rollout_index else None
        _train_on_buffer(
            model=model,
            optimizer=optimizer,
            buffer=buffer,
            config=config,
            removed_row_index=removed_row_index,
        )
    target_buffer = target_buffer or load_buffer(manifest.rollout_buffer_path(lookahead.target_rollout_index))
    return compute_buffer_surrogate_utility(model, target_buffer)


def _recollection_counterfactual_return(
    manifest: ApproxCurriculumManifest,
    occurrence: TrainOccurrenceRef,
    lookahead: LookaheadSpec,
    config: PpoLiteConfig,
    evaluation_seeds: tuple[int, ...] | list[int],
) -> float:
    model = _load_policy_from_checkpoint(manifest.rollout_start_checkpoint_path(lookahead.rollout_index), config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    first_buffer = load_buffer(manifest.rollout_buffer_path(lookahead.rollout_index))
    _train_on_buffer(
        model=model,
        optimizer=optimizer,
        buffer=first_buffer,
        config=config,
        removed_row_index=occurrence.row_index,
    )

    for rollout_index in range(lookahead.rollout_index + 1, lookahead.target_rollout_index + 1):
        buffer = _collect_rollout(config, model, manifest.rollout_seeds[rollout_index])
        _train_on_buffer(model=model, optimizer=optimizer, buffer=buffer, config=config)

    return evaluate_policy_return(model, config, evaluation_seeds)


def _collect_rollout(
    config: PpoLiteConfig,
    model: PolicyValueNet,
    rollout_seed: int,
) -> dict[str, Any]:
    env = gym.make(config.env_name)
    torch.manual_seed(int(rollout_seed))
    observation, _ = env.reset(seed=int(rollout_seed))

    observations: list[Any] = []
    actions: list[int] = []
    rewards: list[float] = []
    dones: list[bool] = []
    log_probs: list[float] = []
    values: list[float] = []

    for _ in range(config.steps_per_rollout):
        observation_tensor = _tensor(observation[None, :])
        with torch.no_grad():
            logits, value = model(observation_tensor)
            distribution = Categorical(logits=logits)
            action_tensor = distribution.sample()
            log_prob = distribution.log_prob(action_tensor)
        action = int(action_tensor.item())
        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)

        observations.append(np.array(observation, dtype=np.float32))
        actions.append(action)
        rewards.append(float(reward))
        dones.append(done)
        log_probs.append(float(log_prob.item()))
        values.append(float(value.item()))

        if done:
            next_observation, _ = env.reset()
        observation = next_observation

    with torch.no_grad():
        if dones[-1]:
            last_value = 0.0
        else:
            _, bootstrap_value = model(_tensor(observation[None, :]))
            last_value = float(bootstrap_value.item())
    env.close()

    advantages, returns = _gae_returns(
        rewards=rewards,
        values=values,
        dones=dones,
        last_value=last_value,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    advantages = np.asarray(advantages, dtype=np.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.bool_),
        "old_log_probs": np.asarray(log_probs, dtype=np.float32),
        "values": np.asarray(values, dtype=np.float32),
        "advantages": advantages,
        "returns": np.asarray(returns, dtype=np.float32),
    }


def _train_on_buffer(
    model: PolicyValueNet,
    optimizer,
    buffer: dict[str, Any],
    config: PpoLiteConfig,
    *,
    global_update_index: int = 0,
    removed_row_index: int | None = None,
    save_dir: str | Path | None = None,
) -> tuple[int, list[Path]]:
    saved_paths: list[Path] = []
    batch_indices = np.arange(len(buffer["actions"]))
    for start in range(0, len(batch_indices), config.minibatch_size):
        indices = batch_indices[start : start + config.minibatch_size]
        if removed_row_index is not None:
            indices = indices[indices != removed_row_index]
            if len(indices) == 0:
                continue
        optimizer.zero_grad()
        loss = _ppo_loss(model, buffer, indices, config)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        global_update_index += 1
        if save_dir is not None:
            checkpoint_path = Path(save_dir) / f"policy_grad_{global_update_index}.pt"
            _save_checkpoint(model, checkpoint_path)
            saved_paths.append(checkpoint_path)
    return global_update_index, saved_paths


def _ppo_loss(
    model: PolicyValueNet,
    buffer: dict[str, Any],
    indices: Any,
    config: PpoLiteConfig,
):
    observations = _tensor(buffer["observations"][indices])
    actions = _tensor(buffer["actions"][indices], dtype=torch.long)
    old_log_probs = _tensor(buffer["old_log_probs"][indices])
    advantages = _tensor(buffer["advantages"][indices])
    returns = _tensor(buffer["returns"][indices])

    logits, values = model(observations)
    distribution = Categorical(logits=logits)
    new_log_probs = distribution.log_prob(actions)
    ratios = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratios * advantages
    clipped = torch.clamp(ratios, 1.0 - config.clip_range, 1.0 + config.clip_range) * advantages
    policy_loss = -torch.mean(torch.minimum(unclipped, clipped))
    value_loss = torch.mean((values - returns) ** 2)
    entropy_bonus = torch.mean(distribution.entropy())
    return policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_bonus


def _gae_returns(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    advantages = [0.0] * len(rewards)
    returns = [0.0] * len(rewards)
    gae = 0.0
    next_value = last_value
    for index in reversed(range(len(rewards))):
        nonterminal = 0.0 if dones[index] else 1.0
        delta = rewards[index] + gamma * next_value * nonterminal - values[index]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[index] = gae
        returns[index] = gae + values[index]
        next_value = values[index]
    return advantages, returns


def _save_checkpoint(model: PolicyValueNet, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, path)


def _load_policy_from_checkpoint(path: str | Path, config: PpoLiteConfig) -> PolicyValueNet:
    model = PolicyValueNet(
        observation_dim=config.observation_dim,
        action_dim=config.action_dim,
        hidden_size=config.hidden_size,
    )
    payload = torch.load(Path(path), map_location="cpu")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _config_from_manifest(manifest: ApproxCurriculumManifest) -> PpoLiteConfig:
    payload = dict(manifest.trainer_config)
    config = PpoLiteConfig(
        env_name=payload["env_name"],
        total_rollouts=int(payload["total_rollouts"]),
        steps_per_rollout=int(payload["steps_per_rollout"]),
        minibatch_size=int(payload["minibatch_size"]),
        hidden_size=int(payload["hidden_size"]),
        learning_rate=float(payload["learning_rate"]),
        gamma=float(payload["gamma"]),
        gae_lambda=float(payload["gae_lambda"]),
        clip_range=float(payload["clip_range"]),
        value_coef=float(payload["value_coef"]),
        entropy_coef=float(payload["entropy_coef"]),
        max_grad_norm=float(payload["max_grad_norm"]),
        seed=int(payload["seed"]),
        evaluation_episodes=int(payload["evaluation_episodes"]),
        observation_dim=int(payload["observation_dim"]),
        action_dim=int(payload["action_dim"]),
    )
    return config


def _tensor(values, *, dtype=None):
    tensor = torch.as_tensor(values)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    else:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


def _require_approx_dependencies() -> None:
    if gym is None or np is None or torch is None or nn is None or Categorical is None:
        raise ApproxDependencyError(
            "Approximation bridge requires optional dependencies: torch and gymnasium."
        )
