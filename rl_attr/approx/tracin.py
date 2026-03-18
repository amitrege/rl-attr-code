from __future__ import annotations

from typing import Any

from .common import LookaheadSpec, TrainOccurrenceRef
from .ppo_lite import (
    ApproxCurriculumManifest,
    PolicyValueNet,
    _config_from_manifest,
    _load_policy_from_checkpoint,
    _ppo_loss,
    _require_approx_dependencies,
    _tensor,
    compute_buffer_surrogate_utility,
    load_buffer,
    torch,
)


def compute_local_snapshot_tracin(
    manifest: ApproxCurriculumManifest,
    lookahead: LookaheadSpec,
) -> dict[TrainOccurrenceRef, float]:
    return _compute_tracin_scores(
        manifest=manifest,
        lookahead=lookahead,
        target_rollout_index=lookahead.rollout_index,
    )


def compute_nonlocal_replay_tracin(
    manifest: ApproxCurriculumManifest,
    lookahead: LookaheadSpec,
) -> dict[TrainOccurrenceRef, float]:
    return _compute_tracin_scores(
        manifest=manifest,
        lookahead=lookahead,
        target_rollout_index=lookahead.target_rollout_index,
    )


def _compute_tracin_scores(
    manifest: ApproxCurriculumManifest,
    lookahead: LookaheadSpec,
    target_rollout_index: int,
) -> dict[TrainOccurrenceRef, float]:
    _require_approx_dependencies()
    config = _config_from_manifest(manifest)
    source_model = _load_policy_from_checkpoint(
        manifest.rollout_start_checkpoint_path(lookahead.rollout_index),
        config,
    )
    target_model = _load_policy_from_checkpoint(
        manifest.rollout_end_checkpoint_path(target_rollout_index),
        config,
    )
    source_buffer = load_buffer(manifest.rollout_buffer_path(lookahead.rollout_index))
    target_buffer = load_buffer(manifest.rollout_buffer_path(target_rollout_index))
    target_gradient = _utility_gradient(target_model, target_buffer)

    scores: dict[TrainOccurrenceRef, float] = {}
    for row_index in range(len(source_buffer["actions"])):
        train_gradient = _row_training_gradient(source_model, source_buffer, row_index, config)
        score = -config.learning_rate * torch.dot(train_gradient, target_gradient).item()
        scores[TrainOccurrenceRef(rollout_index=lookahead.rollout_index, row_index=row_index)] = float(score)
    return scores


def _utility_gradient(model: PolicyValueNet, buffer: dict[str, Any]):
    model.zero_grad(set_to_none=True)
    observations = _tensor(buffer["observations"])
    actions = _tensor(buffer["actions"], dtype=torch.long)
    advantages = _tensor(buffer["advantages"])
    logits, _ = model(observations)
    distribution = torch.distributions.Categorical(logits=logits)
    log_prob = distribution.log_prob(actions)
    utility = torch.mean(advantages * log_prob)
    utility.backward()
    return _flatten_gradients(model)


def _row_training_gradient(
    model: PolicyValueNet,
    buffer: dict[str, Any],
    row_index: int,
    config,
):
    model.zero_grad(set_to_none=True)
    loss = _ppo_loss(model, buffer, [row_index], config)
    loss.backward()
    return _flatten_gradients(model)


def _flatten_gradients(model: PolicyValueNet):
    pieces = []
    for parameter in model.parameters():
        gradient = parameter.grad
        if gradient is None:
            pieces.append(torch.zeros(parameter.numel(), dtype=parameter.dtype))
        else:
            pieces.append(gradient.detach().flatten())
    return torch.cat(pieces)
