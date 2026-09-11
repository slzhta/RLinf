#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluate real-world PnP CNN BC checkpoints on a held-out replay split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.cnn_policy.cnn_policy import CNNConfig, CNNPolicy

ACTION_NAMES = ("x", "y", "z", "rx", "ry", "rz", "gripper")
BINARY_METRICS = (
    "binary_accuracy",
    "binary_close_recall",
    "binary_open_recall",
    "binary_transition_close_recall",
    "binary_transition_open_recall",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--validation-replay", type=Path, required=True)
    parser.add_argument("--config-name", default="realworld_pick_and_place_bc_cnn")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _discover_checkpoints(root: Path) -> list[tuple[int, Path]]:
    checkpoints = []
    for path in root.glob("global_step_*"):
        try:
            step = int(path.name.removeprefix("global_step_"))
        except ValueError:
            continue
        weights = path / "actor" / "model_state_dict" / "full_weights.pt"
        marker = path / "actor" / "bc_policy_state.pt"
        if weights.is_file() and marker.is_file():
            checkpoints.append((step, path))
    if not checkpoints:
        raise FileNotFoundError(f"No complete BC checkpoints found under {root}.")
    return sorted(checkpoints)


def _validate_marker(checkpoint: Path) -> None:
    marker = torch.load(
        checkpoint / "actor" / "bc_policy_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    expected = {
        "format_version": 2,
        "training_stage": "bc",
        "data_format_version": 3,
        "model_type": "cnn_policy",
        "binary_action_indices": (6,),
        "binary_loss": "bce_with_logits",
    }
    actual = {key: marker.get(key) for key in expected}
    actual["binary_action_indices"] = tuple(actual.get("binary_action_indices") or ())
    if actual != expected:
        raise ValueError(f"Incompatible BC checkpoint marker at {checkpoint}: {actual}")


def _build_model(config_name: str, device: torch.device) -> CNNPolicy:
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = repo_root / "examples" / "embodiment" / "config"
    os.environ.setdefault("EMBODIED_PATH", str(config_dir.parent))
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=config_name)
    model_cfg = CNNConfig()
    model_cfg.update_from_dict(OmegaConf.to_container(cfg.actor.model, resolve=True))
    return CNNPolicy(model_cfg).to(device)


def _load_validation_batch(
    path: Path,
) -> tuple[dict[str, torch.Tensor], list[slice]]:
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    trajectory_files = sorted(
        path.glob("trajectory_*.pt"),
        key=lambda item: int(item.name.split("_", maxsplit=2)[1]),
    )
    if len(trajectory_files) != int(metadata["size"]):
        raise ValueError("Validation replay metadata and trajectory count disagree.")

    episode_inputs = []
    episode_slices = []
    offset = 0
    for trajectory_file in trajectory_files:
        payload = torch.load(trajectory_file, map_location="cpu", weights_only=True)
        forward_inputs = payload["forward_inputs"]
        action = forward_inputs["action"]
        if action.ndim != 3 or action.shape[1] != 1:
            raise ValueError(
                f"Expected a single-env trajectory at {trajectory_file}, "
                f"got action shape {tuple(action.shape)}."
            )
        length = int(action.shape[0])
        episode_inputs.append(
            {
                key: value[:, 0]
                for key, value in forward_inputs.items()
                if isinstance(value, torch.Tensor)
            }
        )
        episode_slices.append(slice(offset, offset + length))
        offset += length

    if offset != int(metadata["total_samples"]):
        raise ValueError("Validation replay metadata and sample count disagree.")
    keys = set(episode_inputs[0])
    if any(set(inputs) != keys for inputs in episode_inputs):
        raise ValueError("Validation trajectories have inconsistent input fields.")
    batch = {key: torch.cat([inputs[key] for inputs in episode_inputs]) for key in keys}
    return batch, episode_slices


def _sequence_metrics(
    predicted_open: torch.Tensor,
    target_open: torch.Tensor,
    episode_slices: list[slice],
) -> dict[str, Any]:
    strict_sequences = 0
    tolerant_sequences = 0
    predicted_transition_count = 0
    matches = {False: 0, True: 0}
    targets = {False: 0, True: 0}
    latencies = {False: [], True: []}

    for episode_slice in episode_slices:
        predicted = predicted_open[episode_slice]
        target = target_open[episode_slice]
        predicted_changes = (
            torch.nonzero(predicted[1:] != predicted[:-1]).flatten() + 1
        ).tolist()
        target_changes = (
            torch.nonzero(target[1:] != target[:-1]).flatten() + 1
        ).tolist()
        predicted_transition_count += len(predicted_changes)
        predicted_pattern = (int(predicted[0]),) + tuple(
            int(predicted[index]) for index in predicted_changes
        )
        strict_sequences += int(predicted_pattern == (1, 0, 1))

        episode_matches = 0
        used_predictions: set[int] = set()
        for target_index in target_changes:
            target_state = bool(target[target_index])
            targets[target_state] += 1
            candidates = [
                index
                for index in predicted_changes
                if index not in used_predictions
                and bool(predicted[index]) == target_state
                and target_index - 2 <= index <= target_index + 5
            ]
            if not candidates:
                continue
            predicted_index = min(
                candidates, key=lambda index: abs(index - target_index)
            )
            used_predictions.add(predicted_index)
            matches[target_state] += 1
            latencies[target_state].append(predicted_index - target_index)
            episode_matches += 1
        tolerant_sequences += int(episode_matches == len(target_changes) == 2)

    episode_count = len(episode_slices)
    metrics: dict[str, Any] = {
        "strict_open_close_open_episode_rate": strict_sequences / episode_count,
        "tolerant_open_close_open_episode_rate": tolerant_sequences / episode_count,
        "mean_predicted_transitions_per_episode": (
            predicted_transition_count / episode_count
        ),
    }
    for state, name in ((False, "close"), (True, "open")):
        metrics[f"tolerant_transition_{name}_recall"] = (
            matches[state] / targets[state] if targets[state] else None
        )
        metrics[f"mean_transition_{name}_latency_steps"] = (
            sum(latencies[state]) / len(latencies[state]) if latencies[state] else None
        )
    return metrics


@torch.no_grad()
def _evaluate_checkpoint(
    model: CNNPolicy,
    checkpoint: Path,
    validation_batch: dict[str, torch.Tensor],
    episode_slices: list[slice],
    batch_size: int,
) -> dict[str, Any]:
    weights_path = checkpoint / "actor" / "model_state_dict" / "full_weights.pt"
    model.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True)
    )
    model.eval()

    num_samples = int(validation_batch["action"].shape[0])
    action_loss_sum = torch.zeros(len(ACTION_NAMES), dtype=torch.float64)
    count_sums: dict[str, int] = {}
    predicted_open_batches = []
    for start in range(0, num_samples, batch_size):
        data = {
            key: value[start : start + batch_size]
            for key, value in validation_batch.items()
        }
        action_loss, counts = model(
            forward_type=ForwardType.SFT,
            data=model.prepare_dagger_sft_batch(data),
        )
        action_loss_sum += action_loss.double().sum(dim=0).cpu()
        for key, value in counts.items():
            count_sums[key] = count_sums.get(key, 0) + int(value.item())
        predicted_actions, _ = model.predict_action_batch(
            {
                "main_images": data["main_images"],
                "extra_view_images": data.get("extra_view_images"),
            },
            calculate_logprobs=False,
            calculate_values=False,
            return_obs=False,
            mode="eval",
        )
        predicted_open_batches.append(predicted_actions[:, 0, 6].cpu() > 0)

    action_losses = action_loss_sum / num_samples
    metrics: dict[str, Any] = {
        "num_samples": num_samples,
        "weighted_bc_loss": float(action_losses.mean()),
        "weighted_action_losses": {
            name: float(action_losses[index]) for index, name in enumerate(ACTION_NAMES)
        },
    }
    for name in BINARY_METRICS:
        correct = count_sums.get(f"{name}_correct", 0)
        count = count_sums.get(f"{name}_count", 0)
        metrics[name] = float(correct / count) if count else None
        metrics[f"{name}_count"] = count
    metrics.update(
        _sequence_metrics(
            predicted_open=torch.cat(predicted_open_batches),
            target_open=validation_batch["action"][:, 6] > 0,
            episode_slices=episode_slices,
        )
    )
    return metrics


def main() -> None:
    args = _parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested, but CUDA is unavailable.")

    checkpoints_root = args.checkpoints.expanduser().resolve()
    checkpoints = _discover_checkpoints(checkpoints_root)
    validation_batch, episode_slices = _load_validation_batch(
        args.validation_replay.expanduser().resolve()
    )
    model = _build_model(args.config_name, device)

    results = []
    for step, checkpoint in checkpoints:
        _validate_marker(checkpoint)
        metrics = _evaluate_checkpoint(
            model,
            checkpoint,
            validation_batch,
            episode_slices,
            args.batch_size,
        )
        results.append({"step": step, "checkpoint": str(checkpoint), **metrics})
    best = min(results, key=lambda item: item["weighted_bc_loss"])
    report = {
        "selection_metric": "minimum held-out weighted_bc_loss",
        "best_step": best["step"],
        "best_checkpoint": best["checkpoint"],
        "checkpoints": results,
    }

    output = args.output or checkpoints_root / "validation_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
