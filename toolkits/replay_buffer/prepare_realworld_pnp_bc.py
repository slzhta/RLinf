#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare successful real-world PnP demonstrations for CNN behavior cloning.

The source dataset is left unchanged. The tool writes a clean LeRobot success
subset and RLinf replay buffers with deployment-compatible action targets.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.lerobot_writer import LeRobotDatasetWriter
from rlinf.data.replay_buffer import TrajectoryReplayBuffer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--main-image-field",
        default="extra_view_image",
        help="Fixed third-person view used as CNN encoder 0.",
    )
    parser.add_argument(
        "--extra-image-field",
        default="image",
        help="Optional wrist view used only when policy-image-num is 2.",
    )
    parser.add_argument("--policy-image-num", type=int, choices=(1, 2), default=1)
    parser.add_argument("--validation-modulo", type=int, default=10)
    parser.add_argument("--idle-context-steps", type=int, default=1)
    parser.add_argument("--gripper-width-threshold", type=float, default=0.04)
    parser.add_argument("--gripper-transition-weight", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file_obj:
        return [json.loads(line) for line in file_obj if line.strip()]


def _decode_image(value: dict[str, Any], field: str) -> np.ndarray:
    if not isinstance(value, dict) or not value.get("bytes"):
        raise ValueError(f"Image field {field!r} contains an empty frame.")
    with Image.open(io.BytesIO(value["bytes"])) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8, copy=True)


def select_training_window(actions: np.ndarray, context_steps: int) -> slice:
    """Trim collection-only idle prefixes and suffixes around expert motion."""
    if context_steps < 0:
        raise ValueError("context_steps must be non-negative.")
    active = np.any(np.abs(actions) > 1e-3, axis=-1)
    active_indices = np.flatnonzero(active)
    if active_indices.size == 0:
        raise ValueError("Episode contains no expert action.")
    start = max(0, int(active_indices[0]) - context_steps)
    stop = min(len(actions), int(active_indices[-1]) + context_steps + 1)
    return slice(start, stop)


def prepare_action_targets(
    actions: np.ndarray,
    states: np.ndarray,
    gripper_width_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build bounded continuous actions and persistent binary gripper targets."""
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected actions with shape [T, 7], got {actions.shape}.")
    if states.ndim != 2 or states.shape[0] != actions.shape[0]:
        raise ValueError("State and action trajectory lengths must match.")
    if states.shape[1] != 19:
        raise ValueError(
            "Expected the legacy 19-D real-world state used by this collector, "
            f"got {states.shape[1]}."
        )
    if gripper_width_threshold <= 0:
        raise ValueError("gripper_width_threshold must be positive.")

    targets = actions.astype(np.float32, copy=True)
    targets[:, :6] = np.clip(targets[:, :6], -1.0, 1.0)

    # SpaceMouse gripper commands are pulses: -1 closes, +1 opens, and zero
    # holds the previous state. Integrate those pulses into dense state labels
    # with the deployment convention 0=closed and 1=open.
    gripper_state = float(states[0, 0] >= gripper_width_threshold)
    gripper_targets = np.empty(len(actions), dtype=np.float32)
    for step, command in enumerate(actions[:, 6]):
        if command <= -0.5:
            gripper_state = 0.0
        elif command >= 0.5:
            gripper_state = 1.0
        gripper_targets[step] = gripper_state
    targets[:, 6] = gripper_targets

    transitions = np.zeros(len(targets), dtype=bool)
    transitions[1:] = gripper_targets[1:] != gripper_targets[:-1]
    return targets, transitions


def validate_open_close_open(
    targets: np.ndarray, transitions: np.ndarray, episode_index: int
) -> None:
    """Require the task-specific open -> closed -> open gripper sequence."""
    transition_states = targets[transitions, 6].astype(np.int64).tolist()
    pattern = [int(targets[0, 6]), *transition_states]
    if pattern != [1, 0, 1]:
        raise ValueError(
            f"Episode {episode_index} has gripper pattern {pattern}, expected "
            "[1, 0, 1] (open -> closed -> open)."
        )


def _validate_source(source: Path, info: dict[str, Any]) -> None:
    if int(info.get("total_episodes", -1)) <= 0:
        raise ValueError("Source dataset contains no episodes.")
    features = info.get("features", {})
    expected = {"image", "extra_view_image", "state", "actions", "is_success"}
    missing = sorted(expected - set(features))
    if missing:
        raise ValueError(f"Source dataset is missing features: {missing}.")
    if features["state"].get("shape") != [19]:
        raise ValueError("Source dataset must contain the legacy 19-D state.")
    if features["actions"].get("shape") != [7]:
        raise ValueError("Source dataset must contain 7-D actions.")


def _episode_path(source: Path, episode_index: int) -> Path:
    return source / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"


def _read_episode(source: Path, episode_meta: dict[str, Any]) -> dict[str, Any]:
    episode_index = int(episode_meta["episode_index"])
    parquet_path = _episode_path(source, episode_index)
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)
    data = pq.read_table(parquet_path).to_pydict()
    length = len(data["actions"])
    if length != int(episode_meta["length"]):
        raise ValueError(f"Episode {episode_index} length disagrees with metadata.")
    if set(data["episode_index"]) != {episode_index}:
        raise ValueError(f"Episode {episode_index} has inconsistent row indices.")
    row_success = {bool(value) for value in data["is_success"]}
    if row_success != {bool(episode_meta["is_success"])}:
        raise ValueError(f"Episode {episode_index} has inconsistent success labels.")
    if sum(bool(value) for value in data["done"]) != 1 or not data["done"][-1]:
        raise ValueError(f"Episode {episode_index} must end with exactly one done.")
    return data


def _stack_images(data: dict[str, Any], field: str) -> np.ndarray:
    if field not in data:
        raise ValueError(f"Image field {field!r} is not present in the dataset.")
    return np.stack([_decode_image(value, field) for value in data[field]])


def _build_trajectory(
    data: dict[str, Any],
    source_episode_index: int,
    main_image_field: str,
    extra_image_field: str,
    policy_image_num: int,
    idle_context_steps: int,
    gripper_width_threshold: float,
    gripper_transition_weight: float,
) -> tuple[Trajectory, dict[str, Any]]:
    states = np.asarray(data["state"], dtype=np.float32)
    raw_actions = np.asarray(data["actions"], dtype=np.float32)
    window = select_training_window(raw_actions, idle_context_steps)
    states = states[window]
    raw_actions = raw_actions[window]
    targets, transitions = prepare_action_targets(
        raw_actions, states, gripper_width_threshold
    )
    validate_open_close_open(targets, transitions, source_episode_index)
    main_images = _stack_images(data, main_image_field)[window]

    sample_weights = np.ones(len(targets), dtype=np.float32)
    sample_weights[transitions] = gripper_transition_weight
    actions_t = torch.from_numpy(targets).unsqueeze(1)
    rewards = torch.zeros((len(targets), 1, 1), dtype=torch.float32)
    rewards[-1] = 1.0
    terminations = torch.zeros((len(targets), 1, 1), dtype=torch.bool)
    terminations[-1] = True

    forward_inputs = {
        "main_images": torch.from_numpy(main_images).unsqueeze(1),
        "action": actions_t.clone(),
        "sample_weight": torch.from_numpy(sample_weights).reshape(-1, 1, 1),
        "action_transition": torch.from_numpy(transitions).reshape(-1, 1, 1),
    }
    if policy_image_num == 2:
        extra_images = _stack_images(data, extra_image_field)[window]
        if main_images.shape != extra_images.shape:
            raise ValueError(
                "Main and extra camera trajectories must have equal shapes."
            )
        forward_inputs["extra_view_images"] = (
            torch.from_numpy(extra_images).unsqueeze(1).unsqueeze(2)
        )
    trajectory = Trajectory(
        max_episode_length=len(targets),
        model_weights_id=f"real_expert_source_{source_episode_index}",
        actions=actions_t,
        intervene_flags=torch.ones_like(actions_t, dtype=torch.bool),
        rewards=rewards,
        terminations=terminations,
        truncations=torch.zeros_like(terminations),
        dones=terminations.clone(),
        forward_inputs=forward_inputs,
    )
    audit = {
        "source_episode_index": source_episode_index,
        "source_frames": len(data["actions"]),
        "training_frames": len(targets),
        "trimmed_prefix_frames": int(window.start or 0),
        "trimmed_suffix_frames": len(data["actions"]) - int(window.stop or 0),
        "clipped_continuous_values": int(np.sum(np.abs(raw_actions[:, :6]) > 1.0)),
        "gripper_transition_frames": int(transitions.sum()),
    }
    return trajectory, audit


def _write_replay_buffer(path: Path, trajectories: list[Trajectory]) -> None:
    path.mkdir(parents=True)
    replay = TrajectoryReplayBuffer(
        seed=0,
        enable_cache=False,
        cache_size=0,
        sample_window_size=0,
        auto_save=True,
        auto_save_path=str(path),
        trajectory_format="pt",
    )
    replay.add_trajectories(trajectories)
    replay.close()


def _write_manifest(
    path: Path,
    split: str,
    audits: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    total_transitions = sum(item["training_frames"] for item in audits)
    transition_steps = sum(item["gripper_transition_frames"] for item in audits)
    payload = {
        "format_version": 3,
        "task": "realworld_pick_and_place",
        "split": split,
        "source_dataset": str(args.source.resolve()),
        "collected_episodes": len(audits),
        "total_transitions": total_transitions,
        "source_episode_indices": [item["source_episode_index"] for item in audits],
        "policy_input": {
            "use_state": False,
            "image_num": args.policy_image_num,
            "main_image_field": args.main_image_field,
        },
        "retained_lerobot_images": ["image", "extra_view_image"],
        "action_transform": {
            "continuous_clip": [-1.0, 1.0],
            "persistent_binary_gripper": True,
            "gripper_target_convention": {"closed": 0.0, "open": 1.0},
            "raw_gripper_pulses": {"close": -1.0, "hold": 0.0, "open": 1.0},
            "gripper_width_state_index": 0,
            "gripper_width_threshold": args.gripper_width_threshold,
        },
        "bc_sampling": {
            "weight_key": "forward_inputs.sample_weight",
            "transition_sampling_weight": args.gripper_transition_weight,
            "action_transition_steps": transition_steps,
        },
        "episodes": audits,
    }
    with (path / "collection_summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)


def prepare_dataset(args: argparse.Namespace) -> None:
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    info = _load_json(source / "meta" / "info.json")
    episodes = _load_jsonl(source / "meta" / "episodes.jsonl")
    tasks = {
        int(row["task_index"]): row["task"]
        for row in _load_jsonl(source / "meta" / "tasks.jsonl")
    }
    _validate_source(source, info)
    if len(episodes) != int(info["total_episodes"]):
        raise ValueError("Episode metadata count disagrees with info.json.")

    success_episodes = [row for row in episodes if bool(row["is_success"])]
    failure_indices = [
        int(row["episode_index"]) for row in episodes if not row["is_success"]
    ]
    if not failure_indices:
        raise ValueError(
            "No failed episodes were found; refusing an unexpected rewrite."
        )

    first_data = _read_episode(source, success_episodes[0])
    image_shape = _decode_image(first_data["image"][0], "image").shape
    clean_path = output / "lerobot_success"
    writer = LeRobotDatasetWriter(
        root_dir=str(clean_path),
        robot_type=str(info.get("robot_type", "panda")),
        fps=int(info["fps"]),
        image_shape=image_shape,
        state_dim=19,
        action_dim=7,
        has_wrist_image=False,
        has_extra_view_image=True,
        use_incremental_stats=True,
        stats_sample_ratio=1.0,
    )

    split_trajectories: dict[str, list[Trajectory]] = {"train": [], "validation": []}
    split_audits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    clean_episode_map: list[dict[str, int]] = []
    for clean_index, episode_meta in enumerate(success_episodes):
        source_index = int(episode_meta["episode_index"])
        data = _read_episode(source, episode_meta)
        images = _stack_images(data, "image")
        extra_images = _stack_images(data, "extra_view_image")
        states = np.asarray(data["state"], dtype=np.float32)
        raw_actions = np.asarray(data["actions"], dtype=np.float32)
        actions, gripper_transitions = prepare_action_targets(
            raw_actions, states, args.gripper_width_threshold
        )
        validate_open_close_open(actions, gripper_transitions, source_index)
        task_index = int(data["task_index"][0])
        writer.add_episode(
            images=images,
            wrist_images=None,
            extra_view_images=extra_images,
            states=states,
            actions=actions,
            task=tasks[task_index],
            is_success=True,
            dones=np.asarray(data["done"], dtype=bool),
        )
        clean_episode_map.append(
            {"clean_episode_index": clean_index, "source_episode_index": source_index}
        )

        trajectory, audit = _build_trajectory(
            data=data,
            source_episode_index=source_index,
            main_image_field=args.main_image_field,
            extra_image_field=args.extra_image_field,
            policy_image_num=args.policy_image_num,
            idle_context_steps=args.idle_context_steps,
            gripper_width_threshold=args.gripper_width_threshold,
            gripper_transition_weight=args.gripper_transition_weight,
        )
        is_validation = (
            args.validation_modulo > 0 and source_index % args.validation_modulo == 0
        )
        split = "validation" if is_validation else "train"
        split_trajectories[split].append(trajectory)
        split_audits[split].append(audit)

    writer.finalize()
    if not split_trajectories["train"] or not split_trajectories["validation"]:
        raise ValueError("Both train and validation splits must contain episodes.")

    replay_root = output / "replay"
    for split in ("train", "validation"):
        split_path = replay_root / split
        _write_replay_buffer(split_path, split_trajectories[split])
        _write_manifest(split_path, split, split_audits[split], args)

    summary = {
        "source_dataset": str(source),
        "source_episodes": len(episodes),
        "removed_failure_episode_indices": failure_indices,
        "success_episodes": len(success_episodes),
        "clean_episode_map": clean_episode_map,
        "train_episodes": len(split_trajectories["train"]),
        "validation_episodes": len(split_trajectories["validation"]),
        "train_transitions": sum(
            item["training_frames"] for item in split_audits["train"]
        ),
        "validation_transitions": sum(
            item["training_frames"] for item in split_audits["validation"]
        ),
    }
    with (output / "summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = _parse_args()
    if args.validation_modulo <= 1:
        raise ValueError("validation_modulo must be greater than one.")
    if args.gripper_transition_weight < 1.0:
        raise ValueError("gripper_transition_weight must be at least one.")
    prepare_dataset(args)


if __name__ == "__main__":
    main()
