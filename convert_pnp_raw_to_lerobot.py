#!/usr/bin/env python3
"""Convert successful PnP motion-planning NPZ episodes to RLinf LeRobot v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rlinf.data.lerobot_writer import LeRobotDatasetWriter


REQUIRED_KEYS = ("base_images", "wrist_images", "states", "actions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--robot-type", default="panda_umi")
    parser.add_argument(
        "--default-task",
        default="pick up the cube and place it at the target position",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def scalar_text(value: np.ndarray, fallback: str) -> str:
    if value is None:
        return fallback
    text = str(np.asarray(value).item()).strip()
    return text or fallback


def validate_episode(path: Path, ep: dict[str, np.ndarray]) -> int:
    missing = [key for key in REQUIRED_KEYS if key not in ep]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")

    lengths = {key: len(ep[key]) for key in REQUIRED_KEYS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{path}: inconsistent lengths {lengths}")
    frames = lengths["base_images"]
    if frames == 0:
        raise ValueError(f"{path}: empty episode")

    for key in ("base_images", "wrist_images"):
        value = ep[key]
        if value.shape[1:] != (128, 128, 3) or value.dtype != np.uint8:
            raise ValueError(
                f"{path}: {key} expected (T,128,128,3) uint8, "
                f"got {value.shape} {value.dtype}"
            )
    if ep["states"].shape != (frames, 14):
        raise ValueError(f"{path}: expected states (T,14), got {ep['states'].shape}")
    if ep["actions"].shape != (frames, 7):
        raise ValueError(f"{path}: expected actions (T,7), got {ep['actions'].shape}")
    if not np.isfinite(ep["states"]).all() or not np.isfinite(ep["actions"]).all():
        raise ValueError(f"{path}: state/action contains NaN or Inf")
    return frames


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_dir.glob("episode_*.npz"))
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"no episode_*.npz found in {args.input_dir}")

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir}. "
            "Use a new directory to avoid mixing datasets."
        )

    writer = LeRobotDatasetWriter(
        root_dir=str(args.output_dir),
        robot_type=args.robot_type,
        fps=args.fps,
        image_shape=(128, 128, 3),
        state_dim=14,
        action_dim=7,
        has_wrist_image=True,
        has_extra_view_image=False,
        use_incremental_stats=True,
        stats_sample_ratio=0.1,
    )

    total_frames = 0
    task_counts: dict[str, int] = {}
    for index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as loaded:
            ep = {key: loaded[key] for key in loaded.files}
        frames = validate_episode(path, ep)
        task = scalar_text(ep.get("task"), args.default_task)

        # All files in this collection are successful demonstrations. A single
        # done at the final frame prevents action chunks crossing episodes.
        dones = np.zeros(frames, dtype=bool)
        dones[-1] = True
        writer.add_episode(
            images=ep["base_images"],
            wrist_images=ep["wrist_images"],
            extra_view_images=None,
            states=ep["states"].astype(np.float32, copy=False),
            actions=ep["actions"].astype(np.float32, copy=False),
            task=task,
            is_success=True,
            dones=dones,
        )
        total_frames += frames
        task_counts[task] = task_counts.get(task, 0) + 1
        if (index + 1) % 25 == 0 or index + 1 == len(paths):
            print(f"converted {index + 1}/{len(paths)} episodes")

    writer.finalize()
    summary = {
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "episodes": len(paths),
        "frames": total_frames,
        "fps": args.fps,
        "image_shape": [128, 128, 3],
        "state_dim": 14,
        "action_dim": 7,
        "tasks": task_counts,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
