#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch

import rlinf.envs.maniskill  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="PickAndPlaceDigitalTwin-v1")
    parser.add_argument("--output", default="/tmp/pick_and_place_digital_twin.mp4")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--sim-backend", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--view",
        default="fused-sensor",
        choices=["fused-sensor", "human-render"],
        help="fused-sensor matches the digital-twin training camera with background overlay.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    env = gym.make(
        args.env_id,
        num_envs=1,
        obs_mode="rgb+segmentation",
        control_mode="pd_ee_body_target_delta_pose_real",
        sim_backend=args.sim_backend,
        render_mode="rgb_array",
        reward_mode="dense",
        max_episode_steps=args.steps,
        sensor_configs={"shader_pack": "default"},
    )
    obs, _ = env.reset(seed=args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        for _ in range(args.steps):
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            if args.view == "fused-sensor":
                frame = obs["image"]["3rdview_camera"]["rgb"]
            else:
                frame = env.render()
            if isinstance(frame, torch.Tensor):
                frame = frame.detach().cpu().numpy()
            else:
                frame = np.asarray(frame)
            if frame.ndim == 4:
                frame = frame[0]
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    20,
                    (width, height),
                )
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            obs, *_ = env.step(action)
    finally:
        if writer is not None:
            writer.release()
        env.close()
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
