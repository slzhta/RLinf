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

"""Shared robot-base geometry for the glued U-tool insertion task (metres/radians)."""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation


def pose_matrix(pose):
    pose = np.asarray(pose, dtype=np.float64)
    matrix = np.eye(4)
    matrix[:3, 3] = pose[:3]
    matrix[:3, :3] = (
        Rotation.from_euler("xyz", pose[3:])
        if pose.shape == (6,)
        else Rotation.from_quat(pose[3:])
    ).as_matrix()
    return matrix


def matrix_pose(matrix):
    return np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()]


@dataclass
class PegInsertionGeometry:
    target_ee_pose: list = field(
        default_factory=lambda: [0.545, 0.035, 0.048, np.pi, 0.0, 0.0]
    )
    action_scale: list = field(default_factory=lambda: [0.002, 0.02])
    reset_height: float = 0.10
    random_xy: float = 0.05
    random_yaw: float = np.deg2rad(10.0)
    workspace_xy: float = 0.05
    workspace_above: float = 0.14
    workspace_below: float = 0.001
    rotation_limits: list = field(default_factory=lambda: [0.05, 0.05, 0.30])
    success_xy: float = 0.001
    success_z: float = 0.002
    success_angle: float = 0.03
    success_hold_steps: int = 3
    dense_reward_scale: float = 0.05

    def __post_init__(self):
        values = np.asarray(self.target_ee_pose, dtype=float)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError(
                "target_ee_pose must be finite robot-base xyz + Euler xyz."
            )
        self.target_ee_pose = values.tolist()
        for name, size in (("action_scale", 2), ("rotation_limits", 3)):
            value = np.asarray(getattr(self, name), dtype=float)
            if (
                value.shape != (size,)
                or not np.isfinite(value).all()
                or (value <= 0).any()
            ):
                raise ValueError(f"{name} must contain {size} positive finite values.")
        for name in (
            "reset_height",
            "workspace_xy",
            "workspace_above",
            "success_xy",
            "success_z",
            "success_angle",
            "dense_reward_scale",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        for name in ("random_xy", "random_yaw", "workspace_below"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative and finite.")
        if not 0.05 <= self.reset_height <= self.workspace_above:
            raise ValueError(
                "reset_height must clear the 45 mm board and fit the workspace."
            )
        if (
            self.random_xy > self.workspace_xy
            or self.random_yaw > self.rotation_limits[2]
        ):
            raise ValueError("Reset randomization exceeds the workspace.")
        if (
            self.success_hold_steps < 1
            or int(self.success_hold_steps) != self.success_hold_steps
        ):
            raise ValueError("success_hold_steps must be a positive integer.")
        if self.dense_reward_scale >= 1:
            raise ValueError(
                "dense_reward_scale must be below the terminal reward of 1."
            )

    @property
    def target(self):
        return pose_matrix(self.target_ee_pose)

    @property
    def tcp_to_peg(self):
        return pose_matrix([0, 0, 0.046, np.pi, 0, -np.pi / 2])

    @property
    def insertion_rotation(self):
        return (self.target @ self.tcp_to_peg)[:3, :3]

    def board_pose(self):
        board_to_seated_peg = pose_matrix(
            [0.06448662185668945, -0.07030080127716064, 0.002, 0, 0, 0]
        )
        return self.target @ self.tcp_to_peg @ np.linalg.inv(board_to_seated_peg)

    def relative_state(self, current):
        relative = np.linalg.inv(self.target) @ current
        rotation = Rotation.from_matrix(relative[..., :3, :3]).as_rotvec()
        return np.concatenate([relative[..., :3, 3], rotation], axis=-1).astype(
            np.float32
        )

    def clip_target(self, current):
        clipped = np.array(current, copy=True)
        axes = self.insertion_rotation
        position = (current[..., :3, 3] - self.target[:3, 3]) @ axes
        position = np.clip(
            position,
            [-self.workspace_xy, -self.workspace_xy, -self.workspace_below],
            [self.workspace_xy, self.workspace_xy, self.workspace_above],
        )
        clipped[..., :3, 3] = position @ axes.T + self.target[:3, 3]
        relative = self.target[:3, :3].T @ current[..., :3, :3]
        rotvec = Rotation.from_matrix(relative).as_rotvec()
        rotvec = np.clip(
            rotvec, -np.asarray(self.rotation_limits), self.rotation_limits
        )
        clipped[..., :3, :3] = (
            self.target[:3, :3] @ Rotation.from_rotvec(rotvec).as_matrix()
        )
        return clipped

    def action_target(self, previous, action):
        action = np.asarray(action, dtype=float)
        if action.shape[-1] != 6 or not np.isfinite(action).all():
            raise ValueError("Peg insertion requires six finite action values.")
        action = np.clip(action, -1, 1)
        result = np.array(previous, copy=True)
        result[..., :3, 3] += action[..., :3] * self.action_scale[0]
        result[..., :3, :3] = (
            Rotation.from_rotvec(action[..., 3:] * self.action_scale[1]).as_matrix()
            @ previous[..., :3, :3]
        )
        return self.clip_target(result)

    def reset_pose(self, rng):
        result = self.target.copy()
        offset = np.r_[
            rng.uniform(-self.random_xy, self.random_xy, 2), self.reset_height
        ]
        result[:3, 3] += self.insertion_rotation @ offset
        angle = rng.uniform(-self.random_yaw, self.random_yaw)
        result[:3, :3] = (
            Rotation.from_rotvec(self.insertion_rotation[:, 2] * angle).as_matrix()
            @ result[:3, :3]
        )
        return result

    def metrics(self, current):
        delta = (current[..., :3, 3] - self.target[:3, 3]) @ self.insertion_rotation
        xy = np.linalg.norm(delta[..., :2], axis=-1)
        z = np.abs(delta[..., 2])
        angle = np.linalg.norm(self.relative_state(current)[..., 3:], axis=-1)
        aligned = np.exp(-xy / 0.01 - angle / 0.10)
        dense = self.dense_reward_scale * aligned * (1 + 2 * np.exp(-z / 0.05)) / 3
        candidate = (
            (xy <= self.success_xy)
            & (z <= self.success_z)
            & (angle <= self.success_angle)
        )
        return {
            "xy_error": xy,
            "z_error": z,
            "angle_error": angle,
            "dense_reward": dense,
            "in_target": candidate,
        }
