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

"""Robot-base control geometry for pouring a 16 mm solid ball (metres/radians)."""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from rlinf.envs.peg_insertion_geometry import pose_matrix


@dataclass
class PourWaterGeometry:
    target_ee_pose: list = field(
        default_factory=lambda: [0.49, 0.12448609, 0.18, np.pi, -0.35, 0.0]
    )
    action_scale: list = field(default_factory=lambda: [0.01, 0.05])
    workspace_min: list = field(default_factory=lambda: [-0.15, -0.15, -0.10])
    workspace_max: list = field(default_factory=lambda: [0.15, 0.15, 0.15])
    rotation_limits: list = field(default_factory=lambda: [0.35, 1.15, 0.5])
    reset_position_random_range: list = field(
        default_factory=lambda: [0.015, 0.015, 0.01]
    )
    random_yaw: float = 0.10
    success_hold_steps: int = 3

    def __post_init__(self):
        for name, size in (
            ("target_ee_pose", 6),
            ("action_scale", 2),
            ("workspace_min", 3),
            ("workspace_max", 3),
            ("rotation_limits", 3),
            ("reset_position_random_range", 3),
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain {size} finite numbers.")
            setattr(self, name, value.tolist())
        lower, upper = np.array(self.workspace_min), np.array(self.workspace_max)
        if np.any(lower >= 0) or np.any(upper <= 0):
            raise ValueError("Workspace bounds must contain the reset reference.")
        if np.any(np.array(self.action_scale) <= 0) or np.any(
            np.array(self.rotation_limits) <= 0
        ):
            raise ValueError("Action scales and rotation limits must be positive.")
        random_range = np.array(self.reset_position_random_range)
        if np.any(random_range < 0) or np.any(random_range > np.minimum(-lower, upper)):
            raise ValueError("Reset randomization must fit the workspace.")
        if (
            not np.isfinite(self.random_yaw)
            or not 0 <= self.random_yaw <= self.rotation_limits[2]
        ):
            raise ValueError("Reset yaw must fit the rotation limit.")
        if (
            self.success_hold_steps < 1
            or int(self.success_hold_steps) != self.success_hold_steps
        ):
            raise ValueError("success_hold_steps must be a positive integer.")

    @property
    def target(self):
        return pose_matrix(self.target_ee_pose)

    def relative_state(self, current):
        relative = np.linalg.inv(self.target) @ current
        return np.concatenate(
            [
                relative[..., :3, 3],
                Rotation.from_matrix(relative[..., :3, :3]).as_rotvec(),
            ],
            axis=-1,
        ).astype(np.float32)

    def clip_target(self, current):
        result = np.array(current, copy=True)
        result[..., :3, 3] = self.target[:3, 3] + np.clip(
            current[..., :3, 3] - self.target[:3, 3],
            self.workspace_min,
            self.workspace_max,
        )
        rotation = Rotation.from_matrix(
            self.target[:3, :3].T @ current[..., :3, :3]
        ).as_rotvec()
        rotation = np.clip(
            rotation, -np.array(self.rotation_limits), self.rotation_limits
        )
        result[..., :3, :3] = (
            self.target[:3, :3] @ Rotation.from_rotvec(rotation).as_matrix()
        )
        return result

    def action_target(self, previous, action):
        action = np.asarray(action, dtype=float)
        if action.shape[-1] != 6 or not np.isfinite(action).all():
            raise ValueError("Pour water requires six finite action values.")
        action = np.clip(action, -1, 1)
        result = np.array(previous, copy=True)
        result[..., :3, 3] += action[..., :3] * self.action_scale[0]
        # Match PnP's robot-base Euler increment, not an absolute Euler target.
        result[..., :3, :3] = (
            Rotation.from_euler(
                "xyz", action[..., 3:] * self.action_scale[1]
            ).as_matrix()
            @ previous[..., :3, :3]
        )
        return self.clip_target(result)

    def reset_pose(self, rng):
        result = self.target.copy()
        random_range = np.array(self.reset_position_random_range)
        result[:3, 3] += rng.uniform(-random_range, random_range)
        result[:3, :3] = (
            Rotation.from_euler(
                "z", rng.uniform(-self.random_yaw, self.random_yaw)
            ).as_matrix()
            @ result[:3, :3]
        )
        return self.clip_target(result)


CUP_HEIGHT = 0.09
CUP_BOTTOM_RADIUS = 0.021
CUP_TOP_RADIUS = 0.03
CUP_WALL = 0.001
BALL_RADIUS = 0.008
CUP_SLOPE = (CUP_TOP_RADIUS - CUP_BOTTOM_RADIUS) / CUP_HEIGHT


def ball_inside_cup(local_position, tolerance=0.0003):
    """Require the whole sphere below the rim and inside the tapered inner wall."""
    local_position = np.asarray(local_position)
    z = local_position[..., 2]
    radius = np.linalg.norm(local_position[..., :2], axis=-1)
    wall_distance = (CUP_BOTTOM_RADIUS + CUP_SLOPE * z - radius) / np.sqrt(
        1 + CUP_SLOPE**2
    )
    return (
        (z >= CUP_WALL + BALL_RADIUS - tolerance)
        & (z <= CUP_HEIGHT - BALL_RADIUS)
        & (wall_distance >= CUP_WALL + BALL_RADIUS - tolerance)
    )
