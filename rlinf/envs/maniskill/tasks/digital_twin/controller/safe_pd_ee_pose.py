from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Union

import numpy as np
import torch

from mani_skill.agents.controllers.pd_ee_pose import (
    PDEEPoseController,
    PDEEPoseControllerConfig,
)
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from mani_skill.utils.structs import Pose


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + np.pi, 2 * np.pi) - np.pi


class SafePDEEPoseController(PDEEPoseController):
    """EE pose controller with real-robot style action scaling and safety clipping."""

    config: "SafePDEEPoseControllerConfig"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_safety_config()

    def _init_safety_config(self):
        action_scale = np.asarray(self.config.action_scale, dtype=np.float32).reshape(-1)
        if action_scale.size != 2:
            raise ValueError(
                "action_scale must contain exactly 2 values: [position_scale, rotation_scale]."
            )
        self._position_action_scale = float(action_scale[0])
        self._rotation_action_scale = float(action_scale[1])

        if self.config.ee_pose_limit_min is None or self.config.ee_pose_limit_max is None:
            self._ee_pose_limit_min = None
            self._ee_pose_limit_max = None
            self._target_euler = None
            return

        ee_pose_limit_min = np.asarray(self.config.ee_pose_limit_min, dtype=np.float32).reshape(-1)
        ee_pose_limit_max = np.asarray(self.config.ee_pose_limit_max, dtype=np.float32).reshape(-1)
        if ee_pose_limit_min.size != 6 or ee_pose_limit_max.size != 6:
            raise ValueError(
                "ee_pose_limit_min and ee_pose_limit_max must both contain 6 values "
                "[x, y, z, rx, ry, rz]."
            )
        if np.any(ee_pose_limit_min > ee_pose_limit_max):
            raise ValueError("Each ee_pose_limit_min element must be <= ee_pose_limit_max.")

        self._ee_pose_limit_min = torch.as_tensor(ee_pose_limit_min, device=self.device)
        self._ee_pose_limit_max = torch.as_tensor(ee_pose_limit_max, device=self.device)

        if self.config.target_euler is None:
            self._target_euler = None
            return

        target_euler = np.asarray(self.config.target_euler, dtype=np.float32).reshape(-1)
        if target_euler.size != 3:
            raise ValueError("target_euler must contain 3 values [rx, ry, rz].")
        self._target_euler = torch.as_tensor(target_euler, device=self.device)

    def _scale_action(self, action: torch.Tensor) -> torch.Tensor:
        clipped_action = torch.clamp(
            action,
            min=torch.as_tensor(self.single_action_space.low, device=action.device, dtype=action.dtype),
            max=torch.as_tensor(self.single_action_space.high, device=action.device, dtype=action.dtype),
        )
        scaled_action = clipped_action.clone()
        scaled_action[:, :3] = clipped_action[:, :3] * self._position_action_scale
        scaled_action[:, 3:6] = clipped_action[:, 3:6] * self._rotation_action_scale
        return scaled_action

    def _clip_position_to_safety_box(self, target_pose: Pose) -> Pose:
        if self._ee_pose_limit_min is None or self._ee_pose_limit_max is None:
            return target_pose

        clipped_position = torch.clamp(
            target_pose.p,
            min=self._ee_pose_limit_min[:3],
            max=self._ee_pose_limit_max[:3],
        )

        euler = matrix_to_euler_angles(quaternion_to_matrix(target_pose.q), "XYZ")
        if self._target_euler is None:
            clipped_euler = torch.clamp(
                euler,
                min=self._ee_pose_limit_min[3:],
                max=self._ee_pose_limit_max[3:],
            )
        else:
            target_euler = self._target_euler.view(1, 3)
            delta = _wrap_to_pi(euler - target_euler)
            lower_delta = self._ee_pose_limit_min[3:].view(1, 3) - target_euler
            upper_delta = self._ee_pose_limit_max[3:].view(1, 3) - target_euler
            clipped_delta = torch.minimum(torch.maximum(delta, lower_delta), upper_delta)
            clipped_euler = _wrap_to_pi(target_euler + clipped_delta)

        clipped_quat = matrix_to_quaternion(euler_angles_to_matrix(clipped_euler, "XYZ"))
        return Pose.create_from_pq(p=clipped_position, q=clipped_quat)

    def compute_target_pose(self, prev_ee_pose_at_base: Pose, action: torch.Tensor):
        scaled_action = self._scale_action(action)
        target_pose = super().compute_target_pose(prev_ee_pose_at_base, scaled_action)
        return self._clip_position_to_safety_box(target_pose)


@dataclass
class SafePDEEPoseControllerConfig(PDEEPoseControllerConfig):
    """Config for SafePDEEPoseController."""

    action_scale: Sequence[float] = field(default_factory=lambda: [1.0, 1.0])
    ee_pose_limit_min: Union[None, Sequence[float]] = None
    ee_pose_limit_max: Union[None, Sequence[float]] = None
    target_euler: Union[None, Sequence[float]] = None

    controller_cls = SafePDEEPoseController
