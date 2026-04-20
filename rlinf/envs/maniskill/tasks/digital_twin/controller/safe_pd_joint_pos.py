from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mani_skill.agents.controllers.pd_joint_pos import (
    PDJointPosMimicController,
    PDJointPosMimicControllerConfig,
)
from mani_skill.utils.structs.types import Array


class SafePDJointPosMimicController(PDJointPosMimicController):
    """Mimic gripper controller with explicit clip+scale action preprocessing."""

    config: "SafePDJointPosMimicControllerConfig"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_action_scaling()

    def _init_action_scaling(self):
        action_scale = np.asarray(self.config.action_scale, dtype=np.float32).reshape(-1)
        if action_scale.size != 1:
            raise ValueError("gripper action_scale must contain exactly 1 value.")
        if self.config.binary_gripper_threshold < 0:
            raise ValueError("binary_gripper_threshold must be non-negative.")
        if self.config.use_zero_one_gripper_action and self.config.binary_gripper_threshold > 1:
            raise ValueError(
                "binary_gripper_threshold must be <= 1 when use_zero_one_gripper_action=True."
            )

        self._gripper_action_scale = float(action_scale[0])
        self._binary_gripper_threshold = float(self.config.binary_gripper_threshold)
        self._use_zero_one_gripper_action = bool(self.config.use_zero_one_gripper_action)
        self._open_command = float(self.config.open_command)
        self._close_command = float(self.config.close_command)
        self._binary_state_boundary = 0.5 * (self._open_command + self._close_command)
        self._action_low = torch.as_tensor(
            self.single_action_space.low,
            device=self.device,
            dtype=torch.float32,
        )
        self._action_high = torch.as_tensor(
            self.single_action_space.high,
            device=self.device,
            dtype=torch.float32,
        )
        self._binary_action = None
        self._gripper_open_state = None

    def _to_zero_one_action(self, action: torch.Tensor) -> torch.Tensor:
        """Normalize gripper command to [0, 1].

        If action already lies in [0, 1], keep it. Otherwise treat it as legacy [-1, 1]
        and map with (x + 1) / 2.
        """
        if torch.all((action >= 0.0) & (action <= 1.0)):
            return torch.clamp(action, min=0.0, max=1.0)

        clipped_legacy = torch.clamp(action, min=-1.0, max=1.0)
        return 0.5 * (clipped_legacy + 1.0)

    def _infer_action_from_current_qpos(
        self,
        target_shape: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        current_control_qpos = self.qpos[:, self.control_joint_indices].reshape(
            self.scene.num_envs, -1
        )
        current_control_qpos = current_control_qpos.to(device=device, dtype=dtype)

        if self._normalize_action:
            low_raw = torch.as_tensor(
                self._original_single_action_space.low,
                device=device,
                dtype=dtype,
            ).reshape(1, -1)
            high_raw = torch.as_tensor(
                self._original_single_action_space.high,
                device=device,
                dtype=dtype,
            ).reshape(1, -1)
            denom = torch.clamp(high_raw - low_raw, min=1e-6)
            current_action = 2.0 * (current_control_qpos - low_raw) / denom - 1.0
        else:
            current_action = current_control_qpos

        current_action = torch.clamp(
            current_action,
            min=self._action_low.to(device=device, dtype=dtype),
            max=self._action_high.to(device=device, dtype=dtype),
        )
        if current_action.shape != target_shape:
            current_action = torch.broadcast_to(current_action, target_shape).clone()
        return current_action

    def reset(self):
        super().reset()
        self._binary_action = None
        self._gripper_open_state = None
        if self.config.binary_gripper_action:
            target_shape = (self.scene.num_envs, self.single_action_space.shape[0])
            current_action = self._infer_action_from_current_qpos(
                target_shape=target_shape,
                dtype=self.qpos.dtype,
                device=self.device,
            )
            self._gripper_open_state = current_action >= self._binary_state_boundary
            self._binary_action = torch.where(
                self._gripper_open_state,
                torch.full_like(current_action, self._open_command),
                torch.full_like(current_action, self._close_command),
            )

    def set_action(self, action: Array):
        action = torch.as_tensor(
            action,
            device=self.device,
            dtype=self._action_low.dtype,
        )
        if action.ndim == 1:
            action = action.unsqueeze(0)

        clipped_action = torch.clamp(
            action,
            min=self._action_low.to(device=action.device, dtype=action.dtype),
            max=self._action_high.to(device=action.device, dtype=action.dtype),
        )
        scaled_action = clipped_action * self._gripper_action_scale

        if not self.config.binary_gripper_action:
            super().set_action(scaled_action)
            return

        if (
            self._binary_action is None
            or self._gripper_open_state is None
            or self._binary_action.shape != scaled_action.shape
            or self._gripper_open_state.shape != scaled_action.shape
        ):
            current_action = self._infer_action_from_current_qpos(
                target_shape=scaled_action.shape,
                dtype=scaled_action.dtype,
                device=scaled_action.device,
            )
            self._gripper_open_state = current_action >= self._binary_state_boundary
            self._binary_action = torch.where(
                self._gripper_open_state,
                torch.full_like(scaled_action, self._open_command),
                torch.full_like(scaled_action, self._close_command),
            )
        else:
            self._binary_action = self._binary_action.to(
                device=scaled_action.device,
                dtype=scaled_action.dtype,
            )
            self._gripper_open_state = self._gripper_open_state.to(
                device=scaled_action.device,
                dtype=torch.bool,
            )

        close_mask = (scaled_action <= -self._binary_gripper_threshold) & self._gripper_open_state
        open_mask = (
            (~close_mask)
            & (scaled_action >= self._binary_gripper_threshold)
            & (~self._gripper_open_state)
        )

        if self._use_zero_one_gripper_action:
            zero_one_action = self._to_zero_one_action(scaled_action)
            open_threshold = self._binary_gripper_threshold
            close_threshold = 1.0 - self._binary_gripper_threshold
            close_mask = (zero_one_action <= close_threshold) & self._gripper_open_state
            open_mask = (
                (~close_mask)
                & (zero_one_action >= open_threshold)
                & (~self._gripper_open_state)
            )

        self._binary_action[open_mask] = self._open_command
        self._binary_action[close_mask] = self._close_command
        self._gripper_open_state[open_mask] = True
        self._gripper_open_state[close_mask] = False

        super().set_action(self._binary_action)


@dataclass
class SafePDJointPosMimicControllerConfig(PDJointPosMimicControllerConfig):
    """Config for SafePDJointPosMimicController."""

    action_scale: float = 1.0
    binary_gripper_action: bool = True
    binary_gripper_threshold: float = 0.5
    use_zero_one_gripper_action: bool = False
    open_command: float = 1.0
    close_command: float = -1.0

    controller_cls = SafePDJointPosMimicController
