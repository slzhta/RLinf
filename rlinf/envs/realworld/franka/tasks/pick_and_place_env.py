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

import copy
import time
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from ..franka_env import FrankaEnv
from .co_training_base_env import FrankaCoTrainingBaseConfig, FrankaCoTrainingBaseEnv


@dataclass
class FrankaPickAndPlaceConfig(FrankaCoTrainingBaseConfig):
    """Configuration for the real-world tray-to-tray pick-and-place task."""

    reset_ee_pose: np.ndarray | None = None
    random_xy_range: float = 0.015
    random_z_range: float = 0.01
    random_rz_range: float = 0.35
    step_frequency: float = 10.0
    max_num_steps: int = 120
    gripper_min_command_interval: float = 0.0
    gripper_config: dict[str, float] = field(
        default_factory=lambda: {
            "open_width": 0.08,
            "open_speed": 0.3,
            "close_width": 0.01,
            "close_speed": 0.3,
            "close_force": 130.0,
            "epsilon_inner": 0.005,
            "epsilon_outer": 0.06,
        }
    )
    enable_downward_contact_failure: bool = False
    contact_force_threshold: float = 5.0
    contact_force_confirmation_samples: int = 1
    contact_downward_action_threshold: float = 0.05
    contact_monitor_interval: float = 0.02
    contact_retreat_distance: float = 0.05
    contact_retreat_timeout: float = 0.5
    task_description: str = (
        "Pick up the purple cube from the yellow-marked source tray and place "
        "it in the white target tray"
    )
    use_target_controller: bool = True
    enable_gripper_penalty: bool = False
    enable_inner_safety_box: bool = False
    joint_reset_qpos: list[float] = field(
        default_factory=lambda: [
            0.0,
            np.pi / 8,
            0.0,
            -np.pi * 5 / 8,
            0.0,
            np.pi * 3 / 4,
            np.pi / 4,
        ]
    )

    def __post_init__(self):
        """Preserve an explicitly configured PnP reset pose."""
        reset_ee_pose = self.reset_ee_pose
        super().__post_init__()
        if self.contact_force_threshold <= 0:
            raise ValueError("contact_force_threshold must be positive.")
        if self.contact_force_confirmation_samples < 1:
            raise ValueError("contact_force_confirmation_samples must be positive.")
        if self.contact_downward_action_threshold < 0:
            raise ValueError("contact_downward_action_threshold must be non-negative.")
        if self.gripper_min_command_interval < 0:
            raise ValueError("gripper_min_command_interval must be non-negative.")
        if self.contact_monitor_interval <= 0:
            raise ValueError("contact_monitor_interval must be positive.")
        if self.contact_retreat_distance <= 0:
            raise ValueError("contact_retreat_distance must be positive.")
        if self.contact_retreat_timeout <= 0:
            raise ValueError("contact_retreat_timeout must be positive.")
        if reset_ee_pose is not None:
            reset_ee_pose = np.asarray(reset_ee_pose, dtype=np.float64).reshape(-1)
            if reset_ee_pose.size != 6:
                raise ValueError(
                    "reset_ee_pose must contain 6 values [x, y, z, rx, ry, rz]."
                )
            self.reset_ee_pose = reset_ee_pose


class FrankaPickAndPlaceEnv(FrankaCoTrainingBaseEnv):
    """Real Franka PnP environment with human-provided task rewards."""

    CONFIG_CLS = FrankaPickAndPlaceConfig

    def __init__(self, override_cfg, worker_info=None, hardware_info=None, env_idx=0):
        super().__init__(override_cfg, worker_info, hardware_info, env_idx)
        self._contact_force_baseline: np.ndarray | None = None
        state_space = self.observation_space["state"]
        self.observation_space["state"] = gym.spaces.Dict(
            {
                "arm_joint_position": state_space["arm_joint_position"],
                "tcp_pose": state_space["tcp_pose"],
                "gripper_open_state": gym.spaces.Box(
                    -1.0, 1.0, shape=(1,), dtype=np.float32
                ),
            }
        )
        self._base_observation_space = copy.deepcopy(self.observation_space)

    def _get_observation(self) -> dict:
        observation = super()._get_observation()
        if "state" not in observation:
            return observation

        state = observation["state"]
        observation["state"] = {
            "arm_joint_position": np.asarray(
                state["arm_joint_position"], dtype=np.float32
            ),
            "tcp_pose": np.asarray(state["tcp_pose"], dtype=np.float32),
            "gripper_open_state": np.asarray(
                [1.0 if self._franka_state.gripper_open else -1.0],
                dtype=np.float32,
            ),
        }
        return observation

    def _crop_frame(self, name: str, image: np.ndarray) -> np.ndarray:
        """Apply the centered square crop used during PnP collection."""
        del name
        height, width = image.shape[:2]
        crop_size = min(height, width)
        start_y = (height - crop_size) // 2
        start_x = (width - crop_size) // 2
        return image[
            start_y : start_y + crop_size,
            start_x : start_x + crop_size,
        ]

    def _calc_step_reward(
        self, observation, is_gripper_action_effective=False
    ) -> float:
        """Return zero; task completion is judged only by a human operator."""
        self._success_hold_counter = 0
        return 0.0

    def reset(self, joint_reset=False, **kwargs):
        self._reset_pose = self._build_target_centered_reset_pose()
        observation, info = FrankaEnv.reset(self, joint_reset=joint_reset, **kwargs)
        self._capture_contact_force_baseline()
        return observation, info

    def refresh_observation(self) -> dict:
        """Capture state and images after the operator finishes object reset."""
        if not self.config.is_dummy:
            self._franka_state = self._controller.get_state().wait()[0]
        self._capture_contact_force_baseline()
        return self._get_observation()

    def step(self, action: np.ndarray):
        """End the episode as a trainable failure after downward hard contact."""
        observation, reward, terminated, truncated, info = super().step(action)
        confirmed_force = self._confirm_downward_contact(
            action, self._contact_force_delta_norm()
        )
        if confirmed_force is None:
            return observation, reward, terminated, truncated, info

        commanded_z = float(np.asarray(action).reshape(-1)[2])
        force_delta_norm = float(confirmed_force)
        retreat_distance = self._retreat_from_contact()
        self._logger.warning(
            "PnP downward contact detected during policy motion: "
            "action_z=%.4f force_delta=%.3fN threshold=%.3fN samples=%d; "
            "retreated and ending the episode as failure.",
            commanded_z,
            force_delta_norm,
            self.config.contact_force_threshold,
            self.config.contact_force_confirmation_samples,
        )
        observation = self._get_observation()
        info = dict(info)
        info.update(
            {
                "safety_collision_failure": True,
                "safety_collision_reason": "downward_contact",
                "contact_force_delta_norm": force_delta_norm,
                "contact_force_threshold": float(self.config.contact_force_threshold),
                "contact_commanded_z": commanded_z,
                "contact_retreat_distance": retreat_distance,
                "discard_trajectory": False,
            }
        )
        return observation, 0.0, True, False, info

    def _confirm_downward_contact(
        self, action: np.ndarray, initial_force_delta_norm: float
    ) -> float | None:
        if not self._is_downward_contact(action, initial_force_delta_norm):
            return None

        maximum_force = initial_force_delta_norm
        for _ in range(1, int(self.config.contact_force_confirmation_samples)):
            time.sleep(float(self.config.contact_monitor_interval))
            self._franka_state = self._controller.get_state().wait()[0]
            force_delta_norm = self._contact_force_delta_norm()
            if not self._is_downward_contact(action, force_delta_norm):
                return None
            maximum_force = max(maximum_force, force_delta_norm)
        return maximum_force

    def _capture_contact_force_baseline(self) -> None:
        """Record the current TCP force as the episode-relative baseline."""
        self._contact_force_baseline = (
            np.asarray(self._franka_state.tcp_force, dtype=np.float64)
            .reshape(-1)[:3]
            .copy()
        )

    def _contact_force_delta_norm(self) -> float:
        """Return the translational force change from the reset-ready state."""
        current_force = np.asarray(
            self._franka_state.tcp_force, dtype=np.float64
        ).reshape(-1)[:3]
        if self._contact_force_baseline is None:
            self._contact_force_baseline = current_force.copy()
            return 0.0
        return float(np.linalg.norm(current_force - self._contact_force_baseline))

    def _is_downward_action(self, action: np.ndarray) -> bool:
        """Return whether the policy commands a meaningful downward motion."""
        if not self.config.enable_downward_contact_failure:
            return False
        flat_action = np.asarray(action).reshape(-1)
        if flat_action.size < 3:
            raise ValueError("PnP action must contain at least xyz components.")
        return flat_action[2] <= -float(self.config.contact_downward_action_threshold)

    def _is_downward_contact(self, action: np.ndarray, force_delta_norm: float) -> bool:
        """Check for hard contact during any downward policy motion."""
        return (
            self._franka_state.gripper_open
            and self._is_downward_action(action)
            and force_delta_norm >= float(self.config.contact_force_threshold)
        )

    def _retreat_from_contact(self) -> float:
        """Cancel the stale downward target and raise from the measured pose."""
        measured_pose = np.asarray(self._franka_state.tcp_pose, dtype=np.float64).copy()

        self._target_pose = measured_pose.copy()
        self.next_position = measured_pose.copy()
        self._move_action(measured_pose)

        retreat_pose = measured_pose.copy()
        retreat_pose[2] += float(self.config.contact_retreat_distance)
        retreat_pose = self._clip_position_to_safety_box(retreat_pose)
        self._interpolate_move(
            retreat_pose, timeout=float(self.config.contact_retreat_timeout)
        )

        final_pose = np.asarray(self._franka_state.tcp_pose, dtype=np.float64).copy()
        self._target_pose = final_pose
        self.next_position = final_pose.copy()
        return float(final_pose[2] - measured_pose[2])

    def _build_target_centered_reset_pose(self) -> np.ndarray:
        reset_ee_pose = np.asarray(self.config.reset_ee_pose, dtype=np.float64).reshape(
            -1
        )
        if reset_ee_pose.size != 6:
            raise ValueError(
                "reset_ee_pose must contain 6 values [x, y, z, rx, ry, rz]."
            )
        return np.concatenate(
            [
                reset_ee_pose[:3],
                R.from_euler("xyz", reset_ee_pose[3:].copy()).as_quat(),
            ]
        )
