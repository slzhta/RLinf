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
        return FrankaEnv.reset(self, joint_reset=joint_reset, **kwargs)

    def refresh_observation(self) -> dict:
        """Capture state and images after the operator finishes object reset."""
        if not self.config.is_dummy:
            self._franka_state = self._controller.get_state().wait()[0]
        return self._get_observation()

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
