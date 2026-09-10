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

"""Glued U-tool insertion: no gripper commands, target-relative state and reward."""

import copy
import time
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from rlinf.envs.peg_insertion_geometry import (
    PegInsertionGeometry,
    matrix_pose,
    pose_matrix,
)

from ..franka_env import FrankaRobotConfig
from .co_training_base_env import FrankaCoTrainingBaseConfig, FrankaCoTrainingBaseEnv


@dataclass
class FrankaCoTrainingPegInsertionConfig(FrankaCoTrainingBaseConfig):
    peg_config: dict = field(default_factory=dict)
    max_contact_force: float = 40.0
    max_contact_torque: float = 6.0
    max_num_steps: int = 120
    task_description: str = (
        "Insert the green U-shaped peg into the matching hole in the blue board"
    )

    def __post_init__(self):
        geometry = PegInsertionGeometry(**self.peg_config)
        if not self.is_dummy and self.camera_serials is not None:
            if not self.camera_serials or any(
                not str(serial) or str(serial).startswith("REPLACE_")
                for serial in self.camera_serials
            ):
                raise ValueError(
                    "Set the real wrist camera serial before starting hardware."
                )
        for name in (
            "max_contact_force",
            "max_contact_torque",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        self.target_ee_pose = geometry.target_ee_pose
        self.reset_ee_pose = geometry.target_ee_pose.copy()
        self.reset_ee_pose[:3] = (
            geometry.target[:3, 3]
            + geometry.insertion_rotation[:, 2] * geometry.reset_height
        ).tolist()
        self.action_scale = [*geometry.action_scale, 0.0]
        self.success_hold_steps = geometry.success_hold_steps
        self.enable_inner_safety_box = False
        self.enable_gripper_penalty = False
        self.use_reward_model = False
        self.use_target_controller = True
        self.ee_pose_limit_min = np.asarray(self.target_ee_pose) - [
            0.3,
            0.3,
            0.3,
            0.5,
            0.5,
            0.5,
        ]
        self.ee_pose_limit_max = np.asarray(self.target_ee_pose) + [
            0.3,
            0.3,
            0.3,
            0.5,
            0.5,
            0.5,
        ]
        self.compliance_param = FrankaCoTrainingBaseConfig().compliance_param
        self.precision_param = self.compliance_param.copy()
        FrankaRobotConfig.__post_init__(self)


class FrankaCoTrainingPegInsertionEnv(FrankaCoTrainingBaseEnv):
    CONFIG_CLS = FrankaCoTrainingPegInsertionConfig

    def __init__(self, *args, **kwargs):
        self._safety_fault = None
        super().__init__(*args, **kwargs)
        if self.config.is_dummy:
            self._franka_state.tcp_pose = matrix_pose(self.geometry.target)

    @property
    def geometry(self):
        return PegInsertionGeometry(**self.config.peg_config)

    def _init_action_obs_spaces(self):
        super()._init_action_obs_spaces()
        self.action_space = gym.spaces.Box(-1, 1, (6,), dtype=np.float32)
        self.observation_space["state"]["ee_target_delta"] = gym.spaces.Box(
            -np.inf, np.inf, (6,), dtype=np.float32
        )
        self._base_observation_space = copy.deepcopy(self.observation_space)

    def _get_observation(self):
        observation = super()._get_observation()
        if self.config.is_dummy:
            observation["state"]["tcp_pose"] = self._franka_state.tcp_pose.copy()
        observation["state"]["ee_target_delta"] = self.geometry.relative_state(
            pose_matrix(self._franka_state.tcp_pose)
        )
        return observation

    def _gripper_action(self, *args, **kwargs):
        return False

    def _checked_state(self):
        if self._safety_fault is not None:
            raise RuntimeError(self._safety_fault)
        if self.config.is_dummy:
            return self._franka_state
        if not self._controller.is_robot_up().wait()[0]:
            self._safety_fault = (
                "Insertion controller is unavailable; manual recovery required."
            )
            raise RuntimeError(self._safety_fault)
        state = self._controller.get_state().wait()[0]
        values = np.r_[state.tcp_pose, state.tcp_force, state.tcp_torque]
        if not np.isfinite(values).all() or np.linalg.norm(state.tcp_pose[3:]) < 0.9:
            self._safety_fault = (
                "Invalid robot state; refusing insertion/reset commands."
            )
            raise RuntimeError(self._safety_fault)
        if (
            np.linalg.norm(state.tcp_force) > self.config.max_contact_force
            or np.linalg.norm(state.tcp_torque) > self.config.max_contact_torque
        ):
            self._safety_fault = (
                "Insertion force/torque limit exceeded; manual inspection required."
            )
            # Cancel the accumulated position target; this is not a hardware emergency stop.
            self._controller.move_arm(state.tcp_pose.astype(np.float32)).wait()
            raise RuntimeError(self._safety_fault)
        self._franka_state = state
        return state

    def _move_action(self, position):
        self._checked_state()
        if self.config.is_dummy:
            self._franka_state.tcp_pose = np.asarray(position).copy()
            return
        self._clear_error()
        self._controller.move_arm(np.asarray(position, dtype=np.float32)).wait()

    def _clip_position_to_safety_box(self, pose):
        return matrix_pose(self.geometry.clip_target(pose_matrix(pose)))

    def _initialize_robot_pose(self):
        self._controller.reconfigure_compliance_params(
            self.config.compliance_param
        ).wait()
        super()._initialize_robot_pose()

    def go_to_rest(self, joint_reset=False):
        if joint_reset:
            raise ValueError(
                "Joint reset is not supported with the glued insertion tool."
            )
        geometry = self.geometry
        if self.config.is_dummy:
            self._franka_state.tcp_pose = matrix_pose(
                geometry.reset_pose(self.np_random)
            )
            return
        current = pose_matrix(self._checked_state().tcp_pose)
        clipped = geometry.clip_target(current)
        if (
            np.linalg.norm(current[:3, 3] - clipped[:3, 3]) > 0.003
            or Rotation.from_matrix(current[:3, :3] @ clipped[:3, :3].T).magnitude()
            > 0.03
        ):
            raise RuntimeError(
                "Move the tool manually into the calibrated insertion workspace before startup/reset."
            )
        self._move_action(self._franka_state.tcp_pose)
        time.sleep(0.5)
        self._franka_state = self._controller.get_state().wait()[0]
        lift_pose = self._franka_state.tcp_pose.copy()
        lift_pose[2] += 0.10
        self._interpolate_move(lift_pose, timeout=1)

        reset_pose = matrix_pose(geometry.reset_pose(self.np_random))
        self._franka_state = self._controller.get_state().wait()[0]
        count = 0
        while not np.allclose(self._franka_state.tcp_pose[:3], reset_pose[:3], 0.02):
            count += 1
            self._interpolate_move(reset_pose)
            self._franka_state = self._controller.get_state().wait()[0]
            if count > 2:
                break

    def reset(self, joint_reset=False, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        self._checked_state()
        if not self.config.is_dummy:
            self._controller.reconfigure_compliance_params(
                self.config.compliance_param
            ).wait()
        self.go_to_rest(joint_reset=joint_reset)
        if not self.config.is_dummy:
            self._clear_error()
            self._franka_state = self._controller.get_state().wait()[0]
        self._num_steps = 0
        self._success_hold_counter = 0
        self._target_pose = self._franka_state.tcp_pose.copy()
        return self._get_observation(), {}

    def step(self, action):
        start = time.monotonic()
        action = np.asarray(action, dtype=float)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("Peg insertion expects a finite six-dimensional action.")
        self._checked_state()
        previous = (
            self._franka_state.tcp_pose
            if self._target_pose is None
            else self._target_pose
        )
        self._target_pose = matrix_pose(
            self.geometry.action_target(pose_matrix(previous), action)
        )
        self._move_action(self._target_pose)
        self._num_steps += 1
        if not self.config.is_dummy:
            time.sleep(
                max(0, 1 / self.config.step_frequency - (time.monotonic() - start))
            )
        self._checked_state()
        metrics = self.geometry.metrics(pose_matrix(self._franka_state.tcp_pose))
        self._success_hold_counter = (
            self._success_hold_counter + 1 if metrics["in_target"] else 0
        )
        success = self._success_hold_counter >= self.geometry.success_hold_steps
        reward = 1.0 if success else float(metrics["dense_reward"])
        return (
            self._get_observation(),
            reward,
            success,
            self._num_steps >= self.config.max_num_steps,
            {**metrics, "success": success},
        )
