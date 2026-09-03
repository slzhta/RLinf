# Copyright 2025 The RLinf Authors.
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

"""Franka parallel-jaw gripper controlled via ROS topics.

This module encapsulates the ROS channel setup and message publishing that
were previously embedded in :class:`FrankaController`.
"""

import numpy as np

from rlinf.utils.logging import get_logger

from .base_gripper import BaseGripper


class FrankaGripper(BaseGripper):
    """Franka Emika parallel-jaw gripper (ROS-based).

    Communication uses three ROS topics:

    * ``/franka_gripper/move/goal``   – move to a given width
    * ``/franka_gripper/grasp/goal``  – grasp with configurable force
    * ``/franka_gripper/joint_states`` – finger-joint state feedback

    Args:
        ros: An initialised :class:`ROSController` instance (shared with the
            arm controller).
    """

    def __init__(
        self,
        ros,
        open_width: float = 0.09,
        open_speed: float = 0.3,
        close_width: float = 0.01,
        close_speed: float = 0.3,
        close_force: float = 130.0,
        epsilon_inner: float = 1.0,
        epsilon_outer: float = 1.0,
    ):
        from franka_gripper.msg import GraspActionGoal, MoveActionGoal
        from sensor_msgs.msg import JointState

        self._logger = get_logger()
        self._ros = ros
        self._GraspActionGoal = GraspActionGoal
        self._MoveActionGoal = MoveActionGoal

        self._position_value: float = 0.0
        self._is_open_flag: bool = False
        self._is_ready_flag: bool = False
        self._open_width = float(open_width)
        self._open_speed = float(open_speed)
        self._close_width = float(close_width)
        self._close_speed = float(close_speed)
        self._close_force = float(close_force)
        self._epsilon_inner = float(epsilon_inner)
        self._epsilon_outer = float(epsilon_outer)

        # ROS channels
        self._move_channel = "/franka_gripper/move/goal"
        self._grasp_channel = "/franka_gripper/grasp/goal"
        self._state_channel = "/franka_gripper/joint_states"

        self._ros.create_ros_channel(self._move_channel, MoveActionGoal, queue_size=1)
        self._ros.create_ros_channel(self._grasp_channel, GraspActionGoal, queue_size=1)
        self._ros.connect_ros_channel(
            self._state_channel, JointState, self._on_state_msg
        )

    # ── BaseGripper interface ────────────────────────────────────────

    def open(self, speed: float | None = None) -> None:
        msg = self._MoveActionGoal()
        msg.goal.width = self._open_width
        msg.goal.speed = self._open_speed if speed is None else speed
        self._logger.info(
            "Publishing Franka gripper open command: width=%.4f speed=%.4f",
            msg.goal.width,
            msg.goal.speed,
        )
        self._ros.put_channel(self._move_channel, msg)

    def close(
        self, speed: float | None = None, force: float | None = None
    ) -> None:
        msg = self._GraspActionGoal()
        msg.goal.width = self._close_width
        msg.goal.speed = self._close_speed if speed is None else speed
        msg.goal.epsilon.inner = self._epsilon_inner
        msg.goal.epsilon.outer = self._epsilon_outer
        msg.goal.force = self._close_force if force is None else force
        self._logger.info(
            "Publishing Franka gripper close command: width=%.4f speed=%.4f "
            "force=%.2f epsilon=(%.4f, %.4f)",
            msg.goal.width,
            msg.goal.speed,
            msg.goal.force,
            msg.goal.epsilon.inner,
            msg.goal.epsilon.outer,
        )
        self._ros.put_channel(self._grasp_channel, msg)

    def move(self, position: float, speed: float = 0.3) -> None:
        msg = self._MoveActionGoal()
        msg.goal.width = float(position / 255 * self._open_width)
        msg.goal.speed = speed
        self._ros.put_channel(self._move_channel, msg)

    @property
    def position(self) -> float:
        return self._position_value

    @property
    def is_open(self) -> bool:
        return self._is_open_flag

    def is_ready(self) -> bool:
        return self._ros.get_input_channel_status(self._state_channel)

    # ── ROS callback ─────────────────────────────────────────────────

    def _on_state_msg(self, msg) -> None:
        self._position_value = np.sum(msg.position)
        open_threshold = 0.5 * (self._open_width + self._close_width)
        is_open = self._position_value >= open_threshold
        if not self._is_ready_flag or is_open != self._is_open_flag:
            self._logger.info(
                "Franka gripper state updated from joint feedback: "
                "width=%.6f open=%s threshold=%.6f",
                self._position_value,
                is_open,
                open_threshold,
            )
        self._is_open_flag = bool(is_open)
        self._is_ready_flag = True
