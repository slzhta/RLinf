from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import sapien
import torch
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from sapien.physx import PhysxMaterial

from rlinf.envs.maniskill.tasks.digital_twin.digital_twin_based_env import (
    DigitalTwinBaseEnv,
)
from rlinf.envs.maniskill.tasks.digital_twin.push_button import PushButtonEnv


@register_env("PickAndPlaceDigitalTwin-v1", max_episode_steps=120)
class PickAndPlaceDigitalTwinEnv(PushButtonEnv):
    """Digital-twin tabletop pick-and-place task with simple geometric objects."""

    CUBE_HALF_SIZE = 0.025
    CUBE_COLOR = [0.49019608, 0.45098039, 0.81960784, 1.0]
    TAPE_COLOR = [0.74117647, 0.54509804, 0.10980392, 1.0]
    PLATE_COLOR = [0.94901961, 0.94509804, 0.93333333, 1.0]
    PLATE_LENGTH = 0.24
    PLATE_WIDTH = 0.175
    PLATE_DEPTH = 0.015
    PLATE_BOTTOM_THICKNESS = 0.002
    PLATE_WALL_THICKNESS = 0.005
    PLATE_CORNER_RADIUS = 0.03
    PLATE_TOP_RIM_WIDTH = 0.005
    TAPE_WIDTH = 0.04
    TAPE_LENGTH = 0.125
    TAPE_THICKNESS = 0.0005
    DEFAULT_TRAY_GAP = 0.005
    DEFAULT_TRAY_RANDOM_XY_RANGE = 0.0
    DEFAULT_RESET_POSITION_RANDOM_RANGE = np.array([0.04, 0.04, 0.03], dtype=np.float32)
    DEFAULT_OBJECT_TRAY_MARGIN = 0.005
    DEFAULT_OBJECT_RANDOM_IN_SOURCE_TRAY = True
    DEFAULT_OBJECT_RANDOM_YAW_RANGE = np.pi
    CUBE_DENSITY = 800.0
    CUBE_STATIC_FRICTION = 1.2
    CUBE_DYNAMIC_FRICTION = 1.0
    DEFAULT_USE_DENSE_REWARD = False
    DEFAULT_SPARSE_GRASP_REWARD = 0.1
    DEFAULT_SPARSE_LIFT_REWARD = 0.2
    DEFAULT_SPARSE_PLACE_REWARD = 0.3
    DEFAULT_SPARSE_SUCCESS_REWARD = 1.0
    DEFAULT_SPARSE_DROP_PENALTY = -0.2
    GOAL_HEIGHT_THRESHOLD = 0.025
    DEFAULT_SUCCESS_STABILITY_STEPS = 3
    DEFAULT_SUCCESS_MAX_LINEAR_SPEED = 0.05
    DEFAULT_SUCCESS_MAX_ANGULAR_SPEED = 0.5
    DEFAULT_FAILURE_MIN_Z = -0.02
    DEFAULT_FAILURE_MAX_XY_DISTANCE = 0.4
    LIFT_HEIGHT = 0.055
    OPEN_GRIPPER_QPOS = 0.04
    DEFAULT_OBJECT_RANDOM_XY_RANGE = 0.01
    TASK_DESCRIPTION = (
        "Pick up the purple cube from the yellow-marked source tray and place "
        "it in the white target tray"
    )
    DEFAULT_TRAY_GROUP_CENTER = np.array(
        [-0.12, 0.01, PLATE_DEPTH / 2.0],
        dtype=np.float32,
    )
    DEFAULT_TRAY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    DEFAULT_OBJECT_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reset_sparse_reward_state()
        self._reset_episode_metric_state()

    def _load_task_scene(self, options: dict):
        self.cube = self._build_cube(
            initial_position=self._default_cube_position_np(
                self._default_source_tray_position_np()
            )
        )
        self.source_tray = self._build_tray(
            name="taped_source_tray",
            has_tape=True,
            initial_position=self._default_source_tray_position_np(),
        )
        self.target_tray = self._build_tray(
            name="plain_target_tray",
            has_tape=False,
            initial_position=self._default_target_tray_position_np(),
        )

    def _get_foreground_actors(self):
        return [self.cube, self.source_tray, self.target_tray]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        DigitalTwinBaseEnv._initialize_episode(self, env_idx, options)

        source_tray_pose, target_tray_pose = self._sample_tray_poses(env_idx)
        self.source_tray.set_pose(source_tray_pose)
        self.target_tray.set_pose(target_tray_pose)
        self.cube.set_pose(self._sample_cube_pose(env_idx, source_tray_pose))

        qpos = self.agent.robot.get_qpos().clone()
        reset_ee_pose = self._sample_reset_ee_pose(
            env_idx=env_idx,
            device=qpos.device,
            dtype=qpos.dtype,
        )
        ik_qpos = self._solve_arm_ik_qpos(
            target_pose=reset_ee_pose,
            seed_qpos=qpos[env_idx],
            env_idx=env_idx,
            device=qpos.device,
            dtype=qpos.dtype,
        )
        qpos[env_idx, :7] = ik_qpos
        qpos[env_idx, -2:] = self.OPEN_GRIPPER_QPOS
        self.agent.reset(qpos[env_idx])
        self.agent.robot.set_pose(sapien.Pose(self.ROBOT_INITIAL_POSITION))
        self.sync_gpu_articulation_state()
        self._reset_sparse_reward_state(env_idx)
        self._reset_episode_metric_state(env_idx)

    def _sample_reset_ee_pose(
        self, env_idx: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> Pose:
        controller_cfg = self.controller_alignment

        target_ee_pose = np.asarray(
            controller_cfg.get(
                "target_ee_pose",
                [0.5, 0.0, 0.1, -3.14, 0.0, 0.0],
            ),
            dtype=np.float32,
        ).reshape(-1)
        if target_ee_pose.size != 6:
            raise ValueError(
                "controller_alignment.target_ee_pose must contain 6 values "
                "[x, y, z, rx, ry, rz]."
            )

        reset_ee_pose_cfg = controller_cfg.get("reset_ee_pose", None)
        if reset_ee_pose_cfg is not None:
            reset_ee_pose = np.asarray(reset_ee_pose_cfg, dtype=np.float32).reshape(-1)
            if reset_ee_pose.size != 6:
                raise ValueError(
                    "controller_alignment.reset_ee_pose must contain 6 values "
                    "[x, y, z, rx, ry, rz]."
                )
        else:
            clip_z_range_high = float(controller_cfg.get("clip_z_range_high", 0.3))
            clip_z_range_low = float(controller_cfg.get("clip_z_range_low", 0.001))
            reset_ee_pose = target_ee_pose.copy()
            reset_ee_pose[2] += 0.5 * (clip_z_range_high + clip_z_range_low)

        b = len(env_idx)
        position = np.repeat(reset_ee_pose[:3][None, :], b, axis=0)
        euler = np.repeat(reset_ee_pose[3:][None, :], b, axis=0)

        enable_random_reset = bool(controller_cfg.get("enable_random_reset", True))
        random_position_range = self._get_reset_position_random_range(controller_cfg)
        random_rz_range = float(
            controller_cfg.get("random_rz_range", self.DEFAULT_RESET_RANDOM_RZ_RANGE)
        )
        ee_limit_min, ee_limit_max = self._get_ee_pose_limits(controller_cfg)
        target_rz = float(target_ee_pose[5])

        if enable_random_reset:
            for i, scene_idx_t in enumerate(env_idx):
                scene_idx = int(scene_idx_t.item())
                rng = self._batched_episode_rng[scene_idx]
                position[i] += rng.uniform(
                    -random_position_range, random_position_range, size=(3,)
                )
                euler[i, 2] = target_rz + rng.uniform(-random_rz_range, random_rz_range)
        else:
            euler[:, 2] = target_rz

        if ee_limit_min is not None and ee_limit_max is not None:
            position = np.clip(position, ee_limit_min[:3], ee_limit_max[:3])
            euler = np.clip(euler, ee_limit_min[3:], ee_limit_max[3:])

        position_t = torch.as_tensor(position, device=device, dtype=dtype)
        euler_t = torch.as_tensor(euler, device=device, dtype=dtype)
        quat_t = matrix_to_quaternion(euler_angles_to_matrix(euler_t, "XYZ")).to(
            device=device, dtype=dtype
        )
        return Pose.create_from_pq(p=position_t, q=quat_t)

    def _get_reset_position_random_range(
        self, controller_cfg: dict[str, Any]
    ) -> np.ndarray:
        random_range = np.asarray(
            controller_cfg.get(
                "reset_position_random_range",
                self.DEFAULT_RESET_POSITION_RANDOM_RANGE,
            ),
            dtype=np.float32,
        ).reshape(-1)
        if random_range.size == 1:
            random_range = np.repeat(random_range, 3)
        if random_range.size != 3:
            raise ValueError(
                "controller_alignment.reset_position_random_range must contain "
                "1 or 3 values."
            )
        if np.any(random_range < 0):
            raise ValueError(
                "controller_alignment.reset_position_random_range must be >= 0."
            )
        return random_range

    @staticmethod
    def _get_ee_pose_limits(
        controller_cfg: dict[str, Any],
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        limit_min_cfg = controller_cfg.get("ee_pose_limit_min", None)
        limit_max_cfg = controller_cfg.get("ee_pose_limit_max", None)
        if limit_min_cfg is None or limit_max_cfg is None:
            return None, None

        limit_min = np.asarray(limit_min_cfg, dtype=np.float32).reshape(-1)
        limit_max = np.asarray(limit_max_cfg, dtype=np.float32).reshape(-1)
        if limit_min.size != 6 or limit_max.size != 6:
            raise ValueError(
                "controller_alignment.ee_pose_limit_min and ee_pose_limit_max "
                "must both contain 6 values [x, y, z, rx, ry, rz]."
            )
        if np.any(limit_min > limit_max):
            raise ValueError(
                "Each controller_alignment.ee_pose_limit_min value must be <= "
                "ee_pose_limit_max."
            )
        return limit_min, limit_max

    @classmethod
    def _floor_top_local_z(cls) -> float:
        return -cls.PLATE_DEPTH / 2.0 + cls.PLATE_BOTTOM_THICKNESS

    @classmethod
    def _cube_center_local_z(cls) -> float:
        return cls._floor_top_local_z() + cls.CUBE_HALF_SIZE

    @classmethod
    def _default_tray_spacing(cls) -> float:
        return cls.PLATE_WIDTH + cls.DEFAULT_TRAY_GAP

    @classmethod
    def _default_source_tray_position_np(cls) -> np.ndarray:
        position = cls.DEFAULT_TRAY_GROUP_CENTER.copy()
        position[1] += cls._default_tray_spacing() / 2.0
        return position

    @classmethod
    def _default_target_tray_position_np(cls) -> np.ndarray:
        position = cls.DEFAULT_TRAY_GROUP_CENTER.copy()
        position[1] -= cls._default_tray_spacing() / 2.0
        return position

    @classmethod
    def _default_cube_position_np(cls, source_tray_position: np.ndarray) -> np.ndarray:
        position = np.asarray(source_tray_position, dtype=np.float32).copy()
        position[2] += cls._cube_center_local_z()
        return position

    def _sample_tray_poses(self, env_idx: torch.Tensor) -> tuple[Pose, Pose]:
        source_pose_cfg = self.task_alignment.get("source_tray_initial_pose", None)
        target_pose_cfg = self.task_alignment.get("target_tray_initial_pose", None)
        if source_pose_cfg is not None or target_pose_cfg is not None:
            source_pose = self._sample_explicit_pose(
                env_idx=env_idx,
                key="source_tray_initial_pose",
                default_pose=np.hstack(
                    [self._default_source_tray_position_np(), self.DEFAULT_TRAY_QUAT]
                ),
                random_xy_key="tray_random_xy_range",
                default_random_xy_range=self.DEFAULT_TRAY_RANDOM_XY_RANGE,
            )
            target_pose = self._sample_explicit_pose(
                env_idx=env_idx,
                key="target_tray_initial_pose",
                default_pose=np.hstack(
                    [self._default_target_tray_position_np(), self.DEFAULT_TRAY_QUAT]
                ),
                random_xy_key="tray_random_xy_range",
                default_random_xy_range=self.DEFAULT_TRAY_RANDOM_XY_RANGE,
            )
            return source_pose, target_pose

        group_center = self._get_tray_group_center_np()
        tray_quat = self._get_tray_quat_np()
        spacing = float(
            self.task_alignment.get("tray_center_spacing", self._default_tray_spacing())
        )
        source_offset = np.array([0.0, spacing / 2.0, 0.0], dtype=np.float32)
        target_offset = np.array([0.0, -spacing / 2.0, 0.0], dtype=np.float32)
        random_xy_range = float(
            self.task_alignment.get(
                "tray_random_xy_range", self.DEFAULT_TRAY_RANDOM_XY_RANGE
            )
        )

        b = len(env_idx)
        group_position = np.repeat(group_center[None, :], b, axis=0)
        if random_xy_range > 0:
            xy_offset = self._batched_episode_rng[env_idx].uniform(
                -random_xy_range, random_xy_range, size=(2,)
            )
            group_position[:, :2] += xy_offset

        source_position = group_position + source_offset
        target_position = group_position + target_offset
        quat = np.repeat(tray_quat[None, :], b, axis=0)
        source_pose = Pose.create_from_pq(
            p=torch.as_tensor(source_position, device=self.device, dtype=torch.float32),
            q=torch.as_tensor(quat, device=self.device, dtype=torch.float32),
        )
        target_pose = Pose.create_from_pq(
            p=torch.as_tensor(target_position, device=self.device, dtype=torch.float32),
            q=torch.as_tensor(quat, device=self.device, dtype=torch.float32),
        )
        return source_pose, target_pose

    def _sample_cube_pose(self, env_idx: torch.Tensor, source_tray_pose: Pose) -> Pose:
        object_pose_cfg = self.task_alignment.get("object_initial_pose", None)
        if object_pose_cfg is not None:
            return self._sample_explicit_pose(
                env_idx=env_idx,
                key="object_initial_pose",
                default_pose=np.hstack(
                    [
                        self._default_cube_position_np(
                            self._default_source_tray_position_np()
                        ),
                        self.DEFAULT_OBJECT_QUAT,
                    ]
                ),
                random_xy_key="object_random_xy_range",
                default_random_xy_range=self.DEFAULT_OBJECT_RANDOM_XY_RANGE,
            )

        local_offset, local_yaw = self._sample_cube_local_offset_and_yaw(
            env_idx=env_idx,
            device=source_tray_pose.p.device,
            dtype=source_tray_pose.p.dtype,
        )
        tray_rotation = quaternion_to_matrix(source_tray_pose.q)
        world_offset = torch.matmul(tray_rotation, local_offset.unsqueeze(-1)).squeeze(
            -1
        )
        position = source_tray_pose.p + world_offset

        local_euler = torch.zeros(
            (len(env_idx), 3), device=position.device, dtype=position.dtype
        )
        local_euler[:, 2] = local_yaw
        local_rotation = euler_angles_to_matrix(local_euler, "XYZ")
        object_rotation = torch.matmul(tray_rotation, local_rotation)
        quat = matrix_to_quaternion(object_rotation).to(
            device=position.device, dtype=position.dtype
        )
        return Pose.create_from_pq(
            p=position,
            q=quat,
        )

    def _sample_cube_local_offset_and_yaw(
        self, env_idx: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = len(env_idx)
        local_offset = torch.zeros((b, 3), device=device, dtype=dtype)
        local_offset[:, 2] = self._cube_center_local_z()
        local_yaw = torch.zeros((b,), device=device, dtype=dtype)

        sample_in_tray = bool(
            self.task_alignment.get(
                "object_random_in_source_tray",
                self.DEFAULT_OBJECT_RANDOM_IN_SOURCE_TRAY,
            )
        )
        if sample_in_tray:
            xy_low, xy_high = self._get_source_tray_object_xy_bounds()
        else:
            random_xy_range = float(
                self.task_alignment.get(
                    "object_random_xy_range", self.DEFAULT_OBJECT_RANDOM_XY_RANGE
                )
            )
            xy_low = np.array([-random_xy_range, -random_xy_range], dtype=np.float32)
            xy_high = np.array([random_xy_range, random_xy_range], dtype=np.float32)

        yaw_range = float(
            self.task_alignment.get(
                "object_random_yaw_range", self.DEFAULT_OBJECT_RANDOM_YAW_RANGE
            )
        )
        for i, scene_idx_t in enumerate(env_idx):
            scene_idx = int(scene_idx_t.item())
            rng = self._batched_episode_rng[scene_idx]
            xy = rng.uniform(xy_low, xy_high, size=(2,))
            yaw = rng.uniform(-yaw_range, yaw_range)
            local_offset[i, :2] = torch.as_tensor(xy, device=device, dtype=dtype)
            local_yaw[i] = torch.as_tensor(yaw, device=device, dtype=dtype)

        return local_offset, local_yaw

    def _get_source_tray_object_xy_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        margin = float(
            self.task_alignment.get(
                "object_tray_margin", self.DEFAULT_OBJECT_TRAY_MARGIN
            )
        )
        half_size = np.array(
            [
                self.PLATE_LENGTH / 2.0,
                self.PLATE_WIDTH / 2.0,
            ],
            dtype=np.float32,
        )
        clearance = self.PLATE_WALL_THICKNESS + self.CUBE_HALF_SIZE + margin
        half_range = half_size - clearance
        if np.any(half_range <= 0):
            raise ValueError(
                "Source tray is too small for cube randomization. Reduce "
                "task_alignment.object_tray_margin."
            )
        return -half_range, half_range

    def _sample_explicit_pose(
        self,
        env_idx: torch.Tensor,
        key: str,
        default_pose: np.ndarray,
        random_xy_key: str,
        default_random_xy_range: float,
    ) -> Pose:
        pose = np.asarray(
            self.task_alignment.get(key, default_pose),
            dtype=np.float32,
        ).reshape(-1)
        if pose.size != 7:
            raise ValueError(
                f"task_alignment.{key} must contain 7 values [x, y, z, qw, qx, qy, qz]."
            )

        b = len(env_idx)
        position = np.repeat(pose[:3][None, :], b, axis=0)
        random_xy_range = float(
            self.task_alignment.get(random_xy_key, default_random_xy_range)
        )
        if random_xy_range > 0:
            xy_offset = self._batched_episode_rng[env_idx].uniform(
                -random_xy_range, random_xy_range, size=(2,)
            )
            position[:, :2] += xy_offset
        quat = np.repeat(pose[3:][None, :], b, axis=0)
        return Pose.create_from_pq(
            p=torch.as_tensor(position, device=self.device, dtype=torch.float32),
            q=torch.as_tensor(quat, device=self.device, dtype=torch.float32),
        )

    def _get_tray_group_center_np(self) -> np.ndarray:
        group_center = self.task_alignment.get("tray_group_center", None)
        if group_center is None and bool(
            self.task_alignment.get("tray_group_center_from_target_ee_pose", False)
        ):
            group_center = self._button_pose_from_target_ee_pose()[:3]
            group_center = np.asarray(group_center, dtype=np.float32)
            group_center[2] = float(
                self.task_alignment.get("tray_group_center_z", self.PLATE_DEPTH / 2.0)
            )
        if group_center is None:
            group_center = self.DEFAULT_TRAY_GROUP_CENTER

        group_center = np.asarray(group_center, dtype=np.float32).reshape(-1)
        if group_center.size != 3:
            raise ValueError(
                "task_alignment.tray_group_center must contain 3 values [x, y, z]."
            )
        return group_center

    def _get_tray_quat_np(self) -> np.ndarray:
        tray_quat = np.asarray(
            self.task_alignment.get("tray_quat", self.DEFAULT_TRAY_QUAT),
            dtype=np.float32,
        ).reshape(-1)
        if tray_quat.size != 4:
            raise ValueError(
                "task_alignment.tray_quat must contain 4 values [qw, qx, qy, qz]."
            )
        return tray_quat

    def _get_goal_cube_position(self) -> torch.Tensor:
        target_offset = torch.tensor(
            [0.0, 0.0, self._cube_center_local_z()],
            device=self.target_tray.pose.p.device,
            dtype=self.target_tray.pose.p.dtype,
        )
        return self.target_tray.pose.p + target_offset

    def _get_success_half_size(self) -> torch.Tensor:
        success_half_size = [
            self.PLATE_LENGTH / 2.0 - self.PLATE_WALL_THICKNESS - self.CUBE_HALF_SIZE,
            self.PLATE_WIDTH / 2.0 - self.PLATE_WALL_THICKNESS - self.CUBE_HALF_SIZE,
        ]
        return torch.tensor(
            success_half_size,
            device=self.target_tray.pose.p.device,
            dtype=self.target_tray.pose.p.dtype,
        )

    def evaluate(self):
        cube_position = self.cube.pose.p
        goal_position = self._get_goal_cube_position()
        cube_to_goal = goal_position - cube_position
        cube_to_goal_xy_dist = torch.linalg.norm(cube_to_goal[:, :2], axis=1)
        cube_to_goal_height_delta = torch.abs(cube_to_goal[:, 2])
        cube_to_goal_dist = torch.linalg.norm(cube_to_goal, axis=1)
        is_inside_tray = torch.all(
            torch.abs(cube_to_goal[:, :2]) <= self._get_success_half_size(),
            dim=1,
        )
        is_obj_placed = is_inside_tray & (
            cube_to_goal_height_delta <= self.GOAL_HEIGHT_THRESHOLD
        )
        is_grasped = self.agent.is_grasping(self.cube)
        is_obj_released = ~is_grasped
        is_obj_lifted = cube_position[:, 2] >= self.CUBE_HALF_SIZE + self.LIFT_HEIGHT
        cube_linear_speed = torch.linalg.norm(self.cube.linear_velocity, dim=1)
        cube_angular_speed = torch.linalg.norm(self.cube.angular_velocity, dim=1)
        is_obj_stable = (
            cube_linear_speed
            <= float(
                self.task_alignment.get(
                    "success_max_linear_speed",
                    self.DEFAULT_SUCCESS_MAX_LINEAR_SPEED,
                )
            )
        ) & (
            cube_angular_speed
            <= float(
                self.task_alignment.get(
                    "success_max_angular_speed",
                    self.DEFAULT_SUCCESS_MAX_ANGULAR_SPEED,
                )
            )
        )
        success_candidate = is_obj_placed & is_obj_released & is_obj_stable

        if not hasattr(self, "_episode_grasp_once"):
            self._reset_episode_metric_state()
        success = self._update_success_state(success_candidate)

        task_center = 0.5 * (
            self.source_tray.pose.p[:, :2] + self.target_tray.pose.p[:, :2]
        )
        cube_to_task_center_xy = torch.linalg.norm(
            cube_position[:, :2] - task_center, dim=1
        )
        is_obj_unrecoverable = (
            (~torch.isfinite(cube_position).all(dim=1))
            | (
                cube_position[:, 2]
                < float(
                    self.task_alignment.get("failure_min_z", self.DEFAULT_FAILURE_MIN_Z)
                )
            )
            | (
                cube_to_task_center_xy
                > float(
                    self.task_alignment.get(
                        "failure_max_xy_distance",
                        self.DEFAULT_FAILURE_MAX_XY_DISTANCE,
                    )
                )
            )
        )
        fail = is_obj_unrecoverable & (~success)
        dropped_after_grasp = (
            self._episode_grasp_once & (~is_grasped) & (~is_obj_placed) & (~success)
        )
        self._episode_grasp_once |= is_grasped
        self._episode_lift_once |= is_obj_lifted
        self._episode_place_once |= is_obj_placed
        self._episode_drop_once |= dropped_after_grasp
        return {
            "success": success,
            "fail": fail,
            "success_candidate": success_candidate,
            "success_stable_steps": self._success_stable_steps.clone(),
            "is_grasped": is_grasped,
            "is_obj_released": is_obj_released,
            "is_obj_lifted": is_obj_lifted,
            "is_obj_placed": is_obj_placed,
            "is_obj_stable": is_obj_stable,
            "is_obj_unrecoverable": is_obj_unrecoverable,
            "cube_linear_speed": cube_linear_speed,
            "cube_angular_speed": cube_angular_speed,
            "goal_position": goal_position,
            "cube_to_goal": cube_to_goal,
            "cube_to_goal_dist": cube_to_goal_dist,
            "cube_to_goal_xy_dist": cube_to_goal_xy_dist,
            "cube_to_goal_height_delta": cube_to_goal_height_delta,
            "grasp_once": self._episode_grasp_once.clone(),
            "lift_once": self._episode_lift_once.clone(),
            "place_once": self._episode_place_once.clone(),
            "drop_once": self._episode_drop_once.clone(),
        }

    def _build_extracted_obs(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        extracted_obs = super()._build_extracted_obs(raw_obs)
        qpos = self.agent.robot.get_qpos().to(torch.float32)
        gripper_width = qpos[:, -2:].sum(dim=1, keepdim=True)
        max_gripper_width = 2.0 * self.OPEN_GRIPPER_QPOS
        gripper_open_state = (
            2.0 * torch.clamp(gripper_width / max_gripper_width, min=0.0, max=1.0) - 1.0
        )
        extracted_obs["states"] = torch.cat(
            [extracted_obs["states"], gripper_open_state], dim=1
        )
        return extracted_obs

    def _get_obs_extra(self, info: dict):
        obs = {
            "tcp_pose": self.agent.tcp.pose.raw_pose,
            "goal_pos": info["goal_position"],
        }
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp.pose.p,
                obj_to_goal_pos=info["goal_position"] - self.cube.pose.p,
                is_grasped=info["is_grasped"].unsqueeze(-1),
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        use_dense_reward = bool(
            self.task_alignment.get("use_dense_reward", self.DEFAULT_USE_DENSE_REWARD)
        )
        reward_scale = float(self.task_alignment.get("reward_scale", 1.0))
        success = info["success"]
        reward = torch.zeros_like(info["cube_to_goal_dist"])
        if use_dense_reward:
            tcp_to_obj_dist = torch.linalg.norm(
                self.cube.pose.p - self.agent.tcp.pose.p, axis=1
            )
            reaching_reward = 1.0 - torch.tanh(8.0 * tcp_to_obj_dist)
            lift_progress = torch.clamp(
                (self.cube.pose.p[:, 2] - self.CUBE_HALF_SIZE) / self.LIFT_HEIGHT,
                min=0.0,
                max=1.0,
            )
            place_reward = 1.0 - torch.tanh(8.0 * info["cube_to_goal_dist"])
            can_place = info["is_grasped"] | info["is_obj_lifted"]
            release_reward = (info["is_obj_placed"] & info["is_obj_released"]).to(
                reward.dtype
            )
            dense_reward = (
                0.15 * reaching_reward
                + 0.20 * info["is_grasped"].to(reward.dtype)
                + 0.20 * lift_progress
                + 0.35 * place_reward * can_place.to(reward.dtype)
                + 0.10 * release_reward
            )
            reward = dense_reward * reward_scale
            reward[success] = 1.0
        else:
            reward = self._compute_sparse_milestone_reward(info)
        return reward

    def _reset_sparse_reward_state(self, env_idx: torch.Tensor | None = None):
        if not hasattr(self, "_sparse_grasp_rewarded"):
            self._sparse_grasp_rewarded = torch.zeros(
                (self.num_envs,), dtype=torch.bool, device=self.device
            )
            self._sparse_lift_rewarded = torch.zeros_like(self._sparse_grasp_rewarded)
            self._sparse_place_rewarded = torch.zeros_like(self._sparse_grasp_rewarded)
            self._sparse_success_rewarded = torch.zeros_like(
                self._sparse_grasp_rewarded
            )
            self._sparse_drop_penalized = torch.zeros_like(self._sparse_grasp_rewarded)

        if env_idx is None:
            self._sparse_grasp_rewarded[:] = False
            self._sparse_lift_rewarded[:] = False
            self._sparse_place_rewarded[:] = False
            self._sparse_success_rewarded[:] = False
            self._sparse_drop_penalized[:] = False
            return

        self._sparse_grasp_rewarded[env_idx] = False
        self._sparse_lift_rewarded[env_idx] = False
        self._sparse_place_rewarded[env_idx] = False
        self._sparse_success_rewarded[env_idx] = False
        self._sparse_drop_penalized[env_idx] = False

    def _reset_episode_metric_state(self, env_idx: torch.Tensor | None = None):
        if not hasattr(self, "_episode_grasp_once"):
            self._episode_grasp_once = torch.zeros(
                (self.num_envs,), dtype=torch.bool, device=self.device
            )
            self._episode_lift_once = torch.zeros_like(self._episode_grasp_once)
            self._episode_place_once = torch.zeros_like(self._episode_grasp_once)
            self._episode_drop_once = torch.zeros_like(self._episode_grasp_once)
            self._episode_success_once = torch.zeros_like(self._episode_grasp_once)
            self._success_stable_steps = torch.zeros(
                (self.num_envs,), dtype=torch.int32, device=self.device
            )
            self._success_last_checked_step = torch.full(
                (self.num_envs,), -1, dtype=torch.int32, device=self.device
            )

        if env_idx is None:
            self._episode_grasp_once[:] = False
            self._episode_lift_once[:] = False
            self._episode_place_once[:] = False
            self._episode_drop_once[:] = False
            self._episode_success_once[:] = False
            self._success_stable_steps[:] = 0
            self._success_last_checked_step[:] = -1
            return

        self._episode_grasp_once[env_idx] = False
        self._episode_lift_once[env_idx] = False
        self._episode_place_once[env_idx] = False
        self._episode_drop_once[env_idx] = False
        self._episode_success_once[env_idx] = False
        self._success_stable_steps[env_idx] = 0
        self._success_last_checked_step[env_idx] = -1

    def _update_success_state(self, success_candidate: torch.Tensor) -> torch.Tensor:
        """Latch success after a candidate remains stable for full control steps."""
        required_steps = int(
            self.task_alignment.get(
                "success_stability_steps", self.DEFAULT_SUCCESS_STABILITY_STEPS
            )
        )
        if required_steps < 1:
            raise ValueError("task_alignment.success_stability_steps must be >= 1.")

        elapsed_steps = self.elapsed_steps.to(
            device=self.device, dtype=self._success_last_checked_step.dtype
        )
        is_new_control_step = elapsed_steps != self._success_last_checked_step
        next_stable_steps = torch.where(
            success_candidate,
            self._success_stable_steps + 1,
            torch.zeros_like(self._success_stable_steps),
        )
        self._success_stable_steps = torch.where(
            is_new_control_step,
            next_stable_steps,
            self._success_stable_steps,
        )
        self._success_last_checked_step = torch.where(
            is_new_control_step,
            elapsed_steps,
            self._success_last_checked_step,
        )
        self._episode_success_once |= self._success_stable_steps >= required_steps
        return self._episode_success_once.clone()

    def _compute_sparse_milestone_reward(self, info: dict) -> torch.Tensor:
        reward = torch.zeros_like(info["cube_to_goal_dist"])

        newly_grasped = info["is_grasped"] & (~self._sparse_grasp_rewarded)
        self._sparse_grasp_rewarded = self._sparse_grasp_rewarded | info["is_grasped"]
        reward += newly_grasped.to(reward.dtype) * float(
            self.task_alignment.get(
                "sparse_grasp_reward", self.DEFAULT_SPARSE_GRASP_REWARD
            )
        )

        newly_lifted = info["is_obj_lifted"] & (~self._sparse_lift_rewarded)
        self._sparse_lift_rewarded = self._sparse_lift_rewarded | info["is_obj_lifted"]
        reward += newly_lifted.to(reward.dtype) * float(
            self.task_alignment.get(
                "sparse_lift_reward", self.DEFAULT_SPARSE_LIFT_REWARD
            )
        )

        newly_placed = info["is_obj_placed"] & (~self._sparse_place_rewarded)
        self._sparse_place_rewarded = (
            self._sparse_place_rewarded | info["is_obj_placed"]
        )
        reward += newly_placed.to(reward.dtype) * float(
            self.task_alignment.get(
                "sparse_place_reward", self.DEFAULT_SPARSE_PLACE_REWARD
            )
        )

        was_successed = self._sparse_success_rewarded
        newly_successed = info["success"] & (~was_successed)
        dropped_after_grasp = (
            self._sparse_grasp_rewarded
            & (~info["is_grasped"])
            & (~info["is_obj_placed"])
            & (~info["success"])
        )
        newly_dropped = dropped_after_grasp & (~self._sparse_drop_penalized)

        self._sparse_success_rewarded = was_successed | info["success"]
        self._sparse_drop_penalized |= dropped_after_grasp

        reward += newly_successed.to(reward.dtype) * float(
            self.task_alignment.get(
                "sparse_success_reward", self.DEFAULT_SPARSE_SUCCESS_REWARD
            )
        )
        reward += newly_dropped.to(reward.dtype) * float(
            self.task_alignment.get(
                "sparse_drop_penalty", self.DEFAULT_SPARSE_DROP_PENALTY
            )
        )
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)

    def _build_cube(self, initial_position: np.ndarray):
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(
            half_size=[self.CUBE_HALF_SIZE] * 3,
            material=PhysxMaterial(
                static_friction=self.CUBE_STATIC_FRICTION,
                dynamic_friction=self.CUBE_DYNAMIC_FRICTION,
                restitution=0.0,
            ),
            density=self.CUBE_DENSITY,
        )
        builder.add_box_visual(
            half_size=[self.CUBE_HALF_SIZE] * 3,
            material=sapien.render.RenderMaterial(base_color=self.CUBE_COLOR),
        )
        builder.initial_pose = sapien.Pose(
            p=np.asarray(initial_position, dtype=np.float32).tolist(),
            q=self.DEFAULT_OBJECT_QUAT.tolist(),
        )
        return builder.build(name="pick_cube")

    def _build_tray(
        self,
        name: str,
        has_tape: bool,
        initial_position: np.ndarray,
    ):
        builder = self.scene.create_actor_builder()
        collision_material = PhysxMaterial(
            static_friction=1.0,
            dynamic_friction=0.8,
            restitution=0.0,
        )
        bottom_z = -self.PLATE_DEPTH / 2.0
        builder.add_box_collision(
            pose=sapien.Pose(
                p=[0.0, 0.0, bottom_z + self.PLATE_BOTTOM_THICKNESS / 2.0]
            ),
            half_size=[
                self.PLATE_LENGTH / 2.0,
                self.PLATE_WIDTH / 2.0,
                self.PLATE_BOTTOM_THICKNESS / 2.0,
            ],
            material=collision_material,
        )
        self._add_tray_wall_collisions(builder, collision_material)
        builder.add_visual_from_file(
            filename=str(self._get_tray_mesh_path()),
            material=sapien.render.RenderMaterial(base_color=self.PLATE_COLOR),
        )
        if has_tape:
            self._add_tape_strips(builder)
        builder.initial_pose = sapien.Pose(
            p=np.asarray(initial_position, dtype=np.float32).tolist(),
            q=self.DEFAULT_TRAY_QUAT.tolist(),
        )
        return builder.build_kinematic(name=name)

    def _add_tray_wall_collisions(
        self,
        builder,
        collision_material: PhysxMaterial,
    ):
        wall_half_z = self.PLATE_DEPTH / 2.0
        long_wall_half_size = [
            self.PLATE_LENGTH / 2.0,
            self.PLATE_WALL_THICKNESS / 2.0,
            wall_half_z,
        ]
        short_wall_half_size = [
            self.PLATE_WALL_THICKNESS / 2.0,
            self.PLATE_WIDTH / 2.0 - self.PLATE_WALL_THICKNESS,
            wall_half_z,
        ]
        for y_sign in (-1.0, 1.0):
            builder.add_box_collision(
                pose=sapien.Pose(
                    p=[
                        0.0,
                        y_sign
                        * (self.PLATE_WIDTH / 2.0 - self.PLATE_WALL_THICKNESS / 2.0),
                        0.0,
                    ]
                ),
                half_size=long_wall_half_size,
                material=collision_material,
            )
        for x_sign in (-1.0, 1.0):
            builder.add_box_collision(
                pose=sapien.Pose(
                    p=[
                        x_sign
                        * (self.PLATE_LENGTH / 2.0 - self.PLATE_WALL_THICKNESS / 2.0),
                        0.0,
                        0.0,
                    ]
                ),
                half_size=short_wall_half_size,
                material=collision_material,
            )

    def _add_tape_strips(self, builder):
        tape_material = sapien.render.RenderMaterial(base_color=self.TAPE_COLOR)
        tape_collision_material = PhysxMaterial(
            static_friction=1.2,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        tape_center_x = (
            self.PLATE_LENGTH / 2.0 - self.PLATE_WALL_THICKNESS - self.TAPE_WIDTH / 2.0
        )
        tape_z = self._floor_top_local_z() + self.TAPE_THICKNESS / 2.0
        tape_half_size = [
            self.TAPE_WIDTH / 2.0,
            self.TAPE_LENGTH / 2.0,
            self.TAPE_THICKNESS / 2.0,
        ]
        for x_sign in (-1.0, 1.0):
            tape_pose = sapien.Pose(p=[x_sign * tape_center_x, 0.0, tape_z])
            builder.add_box_collision(
                pose=tape_pose,
                half_size=tape_half_size,
                material=tape_collision_material,
            )
            builder.add_box_visual(
                pose=tape_pose,
                half_size=tape_half_size,
                material=tape_material,
            )

    @classmethod
    def _get_tray_mesh_path(cls) -> Path:
        mesh_dir = Path("/tmp/rlinf_digital_twin_assets")
        mesh_dir.mkdir(parents=True, exist_ok=True)
        mesh_path = mesh_dir / "rectangular_tray_24x17p5x1p5.obj"
        if mesh_path.is_file():
            return mesh_path

        tmp_path = mesh_path.with_suffix(".obj.tmp")
        tmp_path.write_text(cls._build_tray_obj(), encoding="utf-8")
        os.replace(tmp_path, mesh_path)
        return mesh_path

    @classmethod
    def _build_tray_obj(cls) -> str:
        outer = cls._rounded_rect_points(
            half_x=cls.PLATE_LENGTH / 2.0,
            half_y=cls.PLATE_WIDTH / 2.0,
            radius=cls.PLATE_CORNER_RADIUS,
        )
        inner = cls._rounded_rect_points(
            half_x=cls.PLATE_LENGTH / 2.0 - cls.PLATE_TOP_RIM_WIDTH,
            half_y=cls.PLATE_WIDTH / 2.0 - cls.PLATE_TOP_RIM_WIDTH,
            radius=cls.PLATE_CORNER_RADIUS - cls.PLATE_TOP_RIM_WIDTH,
        )
        bottom_z = -cls.PLATE_DEPTH / 2.0
        top_z = cls.PLATE_DEPTH / 2.0
        floor_z = bottom_z + cls.PLATE_BOTTOM_THICKNESS

        vertices: list[list[float]] = []
        faces: list[list[int]] = []

        outer_bottom = cls._add_ring(vertices, outer, bottom_z)
        outer_top = cls._add_ring(vertices, outer, top_z)
        inner_top = cls._add_ring(vertices, inner, top_z)
        inner_floor = cls._add_ring(vertices, inner, floor_z)

        cls._add_side_faces(faces, outer_bottom, outer_top)
        cls._add_side_faces(faces, inner_top, inner_floor)
        cls._add_ring_faces(faces, outer_top, inner_top)
        cls._add_cap_faces(faces, vertices, outer_bottom, bottom_z, upward=False)
        cls._add_cap_faces(faces, vertices, inner_floor, floor_z, upward=True)

        lines = ["# RLinf digital-twin rounded tray mesh"]
        lines.extend(f"v {x:.8f} {y:.8f} {z:.8f}" for x, y, z in vertices)
        lines.extend("f " + " ".join(str(idx + 1) for idx in face) for face in faces)
        return "\n".join(lines) + "\n"

    @classmethod
    def _rounded_rect_points(
        cls,
        half_x: float,
        half_y: float,
        radius: float,
        segments_per_corner: int = 12,
    ) -> list[tuple[float, float]]:
        radius = min(radius, half_x, half_y)
        centers = [
            (half_x - radius, half_y - radius),
            (-half_x + radius, half_y - radius),
            (-half_x + radius, -half_y + radius),
            (half_x - radius, -half_y + radius),
        ]
        angle_ranges = [
            (0.0, np.pi / 2.0),
            (np.pi / 2.0, np.pi),
            (np.pi, 3.0 * np.pi / 2.0),
            (3.0 * np.pi / 2.0, 2.0 * np.pi),
        ]

        points = []
        for center, angle_range in zip(centers, angle_ranges):
            angles = np.linspace(
                angle_range[0],
                angle_range[1],
                segments_per_corner,
                endpoint=False,
            )
            points.extend(
                (
                    center[0] + radius * float(np.cos(angle)),
                    center[1] + radius * float(np.sin(angle)),
                )
                for angle in angles
            )
        return points

    @staticmethod
    def _add_ring(
        vertices: list[list[float]],
        points: list[tuple[float, float]],
        z: float,
    ) -> list[int]:
        indices = []
        for x, y in points:
            indices.append(len(vertices))
            vertices.append([x, y, z])
        return indices

    @staticmethod
    def _add_side_faces(faces: list[list[int]], lower: list[int], upper: list[int]):
        n = len(lower)
        for i in range(n):
            j = (i + 1) % n
            faces.append([lower[i], lower[j], upper[j], upper[i]])

    @staticmethod
    def _add_ring_faces(faces: list[list[int]], outer: list[int], inner: list[int]):
        n = len(outer)
        for i in range(n):
            j = (i + 1) % n
            faces.append([outer[i], outer[j], inner[j], inner[i]])

    @staticmethod
    def _add_cap_faces(
        faces: list[list[int]],
        vertices: list[list[float]],
        ring: list[int],
        z: float,
        upward: bool,
    ):
        center = len(vertices)
        xy = np.mean(np.asarray([vertices[idx][:2] for idx in ring]), axis=0)
        vertices.append([float(xy[0]), float(xy[1]), z])
        n = len(ring)
        for i in range(n):
            j = (i + 1) % n
            face = [center, ring[i], ring[j]] if upward else [center, ring[j], ring[i]]
            faces.append(face)

    def _validate_pick_action(self, action):
        if isinstance(action, torch.Tensor):
            action_dim = action.shape[-1]
        else:
            action_dim = np.asarray(action).shape[-1]
        if action_dim != 7:
            raise ValueError(
                "PickAndPlaceDigitalTwinEnv expects 7-D actions "
                "[dx, dy, dz, droll, dpitch, dyaw, gripper]. "
                f"Got action dim {action_dim}."
            )

    def step(self, action):
        self._validate_pick_action(action)
        raw_obs, reward, terminations, truncations, infos = DigitalTwinBaseEnv.step(
            self, action
        )
        if isinstance(raw_obs, dict) and (
            "sensor_data" in raw_obs or "image" in raw_obs
        ):
            infos["extracted_obs"] = self._build_extracted_obs(raw_obs)
        return raw_obs, reward, terminations, truncations, infos
