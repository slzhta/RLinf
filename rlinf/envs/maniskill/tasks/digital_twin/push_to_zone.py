from __future__ import annotations

from typing import Any

import numpy as np
import sapien
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from sapien.physx import PhysxMaterial

from rlinf.envs.maniskill.tasks.digital_twin.digital_twin_based_env import (
    DigitalTwinBaseEnv,
)
from rlinf.envs.maniskill.tasks.digital_twin.pick_and_place import (
    PickAndPlaceDigitalTwinEnv,
)


@register_env("PushToZoneDigitalTwin-v1", max_episode_steps=80)
class PushToZoneDigitalTwinEnv(PickAndPlaceDigitalTwinEnv):
    """Push a red block across the tabletop into a green target zone.

    This task intentionally reuses the pick-and-place environment's digital-twin
    background, controller alignment, reset IK and observation extraction. It
    does not create trays and does not require grasping. The seventh action
    dimension is retained for compatibility with the existing RLinf policy and
    controller configuration.
    """

    BLOCK_HALF_SIZE = np.array([0.025, 0.025, 0.020], dtype=np.float32)
    BLOCK_COLOR = [0.82, 0.22, 0.18, 1.0]
    BLOCK_DENSITY = 650.0
    BLOCK_STATIC_FRICTION = 0.9
    BLOCK_DYNAMIC_FRICTION = 0.7

    GOAL_HALF_SIZE = np.array([0.070, 0.070, 0.0005], dtype=np.float32)
    GOAL_COLOR = [0.20, 0.78, 0.32, 0.65]

    DEFAULT_BLOCK_POSITION = np.array([-0.12, 0.10, 0.020], dtype=np.float32)
    DEFAULT_GOAL_POSITION = np.array([-0.12, -0.10, 0.0005], dtype=np.float32)
    DEFAULT_OBJECT_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    DEFAULT_BLOCK_RANDOM_XY_RANGE = 0.015
    DEFAULT_GOAL_RANDOM_XY_RANGE = 0.0
    DEFAULT_RESET_POSITION_RANDOM_RANGE = np.array(
        [0.025, 0.025, 0.015], dtype=np.float32
    )

    CONTACT_DISTANCE_THRESHOLD = 0.060
    SUCCESS_SPEED_THRESHOLD = 0.080
    FALL_HEIGHT_THRESHOLD = -0.01
    OPEN_GRIPPER_QPOS = 0.04
    DEFAULT_SPARSE_SUCCESS_REWARD = 1.0
    DEFAULT_SPARSE_FALL_PENALTY = -0.2
    TASK_DESCRIPTION = "push the red block into the green target zone"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reset_sparse_reward_state()
        self._reset_episode_metric_state()

    def _load_task_scene(self, options: dict):
        self.block = self._build_block(self.DEFAULT_BLOCK_POSITION)
        self.goal_zone = self._build_goal_zone(self.DEFAULT_GOAL_POSITION)

    def _get_foreground_actors(self):
        return [self.block, self.goal_zone]

    def _build_block(self, initial_position: np.ndarray):
        builder = self.scene.create_actor_builder()
        material = PhysxMaterial(
            static_friction=self.BLOCK_STATIC_FRICTION,
            dynamic_friction=self.BLOCK_DYNAMIC_FRICTION,
            restitution=0.0,
        )
        builder.add_box_collision(
            half_size=self.BLOCK_HALF_SIZE.tolist(),
            material=material,
            density=self.BLOCK_DENSITY,
        )
        builder.add_box_visual(
            half_size=self.BLOCK_HALF_SIZE.tolist(),
            material=sapien.render.RenderMaterial(base_color=self.BLOCK_COLOR),
        )
        builder.initial_pose = sapien.Pose(
            p=np.asarray(initial_position, dtype=np.float32).tolist(),
            q=self.DEFAULT_OBJECT_QUAT.tolist(),
        )
        return builder.build(name="push_block")

    def _build_goal_zone(self, initial_position: np.ndarray):
        # The goal is visual-only, so it cannot catch or obstruct the block.
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(
            half_size=self.GOAL_HALF_SIZE.tolist(),
            material=sapien.render.RenderMaterial(base_color=self.GOAL_COLOR),
        )
        builder.initial_pose = sapien.Pose(
            p=np.asarray(initial_position, dtype=np.float32).tolist(),
            q=self.DEFAULT_OBJECT_QUAT.tolist(),
        )
        return builder.build_kinematic(name="green_target_zone")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        DigitalTwinBaseEnv._initialize_episode(self, env_idx, options)

        self.block.set_pose(self._sample_actor_pose(
            env_idx=env_idx,
            pose_key="block_initial_pose",
            default_position=self.DEFAULT_BLOCK_POSITION,
            random_xy_key="block_random_xy_range",
            default_random_xy_range=self.DEFAULT_BLOCK_RANDOM_XY_RANGE,
        ))
        self.goal_zone.set_pose(self._sample_actor_pose(
            env_idx=env_idx,
            pose_key="goal_initial_pose",
            default_position=self.DEFAULT_GOAL_POSITION,
            random_xy_key="goal_random_xy_range",
            default_random_xy_range=self.DEFAULT_GOAL_RANDOM_XY_RANGE,
        ))

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

    def _sample_actor_pose(
        self,
        env_idx: torch.Tensor,
        pose_key: str,
        default_position: np.ndarray,
        random_xy_key: str,
        default_random_xy_range: float,
    ) -> Pose:
        default_pose = np.hstack([default_position, self.DEFAULT_OBJECT_QUAT])
        pose = np.asarray(
            self.task_alignment.get(pose_key, default_pose), dtype=np.float32
        ).reshape(-1)
        if pose.size != 7:
            raise ValueError(
                f"task_alignment.{pose_key} must contain 7 values "
                "[x, y, z, qw, qx, qy, qz]."
            )

        b = len(env_idx)
        position = np.repeat(pose[:3][None, :], b, axis=0)
        random_xy_range = float(
            self.task_alignment.get(random_xy_key, default_random_xy_range)
        )
        if random_xy_range < 0:
            raise ValueError(f"task_alignment.{random_xy_key} must be >= 0.")

        if random_xy_range > 0:
            for i, scene_idx_t in enumerate(env_idx):
                scene_idx = int(scene_idx_t.item())
                position[i, :2] += self._batched_episode_rng[scene_idx].uniform(
                    -random_xy_range, random_xy_range, size=(2,)
                )

        quat = np.repeat(pose[3:][None, :], b, axis=0)
        return Pose.create_from_pq(
            p=torch.as_tensor(position, device=self.device, dtype=torch.float32),
            q=torch.as_tensor(quat, device=self.device, dtype=torch.float32),
        )

    def _get_success_half_size(self) -> torch.Tensor:
        # Requiring the block's whole footprint to be inside the target makes
        # the success criterion independent of the visual transparency.
        half_size = self.GOAL_HALF_SIZE[:2] - self.BLOCK_HALF_SIZE[:2]
        margin = float(self.task_alignment.get("success_margin", 0.0))
        half_size = half_size - margin
        if np.any(half_size <= 0):
            raise ValueError("The goal zone is too small for the configured block.")
        return torch.as_tensor(
            half_size, device=self.device, dtype=torch.float32
        )

    def evaluate(self):
        block_position = self.block.pose.p
        goal_position = self.goal_zone.pose.p
        tcp_to_block = block_position - self.agent.tcp.pose.p
        block_to_goal = goal_position - block_position

        tcp_to_block_dist = torch.linalg.norm(tcp_to_block, dim=1)
        block_to_goal_xy_dist = torch.linalg.norm(block_to_goal[:, :2], dim=1)
        block_xy_speed = torch.linalg.norm(self.block.linear_velocity[:, :2], dim=1)

        success_half_size = self._get_success_half_size()
        is_in_goal = torch.all(
            torch.abs(block_to_goal[:, :2]) <= success_half_size, dim=1
        )
        speed_threshold = float(
            self.task_alignment.get(
                "success_speed_threshold", self.SUCCESS_SPEED_THRESHOLD
            )
        )
        is_stable = block_xy_speed <= speed_threshold
        is_contacted = tcp_to_block_dist <= self.CONTACT_DISTANCE_THRESHOLD
        is_fallen = block_position[:, 2] < self.FALL_HEIGHT_THRESHOLD
        success = is_in_goal & is_stable & (~is_fallen)

        if not hasattr(self, "_episode_contact_once"):
            self._reset_episode_metric_state()
        self._episode_contact_once |= is_contacted
        self._episode_goal_once |= is_in_goal
        self._episode_fall_once |= is_fallen

        return {
            "success": success,
            "is_contacted": is_contacted,
            "is_in_goal": is_in_goal,
            "is_stable": is_stable,
            "is_fallen": is_fallen,
            "goal_position": goal_position,
            "tcp_to_block": tcp_to_block,
            "tcp_to_block_dist": tcp_to_block_dist,
            "block_to_goal": block_to_goal,
            "block_to_goal_xy_dist": block_to_goal_xy_dist,
            "block_xy_speed": block_xy_speed,
            "contact_once": self._episode_contact_once.clone(),
            "goal_once": self._episode_goal_once.clone(),
            "fall_once": self._episode_fall_once.clone(),
        }

    def _get_obs_extra(self, info: dict):
        obs = {
            "tcp_pose": self.agent.tcp.pose.raw_pose,
            "goal_pos": info["goal_position"],
        }
        if "state" in self.obs_mode:
            obs.update(
                block_pose=self.block.pose.raw_pose,
                tcp_to_block_pos=info["tcp_to_block"],
                block_to_goal_pos=info["block_to_goal"],
                block_linear_velocity=self.block.linear_velocity,
                is_contacted=info["is_contacted"].unsqueeze(-1),
                is_in_goal=info["is_in_goal"].unsqueeze(-1),
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        """Return event-based sparse rewards only.

        Normal transitions receive zero. Success is rewarded once, and a fall
        is penalized once. No distance, reaching, contact or progress shaping
        is used, so the same reward definition can be reproduced on hardware.
        """
        reward = torch.zeros_like(info["block_to_goal_xy_dist"])

        newly_successful = info["success"] & (~self._sparse_success_rewarded)
        newly_fallen = info["is_fallen"] & (~self._sparse_fall_penalized)

        reward += newly_successful.to(reward.dtype) * float(
            self.task_alignment.get(
                "sparse_success_reward", self.DEFAULT_SPARSE_SUCCESS_REWARD
            )
        )
        reward += newly_fallen.to(reward.dtype) * float(
            self.task_alignment.get(
                "sparse_fall_penalty", self.DEFAULT_SPARSE_FALL_PENALTY
            )
        )

        self._sparse_success_rewarded |= info["success"]
        self._sparse_fall_penalized |= info["is_fallen"]
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)

    def _reset_sparse_reward_state(self, env_idx: torch.Tensor | None = None):
        if not hasattr(self, "_sparse_success_rewarded"):
            self._sparse_success_rewarded = torch.zeros(
                (self.num_envs,), dtype=torch.bool, device=self.device
            )
            self._sparse_fall_penalized = torch.zeros_like(
                self._sparse_success_rewarded
            )

        if env_idx is None:
            self._sparse_success_rewarded[:] = False
            self._sparse_fall_penalized[:] = False
            return

        self._sparse_success_rewarded[env_idx] = False
        self._sparse_fall_penalized[env_idx] = False

    def _reset_episode_metric_state(self, env_idx: torch.Tensor | None = None):
        if not hasattr(self, "_episode_contact_once"):
            self._episode_contact_once = torch.zeros(
                (self.num_envs,), dtype=torch.bool, device=self.device
            )
            self._episode_goal_once = torch.zeros_like(self._episode_contact_once)
            self._episode_fall_once = torch.zeros_like(self._episode_contact_once)

        if env_idx is None:
            self._episode_contact_once[:] = False
            self._episode_goal_once[:] = False
            self._episode_fall_once[:] = False
            return

        self._episode_contact_once[env_idx] = False
        self._episode_goal_once[env_idx] = False
        self._episode_fall_once[env_idx] = False
    def _validate_pick_action(self, action):
        action_dim = (
            action.shape[-1]
            if isinstance(action, torch.Tensor)
            else np.asarray(action).shape[-1]
        )
        if action_dim != 7:
            raise ValueError(
                "PushToZoneDigitalTwinEnv expects 7-D actions "
                "[dx, dy, dz, droll, dpitch, dyaw, gripper]. "
                f"Got action dim {action_dim}."
            )