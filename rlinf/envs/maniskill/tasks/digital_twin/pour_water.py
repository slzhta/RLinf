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

"""Co-training solid-ball pouring with a calibrated rigidly mounted cup."""

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import sapien
import torch
from gymnasium.vector.utils import batch_space
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose

from rlinf.envs.peg_insertion_geometry import matrix_pose
from rlinf.envs.pour_water_geometry import PourWaterGeometry, ball_inside_cup

from .constants import DIGITAL_TWIN_ASSET_DIR
from .controller.safe_pd_ee_pose import (
    SafePDEEPoseController,
    SafePDEEPoseControllerConfig,
)
from .digital_twin_based_env import DigitalTwinBaseEnv
from .rectangular_tray import RectangularTrayMixin
from .robots.panda_umi import PandaUMI


class PourPoseController(SafePDEEPoseController):
    def compute_target_pose(self, previous, action):
        # Share SO(3) composition and target-relative limits with the real controller.
        geometry = PourWaterGeometry(**self.config.pour_config)
        target = geometry.action_target(
            previous.to_transformation_matrix().detach().cpu().numpy(),
            action.detach().cpu().numpy(),
        )
        poses = np.stack([matrix_pose(matrix) for matrix in target])
        return Pose.create_from_pq(
            p=torch.as_tensor(poses[:, :3], device=self.device, dtype=action.dtype),
            q=torch.as_tensor(
                poses[:, [6, 3, 4, 5]], device=self.device, dtype=action.dtype
            ),
        )


@dataclass
class PourPoseControllerConfig(SafePDEEPoseControllerConfig):
    pour_config: dict = field(default_factory=dict)
    controller_cls = PourPoseController


class PandaUMIPourWater(PandaUMI):
    urdf_path = str(DIGITAL_TWIN_ASSET_DIR / "robots/panda_umi_pour_water.urdf")

    def __init__(self, *args, pour_config=None, **kwargs):
        self.pour_config = dict(pour_config or {})
        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        configs = super()._controller_configs
        mode = configs["pd_ee_body_target_delta_pose_real"]
        mode["arm"] = PourPoseControllerConfig(
            **vars(mode["arm"]), pour_config=self.pour_config
        )
        return configs


@register_env("PourWaterDigitalTwin-v1", max_episode_steps=120)
class PourWaterDigitalTwinEnv(RectangularTrayMixin, DigitalTwinBaseEnv):
    TASK_DESCRIPTION = (
        "Pour the light green ball from the fixed cup into the purple cup on the tray"
    )
    FINGER_QPOS = 0.02
    ROBOT_INITIAL_POSITION = np.array([-0.615, 0.0, 0.055], dtype=np.float32)

    def __init__(self, *args, pour_config=None, task_alignment=None, **kwargs):
        self.pour_config = dict(pour_config or {})
        self.pour_scene = dict(task_alignment or {})
        self.geometry = PourWaterGeometry(**self.pour_config)
        self._hold_count = None
        self._last_evaluated_step = None
        if not kwargs.get("use_hand_camera", True):
            raise ValueError("Pour water requires the wrist camera.")
        if kwargs.get("controller_alignment"):
            raise ValueError("Configure control through pour_config.")
        kwargs["controller_alignment"] = {
            "use_target_controller": True,
            "binary_gripper_action": False,
            "action_scale": [*self.geometry.action_scale, 1.0],
            "target_ee_pose": self.geometry.target_ee_pose,
        }
        sim_config = dict(kwargs.pop("sim_config", {}) or {})
        scene_config = dict(sim_config.get("scene_config", {}) or {})
        scene_config.update(contact_offset=0.0001, rest_offset=0.0)
        sim_config.update(scene_config=scene_config)
        kwargs["sim_config"] = sim_config
        super().__init__(*args, **kwargs)
        self._set_policy_action_space()

    def _set_policy_action_space(self):
        self.single_action_space = gym.spaces.Box(-1, 1, (6,), dtype=np.float32)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    def _load_agent(self, options):
        self.agent = PandaUMIPourWater(
            self.scene,
            self._control_freq,
            self._control_mode,
            initial_pose=sapien.Pose(self.ROBOT_INITIAL_POSITION),
            controller_alignment=self.controller_alignment,
            enable_hand_camera=True,
            pour_config=self.pour_config,
        )

    def _load_task_scene(self, options):
        self.cup = self.agent.robot.links_map["pour_cup_fixed_tool"]
        material = sapien.physx.PhysxMaterial(0.20, 0.16, 0.02)
        for name in (
            "pour_cup_fixed_tool",
            "panda_hand",
            "panda_leftfinger",
            "panda_rightfinger",
        ):
            for body in self.agent.robot.links_map[name]._objs:
                for shape in body.get_collision_shapes():
                    groups = list(shape.get_collision_groups())
                    groups[2] |= 1 << 28
                    shape.set_collision_groups(groups)
                    if name == "pour_cup_fixed_tool":
                        shape.set_physical_material(material)
        tray_position = np.asarray(
            self.pour_scene.get(
                "source_tray_position", [-0.06984352, 0.12448609, self.PLATE_DEPTH / 2]
            ),
            dtype=float,
        )
        if tray_position.shape != (3,) or not np.isfinite(tray_position).all():
            raise ValueError("source_tray_position must be a finite world XYZ.")
        self.tray_position = tray_position
        self.source_tray = self._build_tray("plain_source_tray", False, tray_position)
        assets = DIGITAL_TWIN_ASSET_DIR / "objects/pour_water"
        builder = self.scene.create_actor_builder()
        for filename in [*sorted(assets.glob("wall_*.stl")), assets / "bottom.stl"]:
            builder.add_convex_collision_from_file(
                str(filename), material=material, density=970
            )
        builder.add_visual_from_file(
            str(assets / "cup_visual.stl"),
            material=sapien.render.RenderMaterial(
                base_color=[0.45, 0.06, 0.72, 1], roughness=0.7
            ),
        )
        builder.initial_pose = sapien.Pose(
            tray_position + [0, 0, self._floor_top_local_z()]
        )
        self.receiver = builder.build(name="purple_receiver_cup")
        builder = self.scene.create_actor_builder()
        builder.add_sphere_collision(radius=0.008, material=material, density=600)
        builder.add_sphere_visual(
            radius=0.008,
            material=sapien.render.RenderMaterial(
                base_color=[0.55, 0.90, 0.45, 1], roughness=0.6
            ),
        )
        builder.initial_pose = sapien.Pose([0, 0, 0.5])
        self.ball = builder.build(name="light_green_ball")

    def _get_foreground_actors(self):
        return [self.source_tray, self.receiver, self.ball]

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        self.agent.robot.set_pose(sapien.Pose(self.ROBOT_INITIAL_POSITION))
        self.sync_gpu_articulation_state()
        qpos = self.agent.robot.get_qpos().clone()
        targets = np.asarray(
            [
                matrix_pose(
                    self.geometry.reset_pose(
                        self._batched_episode_rng[int(index.item())]
                    )
                )
                for index in env_idx
            ]
        )
        desired = Pose.create_from_pq(
            p=torch.as_tensor(targets[:, :3], device=self.device, dtype=qpos.dtype),
            q=torch.as_tensor(
                targets[:, [6, 3, 4, 5]], device=self.device, dtype=qpos.dtype
            ),
        )
        qpos[env_idx, :7] = self._solve_arm_ik_qpos(
            desired, qpos[env_idx], env_idx, qpos.device, qpos.dtype
        )
        qpos[env_idx, -2:] = self.FINGER_QPOS
        self.agent.reset(qpos[env_idx])
        self.sync_gpu_articulation_state()
        positions = np.tile(self.tray_position, (len(env_idx), 1))
        margin = float(self.pour_scene.get("object_tray_margin", 0.005))
        limits = (
            np.array([self.PLATE_LENGTH, self.PLATE_WIDTH]) / 2
            - self.PLATE_WALL_THICKNESS
            - 0.03
            - margin
        )
        if not np.isfinite(margin) or margin < 0 or np.any(limits <= 0):
            raise ValueError("Receiver margin leaves no valid placement area.")
        if self.pour_scene.get("object_random_in_source_tray", True):
            for row, index in enumerate(env_idx):
                positions[row, :2] += self._batched_episode_rng[
                    int(index.item())
                ].uniform(-limits, limits)
        positions[:, 2] += self._floor_top_local_z() + 0.0002
        self.receiver.set_pose(
            Pose.create_from_pq(
                p=torch.as_tensor(positions, device=self.device, dtype=qpos.dtype),
                q=torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device),
            )
        )
        zeros = torch.zeros((len(env_idx), 3), device=self.device)
        self.receiver.set_linear_velocity(zeros)
        self.receiver.set_angular_velocity(zeros)
        local_ball = Pose.create_from_pq(
            p=torch.tensor([0.0, 0.0, 0.018], device=self.device),
            q=torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device),
        )
        self.ball.set_pose(self.cup.pose[env_idx] * local_ball)
        self.ball.set_linear_velocity(zeros)
        self.ball.set_angular_velocity(zeros)
        self.sync_gpu_articulation_state()
        if self._hold_count is None or len(self._hold_count) != self.num_envs:
            self._hold_count = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self._last_evaluated_step = torch.full_like(self._hold_count, -1)
        self._hold_count[env_idx] = 0
        self._last_evaluated_step[env_idx] = 0

    def evaluate(self):
        local = (self.receiver.pose.inv() * self.ball.pose).p
        inside = torch.as_tensor(
            ball_inside_cup(local.detach().cpu().numpy()), device=self.device
        )
        rotation = self.receiver.pose.to_transformation_matrix()[:, :3, :3]
        upright = rotation[:, 2, 2] > 0.98
        relative_speed = torch.linalg.norm(
            self.ball.linear_velocity - self.receiver.linear_velocity, dim=-1
        )
        stable = inside & upright & (relative_speed < 0.05)
        fresh = self.elapsed_steps > self._last_evaluated_step
        self._hold_count = torch.where(
            fresh, torch.where(stable, self._hold_count + 1, 0), self._hold_count
        )
        self._last_evaluated_step = self.elapsed_steps.clone()
        return {
            "success": self._hold_count >= self.geometry.success_hold_steps,
            "ball_inside_receiver": inside,
            "receiver_upright": upright,
            "ball_relative_speed": relative_speed,
            "success_stable_steps": self._hold_count.clone(),
        }

    def compute_dense_reward(self, obs, action, info):
        return info["success"].float()

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info)

    def _build_extracted_obs(self, raw_obs):
        observation = super()._build_extracted_obs(raw_obs)
        if "extra_view_images" not in observation:
            raise ValueError("Wrist RGB is missing; use obs_mode=rgb+segmentation.")
        # The mounted tool has no gripper action; keep the PnP closed-state slot.
        observation["states"] = torch.cat(
            [observation["states"], -torch.ones_like(observation["states"][:, :1])],
            dim=1,
        )
        return observation

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if "sensor_data" in obs or "image" in obs:
            info["extracted_obs"] = self._build_extracted_obs(obs)
        self._set_policy_action_space()
        return obs, info

    def step(self, action):
        action = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        if action.shape != (self.num_envs, 6) or not torch.isfinite(action).all():
            raise ValueError(f"Expected finite [{self.num_envs}, 6] actions.")
        grip = torch.full_like(action[:, :1], 2 * (self.FINGER_QPOS + 0.01) / 0.05 - 1)
        obs, reward, terminated, truncated, info = super().step(
            torch.cat([action, grip], -1)
        )
        info["extracted_obs"] = self._build_extracted_obs(obs)
        return obs, reward.float(), terminated, truncated, info
