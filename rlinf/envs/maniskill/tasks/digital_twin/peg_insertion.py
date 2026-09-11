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

"""Co-training insertion with the validated rigid, flat-face-mounted U tool."""

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import sapien
import torch
from gymnasium.vector.utils import batch_space
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose

from rlinf.envs.peg_insertion_geometry import PegInsertionGeometry, matrix_pose

from .constants import DIGITAL_TWIN_ASSET_DIR
from .controller.safe_pd_ee_pose import (
    SafePDEEPoseController,
    SafePDEEPoseControllerConfig,
)
from .digital_twin_based_env import DigitalTwinBaseEnv
from .robots.panda_umi import PandaUMI


class PegPoseController(SafePDEEPoseController):
    def compute_target_pose(self, previous, action):
        # Share SO(3) composition and target-relative limits with the real controller.
        geometry = PegInsertionGeometry(**self.config.peg_config)
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
class PegPoseControllerConfig(SafePDEEPoseControllerConfig):
    peg_config: dict = field(default_factory=dict)
    controller_cls = PegPoseController


class PandaUMIPegInsertion(PandaUMI):
    urdf_path = str(DIGITAL_TWIN_ASSET_DIR / "robots/panda_umi_peg_insertion.urdf")

    def __init__(self, *args, peg_config=None, **kwargs):
        self.peg_config = dict(peg_config or {})
        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        configs = super()._controller_configs
        mode = configs["pd_ee_body_target_delta_pose_real"]
        mode["arm"] = PegPoseControllerConfig(
            **vars(mode["arm"]), peg_config=self.peg_config
        )
        return configs


@register_env("PegInsertionDigitalTwin-v1", max_episode_steps=120)
class PegInsertionDigitalTwinEnv(DigitalTwinBaseEnv):
    TASK_DESCRIPTION = (
        "Insert the green U-shaped peg into the matching hole in the blue board"
    )
    FINGER_QPOS = 0.0533628067 / 2

    def __init__(self, *args, peg_config=None, **kwargs):
        self.peg_config = dict(peg_config or {})
        self.geometry = PegInsertionGeometry(**self.peg_config)
        self._hold_count = None
        self._last_evaluated_step = None
        if not kwargs.get("use_hand_camera", True):
            raise ValueError("Peg insertion requires the wrist camera.")
        if kwargs.get("controller_alignment") or kwargs.get("task_alignment"):
            raise ValueError(
                "Configure this task through peg_config, not PnP alignment."
            )
        kwargs["controller_alignment"] = {
            "use_target_controller": True,
            "binary_gripper_action": False,
            "action_scale": [*self.geometry.action_scale, 1.0],
            "target_ee_pose": self.geometry.target_ee_pose,
        }
        sim_config = dict(kwargs.pop("sim_config", {}) or {})
        scene_config = dict(sim_config.get("scene_config", {}) or {})
        scene_config.update(contact_offset=0.0005, rest_offset=0.0)
        sim_config.update(scene_config=scene_config)
        kwargs["sim_config"] = sim_config
        super().__init__(*args, **kwargs)
        self._set_policy_action_space()

    def _set_policy_action_space(self):
        self.single_action_space = gym.spaces.Box(-1, 1, (6,), dtype=np.float32)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    def _load_agent(self, options):
        self.agent = PandaUMIPegInsertion(
            self.scene,
            self._control_freq,
            self._control_mode,
            initial_pose=sapien.Pose([-0.615, 0, 0]),
            controller_alignment=self.controller_alignment,
            enable_hand_camera=True,
            peg_config=self.peg_config,
        )

    def _load_task_scene(self, options):
        self.peg = self.agent.robot.links_map["u_peg_fixed_tool"]
        for name in (
            "u_peg_fixed_tool",
            "panda_hand",
            "panda_leftfinger",
            "panda_rightfinger",
        ):
            for body in self.agent.robot.links_map[name]._objs:
                for shape in body.get_collision_shapes():
                    groups = list(shape.get_collision_groups())
                    groups[2] |= 1 << 28
                    shape.set_collision_groups(groups)
        filename = str(DIGITAL_TWIN_ASSET_DIR / "objects/peg_insertion/board.stl")
        builder = self.scene.create_actor_builder()
        builder.add_nonconvex_collision_from_file(
            filename, material=sapien.physx.PhysxMaterial(0.3, 0.3, 0)
        )
        builder.add_visual_from_file(
            filename,
            material=sapien.render.RenderMaterial(
                base_color=[0.04, 0.17, 0.85, 1], roughness=0.7
            ),
        )
        world_board = self.geometry.board_pose()
        world_board[:3, 3] += [-0.615, 0, 0]
        pose = matrix_pose(world_board)
        builder.initial_pose = sapien.Pose(pose[:3], pose[[6, 3, 4, 5]])
        self.board = builder.build_static(name="u_socket_board")

    def _get_foreground_actors(self):
        return [self.board]

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
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
        if self._hold_count is None or len(self._hold_count) != self.num_envs:
            self._hold_count = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self._last_evaluated_step = torch.full_like(self._hold_count, -1)
        self._hold_count[env_idx] = 0
        self._last_evaluated_step[env_idx] = 0

    def evaluate(self):
        matrices = (
            self.agent.ee_pose_at_robot_base.to_transformation_matrix()
            .detach()
            .cpu()
            .numpy()
        )
        metrics = {
            key: torch.as_tensor(value, device=self.device)
            for key, value in self.geometry.metrics(matrices).items()
        }
        fresh = self.elapsed_steps > self._last_evaluated_step
        self._hold_count = torch.where(
            fresh,
            torch.where(metrics["in_target"], self._hold_count + 1, 0),
            self._hold_count,
        )
        self._last_evaluated_step = self.elapsed_steps.clone()
        metrics["success"] = self._hold_count >= self.geometry.success_hold_steps
        return metrics

    def compute_dense_reward(self, obs, action, info):
        return torch.where(info["success"], 1.0, info["dense_reward"]).float()

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info)

    def _build_extracted_obs(self, raw_obs):
        sensors = raw_obs.get("sensor_data", {})
        images = raw_obs.get("image", {})
        wrist = images.get("hand_camera", sensors.get("hand_camera"))
        if wrist is None:
            raise ValueError("Wrist RGB is missing; use obs_mode=rgb+segmentation.")
        matrices = (
            self.agent.ee_pose_at_robot_base.to_transformation_matrix()
            .detach()
            .cpu()
            .numpy()
        )
        return {
            "main_images": self._pad_and_resize_images(wrist["rgb"].to(torch.uint8)),
            "states": torch.as_tensor(
                self.geometry.relative_state(matrices), device=self.device
            ),
            "task_descriptions": self.get_language_instruction(),
        }

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
        return obs, reward, terminated, truncated, info
