from pathlib import Path
from typing import Any

import sapien
import torch
from sapien.physx import PhysxMaterial

from mani_skill.utils.registration import register_env

from rlinf.envs.maniskill.tasks.digital_twin.digital_twin_based_env import (
    DigitalTwinBaseEnv,
)


@register_env("PushButton-v1", max_episode_steps=100)
class PushButtonEnv(DigitalTwinBaseEnv):
    """Minimal push-button task with contact-force success."""

    BUTTON_POSE = sapien.Pose(p=[0.0, 0.0, 0.02], q=[0, 1, 0, 0])
    BUTTON_FORCE_THRESHOLD = 0.05
    BUTTON_ASSET_DIR = (
        Path(__file__).resolve().parent / "assets" / "objects" / "button"
    )

    def _load_task_scene(self, options: dict):
        self.button = self._build_button()

    def _get_foreground_actors(self):
        return [self.button]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        self.button.set_pose(self.BUTTON_POSE)

    def _get_button_contact_force(self) -> torch.Tensor:
        left_force = self.scene.get_pairwise_contact_forces(
            self.agent.finger1_link, self.button
        )
        right_force = self.scene.get_pairwise_contact_forces(
            self.agent.finger2_link, self.button
        )
        total_force = left_force + right_force
        return torch.linalg.norm(total_force, axis=1)

    def evaluate(self):
        button_contact_force = self._get_button_contact_force()
        is_button_pressed = button_contact_force > self.BUTTON_FORCE_THRESHOLD
        return {
            "button_contact_force": button_contact_force,
            "is_button_pressed": is_button_pressed,
            "success": is_button_pressed.clone(),
        }

    def _get_obs_extra(self, info: dict):
        obs = {"tcp_pose": self.agent.tcp.pose.raw_pose}
        if "state" in self.obs_mode:
            obs.update(
                button_pose=self.button.pose.raw_pose,
                tcp_to_button_pos=self.button.pose.p - self.agent.tcp.pose.p,
                button_contact_force=info["button_contact_force"].unsqueeze(-1),
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_to_button_dist = torch.linalg.norm(
            self.agent.tcp.pose.p - self.button.pose.p, axis=1
        )
        reward = 1.0 - torch.tanh(5.0 * tcp_to_button_dist)
        reward += torch.clamp(info["button_contact_force"], max=1.0)
        reward[info["success"]] = 3.0
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 3.0

    def _build_button(self):
        collision_filename = self.BUTTON_ASSET_DIR / "mesh_w_vertex_color_abs.ply"
        collision_coacd_filename = self.BUTTON_ASSET_DIR / (
            "mesh_w_vertex_color_abs.ply.coacd.ply"
        )
        visual_filename = self.BUTTON_ASSET_DIR / "visual" / "button.obj"
        builder = self.scene.create_actor_builder()

        scale = 1.0
        scale_array = [scale] * 3
        decomposition = "coacd"
        if collision_coacd_filename.is_file():
            collision_filename = collision_coacd_filename
            decomposition = "none"

        builder.add_multiple_convex_collisions_from_file(
            filename=str(collision_filename),
            scale=scale_array,
            material=PhysxMaterial(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            density=1000.0,
            patch_radius=0.1,
            min_patch_radius=0.1,
            decomposition=decomposition,
        )

        builder.add_visual_from_file(
            filename=str(visual_filename),
            scale=scale_array,
        )

        builder.initial_pose = self.BUTTON_POSE
        return builder.build_kinematic(name="button")
