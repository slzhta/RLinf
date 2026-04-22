from pathlib import Path
from typing import Any

import numpy as np
import sapien
import torch
import torch.nn.functional as F
from sapien.physx import PhysxMaterial
from transforms3d.euler import mat2euler

from mani_skill.utils.registration import register_env

from rlinf.envs.maniskill.tasks.digital_twin.digital_twin_based_env import (
    DigitalTwinBaseEnv,
)


@register_env("PushButton-v1", max_episode_steps=100)
class PushButtonEnv(DigitalTwinBaseEnv):
    """Minimal push-button task with contact-force success."""

    BUTTON_POSE = sapien.Pose(p=[0.0, 0.0, 0.02], q=[0, 1, 0, 0])
    BUTTON_FORCE_THRESHOLD = 0.05
    CLOSED_GRIPPER_QPOS = 0.0
    CLOSED_GRIPPER_ACTION = -1.0
    OUTPUT_IMAGE_SIZE = 128
    TASK_DESCRIPTION = "reach the button target and press it"
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
        qpos = self.agent.robot.get_qpos().clone()
        qpos[env_idx, -2:] = self.CLOSED_GRIPPER_QPOS
        self.agent.reset(qpos)
        self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

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

    def _build_extracted_obs(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        sensor_data = raw_obs.get("sensor_data", {})
        main_images = self._pad_and_resize_images(
            sensor_data["3rdview_camera"]["rgb"].to(torch.uint8)
        )

        extracted_obs = {
            "main_images": main_images,
            "task_descriptions": self.get_language_instruction(),
        }

        if "hand_camera" in sensor_data:
            extracted_obs["extra_view_images"] = (
                self._pad_and_resize_images(
                    sensor_data["hand_camera"]["rgb"].to(torch.uint8)
                ).unsqueeze(1)
            )

        gripper_state = self.agent.robot.get_qpos().to(torch.float32)[:, -1:] * 2
        ee_pose_t = (
            self.agent.ee_pose_at_robot_base.to_transformation_matrix().cpu().numpy()
        )
        pos = torch.from_numpy(ee_pose_t[:, :3, 3]).to(gripper_state.device)
        euler = torch.from_numpy(
            np.stack(
                [mat2euler(ee_pose_t[i, :3, :3], "sxyz") for i in range(self.num_envs)],
                axis=0,
            )
        ).to(gripper_state.device, dtype=torch.float32)
        extracted_obs["states"] = torch.cat([pos, euler, gripper_state], dim=1)
        return extracted_obs

    def get_language_instruction(self) -> list[str]:
        return [self.TASK_DESCRIPTION] * self.num_envs

    def _pad_and_resize_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() != 4:
            raise ValueError(f"Expected image tensor with shape [B, H, W, C], got {images.shape}.")

        _, height, width, _ = images.shape
        if height < width:
            pad = width - height
            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = 0
            pad_right = 0
        else:
            pad = height - width
            pad_left = pad // 2
            pad_right = pad - pad_left
            pad_top = 0
            pad_bottom = 0

        nchw_images = images.permute(0, 3, 1, 2).to(torch.float32)
        padded_images = F.pad(
            nchw_images,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0.0,
        )
        resized_images = F.interpolate(
            padded_images,
            size=(self.OUTPUT_IMAGE_SIZE, self.OUTPUT_IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        return resized_images.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)

    def reset(self, seed=None, options=None):
        raw_obs, infos = super().reset(seed=seed, options=options)
        infos["extracted_obs"] = self._build_extracted_obs(raw_obs)
        return raw_obs, infos

    def _append_closed_gripper_action(self, action):
        if isinstance(action, torch.Tensor):
            if action.shape[-1] == 7:
                return action
            if action.shape[-1] != 6:
                raise ValueError(
                    f"Expected action dim 6 or 7 for PushButtonEnv, got {action.shape}."
                )
            close_action = torch.full(
                (*action.shape[:-1], 1),
                self.CLOSED_GRIPPER_ACTION,
                dtype=action.dtype,
                device=action.device,
            )
            return torch.cat([action, close_action], dim=-1)

        action_np = np.asarray(action)
        if action_np.shape[-1] == 7:
            return action_np
        if action_np.shape[-1] != 6:
            raise ValueError(
                f"Expected action dim 6 or 7 for PushButtonEnv, got {action_np.shape}."
            )
        close_action = np.full(
            (*action_np.shape[:-1], 1),
            self.CLOSED_GRIPPER_ACTION,
            dtype=action_np.dtype,
        )
        return np.concatenate([action_np, close_action], axis=-1)

    def step(self, action):
        action = self._append_closed_gripper_action(action)
        raw_obs, reward, terminations, truncations, infos = super().step(action)
        infos["extracted_obs"] = self._build_extracted_obs(raw_obs)
        return raw_obs, reward, terminations, truncations, infos
