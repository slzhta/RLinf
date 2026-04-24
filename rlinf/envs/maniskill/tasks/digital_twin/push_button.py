from pathlib import Path
from typing import Any

import numpy as np
import sapien
import torch
import torch.nn.functional as F
from sapien.physx import PhysxMaterial
from transforms3d.euler import mat2euler

from mani_skill.utils.registration import register_env
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)
from mani_skill.utils.structs.pose import Pose

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
    SUCCESS_TARGET_Z_OFFSET = 0.03
    SUCCESS_POSITION_THRESHOLD = 0.015
    DENSE_REWARD_DECAY = 500.0
    DEFAULT_BUTTON_RANDOM_XY_RANGE = 0.10
    DEFAULT_RESET_RANDOM_XY_RANGE = 0.01
    DEFAULT_RESET_RANDOM_RZ_RANGE = np.pi / 9
    GPU_RESET_IK_MAX_POS_STEP = 0.02
    GPU_RESET_IK_MAX_ROT_STEP = np.pi / 18
    BUTTON_ASSET_DIR = (
        Path(__file__).resolve().parent / "assets" / "objects" / "button"
    )

    def _load_task_scene(self, options: dict):
        self.button = self._build_button()

    def _get_foreground_actors(self):
        return [self.button]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        self.button.set_pose(self._sample_button_pose(env_idx))
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
        qpos[env_idx, -2:] = self.CLOSED_GRIPPER_QPOS
        self.agent.reset(qpos)
        self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        self.sync_gpu_articulation_state()

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
                "controller_alignment.target_ee_pose must contain 6 values [x, y, z, rx, ry, rz]."
            )

        reset_ee_pose_cfg = controller_cfg.get("reset_ee_pose", None)
        if reset_ee_pose_cfg is not None:
            reset_ee_pose = np.asarray(reset_ee_pose_cfg, dtype=np.float32).reshape(-1)
            if reset_ee_pose.size != 6:
                raise ValueError(
                    "controller_alignment.reset_ee_pose must contain 6 values [x, y, z, rx, ry, rz]."
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
        random_xy_range = float(
            controller_cfg.get("random_xy_range", self.DEFAULT_RESET_RANDOM_XY_RANGE)
        )
        random_rz_range = float(
            controller_cfg.get("random_rz_range", self.DEFAULT_RESET_RANDOM_RZ_RANGE)
        )
        target_rz = float(target_ee_pose[5])

        if enable_random_reset:
            for i, scene_idx_t in enumerate(env_idx):
                scene_idx = int(scene_idx_t.item())
                rng = self._batched_episode_rng[scene_idx]
                position[i, :2] += rng.uniform(
                    -random_xy_range, random_xy_range, size=(2,)
                )
                euler[i, 2] = target_rz + rng.uniform(-random_rz_range, random_rz_range)
        else:
            euler[:, 2] = target_rz

        position_t = torch.as_tensor(position, device=device, dtype=dtype)
        euler_t = torch.as_tensor(euler, device=device, dtype=dtype)
        quat_t = matrix_to_quaternion(euler_angles_to_matrix(euler_t, "XYZ")).to(
            device=device, dtype=dtype
        )
        return Pose.create_from_pq(p=position_t, q=quat_t)

    def _solve_arm_ik_qpos(
        self,
        target_pose: Pose,
        seed_qpos: torch.Tensor,
        env_idx: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        arm_controller = self._get_arm_controller()
        if arm_controller is None:
            raise RuntimeError("Cannot find arm controller for IK during reset.")
        kinematics = getattr(arm_controller, "kinematics", None)
        if kinematics is None or not hasattr(kinematics, "compute_ik"):
            raise RuntimeError("Arm controller has no kinematics.compute_ik API.")

        current_pose = self.agent.ee_pose_at_robot_base[env_idx]
        if device.type == "cuda":
            return self._solve_arm_ik_qpos_gpu_incremental(
                kinematics=kinematics,
                arm_controller=arm_controller,
                current_pose=current_pose,
                target_pose=target_pose,
                seed_qpos=seed_qpos,
                device=device,
                dtype=dtype,
            )

        ik_qpos = kinematics.compute_ik(
            pose=target_pose,
            q0=seed_qpos,
            is_delta_pose=False,
            current_pose=current_pose,
            solver_config=arm_controller.config.delta_solver_config,
        )
        return self._normalize_ik_qpos(
            ik_qpos=ik_qpos,
            seed_qpos=seed_qpos,
            device=device,
            dtype=dtype,
        )

    def _solve_arm_ik_qpos_gpu_incremental(
        self,
        kinematics,
        arm_controller,
        current_pose: Pose,
        target_pose: Pose,
        seed_qpos: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        qpos = seed_qpos.clone()
        start_pose = current_pose
        prev_waypoint = current_pose
        num_steps = self._estimate_gpu_reset_ik_steps(
            current_pose=current_pose,
            target_pose=target_pose,
        )

        for step_idx in range(1, num_steps + 1):
            alpha = step_idx / num_steps
            waypoint = self._interpolate_pose(start_pose, target_pose, alpha)
            ik_qpos = kinematics.compute_ik(
                pose=waypoint,
                q0=qpos,
                is_delta_pose=False,
                current_pose=prev_waypoint,
                solver_config=arm_controller.config.delta_solver_config,
            )
            arm_qpos = self._normalize_ik_qpos(
                ik_qpos=ik_qpos,
                seed_qpos=qpos,
                device=device,
                dtype=dtype,
            )
            qpos[:, :7] = arm_qpos
            prev_waypoint = waypoint

        return qpos[:, :7]

    def _normalize_ik_qpos(
        self,
        ik_qpos,
        seed_qpos: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if ik_qpos is None:
            return seed_qpos[:, :7]

        ik_qpos = torch.as_tensor(ik_qpos, device=device, dtype=dtype)
        if ik_qpos.ndim == 1:
            ik_qpos = ik_qpos.unsqueeze(0)
        if ik_qpos.shape[0] == 1 and seed_qpos.shape[0] > 1:
            ik_qpos = ik_qpos.repeat(seed_qpos.shape[0], 1)
        if ik_qpos.shape[1] < 7:
            raise RuntimeError(
                f"IK result has invalid shape {tuple(ik_qpos.shape)}, expected at least 7 joints."
            )
        return ik_qpos[:, :7]

    def _estimate_gpu_reset_ik_steps(
        self,
        current_pose: Pose,
        target_pose: Pose,
    ) -> int:
        position_delta = torch.linalg.norm(target_pose.p - current_pose.p, dim=1)
        quat_dot = torch.sum(current_pose.q * target_pose.q, dim=1).abs().clamp(max=1.0)
        rotation_delta = 2.0 * torch.arccos(quat_dot)
        pos_steps = torch.ceil(position_delta / self.GPU_RESET_IK_MAX_POS_STEP)
        rot_steps = torch.ceil(rotation_delta / self.GPU_RESET_IK_MAX_ROT_STEP)
        num_steps = torch.maximum(pos_steps, rot_steps)
        return max(1, int(num_steps.max().item()))

    def _interpolate_pose(
        self,
        start_pose: Pose,
        target_pose: Pose,
        alpha: float,
    ) -> Pose:
        position = torch.lerp(start_pose.p, target_pose.p, alpha)
        quaternion = self._slerp_quaternion(start_pose.q, target_pose.q, alpha)
        return Pose.create_from_pq(p=position, q=quaternion)

    def _slerp_quaternion(
        self, start_q: torch.Tensor, target_q: torch.Tensor, alpha: float
    ) -> torch.Tensor:
        dot = torch.sum(start_q * target_q, dim=1, keepdim=True)
        target_q = torch.where(dot < 0.0, -target_q, target_q)
        dot = torch.sum(start_q * target_q, dim=1, keepdim=True).clamp(-1.0, 1.0)

        alpha_t = torch.full_like(dot, float(alpha))
        linear_mask = dot.abs() > 0.9995

        theta_0 = torch.arccos(dot)
        sin_theta_0 = torch.sin(theta_0).clamp(min=1e-6)
        theta = theta_0 * alpha_t
        s0 = torch.sin(theta_0 - theta) / sin_theta_0
        s1 = torch.sin(theta) / sin_theta_0
        slerped = s0 * start_q + s1 * target_q
        lerped = (1.0 - alpha_t) * start_q + alpha_t * target_q
        quat = torch.where(linear_mask, lerped, slerped)
        return quat / torch.linalg.norm(quat, dim=1, keepdim=True).clamp(min=1e-6)

    def _get_arm_controller(self):
        controller = getattr(self.agent, "controller", None)
        if controller is None:
            return None

        controllers = getattr(controller, "controllers", None)
        if isinstance(controllers, dict) and "arm" in controllers:
            return controllers["arm"]

        return None

    def _sample_button_pose(self, env_idx: torch.Tensor) -> Pose:
        button_pose = self.task_alignment.get("button_initial_pose", None)
        if button_pose is None:
            button_pose = np.hstack([self.BUTTON_POSE.p, self.BUTTON_POSE.q])
        button_pose = np.asarray(button_pose, dtype=np.float32).reshape(-1)
        if button_pose.size != 7:
            raise ValueError(
                "task_alignment.button_initial_pose must contain 7 values [x, y, z, qx, qy, qz, qw]."
            )

        b = len(env_idx)
        position = np.repeat(button_pose[:3][None, :], b, axis=0)
        random_xy_range = float(
            self.task_alignment.get(
                "button_random_xy_range", self.DEFAULT_BUTTON_RANDOM_XY_RANGE
            )
        )
        xy_offset = self._batched_episode_rng[env_idx].uniform(
            -random_xy_range, random_xy_range, size=(2,)
        )
        position[:, :2] += xy_offset
        quat = np.repeat(button_pose[3:][None, :], b, axis=0)
        return Pose.create_from_pq(
            p=torch.as_tensor(position, device=self.device, dtype=torch.float32),
            q=torch.as_tensor(quat, device=self.device, dtype=torch.float32),
        )

    def _get_button_target_position(self) -> torch.Tensor:
        target_position = self.button.pose.p.clone()
        target_position[:, 2] += self.SUCCESS_TARGET_Z_OFFSET
        return target_position

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
        target_position = self._get_button_target_position()
        tcp_position = self.agent.tcp.pose.p
        target_delta = torch.abs(tcp_position - target_position)
        button_contact_force = self._get_button_contact_force()
        is_button_pressed = button_contact_force > self.BUTTON_FORCE_THRESHOLD
        success = torch.all(
            target_delta
            <= torch.full_like(target_delta, self.SUCCESS_POSITION_THRESHOLD),
            dim=1,
        )
        return {
            "button_contact_force": button_contact_force,
            "is_button_pressed": is_button_pressed,
            "button_target_position": target_position,
            "target_delta": target_delta,
            "success": success,
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
        target_delta = info["target_delta"]
        use_dense_reward = bool(self.task_alignment.get("use_dense_reward", False))
        reward_scale = float(self.task_alignment.get("reward_scale", 1.0))
        return self._compute_aligned_reward(
            target_delta=target_delta,
            success=info["success"],
            use_dense_reward=use_dense_reward,
            reward_scale=reward_scale,
            dense_reward_decay=self.DENSE_REWARD_DECAY,
        )

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)

    @staticmethod
    def _compute_aligned_reward(
        target_delta: torch.Tensor,
        success: torch.Tensor,
        use_dense_reward: bool,
        reward_scale: float,
        dense_reward_decay: float,
    ) -> torch.Tensor:
        squared_position_error = torch.sum(torch.square(target_delta), dim=1)
        dense_reward = torch.exp(-dense_reward_decay * squared_position_error)
        reward = torch.zeros_like(dense_reward)
        if use_dense_reward:
            reward = dense_reward
        reward[success] = 1.0
        return reward * reward_scale

    def _build_button(self):
        mesh_filename = self.BUTTON_ASSET_DIR / "button_mesh_abs.glb"
        builder = self.scene.create_actor_builder()

        scale = 1.0
        scale_array = [scale] * 3

        builder.add_multiple_convex_collisions_from_file(
            filename=str(mesh_filename),
            scale=scale_array,
            material=PhysxMaterial(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            density=1000.0,
            patch_radius=0.1,
            min_patch_radius=0.1,
            decomposition="coacd",
        )

        builder.add_visual_from_file(
            filename=str(mesh_filename),
            scale=scale_array,
        )

        builder.initial_pose = self.BUTTON_POSE
        return builder.build_kinematic(name="button")

    def _build_extracted_obs(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        sensor_data = raw_obs.get("sensor_data", {})
        image_data = raw_obs.get("image", {})
        main_camera_obs = image_data.get("3rdview_camera", sensor_data["3rdview_camera"])
        main_images = self._pad_and_resize_images(
            main_camera_obs["rgb"].to(torch.uint8)
        )

        extracted_obs = {
            "main_images": main_images,
            "task_descriptions": self.get_language_instruction(),
        }

        if "hand_camera" in sensor_data:
            hand_camera_obs = image_data.get("hand_camera", sensor_data["hand_camera"])
            extracted_obs["extra_view_images"] = (
                self._pad_and_resize_images(
                    hand_camera_obs["rgb"].to(torch.uint8)
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
