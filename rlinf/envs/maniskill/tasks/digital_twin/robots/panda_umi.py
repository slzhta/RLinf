# TODO(liangzhi): There may have somethin to change
import sapien
import numpy as np
from copy import deepcopy
from pathlib import Path
from typing import Any

from mani_skill.agents.controllers import deepcopy_dict
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.sensors.camera import CameraConfig

from rlinf.envs.maniskill.tasks.digital_twin.controller import (
    SafePDEEPoseControllerConfig,
    SafePDJointPosMimicControllerConfig,
)


def _as_fixed_float_list(name: str, value: Any, size: int) -> list[float]:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size != size:
        raise ValueError(f"{name} must contain exactly {size} values.")
    return arr.tolist()


def _as_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool.")
    return value


def _normalize_controller_alignment_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError("controller_alignment must be a mapping")
    if "controller_alignment" in cfg and isinstance(cfg["controller_alignment"], dict):
        return dict(cfg["controller_alignment"])
    return dict(cfg)

@register_agent()
class PandaUMI(Panda):
    """Panda arm robot with the real sense camera attached to gripper"""

    uid = "panda_umi"
    urdf_path = str(
        Path(__file__).resolve().parents[1] / "assets" / "robots" / "panda_umi.urdf"
    )

    # Franka-aligned action semantics: action is clipped to [-1, 1], then scaled.
    action_scale = [0.1, 0.1, 1.0]  # [xyz_scale, rpy_scale, gripper_scale]
    arm_action_scale = [0.1, 0.1]  # [xyz_scale, rpy_scale]
    gripper_action_scale = 1.0
    gripper_binary_threshold = 0.5
    target_ee_pose = [0.5, 0.0, 0.1, -3.14, 0.0, 0.0]
    use_target_controller = False
    # Safety limits in robot base frame [x, y, z, rx, ry, rz].
    ee_pose_limit_min = [0.15, -0.45, 0.00, -np.pi, -np.pi, -np.pi]
    ee_pose_limit_max = [0.85, 0.45, 0.60, np.pi, np.pi, np.pi]

    def __init__(
        self,
        *args,
        controller_alignment: dict[str, Any] | None = None,
        **kwargs,
    ):
        self._controller_alignment_cfg = _normalize_controller_alignment_config(
            controller_alignment
        )
        super().__init__(*args, **kwargs)

    @property
    def _sensor_configs(self):
        from rlinf.envs.maniskill.tasks.digital_twin import PANDA_UMI_CAMERA_SETTINGS

        camera_name = "hand_camera"
        camera_settings = PANDA_UMI_CAMERA_SETTINGS[camera_name]
        return [
            CameraConfig(
                uid=camera_name,
                pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
                width=camera_settings["width"],
                height=camera_settings["height"],
                near=camera_settings["near"],
                far=camera_settings["far"],
                fov=np.deg2rad(camera_settings["fov_deg"]),
                mount=self.robot.links_map["camera_link"],
            )
        ]

    @property
    def ee_pose_at_robot_base(self):
        """Return the TCP pose expressed in the robot base frame."""
        to_base = self.robot.pose.inv()
        return to_base * self.tcp.pose

    @property
    def _controller_configs(self):
        alignment_cfg = self._controller_alignment_cfg
        action_scale_cfg = alignment_cfg.get("action_scale", None)
        if action_scale_cfg is not None:
            action_scale = _as_fixed_float_list("action_scale", action_scale_cfg, 3)
            arm_action_scale = action_scale[:2]
            gripper_action_scale = float(action_scale[2])
        else:
            arm_action_scale = _as_fixed_float_list(
                "arm_action_scale",
                alignment_cfg.get("arm_action_scale", self.arm_action_scale),
                2,
            )
            gripper_action_scale = float(
                alignment_cfg.get("gripper_action_scale", self.gripper_action_scale)
            )

        target_ee_pose = _as_fixed_float_list(
            "target_ee_pose",
            alignment_cfg.get("target_ee_pose", self.target_ee_pose),
            6,
        )
        target_euler = target_ee_pose[3:]
        ee_pose_limit_min = _as_fixed_float_list(
            "ee_pose_limit_min",
            alignment_cfg.get("ee_pose_limit_min", self.ee_pose_limit_min),
            6,
        )
        ee_pose_limit_max = _as_fixed_float_list(
            "ee_pose_limit_max",
            alignment_cfg.get("ee_pose_limit_max", self.ee_pose_limit_max),
            6,
        )

        threshold_raw = alignment_cfg.get(
            "gripper_binary_threshold",
            alignment_cfg.get(
                "binary_gripper_threshold", self.gripper_binary_threshold
            ),
        )
        gripper_binary_threshold = float(threshold_raw)
        use_target_controller = _as_bool(
            "use_target_controller",
            alignment_cfg.get("use_target_controller", self.use_target_controller),
        )
        binary_gripper_action = _as_bool(
            "binary_gripper_action",
            alignment_cfg.get("binary_gripper_action", True),
        )
        use_zero_one_gripper_action = _as_bool(
            "use_zero_one_gripper_action",
            alignment_cfg.get("use_zero_one_gripper_action", False),
        )
        open_command = float(alignment_cfg.get("open_command", 1.0))
        close_command = float(alignment_cfg.get("close_command", -1.0))

        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_ee_delta_pose_real_root_frame = SafePDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-1.0,
            pos_upper=1.0,
            rot_lower=-1.0,
            rot_upper=1.0,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            normalize_action=False,
            action_scale=arm_action_scale,
            ee_pose_limit_min=ee_pose_limit_min,
            ee_pose_limit_max=ee_pose_limit_max,
            target_euler=target_euler,
        )
        arm_pd_ee_delta_pose_real = deepcopy(arm_pd_ee_delta_pose_real_root_frame)
        arm_pd_ee_delta_pose_real.frame = "root_translation:root_aligned_body_rotation"
        arm_pd_ee_delta_pose_real.use_target = use_target_controller

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        gripper_pd_joint_pos = SafePDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower=-0.01,
            upper=0.04,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            normalize_action=True,
            drive_mode="force",
            action_scale=gripper_action_scale,
            binary_gripper_action=binary_gripper_action,
            binary_gripper_threshold=gripper_binary_threshold,
            use_zero_one_gripper_action=use_zero_one_gripper_action,
            open_command=open_command,
            close_command=close_command,
        )

        controller_configs = {
            "pd_ee_body_target_delta_pose_real": {
                "arm": arm_pd_ee_delta_pose_real,
                "gripper": gripper_pd_joint_pos,
            },
        }

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)
