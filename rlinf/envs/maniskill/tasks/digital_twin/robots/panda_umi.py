# TODO(liangzhi): There may have somethin to change
import sapien
from copy import deepcopy

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.controllers import deepcopy_dict
from mani_skill.agents.registration import register_agent
from mani_skill.agents.controllers.pd_ee_pose import PDEEPoseControllerConfig
from mani_skill.agents.controllers.pd_joint_pos import PDJointPosMimicControllerConfig
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.sensors.camera import CameraConfig

@register_agent()
class PandaUMI(Panda):
    """Panda arm robot with the real sense camera attached to gripper"""

    uid = "panda_umi"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_umi.urdf"

    @property
    def _sensor_configs(self):
        import numpy as np
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
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_ee_delta_pose_real_root_frame = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,  # -1.0,
            pos_upper=0.1,  # 1.0,
            rot_lower=-0.1,  # -np.pi / 2,
            rot_upper=0.1,  # np.pi / 2,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            normalize_action=False,
        )
        arm_pd_ee_delta_pose_real = deepcopy(arm_pd_ee_delta_pose_real_root_frame)
        arm_pd_ee_delta_pose_real.frame = "root_translation:root_aligned_body_rotation"
        arm_pd_ee_delta_pose_real.use_target = True

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower=-0.01,
            upper=0.04,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            normalize_action=True,
            drive_mode="force",
        )

        controller_configs = {
            "pd_ee_body_target_delta_pose_real": {
                "arm": arm_pd_ee_delta_pose_real,
                "gripper": gripper_pd_joint_pos,
            },
        }

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)
