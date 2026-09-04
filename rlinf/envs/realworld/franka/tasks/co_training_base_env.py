# TODO: This is a base env for co-training method
# We should tuning it's controller

import copy
import queue
import time
from dataclasses import dataclass, field

import cv2
import gymnasium as gym
import numpy as np

from ..franka_env import FrankaEnv, FrankaRobotConfig


@dataclass
class FrankaCoTrainingBaseConfig(FrankaRobotConfig):
    task_description: str = "Pick up the object and put it into another bin"
    random_xy_range: float = 0.01  # for randomization
    clip_x_range: float = 0.2  # for bounding box
    clip_y_range: float = 0.3  # for bounding box
    clip_z_range_high: float = 0.3
    clip_z_range_low: float = 0.001
    random_rz_range: float = np.pi / 9  # for random reset
    clip_rz_range: float = np.pi / 5  # for bounding box
    clip_rp_range: float = np.pi / 6
    enable_random_reset: bool = True
    enable_inner_safety_box: bool = True

    target_ee_pose: np.ndarray = field(default_factory=lambda: np.zeros(6))
    reward_threshold: np.ndarray = field(
        default_factory=lambda: np.array([0.01, 0.01, 0.01, 0.2, 0.2, 0.2])
    )

    def __post_init__(self):
        self.compliance_param = {
            "translational_stiffness": 3000,
            "translational_damping": 89,
            "rotational_stiffness": 300,
            "rotational_damping": 9,
            "translational_Ki": 0.1,
            "translational_clip_x": 0.01,
            "translational_clip_y": 0.01,
            "translational_clip_z": 0.01,
            "translational_clip_neg_x": 0.01,
            "translational_clip_neg_y": 0.01,
            "translational_clip_neg_z": 0.01,
            "rotational_clip_x": 0.05,
            "rotational_clip_y": 0.05,
            "rotational_clip_z": 0.05,
            "rotational_clip_neg_x": 0.05,
            "rotational_clip_neg_y": 0.05,
            "rotational_clip_neg_z": 0.05,
            "rotational_Ki": 0.1,
        }
        self.precision_param = {
            "translational_stiffness": 3000,
            "translational_damping": 89,
            "rotational_stiffness": 300,
            "rotational_damping": 9,
            "translational_Ki": 0.1,
            "translational_clip_x": 0.01,
            "translational_clip_y": 0.01,
            "translational_clip_z": 0.01,
            "translational_clip_neg_x": 0.01,
            "translational_clip_neg_y": 0.01,
            "translational_clip_neg_z": 0.01,
            "rotational_clip_x": 0.05,
            "rotational_clip_y": 0.05,
            "rotational_clip_z": 0.05,
            "rotational_clip_neg_x": 0.05,
            "rotational_clip_neg_y": 0.05,
            "rotational_clip_neg_z": 0.05,
            "rotational_Ki": 0.1,
        }
        self.target_ee_pose = np.array(self.target_ee_pose)
        self.reset_ee_pose = self.target_ee_pose + np.array(
            [0.0, 0.0, (self.clip_z_range_high + self.clip_z_range_low) / 2, 0.0, 0.0, 0.0]
        )
        self.reward_threshold = np.array(self.reward_threshold)
        self.action_scale = np.array([0.01, 0.05, 1])
        self.ee_pose_limit_min = np.array(
            [
                self.target_ee_pose[0] - self.clip_x_range,
                self.target_ee_pose[1] - self.clip_y_range,
                self.target_ee_pose[2] - self.clip_z_range_low,
                self.target_ee_pose[3] - self.clip_rp_range,
                self.target_ee_pose[4] - self.clip_rp_range,
                self.target_ee_pose[5] - self.clip_rz_range,
            ]
        )
        self.ee_pose_limit_max = np.array(
            [
                self.target_ee_pose[0] + self.clip_x_range,
                self.target_ee_pose[1] + self.clip_y_range,
                self.target_ee_pose[2] + self.clip_z_range_high,
                self.target_ee_pose[3] + self.clip_rp_range,
                self.target_ee_pose[4] + self.clip_rp_range,
                self.target_ee_pose[5] + self.clip_rz_range,
            ]
        )


class FrankaCoTrainingBaseEnv(FrankaEnv):
    CONFIG_CLS = FrankaCoTrainingBaseConfig

    def __init__(self, override_cfg, worker_info=None, hardware_info=None, env_idx=0):
        super().__init__(override_cfg, worker_info, hardware_info, env_idx)
        self.task_id = 0  # 0 for forward task, 1 for backward task
        """
        the inner safety box is used to prevent the gripper from hitting the two walls of the bins in the center.
        it is particularly useful when there is things you want to avoid running into within the bounding box.
        it uses the intersect_line_bbox function to detect whether the gripper is going to hit the wall
        and clips actions that will lead to collision.
        """
        self.inner_safety_box = gym.spaces.Box(
            self.config.target_ee_pose[:3] - np.array([0.07, 0.03, 0.001]),
            self.config.target_ee_pose[:3] + np.array([0.07, 0.03, 0.04]),
            dtype=np.float64,
        )

    def intersect_line_bbox(self, p1, p2, bbox_min, bbox_max):
        # Define the parameterized line segment
        # P(t) = p1 + t(p2 - p1)
        tmin = 0
        tmax = 1

        for i in range(3):
            if p1[i] < bbox_min[i] and p2[i] < bbox_min[i]:
                return None
            if p1[i] > bbox_max[i] and p2[i] > bbox_max[i]:
                return None

            # For each axis (x, y, z), compute t values at the intersection points
            if abs(p2[i] - p1[i]) > 1e-10:  # To prevent division by zero
                t1 = (bbox_min[i] - p1[i]) / (p2[i] - p1[i])
                t2 = (bbox_max[i] - p1[i]) / (p2[i] - p1[i])

                # Ensure t1 is smaller than t2
                if t1 > t2:
                    t1, t2 = t2, t1

                tmin = max(tmin, t1)
                tmax = min(tmax, t2)

                if tmin > tmax:
                    return None

        # Compute the intersection point using the t value
        intersection = p1 + tmin * (p2 - p1)

        return intersection

    def _clip_position_to_safety_box(self, pose):
        pose = super()._clip_position_to_safety_box(pose)
        # Clip xyz to inner box
        if (
            self.config.enable_inner_safety_box
            and self.inner_safety_box.contains(pose[:3])
        ):
            pose[:3] = self.intersect_line_bbox(
                self._franka_state.tcp_pose[:3],
                pose[:3],
                self.inner_safety_box.low,
                self.inner_safety_box.high,
            )
        return pose

    def _crop_frame(self, name, image):
        """Pad realsense images to a square with symmetric black borders."""
        h, w, _ = image.shape
        if h == w:
            return image
        if h < w:
            pad = w - h
            pad_top = pad // 2
            pad_bottom = pad - pad_top
            return cv2.copyMakeBorder(
                image,
                pad_top,
                pad_bottom,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )

        pad = h - w
        pad_left = pad // 2
        pad_right = pad - pad_left
        return cv2.copyMakeBorder(
            image,
            0,
            0,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

    def _get_camera_frames(self):
        images = {}
        display_images = {}
        for camera in self._cameras:
            try:
                rgb = camera.get_frame()
                cropped_rgb = self._crop_frame(camera.name, rgb)
                resized = cv2.resize(
                    cropped_rgb,
                    self.observation_space["frames"][camera.name].shape[:2][::-1],
                )
                images[camera.name] = resized[..., ::-1]
                display_images[camera.name] = resized
                if camera.name == "front":
                    display_images[camera.name + "_full"] = cv2.resize(
                        cropped_rgb, (480, 480)
                    )
                elif camera.name == "wrist_1":
                    display_images[camera.name + "_full"] = cropped_rgb
            except queue.Empty:
                time.sleep(5)
                camera.close()
                self._open_cameras()
                return self._get_camera_frames()

        self.camera_player.put_frame(display_images)
        return images

    def task_graph(self, obs=None):
        if obs is None:
            return (self.task_id + 1) % 2

    def set_task_id(self, task_id):
        self.task_id = task_id

    def reset(self, joint_reset=False, **kwargs):
        raise NotImplementedError(
            "FrankaCoTrainingBaseEnv does not define task-specific reset logic. "
            "Subclasses must implement reset()."
        )

    def go_to_rest(self, joint_reset=False):
        """
        Move to the rest position defined in base class.
        Add a small z offset before going to rest to avoid collision with object.
        """
        self._controller.open_gripper().wait()
        time.sleep(0.6)
        self._franka_state = self._controller.get_state().wait()[0]
        self._move_action(self._franka_state.tcp_pose)
        time.sleep(0.5)
        self._franka_state = self._controller.get_state().wait()[0]

        # Move up to clear the slot
        reset_pose = copy.deepcopy(self._franka_state.tcp_pose)
        reset_pose[2] += 0.10
        self._interpolate_move(reset_pose, timeout=1)

        # execute the go_to_rest method from the parent class
        super().go_to_rest(joint_reset)
