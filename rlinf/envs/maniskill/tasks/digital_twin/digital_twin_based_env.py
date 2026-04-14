import os
from typing import Any, Optional, Union

import cv2
import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import Camera, CameraConfig
from mani_skill.utils import common
from mani_skill.utils.structs import Actor, Articulation, Link
from mani_skill.utils.structs.pose import Pose

from rlinf.envs.maniskill.tasks.digital_twin import (
    DIGITAL_TWIN_BACKGROUND_IMAGES,
    PANDA_UMI_CAMERA_POSES,
    PANDA_UMI_CAMERA_SETTINGS,
)
from rlinf.envs.maniskill.tasks.digital_twin.robots import PandaUMI
from rlinf.envs.maniskill.tasks.digital_twin.scene import TableSceneBuilder


ForegroundActor = Actor | Articulation | Link


class DigitalTwinBaseEnv(BaseEnv):
    """Minimal tabletop base env with optional fused RGB observations."""

    SUPPORTED_ROBOTS = ["panda_umi"]
    CAMERA_NAMES = ("3rdview_camera", "hand_camera")
    agent: PandaUMI

    def __init__(
        self,
        *args,
        robot_uids="panda_umi",
        robot_init_qpos_noise=0.02,
        overwrite_rgb_in_obs: bool = True,
        **kwargs,
    ):
        if robot_uids != "panda_umi":
            raise NotImplementedError(
                "DigitalTwinBaseEnv currently only supports robot_uids='panda_umi'"
            )

        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.overwrite_rgb_in_obs = overwrite_rgb_in_obs
        self._background_images_np = self._load_background_images()
        self._background_images: dict[str, torch.Tensor] = {}
        self._robot_segmentation_ids: torch.Tensor | None = None
        self._object_segmentation_ids: torch.Tensor | None = None

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_background_images(self) -> dict[str, Any]:
        images = {}
        for camera_name, path in DIGITAL_TWIN_BACKGROUND_IMAGES.items():
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Failed to load background image for {camera_name}: {path}")
            images[camera_name] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return images

    def _camera_pose_from_constants(self, camera_name: str) -> sapien.Pose:
        camera_cfg = PANDA_UMI_CAMERA_POSES[camera_name]
        return sapien.Pose(p=camera_cfg["p"], q=camera_cfg["q"])

    def _camera_settings_from_constants(self, camera_name: str) -> dict[str, float | int]:
        return dict(PANDA_UMI_CAMERA_SETTINGS[camera_name])

    @property
    def _default_sensor_configs(self):
        camera_name = "3rdview_camera"
        camera_settings = self._camera_settings_from_constants(camera_name)
        return [
            CameraConfig(
                uid=camera_name,
                pose=self._camera_pose_from_constants(camera_name),
                width=camera_settings["width"],
                height=camera_settings["height"],
                near=camera_settings["near"],
                far=camera_settings["far"],
                fov=np.deg2rad(camera_settings["fov_deg"]),
                mount=self.agent.robot.links_map[
                    PANDA_UMI_CAMERA_POSES[camera_name]["mount_link"]
                ],
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        camera_name = "3rdview_camera"
        camera_settings = self._camera_settings_from_constants(camera_name)
        return CameraConfig(
            "render_camera",
            pose=self._camera_pose_from_constants(camera_name),
            width=camera_settings["width"],
            height=camera_settings["height"],
            near=camera_settings["near"],
            far=camera_settings["far"],
            fov=np.deg2rad(camera_settings["fov_deg"]),
            shader_pack="default",
            mount=self.agent.robot.links_map[
                PANDA_UMI_CAMERA_POSES[camera_name]["mount_link"]
            ],
        )

    @property
    def _default_viewer_camera_configs(self):
        return CameraConfig(
            uid="viewer",
            pose=self._camera_pose_from_constants("3rdview_camera"),
            width=1920,
            height=1080,
            near=0.01,
            far=100,
            fov=np.pi / 2,
            shader_pack="default",
        )

    def _load_lighting(self, options: dict):
        shadow = self.enable_shadow
        
        self.scene.set_ambient_light([0.05, 0.05, 0.25])
        self.scene.add_directional_light(
            [1, -0.2, -0.2], [1.0, 1.0, 1.0], shadow=shadow, shadow_scale=5, shadow_map_size=2048
        )
        self.scene.add_directional_light(
            [0, 1, -1], [1.8, 1.8, 1.8], shadow=shadow, shadow_scale=5, shadow_map_size=2048
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self,
            robot_init_qpos_noise=self.robot_init_qpos_noise,
        )
        self.table_scene.build()
        self._load_task_scene(options)

    def _load_task_scene(self, options: dict):
        """Hook for subclasses to create task-specific actors."""

    def _get_foreground_actors(self) -> list[ForegroundActor]:
        """Hook for subclasses to mark task objects kept in fused RGB."""
        return []

    def _after_reconfigure(self, options):
        super()._after_reconfigure(options)
        self._resize_background_images()
        self._refresh_segmentation_ids()
        self._apply_hand_camera_pose_constant()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.table_scene.initialize(env_idx)
        self._apply_hand_camera_pose_constant()

    # TODO important to check. You should collect real image with the size same with sim camera size.
    def _resize_background_images(self):
        # Real background images may have arbitrary resolutions from captured photos.
        # Resize them once after camera configs are finalized so they can be blended
        # pixel-wise with simulated RGB without doing per-frame resize work.
        self._background_images = {}
        for camera_name, image in self._background_images_np.items():
            if camera_name not in self._sensor_configs:
                continue
            sensor_config = self._sensor_configs[camera_name]
            resized = cv2.resize(
                image,
                (sensor_config.width, sensor_config.height),
                interpolation=cv2.INTER_LINEAR,
            )
            self._background_images[camera_name] = common.to_tensor(
                resized, device=self.device
            )

    def _collect_segmentation_ids(self, actors: list[ForegroundActor]) -> torch.Tensor:
        # Segmentation images store per-scene ids, not semantic labels. Convert the
        # configured robot/task entities into a flat id list so masks can be built
        # with a simple isin() against the segmentation buffer.
        ids = []
        for actor in actors:
            if isinstance(actor, Articulation):
                ids.extend(
                    common.to_tensor(link.per_scene_id, device=self.device).reshape(-1)
                    for link in actor.get_links()
                )
            else:
                ids.append(common.to_tensor(actor.per_scene_id, device=self.device).reshape(-1))
        if len(ids) == 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        return torch.unique(torch.concatenate(ids))

    def _refresh_segmentation_ids(self):
        self._robot_segmentation_ids = self._collect_segmentation_ids([self.agent.robot])
        self._object_segmentation_ids = self._collect_segmentation_ids(
            self._get_foreground_actors()
        )

    def _apply_hand_camera_pose_constant(self):
        if "hand_camera" not in self._sensors:
            return
        pose = self._camera_pose_from_constants("hand_camera")
        self.set_camera_local_pose("hand_camera", pose)

    def set_camera_local_pose(self, camera_name: str, pose: sapien.Pose):
        if camera_name not in self._sensors:
            raise KeyError(f"Unknown camera: {camera_name}")
        sensor = self._sensors[camera_name]
        sensor.camera.set_local_pose(pose)
        self._sensor_configs[camera_name].pose = Pose.create(pose, device=self.device)

    def get_camera_local_pose(self, camera_name: str) -> sapien.Pose:
        if camera_name not in self._sensors:
            raise KeyError(f"Unknown camera: {camera_name}")
        return Pose.create(self._sensors[camera_name].camera.get_local_pose(), device=self.device).sp
    
    # overwrite for get segmentation for env.get_obs()
    def _get_obs_sensor_data(self, apply_texture_transforms: bool = True) -> dict:
        # This overrides ManiSkill's default camera capture path so fused observations
        # can work with plain obs_mode="rgb". When rgb is requested, we internally
        # also grab segmentation for the third-view / hand cameras to build masks and
        # fused RGB, then optionally strip segmentation back out if the user did not
        # explicitly request it in the final obs.
        #
        # apply_texture_transforms=True returns standard processed outputs such as
        # rgb/depth/segmentation. False returns lower-level raw textures, so in that
        # case we defer to ManiSkill's default implementation.
        if not apply_texture_transforms:
            return super()._get_obs_sensor_data(apply_texture_transforms=False)

        for obj in self._hidden_objects:
            obj.hide_visual()
        self.scene.update_render(update_sensors=True, update_human_render_cameras=False)
        self.capture_sensor_data()

        sensor_obs = {}
        for name, sensor in self.scene.sensors.items():
            if not isinstance(sensor, Camera):
                continue
            if self.obs_mode in ["state", "state_dict"]:
                sensor_obs[name] = sensor.get_obs(
                    position=False,
                    segmentation=False,
                    apply_texture_transforms=apply_texture_transforms,
                )
                continue
            # modify the bool to get the segmentation for env.get_obs()
            need_internal_seg = name in self.CAMERA_NAMES and self.obs_mode_struct.visual.rgb
            need_internal_rgb = name in self.CAMERA_NAMES and self.obs_mode_struct.visual.rgb
            sensor_obs[name] = sensor.get_obs(
                rgb=self.obs_mode_struct.visual.rgb or need_internal_rgb,
                depth=self.obs_mode_struct.visual.depth,
                position=self.obs_mode_struct.visual.position,
                segmentation=self.obs_mode_struct.visual.segmentation or need_internal_seg,
                normal=self.obs_mode_struct.visual.normal,
                albedo=self.obs_mode_struct.visual.albedo,
                apply_texture_transforms=apply_texture_transforms,
            )

        if self.backend.render_device.is_cuda():
            torch.cuda.synchronize()
        return sensor_obs

    def _make_mask(self, segmentation: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        actor_seg = segmentation[..., 0]
        if len(ids) == 0:
            return torch.zeros_like(actor_seg, dtype=torch.bool)
        return torch.isin(actor_seg, ids.to(device=actor_seg.device, dtype=actor_seg.dtype))

    def _background_for_camera(self, camera_name: str, rgb: torch.Tensor) -> torch.Tensor:
        if camera_name not in self._background_images: # for wrist camera
            return rgb
        background = self._background_images[camera_name]
        if background.ndim == 3:
            background = background.unsqueeze(0)
        if background.shape[0] == 1 and rgb.shape[0] > 1:
            background = background.repeat(rgb.shape[0], 1, 1, 1)
        return background.to(device=rgb.device, dtype=rgb.dtype)

    def _build_camera_image_obs(self, camera_name: str, camera_obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raw_rgb = camera_obs["rgb"]
        segmentation = camera_obs["segmentation"]
        robot_mask = self._make_mask(segmentation, self._robot_segmentation_ids)
        object_mask = self._make_mask(segmentation, self._object_segmentation_ids)
        foreground_mask = robot_mask | object_mask
        background_rgb = self._background_for_camera(camera_name, raw_rgb)
        fused_rgb = torch.where(foreground_mask[..., None], raw_rgb, background_rgb)
        return {
            "rgb": fused_rgb,
            "fused_rgb": fused_rgb,
            "raw_rgb": raw_rgb,
            "robot_mask": robot_mask,
            "object_mask": object_mask,
        }

    def get_fused_camera_obs(
        self, camera_name: str, obs: dict | None = None, info: dict | None = None
    ) -> dict[str, torch.Tensor]:
        if camera_name not in self.CAMERA_NAMES:
            raise KeyError(f"Unsupported camera_name={camera_name}")
        if obs is None:
            obs = super().get_obs(info=info, unflattened=True)
        camera_obs = obs["sensor_data"][camera_name]
        if "segmentation" not in camera_obs:
            obs = super().get_obs(info=info, unflattened=True)
            camera_obs = obs["sensor_data"][camera_name]
        if "rgb" not in camera_obs or "segmentation" not in camera_obs:
            raise RuntimeError(
                f"Camera {camera_name} does not contain rgb+segmentation under obs_mode={self.obs_mode}"
            )
        return self._build_camera_image_obs(camera_name, camera_obs)

    def _strip_internal_segmentation(self, obs: dict):
        if self.obs_mode_struct.visual.segmentation or "sensor_data" not in obs:
            return
        for camera_name in self.CAMERA_NAMES:
            if camera_name in obs["sensor_data"]:
                obs["sensor_data"][camera_name].pop("segmentation", None)

    def _inject_image_obs(self, obs: dict):
        if not self.overwrite_rgb_in_obs or "sensor_data" not in obs:
            return
        image_obs = {}
        for camera_name in self.CAMERA_NAMES:
            if camera_name not in obs["sensor_data"]:
                continue
            camera_obs = obs["sensor_data"][camera_name]
            if "rgb" not in camera_obs or "segmentation" not in camera_obs:
                continue
            image_obs[camera_name] = self._build_camera_image_obs(camera_name, camera_obs)
        if len(image_obs) > 0:
            obs["image"] = image_obs

    def get_obs(self, info: dict | None = None, unflattened: bool = False):
        obs = super().get_obs(info=info, unflattened=True)
        if isinstance(obs, dict) and "sensor_data" in obs:
            self._inject_image_obs(obs)
            self._strip_internal_segmentation(obs)
        return obs if unflattened else self._flatten_raw_obs(obs)

    def _get_obs_extra(self, info: dict):
        return {}
    
    def step(self, action: Union[None, np.ndarray, torch.Tensor, dict]):
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(self.device)
        else:
            action = action.to(self.device)
        
        
        
        return super().step(action)
