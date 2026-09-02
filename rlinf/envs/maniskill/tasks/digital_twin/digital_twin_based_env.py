from typing import Any, Sequence, Union

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
    DR_BOOL_KEYS = (
        "randomize_lighting",
        "randomize_joint_control",
        "randomize_joint_stiffness",
        "randomize_joint_damping",
        "randomize_joint_force_limit",
        "randomize_joint_friction",
        "randomize_joint_drive_mode",
    )
    DR_RANGE_DEFAULTS: dict[str, tuple[float, float]] = {
        "joint_stiffness_scale_range": (0.7, 1.3),
        "joint_damping_scale_range": (0.7, 1.3),
        "joint_force_limit_scale_range": (0.8, 1.2),
        "joint_friction_abs_range": (0.0, 0.3),
    }
    DR_CHOICE_DEFAULTS: dict[str, tuple[str, ...]] = {
        "joint_drive_mode_choices": ("force", "acceleration"),
    }
    agent: PandaUMI

    def __init__(
        self,
        *args,
        robot_uids="panda_umi",
        robot_init_qpos_noise=0.02,
        overwrite_rgb_in_obs: bool = True,
        use_hand_camera: bool = True,
        synchronize_render: bool = True,
        controller_alignment: dict[str, Any] | None = None,
        reset_alignment: dict[str, Any] | None = None,
        task_alignment: dict[str, Any] | None = None,
        domain_randomization: bool | dict[str, Any] | None = None,
        randomize_lighting: bool | None = None,
        randomize_joint_control: bool | None = None,
        randomize_joint_stiffness: bool | None = None,
        randomize_joint_damping: bool | None = None,
        randomize_joint_force_limit: bool | None = None,
        randomize_joint_friction: bool | None = None,
        randomize_joint_drive_mode: bool | None = None,
        joint_stiffness_scale_range: Sequence[float] | None = None,
        joint_damping_scale_range: Sequence[float] | None = None,
        joint_force_limit_scale_range: Sequence[float] | None = None,
        joint_friction_abs_range: Sequence[float] | None = None,
        joint_drive_mode_choices: Sequence[str] | None = None,
        **kwargs,
    ):
        if robot_uids != "panda_umi":
            raise NotImplementedError(
                "DigitalTwinBaseEnv currently only supports robot_uids='panda_umi'"
            )

        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.overwrite_rgb_in_obs = overwrite_rgb_in_obs
        self.use_hand_camera = use_hand_camera
        self.synchronize_render = synchronize_render
        self.controller_alignment = self._normalize_controller_alignment_arg(
            controller_alignment
        )
        self.reset_alignment = self._normalize_reset_alignment_arg(reset_alignment)
        self.task_alignment = self._normalize_task_alignment_arg(task_alignment)

        domain_randomization_flag, domain_randomization_inline = (
            self._normalize_domain_randomization_arg(domain_randomization)
        )

        dr_cfg = (
            domain_randomization_inline
            if domain_randomization_inline is not None
            else {}
        )

        cfg_master_toggle = self._infer_master_domain_randomization_toggle(dr_cfg)

        # Priority for each DR field: explicit env arg > inline mapping > existing/default.
        self.domain_randomization = self._resolve_bool_attr(
            name="domain_randomization",
            explicit_value=(
                domain_randomization_flag
                if domain_randomization_flag is not None
                else cfg_master_toggle
            ),
            default=False,
        )

        bool_overrides = {
            "randomize_lighting": randomize_lighting,
            "randomize_joint_control": randomize_joint_control,
            "randomize_joint_stiffness": randomize_joint_stiffness,
            "randomize_joint_damping": randomize_joint_damping,
            "randomize_joint_force_limit": randomize_joint_force_limit,
            "randomize_joint_friction": randomize_joint_friction,
            "randomize_joint_drive_mode": randomize_joint_drive_mode,
        }
        for key in self.DR_BOOL_KEYS:
            explicit_value = bool_overrides[key]
            resolved_input = explicit_value if explicit_value is not None else dr_cfg.get(key)
            setattr(
                self,
                key,
                self._resolve_bool_attr(
                    name=key,
                    explicit_value=resolved_input,
                    default=False,
                ),
            )

        range_overrides = {
            "joint_stiffness_scale_range": joint_stiffness_scale_range,
            "joint_damping_scale_range": joint_damping_scale_range,
            "joint_force_limit_scale_range": joint_force_limit_scale_range,
            "joint_friction_abs_range": joint_friction_abs_range,
        }
        for key, default_range in self.DR_RANGE_DEFAULTS.items():
            explicit_value = range_overrides[key]
            resolved_input = explicit_value if explicit_value is not None else dr_cfg.get(key)
            setattr(
                self,
                key,
                self._resolve_range_attr(
                    name=key,
                    explicit_value=resolved_input,
                    default=default_range,
                ),
            )

        choice_overrides = {
            "joint_drive_mode_choices": joint_drive_mode_choices,
        }
        for key, default_choices in self.DR_CHOICE_DEFAULTS.items():
            explicit_value = choice_overrides[key]
            resolved_input = explicit_value if explicit_value is not None else dr_cfg.get(key)
            setattr(
                self,
                key,
                self._resolve_choices_attr(
                    name=key,
                    explicit_value=resolved_input,
                    default=default_choices,
                ),
            )

        if not hasattr(self, "_joint_drive_defaults"):
            self._joint_drive_defaults: dict[
                str, list[tuple[float, float, float, str, float]]
            ] = {}
        if not hasattr(self, "_parallel_randomization_info"):
            self._parallel_randomization_info: list[dict[str, Any]] = []

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
        if not self._is_enabled(self.randomize_lighting):
            shadow = self.enable_shadow
            self.scene.set_ambient_light([0.05, 0.05, 0.25])
            self.scene.add_directional_light(
                [1, -0.2, -0.2],
                [1.0, 1.0, 1.0],
                shadow=shadow,
                shadow_scale=5,
                shadow_map_size=2048,
            )
            self.scene.add_directional_light(
                [0, 1, -1],
                [1.8, 1.8, 1.8],
                shadow=shadow,
                shadow_scale=5,
                shadow_map_size=2048,
            )
            return

        self.scene.set_ambient_light([0.08, 0.08, 0.08])
        self.scene.add_directional_light(
            [0.0, 0.0, -1.0],
            [0.25, 0.25, 0.25],
            shadow=False,
        )

        for scene_idx in range(self.num_envs):
            rng = self._batched_episode_rng[scene_idx]

            spot_position = np.array(
                [
                    rng.uniform(-0.45, 0.45),
                    rng.uniform(-0.45, 0.45),
                    rng.uniform(0.55, 1.25),
                ],
                dtype=np.float32,
            )
            spot_direction = np.array(
                [
                    rng.uniform(-0.5, 0.5),
                    rng.uniform(-0.5, 0.5),
                    rng.uniform(-1.0, -0.2),
                ],
                dtype=np.float32,
            )
            spot_direction = spot_direction / (np.linalg.norm(spot_direction) + 1e-6)
            spot_color = rng.uniform(0.6, 2.4, size=(3,)).tolist()
            inner_fov = np.deg2rad(float(rng.uniform(18.0, 30.0)))
            outer_fov = np.deg2rad(float(rng.uniform(38.0, 68.0)))

            self._set_randomization_value(
                scene_idx,
                "lighting.spot.position",
                [float(v) for v in spot_position.tolist()],
            )
            self._set_randomization_value(
                scene_idx,
                "lighting.spot.direction",
                [float(v) for v in spot_direction.tolist()],
            )
            self._set_randomization_value(
                scene_idx,
                "lighting.spot.color",
                [float(v) for v in spot_color],
            )
            self._set_randomization_value(
                scene_idx,
                "lighting.spot.inner_fov_deg",
                float(np.rad2deg(inner_fov)),
            )
            self._set_randomization_value(
                scene_idx,
                "lighting.spot.outer_fov_deg",
                float(np.rad2deg(outer_fov)),
            )

            self.scene.add_spot_light(
                position=spot_position,
                direction=spot_direction,
                inner_fov=inner_fov,
                outer_fov=outer_fov,
                color=spot_color,
                shadow=self.enable_shadow,
                shadow_map_size=1024,
                scene_idxs=[scene_idx],
            )

            fill_position = np.array(
                [
                    rng.uniform(-0.6, 0.6),
                    rng.uniform(-0.6, 0.6),
                    rng.uniform(0.4, 0.9),
                ],
                dtype=np.float32,
            )
            fill_color = rng.uniform(0.15, 0.9, size=(3,)).tolist()
            self._set_randomization_value(
                scene_idx,
                "lighting.fill.position",
                [float(v) for v in fill_position.tolist()],
            )
            self._set_randomization_value(
                scene_idx,
                "lighting.fill.color",
                [float(v) for v in fill_color],
            )
            self.scene.add_point_light(
                position=fill_position,
                color=fill_color,
                shadow=False,
                scene_idxs=[scene_idx],
            )

    def _load_agent(self, options: dict):
        self.agent = PandaUMI(
            self.scene,
            self._control_freq,
            self._control_mode,
            initial_pose=sapien.Pose(p=[-0.615, 0, 0]),
            controller_alignment=self.controller_alignment,
            enable_hand_camera=self.use_hand_camera,
        )

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self,
            robot_init_qpos_noise=self.robot_init_qpos_noise,
        )
        self.table_scene.build()
        self._load_task_scene(options)

    def _normalize_domain_randomization_arg(
        self,
        domain_randomization: bool | dict[str, Any] | None,
    ) -> tuple[bool | None, dict[str, Any] | None]:
        if domain_randomization is None:
            return None, None
        if isinstance(domain_randomization, bool):
            return domain_randomization, None
        if isinstance(domain_randomization, dict):
            return None, dict(domain_randomization)
        raise TypeError(
            "domain_randomization must be bool or a mapping of randomization settings"
        )

    def _normalize_controller_alignment_arg(
        self,
        controller_alignment: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if controller_alignment is None:
            return {}
        if not isinstance(controller_alignment, dict):
            raise TypeError(
                "controller_alignment must be a mapping of controller settings"
            )
        return dict(controller_alignment)

    def _normalize_reset_alignment_arg(
        self,
        reset_alignment: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if reset_alignment is None:
            return {}
        if not isinstance(reset_alignment, dict):
            raise TypeError("reset_alignment must be a mapping of reset settings")
        return dict(reset_alignment)

    def _normalize_task_alignment_arg(
        self,
        task_alignment: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if task_alignment is None:
            return {}
        if not isinstance(task_alignment, dict):
            raise TypeError("task_alignment must be a mapping of task settings")
        return dict(task_alignment)

    def _infer_master_domain_randomization_toggle(
        self, config: dict[str, Any]
    ) -> bool | None:
        if "enabled" in config:
            return self._resolve_bool_attr(
                name="enabled", explicit_value=config["enabled"], default=False
            )
        if "domain_randomization" in config:
            return self._resolve_bool_attr(
                name="domain_randomization",
                explicit_value=config["domain_randomization"],
                default=False,
            )

        if "randomize_lighting" in config or "randomize_joint_control" in config:
            randomize_lighting = self._resolve_bool_attr(
                name="randomize_lighting",
                explicit_value=config.get("randomize_lighting", False),
                default=False,
            )
            randomize_joint_control = self._resolve_bool_attr(
                name="randomize_joint_control",
                explicit_value=config.get("randomize_joint_control", False),
                default=False,
            )
            return bool(randomize_lighting or randomize_joint_control)

        return None

    def _resolve_bool_attr(
        self,
        name: str,
        explicit_value: bool | None,
        default: bool,
    ) -> bool:
        if explicit_value is not None:
            if not isinstance(explicit_value, bool):
                raise TypeError(f"{name} must be bool, got {type(explicit_value).__name__}")
            return explicit_value
        existing = getattr(self, name, default)
        return bool(existing)

    def _resolve_range_attr(
        self,
        name: str,
        explicit_value: Sequence[float] | None,
        default: tuple[float, float],
    ) -> tuple[float, float]:
        raw: Any = explicit_value if explicit_value is not None else getattr(self, name, default)
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or len(raw) != 2:
            raise ValueError(f"{name} expects exactly 2 values, got: {raw}")
        return float(raw[0]), float(raw[1])

    def _resolve_choices_attr(
        self,
        name: str,
        explicit_value: Sequence[str] | None,
        default: tuple[str, ...],
    ) -> tuple[str, ...]:
        raw: Any = explicit_value if explicit_value is not None else getattr(self, name, default)
        if isinstance(raw, (str, bytes)):
            raise TypeError(f"{name} must be a sequence of strings, got: {raw}")
        if not isinstance(raw, Sequence):
            raise TypeError(f"{name} must be a sequence of strings, got: {raw}")
        choices = tuple(raw)
        if len(choices) == 0:
            raise ValueError(f"{name} must not be empty")
        if not all(isinstance(item, str) for item in choices):
            raise TypeError(f"{name} must contain only strings, got: {choices}")
        return choices

    def _is_enabled(self, flag: bool) -> bool:
        return bool(self.domain_randomization and flag)

    def _ensure_randomization_info(self):
        if len(self._parallel_randomization_info) != self.num_envs:
            self._parallel_randomization_info = [{} for _ in range(self.num_envs)]

    def _set_randomization_value(self, scene_idx: int, key: str, value: Any):
        self._ensure_randomization_info()
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy().tolist()
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        self._parallel_randomization_info[scene_idx][key] = value

    def get_parallel_randomization_info(self) -> list[dict[str, Any]]:
        self._ensure_randomization_info()
        return [dict(item) for item in self._parallel_randomization_info]

    def _cache_joint_drive_defaults(self):
        self._joint_drive_defaults = {}
        for joint in self.agent.robot.active_joints:
            defaults = []
            for obj in joint._objs:
                defaults.append(
                    (
                        float(obj.stiffness),
                        float(obj.damping),
                        float(obj.force_limit),
                        str(obj.drive_mode),
                        float(obj.friction),
                    )
                )
            self._joint_drive_defaults[joint.name] = defaults

    def _randomize_joint_drive(self, env_idx: torch.Tensor):
        if len(self._joint_drive_defaults) == 0:
            self._cache_joint_drive_defaults()

        for scene_idx_t in env_idx:
            scene_idx = int(scene_idx_t.item())
            rng = self._batched_episode_rng[scene_idx]

            stiffness_scale = (
                float(rng.uniform(*self.joint_stiffness_scale_range))
                if self._is_enabled(self.randomize_joint_stiffness)
                else 1.0
            )
            damping_scale = (
                float(rng.uniform(*self.joint_damping_scale_range))
                if self._is_enabled(self.randomize_joint_damping)
                else 1.0
            )
            force_limit_scale = (
                float(rng.uniform(*self.joint_force_limit_scale_range))
                if self._is_enabled(self.randomize_joint_force_limit)
                else 1.0
            )
            friction_abs = (
                float(rng.uniform(*self.joint_friction_abs_range))
                if self._is_enabled(self.randomize_joint_friction)
                else None
            )
            drive_mode = (
                str(rng.choice(self.joint_drive_mode_choices))
                if self._is_enabled(self.randomize_joint_drive_mode)
                else None
            )

            if self._is_enabled(self.randomize_joint_stiffness):
                self._set_randomization_value(scene_idx, "joint.stiffness_scale", stiffness_scale)
            if self._is_enabled(self.randomize_joint_damping):
                self._set_randomization_value(scene_idx, "joint.damping_scale", damping_scale)
            if self._is_enabled(self.randomize_joint_force_limit):
                self._set_randomization_value(scene_idx, "joint.force_limit_scale", force_limit_scale)
            if self._is_enabled(self.randomize_joint_friction) and friction_abs is not None:
                self._set_randomization_value(scene_idx, "joint.friction_abs", friction_abs)
            if self._is_enabled(self.randomize_joint_drive_mode) and drive_mode is not None:
                self._set_randomization_value(scene_idx, "joint.drive_mode", drive_mode)

            for joint in self.agent.robot.active_joints:
                defaults = self._joint_drive_defaults.get(joint.name)
                if defaults is None or scene_idx >= len(defaults):
                    continue
                (
                    base_stiffness,
                    base_damping,
                    base_force_limit,
                    base_mode,
                    base_friction,
                ) = defaults[scene_idx]

                stiffness = max(base_stiffness * stiffness_scale, 1e-6)
                damping = max(base_damping * damping_scale, 1e-6)
                force_limit = max(base_force_limit * force_limit_scale, 1e-6)
                friction = base_friction if friction_abs is None else max(friction_abs, 0.0)
                mode = drive_mode if drive_mode is not None else base_mode

                joint._objs[scene_idx].set_drive_properties(
                    stiffness,
                    damping,
                    force_limit,
                    mode,
                )
                joint._objs[scene_idx].friction = friction

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
        self._cache_joint_drive_defaults()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.table_scene.initialize(env_idx)
        self._apply_hand_camera_pose_constant()
        if self._is_enabled(self.randomize_joint_control):
            self._randomize_joint_drive(env_idx)

    def sync_gpu_articulation_state(self):
        if self.device.type != "cuda":
            return
        self.scene._gpu_apply_all()
        self.scene.px.gpu_update_articulation_kinematics()
        self.scene._gpu_fetch_all()

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

        if self.synchronize_render and self.backend.render_device.is_cuda():
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
            background = background.expand(rgb.shape[0], -1, -1, -1)
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
