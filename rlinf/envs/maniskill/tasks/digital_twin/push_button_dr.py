from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml

from mani_skill.utils.registration import register_env

from rlinf.envs.maniskill.tasks.digital_twin.push_button import PushButtonEnv


@register_env("PushButtonDomainRandomized-v1", max_episode_steps=100)
class PushButtonDomainRandomizedEnv(PushButtonEnv):
    """Push-button task with YAML-driven lighting and joint-control randomization."""

    def __init__(
        self,
        *args,
        dr_yaml_path: str | Path | None = None,
        domain_randomization: bool | None = None,
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
        dr_cfg, resolved_yaml_path = self._load_domain_randomization_config(dr_yaml_path)
        self.domain_randomization_yaml_path = (
            str(resolved_yaml_path) if resolved_yaml_path is not None else None
        )

        self.domain_randomization = self._resolve_master_toggle(
            explicit_value=domain_randomization,
            config=dr_cfg,
        )
        self.randomize_lighting = self._resolve_bool(
            key="randomize_lighting",
            explicit_value=randomize_lighting,
            config=dr_cfg,
            default=False,
        )
        self.randomize_joint_control = self._resolve_bool(
            key="randomize_joint_control",
            explicit_value=randomize_joint_control,
            config=dr_cfg,
            default=False,
        )
        self.randomize_joint_stiffness = self._resolve_bool(
            key="randomize_joint_stiffness",
            explicit_value=randomize_joint_stiffness,
            config=dr_cfg,
            default=False,
        )
        self.randomize_joint_damping = self._resolve_bool(
            key="randomize_joint_damping",
            explicit_value=randomize_joint_damping,
            config=dr_cfg,
            default=False,
        )
        self.randomize_joint_force_limit = self._resolve_bool(
            key="randomize_joint_force_limit",
            explicit_value=randomize_joint_force_limit,
            config=dr_cfg,
            default=False,
        )
        self.randomize_joint_friction = self._resolve_bool(
            key="randomize_joint_friction",
            explicit_value=randomize_joint_friction,
            config=dr_cfg,
            default=False,
        )
        self.randomize_joint_drive_mode = self._resolve_bool(
            key="randomize_joint_drive_mode",
            explicit_value=randomize_joint_drive_mode,
            config=dr_cfg,
            default=False,
        )

        self.joint_stiffness_scale_range = self._resolve_range(
            key="joint_stiffness_scale_range",
            explicit_value=joint_stiffness_scale_range,
            config=dr_cfg,
            default=(0.7, 1.3),
        )
        self.joint_damping_scale_range = self._resolve_range(
            key="joint_damping_scale_range",
            explicit_value=joint_damping_scale_range,
            config=dr_cfg,
            default=(0.7, 1.3),
        )
        self.joint_force_limit_scale_range = self._resolve_range(
            key="joint_force_limit_scale_range",
            explicit_value=joint_force_limit_scale_range,
            config=dr_cfg,
            default=(0.8, 1.2),
        )
        self.joint_friction_abs_range = self._resolve_range(
            key="joint_friction_abs_range",
            explicit_value=joint_friction_abs_range,
            config=dr_cfg,
            default=(0.0, 0.3),
        )
        self.joint_drive_mode_choices = self._resolve_choices(
            key="joint_drive_mode_choices",
            explicit_value=joint_drive_mode_choices,
            config=dr_cfg,
            default=("force", "acceleration"),
        )

        self._joint_drive_defaults: dict[
            str, list[tuple[float, float, float, str, float]]
        ] = {}
        self._parallel_randomization_info: list[dict[str, Any]] = []

        super().__init__(*args, **kwargs)

    @staticmethod
    def _require_bool(name: str, value: Any) -> bool:
        if not isinstance(value, bool):
            raise TypeError(
                f"{name} must be a bool (true/false in yaml), but got {type(value).__name__}: {value}"
            )
        return value

    def _resolve_master_toggle(
        self, explicit_value: bool | None, config: dict[str, Any]
    ) -> bool:
        if explicit_value is not None:
            return self._require_bool("domain_randomization", explicit_value)
        if "enabled" in config:
            return self._require_bool("enabled", config["enabled"])
        if "domain_randomization" in config:
            return self._require_bool("domain_randomization", config["domain_randomization"])

        randomize_lighting = self._require_bool(
            "randomize_lighting",
            config.get("randomize_lighting", False),
        )
        randomize_joint_control = self._require_bool(
            "randomize_joint_control",
            config.get("randomize_joint_control", False),
        )
        return randomize_lighting or randomize_joint_control

    def _resolve_bool(
        self,
        key: str,
        explicit_value: bool | None,
        config: dict[str, Any],
        default: bool,
    ) -> bool:
        if explicit_value is not None:
            return self._require_bool(key, explicit_value)
        return self._require_bool(key, config.get(key, default))

    def _resolve_range(
        self,
        key: str,
        explicit_value: Sequence[float] | None,
        config: dict[str, Any],
        default: tuple[float, float],
    ) -> tuple[float, float]:
        raw: Any = explicit_value if explicit_value is not None else config.get(key, default)
        if isinstance(raw, (str, bytes)):
            raise TypeError(f"{key} must be a sequence like [min, max], but got: {raw}")
        if not isinstance(raw, Sequence) or len(raw) != 2:
            raise ValueError(f"{key} expects exactly 2 values, but got: {raw}")
        return float(raw[0]), float(raw[1])

    def _resolve_choices(
        self,
        key: str,
        explicit_value: Sequence[str] | None,
        config: dict[str, Any],
        default: tuple[str, ...],
    ) -> tuple[str, ...]:
        raw: Any = explicit_value if explicit_value is not None else config.get(key, default)
        if isinstance(raw, (str, bytes)):
            raise TypeError(f"{key} must be a sequence of strings, but got: {raw}")
        if not isinstance(raw, Sequence):
            raise TypeError(f"{key} must be a sequence of strings, but got: {raw}")
        choices = tuple(raw)
        if len(choices) == 0:
            raise ValueError(f"{key} must not be empty")
        if not all(isinstance(item, str) for item in choices):
            raise TypeError(f"{key} must contain only strings, but got: {choices}")
        return choices

    def _resolve_domain_randomization_yaml_path(
        self, dr_yaml_path: str | Path | None
    ) -> Path:
        if dr_yaml_path is not None:
            path = Path(dr_yaml_path).expanduser()
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
            if not path.is_file():
                raise FileNotFoundError(
                    f"Domain randomization yaml not found: {path}"
                )
            return path.resolve()

        default_path = (
            Path(__file__).resolve().parent
            / "config"
            / "domain_randomization.yaml"
        )
        if not default_path.is_file():
            raise FileNotFoundError(
                f"Domain randomization yaml not found: {default_path}"
            )
        return default_path.resolve()

    def _load_domain_randomization_config(
        self, dr_yaml_path: str | Path | None
    ) -> tuple[dict[str, Any], Path]:
        resolved_path = self._resolve_domain_randomization_yaml_path(dr_yaml_path)

        with resolved_path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(
                f"Expected mapping at the top level of yaml file: {resolved_path}"
            )

        if "domain_randomization" not in loaded:
            raise KeyError(
                f"Missing required top-level key 'domain_randomization' in yaml: {resolved_path}"
            )
        dr_config = loaded["domain_randomization"]
        if not isinstance(dr_config, dict):
            raise ValueError(
                "Field 'domain_randomization' in yaml must be a mapping if provided"
            )
        return dict(dr_config), resolved_path

    def _is_enabled(self, flag: bool) -> bool:
        return bool(self.domain_randomization and flag)

    def _ensure_randomization_info(self):
        if len(self._parallel_randomization_info) != self.num_envs:
            self._parallel_randomization_info = [dict() for _ in range(self.num_envs)]

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
                self._set_randomization_value(
                    scene_idx,
                    "joint.stiffness_scale",
                    stiffness_scale,
                )
            if self._is_enabled(self.randomize_joint_damping):
                self._set_randomization_value(
                    scene_idx,
                    "joint.damping_scale",
                    damping_scale,
                )
            if self._is_enabled(self.randomize_joint_force_limit):
                self._set_randomization_value(
                    scene_idx,
                    "joint.force_limit_scale",
                    force_limit_scale,
                )
            if self._is_enabled(self.randomize_joint_friction) and friction_abs is not None:
                self._set_randomization_value(
                    scene_idx,
                    "joint.friction_abs",
                    friction_abs,
                )
            if self._is_enabled(self.randomize_joint_drive_mode) and drive_mode is not None:
                self._set_randomization_value(
                    scene_idx,
                    "joint.drive_mode",
                    drive_mode,
                )

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

    def _after_reconfigure(self, options):
        super()._after_reconfigure(options)
        self._cache_joint_drive_defaults()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        if self._is_enabled(self.randomize_joint_control):
            self._randomize_joint_drive(env_idx)

    def _load_lighting(self, options: dict):
        if not self._is_enabled(self.randomize_lighting):
            return super()._load_lighting(options)

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