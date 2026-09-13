# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Separate RGB/state/action features with an optional Bernoulli gripper head."""

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from torch import nn
from torch.nn import functional as F

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.cnn_policy.cnn_policy import CNNPolicy
from rlinf.models.embodiment.modules.utils import init_mlp_weights, make_mlp
from rlinf.models.embodiment.residual_policy.residual_policy import (
    ResidualConfig,
    ResidualPolicy,
)


@dataclass
class SharedGripperConfig(ResidualConfig):
    gripper_mode: str = "cnn"
    gripper_architecture: str = "shared"
    gripper_share_features: bool = True
    action_latent_dim: int = 64
    gripper_hidden_dim: int = 128
    shared_feature_checkpoint: str | None = None
    gripper_checkpoint: str | None = None
    gripper: dict[str, Any] = field(default_factory=dict)


class SharedGripperPolicy(CNNPolicy):
    """Use native CNN PPO statistics for six residuals and an optional gripper."""

    def __init__(self, cfg: SharedGripperConfig) -> None:
        if cfg.gripper_mode not in ("cnn", "none"):
            raise ValueError("Shared feature policy supports cnn or none gripper mode.")
        self.has_gripper = cfg.gripper_mode == "cnn"
        if not self.has_gripper and (
            not cfg.gripper_share_features
            or cfg.gripper_checkpoint
            or cfg.shared_feature_checkpoint
        ):
            raise ValueError(
                "No-gripper policy does not load gripper BC or separate gripper features."
            )
        if not isinstance(cfg.gripper_share_features, bool):
            raise ValueError("gripper_share_features must be a boolean.")
        if (
            cfg.action_dim != (7 if self.has_gripper else 6)
            or list(cfg.residual_action_indices) != list(range(6))
            or cfg.num_action_chunks != 1
            or not cfg.add_value_head
            or cfg.add_q_head
            or not cfg.use_state
            or not cfg.independent_std
            or cfg.action_scale is not None
            or cfg.binary_action_indices
            or cfg.encoder_input_size != 128
            or min(
                cfg.state_dim,
                cfg.base_horizon,
                cfg.action_latent_dim,
                cfg.gripper_hidden_dim,
            )
            < 1
        ):
            raise ValueError(
                "Shared gripper requires 128px CNN PPO, state input, six arm "
                "residuals, one action chunk, a value head and independent std. "
                "Leave binary_action_indices empty; gripper_mode selects the binary head."
            )
        if cfg.gripper:
            raise ValueError(
                "Shared gripper uses gripper_hidden_dim, not the independent "
                "gripper configuration. Use gripper_architecture=independent "
                "for existing scalar/Bernoulli CNN checkpoints."
            )
        if cfg.encoder_config.get("dropout", 0.0) != 0.0:
            raise ValueError("Shared gripper requires encoder dropout=0 for PPO.")
        cnn_cfg = deepcopy(cfg)
        cnn_cfg.binary_action_indices = [6] if self.has_gripper else []
        cnn_cfg.encoder_config.setdefault("dropout", 0.0)
        cnn_cfg.encoder_config.setdefault("freeze_backbone", False)
        super().__init__(cnn_cfg)
        self.residual_cfg = deepcopy(cfg)
        feature_dim = (
            sum(encoder.out_dim for encoder in self.encoders) + cfg.state_latent_dim
        )
        self.action_proj = nn.Sequential(
            *make_mlp(
                in_channels=cfg.base_horizon * 7,
                mlp_channels=[cfg.action_latent_dim],
                act_builder=nn.Tanh,
                last_act=True,
                use_layer_norm=True,
            )
        )
        self.mix_proj = nn.Sequential(
            *make_mlp(
                in_channels=feature_dim + cfg.action_latent_dim,
                mlp_channels=[256, 256],
                act_builder=nn.Tanh,
                last_act=True,
                use_layer_norm=True,
            )
        )
        self.actor_mean = nn.Linear(256, 6)
        nn.init.zeros_(self.actor_mean.weight)
        nn.init.zeros_(self.actor_mean.bias)
        if self.has_gripper:
            self.gripper_head = nn.Sequential(
                nn.Linear(feature_dim, cfg.gripper_hidden_dim),
                nn.Tanh(),
                nn.Linear(cfg.gripper_hidden_dim, 1),
            )
        modules = [self.action_proj, self.mix_proj]
        if self.has_gripper:
            modules.append(self.gripper_head)
        for module in modules:
            init_mlp_weights(module, nonlinearity="tanh")
        if not cfg.gripper_share_features:
            self.gripper_encoders = deepcopy(self.encoders)
            self.gripper_state_proj = deepcopy(self.state_proj)
        self._shared_feature_id: str | None = None
        if cfg.shared_feature_checkpoint:
            self.load_shared_features(cfg.shared_feature_checkpoint)
        if cfg.gripper_checkpoint:
            self.load_gripper_bc(cfg.gripper_checkpoint)

    def _prepare_cnn_obs(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        base = obs["base_actions"].clone()
        if self.has_gripper:
            base[..., 6] = 0
        return ResidualPolicy._prepare_cnn_obs(self, {**obs, "base_actions": base})

    def _actor_forward_from_processed_tensors(
        self,
        main_images: torch.Tensor,
        states: torch.Tensor | None,
        extra_view_images: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        horizon = self.residual_cfg.base_horizon
        action_dim = self.residual_cfg.action_dim
        plan_end = horizon * action_dim
        condition_end = plan_end + horizon
        if (
            states is None
            or states.ndim != 2
            or states.shape[1] != condition_end + self.cfg.state_dim
        ):
            raise ValueError("PPO replay requires the saved base plan, mask and state.")
        plan = states[:, :plan_end].reshape(-1, horizon, action_dim)
        mask = states[:, plan_end:condition_end]
        action_input = torch.cat(
            ((plan[..., :6] * mask.unsqueeze(-1)).flatten(1), mask), dim=1
        )
        shared, _ = self._get_feature_from_processed_tensors(
            main_images, states[:, condition_end:], extra_view_images
        )
        full_feature = torch.cat((shared, self.action_proj(action_input)), dim=1)
        mix_feature = self.mix_proj(full_feature)
        arm_mean = self.actor_mean(mix_feature)
        action_mean = arm_mean
        if self.has_gripper:
            gripper_feature = (
                shared
                if self.cfg.gripper_share_features
                else self._get_gripper_feature(
                    main_images, states[:, condition_end:], extra_view_images
                )
            )
            action_mean = torch.cat(
                (arm_mean, self.gripper_head(gripper_feature)), dim=-1
            )
        action_logstd = self.actor_logstd.expand_as(action_mean)
        return full_feature, mix_feature, action_mean, action_logstd

    def forward(
        self, forward_type: ForwardType = ForwardType.DEFAULT, **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        if forward_type != ForwardType.DEFAULT:
            raise NotImplementedError(
                "Shared gripper supports PPO default forward only."
            )
        return self.default_forward(**kwargs)

    def default_forward(
        self, forward_inputs: dict[str, torch.Tensor], **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        result = super().default_forward(forward_inputs, **kwargs)
        if "entropy" in result:
            result["entropy"] = result["entropy"].reshape(-1, 1, self.cfg.action_dim)
        return result

    @torch.no_grad()
    def predict_action_batch(
        self, env_obs: dict[str, torch.Tensor], **kwargs: Any
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if kwargs.get("return_obs", True) is False:
            raise ValueError(
                "Shared gripper requires saved observations for PPO replay."
            )
        if not self.residual_cfg.enabled and kwargs.get("mode", "train") != "eval":
            raise ValueError("Disable residual only for evaluation.")
        actions, result = super().predict_action_batch(
            self._prepare_cnn_obs(env_obs), **kwargs
        )
        result["forward_inputs"] = {
            key: value.detach().clone()
            for key, value in result["forward_inputs"].items()
        }
        if not self.residual_cfg.enabled:
            actions = actions.clone()
            actions[..., :6] = 0
        return actions, result

    def _get_gripper_feature(
        self,
        main_images: torch.Tensor,
        states: torch.Tensor,
        extra_view_images: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.cfg.gripper_share_features:
            feature, _ = self._get_feature_from_processed_tensors(
                main_images, states, extra_view_images
            )
            return feature
        features = []
        for i, encoder in enumerate(self.gripper_encoders):
            if i > 0 and extra_view_images is None:
                raise ValueError(
                    "Independent gripper features require all camera views."
                )
            images = main_images if i == 0 else extra_view_images[:, i - 1]
            if images.shape[3] == 3:
                images = images.permute(0, 3, 1, 2)
            size = (self.cfg.encoder_input_size,) * 2
            if images.shape[-2:] != size:
                images = F.interpolate(
                    images, size=size, mode="bilinear", align_corners=False
                )
            features.append(encoder(images))
        return torch.cat((*features, self.gripper_state_proj(states)), dim=-1)

    def gripper_bc_logits(self, env_obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run only shared RGB/state features and the head; no OpenPI is needed."""
        if not self.has_gripper:
            raise ValueError("No-gripper policy has no gripper BC head.")
        if (
            env_obs["states"].ndim != 2
            or env_obs["states"].shape[1] != self.cfg.state_dim
        ):
            raise ValueError(
                "BC requires original proprioception, not augmented state."
            )
        obs = self.preprocess_env_obs(env_obs)
        shared = self._get_gripper_feature(
            obs["main_images"], obs["states"], obs.get("extra_view_images")
        )
        return self.gripper_head(shared)

    def set_gripper_bc_mode(self, train_shared_features: bool = False) -> None:
        """Train the gripper, optionally including its RGB/state frontend."""
        if not self.has_gripper:
            raise ValueError("No-gripper policy has no gripper BC head.")
        self._shared_feature_id = (
            None if train_shared_features else self._shared_feature_id
        )
        self.requires_grad_(False)
        self.eval()
        if train_shared_features:
            for encoder in self._feature_modules()["encoders"]:
                encoder.freeze_backbone = False
            for module in self._feature_modules().values():
                module.requires_grad_(True)
                module.train()
        self.gripper_head.requires_grad_(True)
        self.gripper_head.train()

    def set_rl_mode(self) -> None:
        """Unfreeze both policy heads and their shared feature extractors."""
        self._shared_feature_id = None
        self.requires_grad_(True)
        for encoder in self.encoders:
            encoder.freeze_backbone = False
        for encoder in self._feature_modules()["encoders"]:
            encoder.freeze_backbone = False
        self.train()

    def _feature_contract(self) -> dict[str, Any]:
        return {
            "image_num": self.cfg.image_num,
            "encoder_input_size": self.cfg.encoder_input_size,
            "state_dim": self.cfg.state_dim,
            "state_latent_dim": self.cfg.state_latent_dim,
            "normalization": "imagenet_rgb",
            "backbone": self.cfg.backbone,
        }

    def _feature_modules(self) -> dict[str, nn.Module]:
        if not self.cfg.gripper_share_features:
            return {
                "encoders": self.gripper_encoders,
                "state_proj": self.gripper_state_proj,
            }
        return {"encoders": self.encoders, "state_proj": self.state_proj}

    def save_shared_features(self, path: str | Path) -> None:
        """Save the exact BC frontend, including randomly initialized projections."""
        self._shared_feature_id = uuid4().hex
        torch.save(
            {
                "format": "shared_gripper_features_v1",
                "feature_id": self._shared_feature_id,
                "config": self._feature_contract(),
                "state_dict": {
                    name: module.state_dict()
                    for name, module in self._feature_modules().items()
                },
            },
            Path(path).expanduser(),
        )

    def save_gripper_bc_pair(
        self, feature_path: str | Path, head_path: str | Path
    ) -> None:
        """Snapshot both trained BC parts without exporting the residual branch."""
        if (
            Path(feature_path).expanduser().resolve()
            == Path(head_path).expanduser().resolve()
        ):
            raise ValueError("Feature and head checkpoints must use different paths.")
        parameters = [
            p
            for module in self._feature_modules().values()
            for p in module.parameters()
        ]
        original_flags = [p.requires_grad for p in parameters]
        try:
            for p in parameters:
                p.requires_grad_(False)
            self.save_shared_features(feature_path)
            self.save_gripper_bc(head_path)
        finally:
            for p, requires_grad in zip(parameters, original_flags):
                p.requires_grad_(requires_grad)
            if any(original_flags):
                self._shared_feature_id = None

    def load_shared_features(self, path: str | Path) -> None:
        checkpoint = torch.load(
            Path(path).expanduser(), map_location="cpu", weights_only=True
        )
        if (
            checkpoint.get("format") != "shared_gripper_features_v1"
            or checkpoint.get("config") != self._feature_contract()
            or not checkpoint.get("feature_id")
        ):
            raise ValueError("Shared feature checkpoint format/configuration mismatch.")
        for name, module in self._feature_modules().items():
            module.load_state_dict(checkpoint["state_dict"][name], strict=True)
        if not self.cfg.gripper_share_features:
            self.encoders.load_state_dict(
                checkpoint["state_dict"]["encoders"], strict=True
            )
            self.state_proj.load_state_dict(
                checkpoint["state_dict"]["state_proj"], strict=True
            )
        self._shared_feature_id = checkpoint["feature_id"]

    def save_gripper_bc(self, path: str | Path) -> None:
        """Export only the head, tied to an explicitly saved frontend."""
        if self._shared_feature_id is None:
            raise ValueError("Save/load shared features before saving the BC head.")
        if any(
            p.requires_grad
            for module in self._feature_modules().values()
            for p in module.parameters()
        ):
            raise ValueError(
                "Freeze shared features with set_gripper_bc_mode before BC."
            )
        torch.save(
            {
                "format": "shared_gripper_head_v1",
                "feature_id": self._shared_feature_id,
                "config": self._feature_contract(),
                "hidden_dim": self.residual_cfg.gripper_hidden_dim,
                "action_convention": "minus_one_close_plus_one_open",
                "state_dict": self.gripper_head.state_dict(),
            },
            Path(path).expanduser(),
        )

    def load_gripper_bc(self, path: str | Path) -> None:
        checkpoint = torch.load(
            Path(path).expanduser(), map_location="cpu", weights_only=True
        )
        if (
            checkpoint.get("format") != "shared_gripper_head_v1"
            or self._shared_feature_id is None
            or checkpoint.get("feature_id") != self._shared_feature_id
            or checkpoint.get("config") != self._feature_contract()
            or checkpoint.get("hidden_dim") != self.residual_cfg.gripper_hidden_dim
            or checkpoint.get("action_convention") != "minus_one_close_plus_one_open"
        ):
            raise ValueError(
                "BC head requires matching shared_feature_checkpoint; independent "
                "CNN gripper checkpoints cannot initialize the shared architecture."
            )
        self.gripper_head.load_state_dict(checkpoint["state_dict"], strict=True)
