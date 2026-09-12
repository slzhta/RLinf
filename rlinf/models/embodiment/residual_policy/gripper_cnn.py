# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Independent gripper head shared by offline BC and online PPO."""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GripperCNNConfig:
    """Observation contract saved with the BC weights."""

    image_num: int = 2
    image_size: int = 96
    state_dim: int = 14
    hidden_dim: int = 128
    output_mode: str = "bernoulli"


class GripperCNN(nn.Module):
    """Predict a binary logit or a scalar pre-tanh mean from RGB and state.

    Images are NHWC RGB uint8 (or floats in [0, 255]) in both BC and PPO.
    There is no OpenPI input, Gaussian noise, dropout or running batch statistic.
    Positive commands mean open (+1); negative commands mean close (-1).
    """

    def __init__(self, cfg: GripperCNNConfig) -> None:
        super().__init__()
        if cfg.image_num < 1 or cfg.image_size < 16 or cfg.state_dim < 0:
            raise ValueError("Invalid gripper image/state dimensions.")
        if cfg.hidden_dim < 1:
            raise ValueError("gripper hidden_dim must be positive.")
        if cfg.output_mode not in ("bernoulli", "scalar"):
            raise ValueError("gripper output_mode must be bernoulli or scalar.")
        self.cfg = cfg
        layers = []
        channels = 3
        for width in (16, 32, 64):
            layers.extend(
                [
                    nn.Conv2d(channels, width, 3, stride=2, padding=1),
                    nn.GroupNorm(4, width),
                    nn.SiLU(),
                ]
            )
            channels = width
        layers.extend([nn.AdaptiveAvgPool2d((3, 3)), nn.Flatten()])
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(64 * 9 * cfg.image_num + cfg.state_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return [B, 1] logits or pre-tanh means using the saved input contract."""
        main = obs["main_images"]
        if main.ndim != 4 or main.shape[-1] != 3:
            raise ValueError("Gripper main_images must be NHWC RGB.")
        views = main.unsqueeze(1)
        if self.cfg.image_num > 1:
            extra = obs.get("extra_view_images")
            if extra is None or extra.shape != (
                main.shape[0],
                self.cfg.image_num - 1,
                *main.shape[1:],
            ):
                raise ValueError("Missing or mismatched gripper camera views.")
            views = torch.cat((views, extra), dim=1)
        x = views.flatten(0, 1).permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(
            x,
            (self.cfg.image_size, self.cfg.image_size),
            mode="bilinear",
            align_corners=False,
        )
        features = self.encoder(x).reshape(main.shape[0], -1)
        if self.cfg.state_dim:
            states = obs.get("states")
            if states is None or states.shape != (main.shape[0], self.cfg.state_dim):
                raise ValueError("Gripper states must match the BC state dimension.")
            features = torch.cat((features, states.float()), dim=-1)
        return self.head(features)

    def save_bc(self, path: str | Path, **metadata: object) -> None:
        """Save portable weights and the exact BC input/action convention."""
        torch.save(
            {
                "format_version": 1,
                "config": asdict(self.cfg),
                "action_convention": "minus_one_close_plus_one_open",
                "state_dict": self.state_dict(),
                "metadata": metadata,
            },
            path,
        )

    def load_bc(self, path: str | Path) -> None:
        """Reject incompatible BC checkpoints instead of silently reinitializing."""
        checkpoint = torch.load(
            Path(path).expanduser(), map_location="cpu", weights_only=True
        )
        saved_config = dict(checkpoint.get("config", {}))
        saved_config.setdefault("output_mode", "bernoulli")
        if (
            checkpoint.get("format_version") != 1
            or saved_config != asdict(self.cfg)
            or checkpoint.get("action_convention") != "minus_one_close_plus_one_open"
        ):
            raise ValueError("Gripper BC checkpoint configuration/convention mismatch.")
        self.load_state_dict(checkpoint["state_dict"], strict=True)
