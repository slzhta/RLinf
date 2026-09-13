# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf


def get_model(
    cfg: DictConfig, torch_dtype: torch.dtype = torch.float32
) -> torch.nn.Module:
    """Build the small PPO policy; the frozen base belongs to rollout."""
    from rlinf.models.embodiment.residual_policy.residual_policy import (
        ResidualConfig,
        ResidualPolicy,
    )

    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if cfg.get("model_path"):
        cfg.model_path = str(Path(cfg.model_path).expanduser())
    mode = cfg.get("gripper_mode", "cnn")
    if mode in ("cnn", "none"):
        architecture = cfg.get("gripper_architecture", "shared")
        if mode == "none" and architecture != "shared":
            raise ValueError(
                "No-gripper residual requires the shared feature architecture."
            )
        if architecture == "shared":
            from rlinf.models.embodiment.residual_policy.shared_gripper_policy import (
                SharedGripperConfig,
                SharedGripperPolicy,
            )

            model_config = SharedGripperConfig()
            model_config.update_from_dict(OmegaConf.to_container(cfg, resolve=True))
            return SharedGripperPolicy(model_config)
        if architecture != "independent":
            raise ValueError(f"Unsupported gripper_architecture: {architecture}")
        from rlinf.models.embodiment.residual_policy.split_gripper_policy import (
            SplitGripperConfig,
            SplitGripperPolicy,
        )

        model_config = SplitGripperConfig()
        model_config.update_from_dict(OmegaConf.to_container(cfg, resolve=True))
        return SplitGripperPolicy(model_config)
    if mode != "residual":
        raise ValueError(f"Unsupported gripper_mode: {mode}")
    model_config = ResidualConfig()
    model_config.update_from_dict(OmegaConf.to_container(cfg, resolve=True))
    return ResidualPolicy(model_config)
